use anyhow::{bail, ensure, Result};
use byteorder::{BigEndian, ByteOrder};
use bytes::{Bytes, BytesMut};
use std::io::Read;
use tokio::io::AsyncReadExt;

pub const ACTIVATION_MAGIC: [u8; 4] = *b"AERO";
pub const TOKEN_RESP_MAGIC: [u8; 4] = *b"ATOK";
pub const HANDSHAKE_REQ_MAGIC: [u8; 4] = *b"AHSK";
pub const HANDSHAKE_RESP_MAGIC: [u8; 4] = *b"AHSR";
pub const PROTOCOL_VERSION: u16 = 4;

// Hard Resource Ceilings to Prevent Unbounded Network Memory Allocations & DoS
pub const MAX_ACTIVATION_PAYLOAD_BYTES: u32 = 128 * 1024 * 1024; // 128 MB max per activation frame
pub const MAX_SEQUENCE_LENGTH: u32 = 131_072; // 128k max sequence length
pub const MAX_HIDDEN_DIM: u32 = 65_536; // 64k max hidden dimension
pub const MAX_TOKEN_TEXT_BYTES: u32 = 64 * 1024; // 64 KB max token text piece
pub const MAX_HANDSHAKE_STR_LEN: usize = 512; // 512 B max architecture or checksum string
pub const MAX_HANDSHAKE_ERR_LEN: usize = 8192; // 8 KB max error message length

// Flags for ActivationHeader
pub const FLAG_CLEAR_KV: u8 = 0x01;
pub const FLAG_IS_PROMPT: u8 = 0x02;
pub const FLAG_EOS_SIGNAL: u8 = 0x04;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationDtype {
    RawF32 = 0,
    Fp16 = 1,
    Bf16 = 2,
    Int8PerRow = 3,
    Fp8E4M3 = 4,
}

impl ActivationDtype {
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(Self::RawF32),
            1 => Ok(Self::Fp16),
            2 => Ok(Self::Bf16),
            3 => Ok(Self::Int8PerRow),
            4 => Ok(Self::Fp8E4M3),
            _ => bail!("Unsupported activation dtype: {}", val),
        }
    }

    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::RawF32 => 4,
            Self::Fp16 | Self::Bf16 => 2,
            Self::Int8PerRow | Self::Fp8E4M3 => 1,
        }
    }
}

/// Binary frame header for high-speed P2P activation vector streaming with multi-stage session & KV tracking.
///
/// Fixed size: 44 bytes (explicit BigEndian serialization, independent of compiler padding).
///
/// # Sampling configuration
///
/// Per-request `temperature` / `top_p` are **not** carried on the wire: the
/// 44-byte header is fully allocated (offsets 40..44 hold `payload_bytes`, which
/// is required to frame the payload). In distributed mode the terminal stage
/// samples with the parameters baked in when it loaded the model; in the
/// single-node path (`start --role all` / no peers) the coordinator applies the
/// request's values directly. Extending the frame with a trailer is tracked as
/// future work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationHeader {
    pub magic: [u8; 4],       // b"AERO"
    pub version: u16,         // 4
    pub session_id: u64,      // Unique multi-turn chat session ID
    pub sequence_id: u64,     // Token generation sequence counter
    pub sequence_length: u32, // S: Number of tokens in sequence (1 for decode, >1 for prefill)
    pub token_position: u32,  // Context position offset in KV-cache (pos)
    pub stage_index: u8,      // Current stage index in cluster (0 .. total_stages - 1)
    pub total_stages: u8,     // Total stages in pipeline (N >= 2)
    pub layer_index: u16,     // Boundary layer index (e.g. 24)
    pub hidden_dim: u32,      // Hidden dimension (e.g. 1536 / 4096 / 5120)
    pub dtype: u8,            // 0 = RawF32, 1 = FP16, 2 = BF16, 3 = INT8-Row, 4 = FP8-E4M3
    pub flags: u8,            // Bitmask: FLAG_CLEAR_KV (0x01), FLAG_IS_PROMPT (0x02), etc.
    pub payload_bytes: u32,   // Length of payload following header
}

impl ActivationHeader {
    pub const SIZE: usize = 44;

    /// Calculates mathematically expected payload size for given sequence length, hidden dimension, and dtype.
    /// Returns an error on integer overflow or unsupported dtype.
    pub fn calculate_expected_payload_bytes(
        sequence_length: u32,
        hidden_dim: u32,
        dtype: u8,
    ) -> Result<u32> {
        let seq_len = sequence_length as u64;
        let d = hidden_dim as u64;
        let activation_dtype = ActivationDtype::from_u8(dtype)?;

        let total_bytes: u64 = match activation_dtype {
            ActivationDtype::RawF32 => seq_len
                .checked_mul(d)
                .and_then(|elems| elems.checked_mul(4))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "RawF32 payload size overflow (seq_len={}, dim={})",
                        seq_len,
                        d
                    )
                })?,
            ActivationDtype::Fp16 | ActivationDtype::Bf16 => seq_len
                .checked_mul(d)
                .and_then(|elems| elems.checked_mul(2))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "FP16/BF16 payload size overflow (seq_len={}, dim={})",
                        seq_len,
                        d
                    )
                })?,
            ActivationDtype::Fp8E4M3 => seq_len.checked_mul(d).ok_or_else(|| {
                anyhow::anyhow!("FP8 payload size overflow (seq_len={}, dim={})", seq_len, d)
            })?,
            ActivationDtype::Int8PerRow => {
                let k_outliers: u64 = if d >= 256 { 8 } else { 0 };
                let row_bytes = d
                    .checked_add(4 + k_outliers * 6)
                    .ok_or_else(|| anyhow::anyhow!("Int8PerRow row size overflow (dim={})", d))?;
                seq_len.checked_mul(row_bytes).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Int8PerRow payload size overflow (seq_len={}, dim={})",
                        seq_len,
                        d
                    )
                })?
            }
        };

        if total_bytes > (u32::MAX as u64) {
            bail!("Calculated payload size exceeds u32::MAX: {}", total_bytes);
        }

        Ok(total_bytes as u32)
    }

    pub fn encode(&self, out: &mut [u8]) {
        assert!(out.len() >= Self::SIZE);
        out[0..4].copy_from_slice(&self.magic);
        BigEndian::write_u16(&mut out[4..6], self.version);
        BigEndian::write_u64(&mut out[6..14], self.session_id);
        BigEndian::write_u64(&mut out[14..22], self.sequence_id);
        BigEndian::write_u32(&mut out[22..26], self.sequence_length);
        BigEndian::write_u32(&mut out[26..30], self.token_position);
        out[30] = self.stage_index;
        out[31] = self.total_stages;
        BigEndian::write_u16(&mut out[32..34], self.layer_index);
        BigEndian::write_u32(&mut out[34..38], self.hidden_dim);
        out[38] = self.dtype;
        out[39] = self.flags;
        BigEndian::write_u32(&mut out[40..44], self.payload_bytes);
    }

    pub fn decode(src: &[u8]) -> Result<Self> {
        ensure!(
            src.len() >= Self::SIZE,
            "Buffer too short for ActivationHeader (got {}, expected {})",
            src.len(),
            Self::SIZE
        );
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&src[0..4]);
        ensure!(
            magic == ACTIVATION_MAGIC,
            "Invalid activation magic: {:?}",
            magic
        );

        let version = BigEndian::read_u16(&src[4..6]);
        ensure!(
            version == PROTOCOL_VERSION,
            "Unsupported activation protocol version: {}",
            version
        );

        let session_id = BigEndian::read_u64(&src[6..14]);
        let sequence_id = BigEndian::read_u64(&src[14..22]);
        let sequence_length = BigEndian::read_u32(&src[22..26]);
        let token_position = BigEndian::read_u32(&src[26..30]);
        let stage_index = src[30];
        let total_stages = src[31];
        let layer_index = BigEndian::read_u16(&src[32..34]);
        let hidden_dim = BigEndian::read_u32(&src[34..38]);
        let dtype = src[38];
        let flags = src[39];
        let payload_bytes = BigEndian::read_u32(&src[40..44]);

        ensure!(
            total_stages >= 2,
            "Activation total_stages must be >= 2 (got {})",
            total_stages
        );
        ensure!(
            stage_index < total_stages,
            "Activation stage_index ({}) out of bounds (total_stages={})",
            stage_index,
            total_stages
        );

        // Validate protocol boundaries against malicious inputs before accepting
        ensure!(
            sequence_length <= MAX_SEQUENCE_LENGTH,
            "Activation sequence_length ({}) exceeds maximum allowed ({})",
            sequence_length,
            MAX_SEQUENCE_LENGTH
        );
        ensure!(
            hidden_dim <= MAX_HIDDEN_DIM && hidden_dim > 0,
            "Activation hidden_dim ({}) is invalid (max allowed: {})",
            hidden_dim,
            MAX_HIDDEN_DIM
        );
        ensure!(
            payload_bytes <= MAX_ACTIVATION_PAYLOAD_BYTES,
            "Activation payload_bytes ({}) exceeds maximum safety ceiling ({} bytes)",
            payload_bytes,
            MAX_ACTIVATION_PAYLOAD_BYTES
        );

        // Mathematical consistency guard: payload_bytes must match tensor dimensions and dtype
        let expected_bytes =
            Self::calculate_expected_payload_bytes(sequence_length, hidden_dim, dtype)?;
        ensure!(
            payload_bytes == expected_bytes,
            "Mismatched activation payload size in header: header claims {} bytes, expected {} bytes (seq_len={}, dim={}, dtype={})",
            payload_bytes,
            expected_bytes,
            sequence_length,
            hidden_dim,
            dtype
        );

        Ok(Self {
            magic,
            version,
            session_id,
            sequence_id,
            sequence_length,
            token_position,
            stage_index,
            total_stages,
            layer_index,
            hidden_dim,
            dtype,
            flags,
            payload_bytes,
        })
    }

    #[inline]
    pub fn has_flag(&self, flag: u8) -> bool {
        (self.flags & flag) != 0
    }
}

/// Activation frame containing the 42-byte header and binary payload (raw FP32 or quantized INT8/FP8).
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationFrame {
    pub header: ActivationHeader,
    pub payload: Bytes,
}

impl ActivationFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn new_stage(
        session_id: u64,
        sequence_id: u64,
        sequence_length: u32,
        token_position: u32,
        stage_index: u8,
        total_stages: u8,
        layer_index: u16,
        hidden_dim: u32,
        dtype: ActivationDtype,
        flags: u8,
        payload: Bytes,
    ) -> Result<Self> {
        let payload_bytes = payload.len() as u32;
        let header = ActivationHeader {
            magic: ACTIVATION_MAGIC,
            version: PROTOCOL_VERSION,
            session_id,
            sequence_id,
            sequence_length,
            token_position,
            stage_index,
            total_stages,
            layer_index,
            hidden_dim,
            dtype: dtype as u8,
            flags,
            payload_bytes,
        };

        Ok(Self { header, payload })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session_id: u64,
        sequence_id: u64,
        sequence_length: u32,
        token_position: u32,
        layer_index: u16,
        hidden_dim: u32,
        dtype: ActivationDtype,
        flags: u8,
        payload: Bytes,
    ) -> Result<Self> {
        Self::new_stage(
            session_id,
            sequence_id,
            sequence_length,
            token_position,
            0,
            2,
            layer_index,
            hidden_dim,
            dtype,
            flags,
            payload,
        )
    }

    pub fn from_f32_matrix(
        session_id: u64,
        sequence_id: u64,
        sequence_length: u32,
        token_position: u32,
        layer_index: u16,
        hidden_dim: u32,
        matrix: &[f32],
    ) -> Self {
        Self::from_f32_matrix_quantized(
            session_id,
            sequence_id,
            sequence_length,
            token_position,
            layer_index,
            hidden_dim,
            matrix,
            ActivationDtype::Int8PerRow,
            0,
        )
    }

    pub fn dequantize(&self, out: &mut [f32]) -> Result<()> {
        self.dequantize_into(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_f32_matrix_quantized(
        session_id: u64,
        sequence_id: u64,
        sequence_length: u32,
        token_position: u32,
        layer_index: u16,
        hidden_dim: u32,
        matrix: &[f32],
        dtype: ActivationDtype,
        flags: u8,
    ) -> Self {
        Self::from_f32_matrix_quantized_stage(
            session_id,
            sequence_id,
            sequence_length,
            token_position,
            0,
            2,
            layer_index,
            hidden_dim,
            matrix,
            dtype,
            flags,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_f32_matrix_quantized_stage(
        session_id: u64,
        sequence_id: u64,
        sequence_length: u32,
        token_position: u32,
        stage_index: u8,
        total_stages: u8,
        layer_index: u16,
        hidden_dim: u32,
        matrix: &[f32],
        dtype: ActivationDtype,
        flags: u8,
    ) -> Self {
        let seq_len = sequence_length as usize;
        let d = hidden_dim as usize;
        assert_eq!(
            matrix.len(),
            seq_len * d,
            "Matrix size mismatch: expected {}, got {}",
            seq_len * d,
            matrix.len()
        );

        match dtype {
            ActivationDtype::Int8PerRow => {
                // Outlier-Preserving Per-Row INT8 for LLM hidden dims (d >= 256):
                // Extracts top-8 outlier channels per row as exact (u16 idx, f32 val) pairs
                // so Qwen2/DeepSeek/Llama residual outliers never crush small channel resolution.
                let k_outliers = if d >= 256 { 8usize } else { 0usize };
                let row_bytes = 4 + k_outliers * 6 + d;
                let mut payload = BytesMut::with_capacity(seq_len * row_bytes);

                for row in matrix.chunks_exact(d) {
                    let mut outlier_idx = [usize::MAX; 8];
                    if k_outliers > 0 {
                        let mut top_vals = [0.0f32; 8];
                        for (idx, &val) in row.iter().enumerate() {
                            let a = val.abs();
                            if a > top_vals[7] {
                                let mut pos = 7;
                                while pos > 0 && a > top_vals[pos - 1] {
                                    top_vals[pos] = top_vals[pos - 1];
                                    outlier_idx[pos] = outlier_idx[pos - 1];
                                    pos -= 1;
                                }
                                top_vals[pos] = a;
                                outlier_idx[pos] = idx;
                            }
                        }
                    }

                    let mut max_abs = 0.0f32;
                    for (idx, &val) in row.iter().enumerate() {
                        if k_outliers > 0 && outlier_idx.contains(&idx) {
                            continue;
                        }
                        let a = val.abs();
                        if a > max_abs {
                            max_abs = a;
                        }
                    }

                    let scale = if max_abs > 1e-12 {
                        max_abs / 127.0
                    } else {
                        1.0
                    };
                    let inv_scale = if max_abs > 1e-12 {
                        127.0 / max_abs
                    } else {
                        0.0
                    };
                    payload.extend_from_slice(&scale.to_le_bytes());

                    for &idx in outlier_idx.iter().take(k_outliers) {
                        let (u_idx, exact_val) = if idx < d {
                            (idx as u16, row[idx])
                        } else {
                            (0u16, 0.0f32)
                        };
                        payload.extend_from_slice(&u_idx.to_le_bytes());
                        payload.extend_from_slice(&exact_val.to_le_bytes());
                    }

                    let base = payload.len();
                    payload.resize(base + d, 0);
                    for (dst, &val) in payload[base..base + d].iter_mut().zip(row.iter()) {
                        let scaled = (val * inv_scale).clamp(-127.0, 127.0);
                        let q = if scaled >= 0.0 {
                            (scaled + 0.5) as i8
                        } else {
                            (scaled - 0.5) as i8
                        };
                        *dst = q as u8;
                    }
                }

                let payload = payload.freeze();
                let header = ActivationHeader {
                    magic: ACTIVATION_MAGIC,
                    version: PROTOCOL_VERSION,
                    session_id,
                    sequence_id,
                    sequence_length,
                    token_position,
                    stage_index,
                    total_stages,
                    layer_index,
                    hidden_dim,
                    dtype: ActivationDtype::Int8PerRow as u8,
                    flags,
                    payload_bytes: payload.len() as u32,
                };

                Self { header, payload }
            }
            ActivationDtype::Fp8E4M3 => {
                let mut payload = BytesMut::with_capacity(seq_len * d);
                for &val in matrix {
                    payload.extend_from_slice(&[f32_to_fp8_e4m3(val)]);
                }

                let payload = payload.freeze();
                let header = ActivationHeader {
                    magic: ACTIVATION_MAGIC,
                    version: PROTOCOL_VERSION,
                    session_id,
                    sequence_id,
                    sequence_length,
                    token_position,
                    stage_index,
                    total_stages,
                    layer_index,
                    hidden_dim,
                    dtype: ActivationDtype::Fp8E4M3 as u8,
                    flags,
                    payload_bytes: payload.len() as u32,
                };

                Self { header, payload }
            }
            ActivationDtype::Bf16 => {
                // BF16 keeps the full FP32 exponent range, so outlier channels
                // (thousands in magnitude) survive alongside small values.
                let mut payload = BytesMut::with_capacity(seq_len * d * 2);
                for &val in matrix {
                    let bits = val.to_bits();
                    let bf = if val.is_nan() {
                        0x7FC0u16
                    } else {
                        // Round-to-nearest-even on the truncated 16 mantissa bits
                        let rounding = 0x7FFFu32 + ((bits >> 16) & 1);
                        (bits.wrapping_add(rounding) >> 16) as u16
                    };
                    payload.extend_from_slice(&bf.to_le_bytes());
                }

                let payload = payload.freeze();
                let header = ActivationHeader {
                    magic: ACTIVATION_MAGIC,
                    version: PROTOCOL_VERSION,
                    session_id,
                    sequence_id,
                    sequence_length,
                    token_position,
                    stage_index,
                    total_stages,
                    layer_index,
                    hidden_dim,
                    dtype: ActivationDtype::Bf16 as u8,
                    flags,
                    payload_bytes: payload.len() as u32,
                };

                Self { header, payload }
            }
            ActivationDtype::Fp16 => {
                let mut payload = BytesMut::with_capacity(seq_len * d * 2);
                for &val in matrix {
                    let half = float_to_half(val);
                    payload.extend_from_slice(&half.to_le_bytes());
                }

                let payload = payload.freeze();
                let header = ActivationHeader {
                    magic: ACTIVATION_MAGIC,
                    version: PROTOCOL_VERSION,
                    session_id,
                    sequence_id,
                    sequence_length,
                    token_position,
                    stage_index,
                    total_stages,
                    layer_index,
                    hidden_dim,
                    dtype: ActivationDtype::Fp16 as u8,
                    flags,
                    payload_bytes: payload.len() as u32,
                };

                Self { header, payload }
            }
            _ => {
                // Raw uncompressed FP32 (endian-safe LittleEndian serialization)
                let mut payload = BytesMut::with_capacity(matrix.len() * 4);
                for &val in matrix {
                    payload.extend_from_slice(&val.to_le_bytes());
                }
                let payload = payload.freeze();
                let header = ActivationHeader {
                    magic: ACTIVATION_MAGIC,
                    version: PROTOCOL_VERSION,
                    session_id,
                    sequence_id,
                    sequence_length,
                    token_position,
                    stage_index,
                    total_stages,
                    layer_index,
                    hidden_dim,
                    dtype: ActivationDtype::RawF32 as u8,
                    flags,
                    payload_bytes: payload.len() as u32,
                };
                Self { header, payload }
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn dequantize_into(&self, out: &mut [f32]) -> Result<()> {
        let seq_len = self.header.sequence_length as usize;
        let d = self.header.hidden_dim as usize;
        let total_elements = seq_len * d;
        ensure!(
            out.len() >= total_elements,
            "Output buffer too small for dequantization (got {}, required {})",
            out.len(),
            total_elements
        );

        match ActivationDtype::from_u8(self.header.dtype)? {
            ActivationDtype::Int8PerRow => {
                let k_outliers = if d >= 256 { 8usize } else { 0usize };
                let expected_row_bytes = 4 + k_outliers * 6 + d;
                ensure!(
                    self.payload.len() == seq_len * expected_row_bytes,
                    "Invalid INT8-per-row payload length (got {}, expected {})",
                    self.payload.len(),
                    seq_len * expected_row_bytes
                );

                for (s, row_bytes) in self.payload.chunks_exact(expected_row_bytes).enumerate() {
                    let scale = f32::from_le_bytes([
                        row_bytes[0],
                        row_bytes[1],
                        row_bytes[2],
                        row_bytes[3],
                    ]);
                    let q_offset = 4 + k_outliers * 6;
                    let row_quant = &row_bytes[q_offset..q_offset + d];
                    let out_row = &mut out[s * d..(s + 1) * d];
                    for (dst, &qb) in out_row.iter_mut().zip(row_quant.iter()) {
                        *dst = (qb as i8 as f32) * scale;
                    }
                    for i in 0..k_outliers {
                        let off = 4 + i * 6;
                        let u_idx =
                            u16::from_le_bytes([row_bytes[off], row_bytes[off + 1]]) as usize;
                        let exact_val = f32::from_le_bytes([
                            row_bytes[off + 2],
                            row_bytes[off + 3],
                            row_bytes[off + 4],
                            row_bytes[off + 5],
                        ]);
                        if u_idx < d {
                            out_row[u_idx] = exact_val;
                        }
                    }
                }
            }
            ActivationDtype::Fp8E4M3 => {
                ensure!(
                    self.payload.len() == total_elements,
                    "Invalid FP8 payload length"
                );
                for i in 0..total_elements {
                    out[i] = fp8_e4m3_to_f32(self.payload[i]);
                }
            }
            ActivationDtype::RawF32 => {
                ensure!(
                    self.payload.len() == total_elements * 4,
                    "Invalid FP32 payload length"
                );
                for i in 0..total_elements {
                    let chunk: [u8; 4] = self.payload[i * 4..(i + 1) * 4].try_into()?;
                    out[i] = f32::from_le_bytes(chunk);
                }
            }
            ActivationDtype::Fp16 => {
                ensure!(
                    self.payload.len() == total_elements * 2,
                    "Invalid FP16 payload length"
                );
                for i in 0..total_elements {
                    let chunk: [u8; 2] = self.payload[i * 2..(i + 1) * 2].try_into()?;
                    let half = u16::from_le_bytes(chunk);
                    out[i] = half_to_float(half);
                }
            }
            ActivationDtype::Bf16 => {
                ensure!(
                    self.payload.len() == total_elements * 2,
                    "Invalid BF16 payload length"
                );
                for i in 0..total_elements {
                    let chunk: [u8; 2] = self.payload[i * 2..(i + 1) * 2].try_into()?;
                    let bits = (u16::from_le_bytes(chunk) as u32) << 16;
                    out[i] = f32::from_bits(bits);
                }
            }
        }
        Ok(())
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        let total_elements = (self.header.sequence_length as usize)
            .checked_mul(self.header.hidden_dim as usize)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Integer overflow calculating total f32 elements (seq_len={}, dim={})",
                    self.header.sequence_length,
                    self.header.hidden_dim
                )
            })?;
        let max_f32_elements = (MAX_ACTIVATION_PAYLOAD_BYTES as usize) / std::mem::size_of::<f32>();
        ensure!(
            total_elements <= max_f32_elements,
            "Total elements ({}) exceeds safety limit ({} elements / {} bytes)",
            total_elements,
            max_f32_elements,
            MAX_ACTIVATION_PAYLOAD_BYTES
        );
        let mut out = vec![0.0f32; total_elements];
        self.dequantize_into(&mut out)?;
        Ok(out)
    }

    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(ActivationHeader::SIZE + self.payload.len());
        let mut header_buf = [0u8; ActivationHeader::SIZE];
        self.header.encode(&mut header_buf);
        buf.extend_from_slice(&header_buf);
        buf.extend_from_slice(&self.payload);
        buf.freeze()
    }

    pub async fn decode_async<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Self> {
        let mut header_buf = [0u8; ActivationHeader::SIZE];
        reader.read_exact(&mut header_buf).await?;
        let header = ActivationHeader::decode(&header_buf)?;

        ensure!(
            header.payload_bytes <= MAX_ACTIVATION_PAYLOAD_BYTES,
            "Activation payload_bytes ({}) exceeds maximum safety ceiling ({} bytes)",
            header.payload_bytes,
            MAX_ACTIVATION_PAYLOAD_BYTES
        );

        let mut payload_buf = vec![0u8; header.payload_bytes as usize];
        reader.read_exact(&mut payload_buf).await?;

        Ok(Self {
            header,
            payload: Bytes::from(payload_buf),
        })
    }

    pub fn decode_sync<R: Read>(reader: &mut R) -> Result<Self> {
        let mut header_buf = [0u8; ActivationHeader::SIZE];
        reader.read_exact(&mut header_buf)?;
        let header = ActivationHeader::decode(&header_buf)?;

        ensure!(
            header.payload_bytes <= MAX_ACTIVATION_PAYLOAD_BYTES,
            "Activation payload_bytes ({}) exceeds maximum safety ceiling ({} bytes)",
            header.payload_bytes,
            MAX_ACTIVATION_PAYLOAD_BYTES
        );

        let mut payload_buf = vec![0u8; header.payload_bytes as usize];
        reader.read_exact(&mut payload_buf)?;

        Ok(Self {
            header,
            payload: Bytes::from(payload_buf),
        })
    }
}

// ---------------------------------------------------------------------------
// Tensor Diagnostic Metrics (L2 Norm, Mean, Head Preview)
// ---------------------------------------------------------------------------

pub fn compute_l2_norm(slice: &[f32]) -> f32 {
    let mut sum_sq = 0.0f64;
    for &v in slice {
        sum_sq += (v as f64) * (v as f64);
    }
    sum_sq.sqrt() as f32
}

pub fn compute_mean(slice: &[f32]) -> f32 {
    if slice.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for &v in slice {
        sum += v as f64;
    }
    (sum / (slice.len() as f64)) as f32
}

pub fn get_head_preview(slice: &[f32], n: usize) -> Vec<f32> {
    slice.iter().take(n).copied().collect()
}

// ---------------------------------------------------------------------------
// FP8 (E4M3) Conversion
// ---------------------------------------------------------------------------

pub fn f32_to_fp8_e4m3(val: f32) -> u8 {
    if val.is_nan() {
        return 0x7F;
    }
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let mant = bits & 0x7FFFFF;

    if val == 0.0 {
        return (sign << 7) as u8;
    }

    let target_exp = exp + 7;
    if target_exp <= 0 {
        let shift = 1 - target_exp;
        if shift > 3 {
            return (sign << 7) as u8;
        }
        let full_mant = (1 << 3) | (mant >> 20);
        let q_mant = (full_mant >> shift) & 0x07;
        return ((sign << 7) | q_mant) as u8;
    }

    if target_exp >= 15 {
        return ((sign << 7) | 0x7E) as u8;
    }

    let q_exp = (target_exp as u32) & 0x0F;
    let q_mant = (mant >> 20) & 0x07;
    ((sign << 7) | (q_exp << 3) | q_mant) as u8
}

pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0x0F;
    let mant = byte & 0x07;

    if exp == 0 && mant == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }

    if exp == 15 && mant == 7 {
        return f32::NAN;
    }

    if exp == 0 {
        let val = (mant as f32) / 8.0 * (2.0f32.powi(-6));
        return if sign == 1 { -val } else { val };
    }

    let mant_f = 1.0 + (mant as f32) / 8.0;
    let exp_f = 2.0f32.powi(exp as i32 - 7);
    let val = mant_f * exp_f;
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// Token response frame streamed from final stage worker back to coordinator.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenResponseFrame {
    pub session_id: u64,
    pub sequence_id: u64,
    pub token_id: i32,
    pub is_eos: bool,
    pub token_text: String,
    pub eval_time_ms: f32,
}

impl TokenResponseFrame {
    pub const HEADER_SIZE: usize = 4 + 2 + 8 + 8 + 4 + 1 + 4 + 4; // 35 bytes

    pub fn encode(&self) -> Bytes {
        let text_bytes = self.token_text.as_bytes();
        let mut buf = BytesMut::with_capacity(Self::HEADER_SIZE + text_bytes.len());
        buf.extend_from_slice(&TOKEN_RESP_MAGIC);
        buf.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
        buf.extend_from_slice(&self.session_id.to_be_bytes());
        buf.extend_from_slice(&self.sequence_id.to_be_bytes());
        buf.extend_from_slice(&self.token_id.to_be_bytes());
        buf.extend_from_slice(&[if self.is_eos { 1 } else { 0 }]);
        buf.extend_from_slice(&self.eval_time_ms.to_bits().to_be_bytes());
        buf.extend_from_slice(&(text_bytes.len() as u32).to_be_bytes());
        buf.extend_from_slice(text_bytes);
        buf.freeze()
    }

    pub async fn decode_async<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Self> {
        let mut hdr = [0u8; Self::HEADER_SIZE];
        reader.read_exact(&mut hdr).await?;
        ensure!(
            hdr[0..4] == TOKEN_RESP_MAGIC,
            "Invalid token response magic: {:?}",
            &hdr[0..4]
        );

        let version = u16::from_be_bytes([hdr[4], hdr[5]]);
        ensure!(
            version == PROTOCOL_VERSION,
            "Unsupported token response protocol version: {}",
            version
        );

        let session_id = u64::from_be_bytes(hdr[6..14].try_into().unwrap());
        let sequence_id = u64::from_be_bytes(hdr[14..22].try_into().unwrap());
        let token_id = i32::from_be_bytes(hdr[22..26].try_into().unwrap());
        let is_eos = hdr[26] != 0;
        let eval_time_ms = f32::from_bits(u32::from_be_bytes(hdr[27..31].try_into().unwrap()));
        let text_len = u32::from_be_bytes(hdr[31..35].try_into().unwrap()) as usize;

        ensure!(
            text_len <= (MAX_TOKEN_TEXT_BYTES as usize),
            "Token text length ({}) exceeds maximum safety ceiling ({} bytes)",
            text_len,
            MAX_TOKEN_TEXT_BYTES
        );

        let mut text_buf = vec![0u8; text_len];
        reader.read_exact(&mut text_buf).await?;
        let token_text = String::from_utf8(text_buf)?;

        Ok(Self {
            session_id,
            sequence_id,
            token_id,
            is_eos,
            token_text,
            eval_time_ms,
        })
    }

    pub fn decode_sync<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        ensure!(
            magic == TOKEN_RESP_MAGIC,
            "Invalid token response magic: {:?}",
            magic
        );

        let mut ver_buf = [0u8; 2];
        reader.read_exact(&mut ver_buf)?;
        let version = u16::from_be_bytes(ver_buf);
        ensure!(
            version == PROTOCOL_VERSION,
            "Unsupported token response protocol version: {}",
            version
        );

        let mut u64_buf = [0u8; 8];
        reader.read_exact(&mut u64_buf)?;
        let session_id = u64::from_be_bytes(u64_buf);

        reader.read_exact(&mut u64_buf)?;
        let sequence_id = u64::from_be_bytes(u64_buf);

        let mut u32_buf = [0u8; 4];
        reader.read_exact(&mut u32_buf)?;
        let token_id = i32::from_be_bytes(u32_buf);

        let mut u8_buf = [0u8; 1];
        reader.read_exact(&mut u8_buf)?;
        let is_eos = u8_buf[0] != 0;

        reader.read_exact(&mut u32_buf)?;
        let eval_time_ms = f32::from_bits(u32::from_be_bytes(u32_buf));

        reader.read_exact(&mut u32_buf)?;
        let text_len = u32::from_be_bytes(u32_buf) as usize;

        ensure!(
            text_len <= (MAX_TOKEN_TEXT_BYTES as usize),
            "Token text length ({}) exceeds maximum safety ceiling ({} bytes)",
            text_len,
            MAX_TOKEN_TEXT_BYTES
        );

        let mut text_buf = vec![0u8; text_len];
        reader.read_exact(&mut text_buf)?;
        let token_text = String::from_utf8(text_buf)?;

        Ok(Self {
            session_id,
            sequence_id,
            token_id,
            is_eos,
            token_text,
            eval_time_ms,
        })
    }
}

/// Initial Handshake Request sent from Coordinator to Worker to verify matching tensor shapes, layer slices, and chain topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandshakeRequest {
    pub version: u16,
    pub auth_token: [u8; 32],
    pub stage_index: u8,
    pub total_stages: u8,
    pub downstream_peer: String,
    pub model_architecture: String,
    pub hidden_dim: u32,
    pub total_layers: u32,
    pub worker_layer_start: u32,
    pub worker_layer_end: u32,
    pub checksum_prefix: String,
}

impl HandshakeRequest {
    pub fn encode(&self) -> Bytes {
        let arch_bytes = self.model_architecture.as_bytes();
        let csum_bytes = self.checksum_prefix.as_bytes();
        let peer_bytes = self.downstream_peer.as_bytes();
        assert!(
            peer_bytes.len() <= MAX_HANDSHAKE_STR_LEN,
            "downstream_peer length {} exceeds limit {}",
            peer_bytes.len(),
            MAX_HANDSHAKE_STR_LEN
        );
        assert!(
            arch_bytes.len() <= MAX_HANDSHAKE_STR_LEN,
            "model_architecture length {} exceeds limit {}",
            arch_bytes.len(),
            MAX_HANDSHAKE_STR_LEN
        );
        assert!(
            csum_bytes.len() <= MAX_HANDSHAKE_STR_LEN,
            "checksum_prefix length {} exceeds limit {}",
            csum_bytes.len(),
            MAX_HANDSHAKE_STR_LEN
        );
        let mut buf =
            BytesMut::with_capacity(64 + arch_bytes.len() + csum_bytes.len() + peer_bytes.len());

        buf.extend_from_slice(&HANDSHAKE_REQ_MAGIC);
        buf.extend_from_slice(&self.version.to_be_bytes());
        buf.extend_from_slice(&self.auth_token);
        buf.extend_from_slice(&[self.stage_index]);
        buf.extend_from_slice(&[self.total_stages]);
        buf.extend_from_slice(&self.hidden_dim.to_be_bytes());
        buf.extend_from_slice(&self.total_layers.to_be_bytes());
        buf.extend_from_slice(&self.worker_layer_start.to_be_bytes());
        buf.extend_from_slice(&self.worker_layer_end.to_be_bytes());

        buf.extend_from_slice(&(peer_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(peer_bytes);

        buf.extend_from_slice(&(arch_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(arch_bytes);

        buf.extend_from_slice(&(csum_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(csum_bytes);

        buf.freeze()
    }

    pub async fn decode_async<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).await?;
        ensure!(
            magic == HANDSHAKE_REQ_MAGIC,
            "Invalid handshake request magic: {:?}",
            magic
        );

        let version = reader.read_u16().await?;
        if version < 4 {
            bail!("Handshake rejected: received legacy protocol version {} (< 4). AeroMesh requires PROTOCOL_VERSION 4 for multi-node pipeline support.", version);
        }
        ensure!(
            version == PROTOCOL_VERSION,
            "Unsupported handshake protocol version: {}",
            version
        );

        let mut auth_token = [0u8; 32];
        reader.read_exact(&mut auth_token).await?;

        let stage_index = reader.read_u8().await?;
        let total_stages = reader.read_u8().await?;

        let hidden_dim = reader.read_u32().await?;
        let total_layers = reader.read_u32().await?;
        let worker_layer_start = reader.read_u32().await?;
        let worker_layer_end = reader.read_u32().await?;

        let peer_len = reader.read_u16().await? as usize;
        ensure!(
            peer_len <= MAX_HANDSHAKE_STR_LEN,
            "Handshake downstream peer string length ({}) exceeds limit ({} bytes)",
            peer_len,
            MAX_HANDSHAKE_STR_LEN
        );
        let mut peer_buf = vec![0u8; peer_len];
        reader.read_exact(&mut peer_buf).await?;
        let downstream_peer = String::from_utf8(peer_buf)?;

        let arch_len = reader.read_u16().await? as usize;
        ensure!(
            arch_len <= MAX_HANDSHAKE_STR_LEN,
            "Handshake architecture string length ({}) exceeds limit ({} bytes)",
            arch_len,
            MAX_HANDSHAKE_STR_LEN
        );
        let mut arch_buf = vec![0u8; arch_len];
        reader.read_exact(&mut arch_buf).await?;
        let model_architecture = String::from_utf8(arch_buf)?;

        let csum_len = reader.read_u16().await? as usize;
        ensure!(
            csum_len <= MAX_HANDSHAKE_STR_LEN,
            "Handshake checksum string length ({}) exceeds limit ({} bytes)",
            csum_len,
            MAX_HANDSHAKE_STR_LEN
        );
        let mut csum_buf = vec![0u8; csum_len];
        reader.read_exact(&mut csum_buf).await?;
        let checksum_prefix = String::from_utf8(csum_buf)?;

        ensure!(
            total_stages >= 2,
            "Invalid total_stages in handshake request: {}",
            total_stages
        );
        ensure!(
            stage_index < total_stages,
            "Invalid stage_index {} >= total_stages {}",
            stage_index,
            total_stages
        );

        Ok(Self {
            version,
            auth_token,
            stage_index,
            total_stages,
            downstream_peer,
            model_architecture,
            hidden_dim,
            total_layers,
            worker_layer_start,
            worker_layer_end,
            checksum_prefix,
        })
    }
}

/// Handshake Response returned from Worker to Coordinator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandshakeResponse {
    pub version: u16,
    pub auth_token: [u8; 32],
    pub accepted: bool,
    pub worker_layer_count: u32,
    pub error_message: String,
}

impl HandshakeResponse {
    pub fn ok(auth_token: [u8; 32]) -> Self {
        Self::ok_with_layers(auth_token, 0)
    }

    pub fn ok_with_layers(auth_token: [u8; 32], worker_layer_count: u32) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            auth_token,
            accepted: true,
            worker_layer_count,
            error_message: String::new(),
        }
    }

    pub fn reject(error_message: impl Into<String>) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            auth_token: [0u8; 32],
            accepted: false,
            worker_layer_count: 0,
            error_message: error_message.into(),
        }
    }

    pub fn encode(&self) -> Bytes {
        let err_bytes = self.error_message.as_bytes();
        let mut buf = BytesMut::with_capacity(48 + err_bytes.len());

        buf.extend_from_slice(&HANDSHAKE_RESP_MAGIC);
        buf.extend_from_slice(&self.version.to_be_bytes());
        buf.extend_from_slice(&self.auth_token);
        buf.extend_from_slice(&[if self.accepted { 1 } else { 0 }]);
        buf.extend_from_slice(&self.worker_layer_count.to_be_bytes());

        buf.extend_from_slice(&(err_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(err_bytes);

        buf.freeze()
    }

    pub async fn decode_async<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).await?;
        ensure!(
            magic == HANDSHAKE_RESP_MAGIC,
            "Invalid handshake response magic: {:?}",
            magic
        );

        let version = reader.read_u16().await?;
        if version < 4 {
            bail!("Handshake response rejected: received legacy protocol version {} (< 4). AeroMesh requires PROTOCOL_VERSION 4.", version);
        }
        ensure!(
            version == PROTOCOL_VERSION,
            "Unsupported handshake response protocol version: {}",
            version
        );

        let mut auth_token = [0u8; 32];
        reader.read_exact(&mut auth_token).await?;

        let accepted = reader.read_u8().await? != 0;
        let worker_layer_count = reader.read_u32().await?;

        let err_len = reader.read_u16().await? as usize;
        ensure!(
            err_len <= MAX_HANDSHAKE_ERR_LEN,
            "Handshake error message length ({}) exceeds limit ({} bytes)",
            err_len,
            MAX_HANDSHAKE_ERR_LEN
        );
        let mut err_buf = vec![0u8; err_len];
        reader.read_exact(&mut err_buf).await?;
        let error_message = String::from_utf8(err_buf)?;

        Ok(Self {
            version,
            auth_token,
            accepted,
            worker_layer_count,
            error_message,
        })
    }
}

fn half_to_float(h: u16) -> f32 {
    let sign = ((h >> 15) & 0x0001) as u32;
    let exp = ((h >> 10) & 0x001f) as u32;
    let mant = (h & 0x03ff) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut e = 1;
            let mut m = mant;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e += 1;
            }
            let exp_f = (127 - 15 - e + 1) as u32;
            let mant_f = (m & 0x03ff) << 13;
            f32::from_bits((sign << 31) | (exp_f << 23) | mant_f)
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits((sign << 31) | 0x7f800000)
        } else {
            f32::from_bits((sign << 31) | 0x7f800000 | (mant << 13))
        }
    } else {
        let exp_f = exp + 127 - 15;
        let mant_f = mant << 13;
        f32::from_bits((sign << 31) | (exp_f << 23) | mant_f)
    }
}

pub fn float_to_half(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007FFFFF;

    if exp == 255 {
        if mant != 0 {
            0x7E00 | ((mant >> 13) as u16)
        } else {
            sign | 0x7C00
        }
    } else {
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            sign | 0x7C00
        } else if new_exp <= 0 {
            if 10 - new_exp > 24 {
                sign
            } else {
                let m = mant | 0x00800000;
                let shift = (14 - new_exp) as u32;
                let half_mant = (m >> shift) as u16;
                sign | half_mant
            }
        } else {
            let half_exp = (new_exp as u16) << 10;
            let half_mant = (mant >> 13) as u16;
            sign | half_exp | half_mant
        }
    }
}

/// Telemetry stats computed by the Activation Boundary Watchdog.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationStats {
    pub tag: String,
    pub count: usize,
    pub mean: f32,
    pub variance: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub l2_norm: f32,
    pub has_nan: bool,
    pub has_inf: bool,
}

/// Boundary Watchdog: Inspects intermediate activation tensors at node egress/ingress boundaries.
pub fn inspect_activations(tag: &str, tensor: &[f32], shape: &[usize]) -> Result<ActivationStats> {
    if tensor.is_empty() {
        bail!(
            "🚨 ActivationWatchdog [{}]: tensor buffer is empty! Shape: {:?}",
            tag,
            shape
        );
    }

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut has_nan = false;
    let mut has_inf = false;

    for &v in tensor {
        if v.is_nan() {
            has_nan = true;
        } else if v.is_infinite() {
            has_inf = true;
        } else {
            sum += v as f64;
            sum_sq += (v as f64) * (v as f64);
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
    }

    let count = tensor.len();
    let mean = (sum / count as f64) as f32;
    let variance = ((sum_sq / count as f64) - (mean as f64 * mean as f64)).max(0.0) as f32;
    let std_dev = variance.sqrt();
    let l2_norm = sum_sq.sqrt() as f32;

    if has_nan {
        bail!(
            "🚨 ActivationWatchdog [{}]: NaN detected in activation tensor! Shape: {:?}",
            tag,
            shape
        );
    }
    if has_inf {
        bail!(
            "🚨 ActivationWatchdog [{}]: Inf detected in activation tensor! Shape: {:?}",
            tag,
            shape
        );
    }
    if l2_norm == 0.0 {
        tracing::warn!(
            "⚠️ ActivationWatchdog [{}]: L2 norm is 0.0 (all-zero tensor)! Shape: {:?}",
            tag,
            shape
        );
    }
    if max.abs() > 500.0 || min.abs() > 500.0 {
        tracing::warn!(
            "⚠️ ActivationWatchdog [{}]: High activation magnitude detected: min={:.3}, max={:.3}",
            tag,
            min,
            max
        );
    }

    tracing::debug!(
        tag = %tag,
        count = count,
        shape = ?shape,
        mean = mean,
        std_dev = std_dev,
        min = min,
        max = max,
        l2_norm = l2_norm,
        "🔍 ActivationWatchdog telemetry check passed"
    );

    Ok(ActivationStats {
        tag: tag.to_string(),
        count,
        mean,
        variance,
        std_dev,
        min,
        max,
        l2_norm,
        has_nan,
        has_inf,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_f32_fidelity() {
        let hidden_dim = 1536;
        let seq_len = 2;
        let data = vec![1.2345f32; seq_len * hidden_dim];

        let frame = ActivationFrame::from_f32_matrix_quantized(
            123,
            1,
            seq_len as u32,
            0,
            13,
            hidden_dim as u32,
            &data,
            ActivationDtype::RawF32,
            0,
        );

        let mut recovered = vec![0.0f32; data.len()];
        frame.dequantize_into(&mut recovered).unwrap();
        assert_eq!(data, recovered);
    }

    #[test]
    fn test_per_row_quantization_accuracy_and_outlier_resilience() {
        let hidden_dim = 1536;
        let seq_len = 4;
        let mut data = vec![0.0f32; seq_len * hidden_dim];

        // Fill with synthetic activations + deliberate outlier at row 2, col 50
        for (i, item) in data.iter_mut().enumerate() {
            *item = ((i % 100) as f32 - 50.0) / 50.0; // [-1.0, 1.0]
        }
        data[2 * hidden_dim + 50] = 42.5; // Outlier

        let frame = ActivationFrame::from_f32_matrix_quantized(
            1234,
            1,
            seq_len as u32,
            0,
            13,
            hidden_dim as u32,
            &data,
            ActivationDtype::Int8PerRow,
            0,
        );

        let mut recovered = vec![0.0f32; data.len()];
        frame.dequantize_into(&mut recovered).unwrap();

        // Check outlier recovery accuracy
        let outlier_orig = data[2 * hidden_dim + 50];
        let outlier_rec = recovered[2 * hidden_dim + 50];
        assert!(
            (outlier_orig - outlier_rec).abs() < 0.5,
            "Outlier should be preserved accurately"
        );

        // Check average row error
        let mut max_err = 0.0f32;
        for i in 0..data.len() {
            let diff = (data[i] - recovered[i]).abs();
            if diff > max_err {
                max_err = diff;
            }
        }
        assert!(
            max_err < 0.6,
            "Max error under INT8 scaling should be minimal"
        );
    }

    #[test]
    fn test_fp8_e4m3_roundtrip() {
        let values = [-5.0f32, -1.0, -0.25, 0.0, 0.25, 1.0, 3.5, 15.0];
        for &v in &values {
            let fp8 = f32_to_fp8_e4m3(v);
            let rec = fp8_e4m3_to_f32(fp8);
            let diff = (v - rec).abs();
            assert!(
                diff <= 0.5 * v.abs().max(0.1),
                "FP8 roundtrip error too large for {}",
                v
            );
        }
    }

    #[test]
    fn test_bf16_preserves_outlier_activations() {
        // Qwen2-style residual: one massive outlier channel + many small values
        let d = 64usize;
        let mut data: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.37).sin() * 2.0).collect();
        data[7] = 4000.0;

        let int8 = ActivationFrame::from_f32_matrix_quantized(
            1,
            0,
            1,
            0,
            23,
            d as u32,
            &data,
            ActivationDtype::Int8PerRow,
            0,
        );
        let int8_rec = int8.to_f32_vec().unwrap();
        let int8_zeroed = (0..d).filter(|&i| i != 7 && int8_rec[i] == 0.0).count();
        assert!(
            int8_zeroed > d / 2,
            "INT8 per-row is expected to destroy small channels next to outliers"
        );

        let bf16 = ActivationFrame::from_f32_matrix_quantized(
            1,
            0,
            1,
            0,
            23,
            d as u32,
            &data,
            ActivationDtype::Bf16,
            0,
        );
        assert_eq!(bf16.header.dtype, ActivationDtype::Bf16 as u8);
        let bf16_rec = bf16.to_f32_vec().unwrap();
        for i in 0..d {
            let tol = data[i].abs() * 0.01 + 1e-3;
            assert!(
                (data[i] - bf16_rec[i]).abs() <= tol,
                "BF16 error too large at {}: {} vs {}",
                i,
                data[i],
                bf16_rec[i]
            );
        }
    }

    #[test]
    fn test_activation_watchdog_telemetry() {
        let clean_data = vec![1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5];
        let stats = inspect_activations("Test_Clean", &clean_data, &[1, 6]).unwrap();
        assert_eq!(stats.count, 6);
        assert!(!stats.has_nan);
        assert!(!stats.has_inf);
        assert!(stats.l2_norm > 0.0);

        let nan_data = vec![1.0f32, f32::NAN, 2.0];
        assert!(inspect_activations("Test_NaN", &nan_data, &[1, 3]).is_err());

        let inf_data = vec![1.0f32, f32::INFINITY, 2.0];
        assert!(inspect_activations("Test_Inf", &inf_data, &[1, 3]).is_err());
    }

    #[test]
    fn test_activation_header_rejects_oversized_payload_ceiling() {
        let mut buf = [0u8; ActivationHeader::SIZE];
        buf[0..4].copy_from_slice(&ACTIVATION_MAGIC);
        BigEndian::write_u16(&mut buf[4..6], PROTOCOL_VERSION);
        BigEndian::write_u32(&mut buf[22..26], 1); // seq_len = 1
        BigEndian::write_u32(&mut buf[26..30], 0); // token_position = 0
        buf[30] = 0; // stage_index
        buf[31] = 2; // total_stages
        BigEndian::write_u16(&mut buf[32..34], 0); // layer_index
        BigEndian::write_u32(&mut buf[34..38], 1024); // hidden_dim = 1024
        buf[38] = ActivationDtype::RawF32 as u8;
        // Inject payload_bytes = MAX_ACTIVATION_PAYLOAD_BYTES + 1
        BigEndian::write_u32(&mut buf[40..44], MAX_ACTIVATION_PAYLOAD_BYTES + 1);

        let res = ActivationHeader::decode(&buf);
        assert!(
            res.is_err(),
            "Must reject payload exceeding MAX_ACTIVATION_PAYLOAD_BYTES"
        );
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("exceeds maximum safety ceiling"));

        // Inject payload_bytes = 0xFFFFFFFF
        BigEndian::write_u32(&mut buf[40..44], u32::MAX);
        let res_max = ActivationHeader::decode(&buf);
        assert!(res_max.is_err(), "Must reject u32::MAX payload_bytes");
    }

    #[test]
    fn test_activation_header_rejects_mismatched_payload_bytes() {
        let mut buf = [0u8; ActivationHeader::SIZE];
        buf[0..4].copy_from_slice(&ACTIVATION_MAGIC);
        BigEndian::write_u16(&mut buf[4..6], PROTOCOL_VERSION);
        BigEndian::write_u32(&mut buf[22..26], 1); // seq_len = 1
        BigEndian::write_u32(&mut buf[26..30], 0); // token_position = 0
        buf[30] = 0; // stage_index
        buf[31] = 2; // total_stages
        BigEndian::write_u16(&mut buf[32..34], 0); // layer_index
        BigEndian::write_u32(&mut buf[34..38], 16); // hidden_dim = 16
        buf[38] = ActivationDtype::RawF32 as u8; // expected: 1 * 16 * 4 = 64 bytes
        BigEndian::write_u32(&mut buf[40..44], 128); // malicious/corrupted claim: 128 bytes

        let res = ActivationHeader::decode(&buf);
        assert!(
            res.is_err(),
            "Must reject mathematically mismatched payload length"
        );
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("Mismatched activation payload size"));
    }

    #[test]
    fn test_activation_header_rejects_oversized_sequence_or_dim() {
        let mut buf = [0u8; ActivationHeader::SIZE];
        buf[0..4].copy_from_slice(&ACTIVATION_MAGIC);
        BigEndian::write_u16(&mut buf[4..6], PROTOCOL_VERSION);
        BigEndian::write_u32(&mut buf[22..26], MAX_SEQUENCE_LENGTH + 1); // seq_len too large
        BigEndian::write_u32(&mut buf[26..30], 0);
        buf[30] = 0;
        buf[31] = 2;
        BigEndian::write_u16(&mut buf[32..34], 0);
        BigEndian::write_u32(&mut buf[34..38], 1024);
        buf[38] = ActivationDtype::RawF32 as u8;
        BigEndian::write_u32(&mut buf[40..44], 4096);

        let res = ActivationHeader::decode(&buf);
        assert!(
            res.is_err(),
            "Must reject sequence_length > MAX_SEQUENCE_LENGTH"
        );

        // Now test hidden_dim > MAX_HIDDEN_DIM
        BigEndian::write_u32(&mut buf[22..26], 1);
        BigEndian::write_u32(&mut buf[34..38], MAX_HIDDEN_DIM + 1);
        let res_dim = ActivationHeader::decode(&buf);
        assert!(res_dim.is_err(), "Must reject hidden_dim > MAX_HIDDEN_DIM");

        // Now test hidden_dim == 0
        BigEndian::write_u32(&mut buf[34..38], 0);
        let res_zero = ActivationHeader::decode(&buf);
        assert!(res_zero.is_err(), "Must reject hidden_dim == 0");
    }

    #[test]
    fn test_activation_header_rejects_integer_overflow() {
        let res = ActivationHeader::calculate_expected_payload_bytes(
            u32::MAX,
            u32::MAX,
            ActivationDtype::RawF32 as u8,
        );
        assert!(
            res.is_err(),
            "Must fail on integer overflow in dimensions calculation"
        );
    }

    #[tokio::test]
    async fn test_token_response_rejects_oversized_text_async() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&TOKEN_RESP_MAGIC);
        stream.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
        stream.extend_from_slice(&1u64.to_be_bytes()); // session_id
        stream.extend_from_slice(&1u64.to_be_bytes()); // sequence_id
        stream.extend_from_slice(&42i32.to_be_bytes()); // token_id
        stream.push(0); // is_eos = false
        stream.extend_from_slice(&1.5f32.to_bits().to_be_bytes()); // eval_time_ms
                                                                   // Malicious length: 1 MB text (> MAX_TOKEN_TEXT_BYTES = 64 KB)
        stream.extend_from_slice(&(1024 * 1024u32).to_be_bytes());

        let mut cursor = std::io::Cursor::new(stream);
        let res = TokenResponseFrame::decode_async(&mut cursor).await;
        assert!(res.is_err(), "Must reject token text exceeding ceiling");
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("exceeds maximum safety ceiling"));
    }

    #[test]
    fn test_token_response_rejects_oversized_text_sync() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&TOKEN_RESP_MAGIC);
        stream.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
        stream.extend_from_slice(&1u64.to_be_bytes());
        stream.extend_from_slice(&1u64.to_be_bytes());
        stream.extend_from_slice(&42i32.to_be_bytes());
        stream.push(0);
        stream.extend_from_slice(&1.5f32.to_bits().to_be_bytes());
        // Malicious length: 1 MB
        stream.extend_from_slice(&(1024 * 1024u32).to_be_bytes());

        let mut cursor = std::io::Cursor::new(stream);
        let res = TokenResponseFrame::decode_sync(&mut cursor);
        assert!(res.is_err(), "Must reject token text exceeding ceiling");
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("exceeds maximum safety ceiling"));
    }

    #[tokio::test]
    async fn test_handshake_request_rejects_oversized_strings() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&HANDSHAKE_REQ_MAGIC);
        stream.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
        stream.extend_from_slice(&[0u8; 32]); // auth_token
        stream.push(0); // stage_index
        stream.push(2); // total_stages
        stream.extend_from_slice(&4096u32.to_be_bytes()); // hidden_dim
        stream.extend_from_slice(&32u32.to_be_bytes()); // total_layers
        stream.extend_from_slice(&0u32.to_be_bytes()); // worker_layer_start
        stream.extend_from_slice(&16u32.to_be_bytes()); // worker_layer_end
        stream.extend_from_slice(&0u16.to_be_bytes()); // downstream_peer len = 0
                                                       // Malicious arch length: 1000 bytes (> MAX_HANDSHAKE_STR_LEN = 512)
        stream.extend_from_slice(&1000u16.to_be_bytes());

        let mut cursor = std::io::Cursor::new(stream);
        let res = HandshakeRequest::decode_async(&mut cursor).await;
        assert!(
            res.is_err(),
            "Must reject architecture string exceeding limit"
        );
        assert!(res.unwrap_err().to_string().contains("exceeds limit"));
    }

    #[tokio::test]
    async fn test_handshake_response_rejects_oversized_error() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&HANDSHAKE_RESP_MAGIC);
        stream.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
        stream.extend_from_slice(&[0u8; 32]); // auth_token
        stream.push(0); // accepted = false
        stream.extend_from_slice(&0u32.to_be_bytes()); // worker_layer_count
                                                       // Malicious error string length: 10000 bytes (> MAX_HANDSHAKE_ERR_LEN = 8192)
        stream.extend_from_slice(&10000u16.to_be_bytes());

        let mut cursor = std::io::Cursor::new(stream);
        let res = HandshakeResponse::decode_async(&mut cursor).await;
        assert!(
            res.is_err(),
            "Must reject handshake error string exceeding limit"
        );
        assert!(res.unwrap_err().to_string().contains("exceeds limit"));
    }

    #[tokio::test]
    async fn test_activation_frame_truncated_stream_error() {
        let mut header_bytes = [0u8; ActivationHeader::SIZE];
        header_bytes[0..4].copy_from_slice(&ACTIVATION_MAGIC);
        BigEndian::write_u16(&mut header_bytes[4..6], PROTOCOL_VERSION);
        BigEndian::write_u32(&mut header_bytes[22..26], 1);
        BigEndian::write_u32(&mut header_bytes[26..30], 0);
        header_bytes[30] = 0; // stage_index
        header_bytes[31] = 2; // total_stages
        BigEndian::write_u16(&mut header_bytes[32..34], 0);
        BigEndian::write_u32(&mut header_bytes[34..38], 4);
        header_bytes[38] = ActivationDtype::RawF32 as u8; // expected 16 bytes
        BigEndian::write_u32(&mut header_bytes[40..44], 16);

        // Feed only 8 bytes of payload instead of 16
        let mut stream = header_bytes.to_vec();
        stream.extend_from_slice(&[0u8; 8]);

        let mut cursor = std::io::Cursor::new(stream);
        let res = ActivationFrame::decode_async(&mut cursor).await;
        assert!(res.is_err(), "Must error on truncated payload stream");
    }

    #[tokio::test]
    async fn test_handshake_v4_multi_node_and_legacy_rejection() {
        let auth_token = [7u8; 32];
        let req = HandshakeRequest {
            version: PROTOCOL_VERSION,
            auth_token,
            stage_index: 1,
            total_stages: 3,
            downstream_peer: "192.168.1.50:9002".to_string(),
            model_architecture: "qwen2".to_string(),
            hidden_dim: 5120,
            total_layers: 48,
            worker_layer_start: 16,
            worker_layer_end: 31,
            checksum_prefix: "test-mesh".to_string(),
        };

        let encoded = req.encode();
        let mut cursor = std::io::Cursor::new(encoded.to_vec());
        let decoded = HandshakeRequest::decode_async(&mut cursor).await.unwrap();
        assert_eq!(req, decoded);
        assert_eq!(decoded.auth_token, auth_token);
        assert_eq!(decoded.stage_index, 1);
        assert_eq!(decoded.total_stages, 3);
        assert_eq!(decoded.downstream_peer, "192.168.1.50:9002");

        let resp = HandshakeResponse::ok(auth_token);
        let resp_encoded = resp.encode();
        let mut resp_cursor = std::io::Cursor::new(resp_encoded.to_vec());
        let resp_decoded = HandshakeResponse::decode_async(&mut resp_cursor)
            .await
            .unwrap();
        assert_eq!(resp, resp_decoded);
        assert_eq!(resp_decoded.auth_token, auth_token);

        // Verify rejection of legacy v3 handshake request
        let mut legacy_v3_bytes = vec![];
        legacy_v3_bytes.extend_from_slice(&HANDSHAKE_REQ_MAGIC);
        legacy_v3_bytes.extend_from_slice(&3u16.to_be_bytes()); // Version 3
        legacy_v3_bytes.extend_from_slice(&[0u8; 64]);
        let mut legacy_cursor = std::io::Cursor::new(legacy_v3_bytes);
        let legacy_res = HandshakeRequest::decode_async(&mut legacy_cursor).await;
        assert!(
            legacy_res.is_err(),
            "Legacy protocol v3 must be explicitly rejected"
        );
        let err_str = legacy_res.unwrap_err().to_string();
        assert!(
            err_str.contains("legacy protocol version 3"),
            "Error message should mention legacy version: {}",
            err_str
        );
    }

    #[tokio::test]
    async fn test_multi_stage_activation_frame_roundtrip() {
        let hidden_dim = 128;
        let seq_len = 2;
        let data = vec![0.5f32; seq_len * hidden_dim];

        let frame = ActivationFrame::from_f32_matrix_quantized_stage(
            999,
            42,
            seq_len as u32,
            10,
            1,
            3,
            16,
            hidden_dim as u32,
            &data,
            ActivationDtype::RawF32,
            0,
        );

        assert_eq!(frame.header.stage_index, 1);
        assert_eq!(frame.header.total_stages, 3);
        assert_eq!(frame.header.token_position, 10);

        let encoded = frame.encode();
        let mut cursor = std::io::Cursor::new(encoded.to_vec());
        let decoded = ActivationFrame::decode_async(&mut cursor).await.unwrap();

        assert_eq!(decoded.header.stage_index, 1);
        assert_eq!(decoded.header.total_stages, 3);
        assert_eq!(decoded.header.token_position, 10);
        assert_eq!(decoded.header.sequence_length, 2);

        let recovered = decoded.to_f32_vec().unwrap();
        assert_eq!(data, recovered);
    }

    #[test]
    fn test_fp16_roundtrip() {
        let hidden_dim = 64;
        let seq_len = 1;
        let data = vec![1.0f32, -2.5f32, 0.125f32, 65504.0f32];
        let mut full_data = vec![0.0f32; hidden_dim];
        full_data[..data.len()].copy_from_slice(&data);

        let frame = ActivationFrame::from_f32_matrix_quantized_stage(
            123,
            1,
            seq_len as u32,
            0,
            0,
            2,
            8,
            hidden_dim as u32,
            &full_data,
            ActivationDtype::Fp16,
            0,
        );

        assert_eq!(frame.header.dtype, ActivationDtype::Fp16 as u8);
        assert_eq!(frame.payload.len(), hidden_dim * 2);

        let recovered = frame.to_f32_vec().unwrap();
        for (orig, rec) in full_data.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 1e-3,
                "Mismatch: orig {} vs rec {}",
                orig,
                rec
            );
        }
    }
}
