use std::io::Read;
use anyhow::{bail, ensure, Result};
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use bytes::{Bytes, BytesMut};
use tokio::io::AsyncReadExt;

pub const ACTIVATION_MAGIC: [u8; 4] = *b"AERO";
pub const TOKEN_RESP_MAGIC: [u8; 4] = *b"ATOK";
pub const HANDSHAKE_REQ_MAGIC: [u8; 4] = *b"AHSK";
pub const HANDSHAKE_RESP_MAGIC: [u8; 4] = *b"AHSR";
pub const PROTOCOL_VERSION: u16 = 2;

// Flags for ActivationHeader
pub const FLAG_CLEAR_KV: u8 = 0x01;
pub const FLAG_IS_PROMPT: u8 = 0x02;
pub const FLAG_EOS_SIGNAL: u8 = 0x04;
pub const FLAG_FULL_PRECISION: u8 = 0x08;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationDtype {
    Fp32 = 0,
    Fp16 = 1,
    Bf16 = 2,
    Int8PerRow = 3,
    Fp8E4M3 = 4,
}

impl ActivationDtype {
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(Self::Fp32),
            1 => Ok(Self::Fp16),
            2 => Ok(Self::Bf16),
            3 => Ok(Self::Int8PerRow),
            4 => Ok(Self::Fp8E4M3),
            _ => bail!("Unsupported activation dtype: {}", val),
        }
    }

    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::Fp32 => 4,
            Self::Fp16 | Self::Bf16 => 2,
            Self::Int8PerRow | Self::Fp8E4M3 => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// FP8 (E4M3) Lookup & Vectorized Utilities
// ---------------------------------------------------------------------------

const FP8_E4M3_MAX_NORMAL: f32 = 448.0;

/// Pre-computed 256-entry lookup table mapping FP8 E4M3 bytes to f32.
static FP8_E4M3_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        let b = i as u8;
        let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (b >> 3) & 0x0F;
        let mant = (b & 0x07) as f32;

        lut[i] = if exp == 0 {
            // Subnormal: (-1)^sign * 2^(-6) * (mant / 8.0)
            sign * (1.0 / 64.0) * (mant / 8.0)
        } else if exp == 15 && mant == 7.0 {
            // NaN in E4M3
            f32::NAN
        } else {
            // Normalized: (-1)^sign * 2^(exp - 7) * (1.0 + mant / 8.0)
            let exp_val = 1 << (exp - 1); // 2^(exp - 1)
            let base_pow = (exp_val as f32) / 64.0; // 2^(exp - 7)
            sign * base_pow * (1.0 + mant / 8.0)
        };
        i += 1;
    }
    lut
};

/// Converts a 32-bit float to FP8 E4M3 representation with rounding.
#[inline(always)]
pub fn f32_to_fp8_e4m3(val: f32) -> u8 {
    if val.is_nan() {
        return 0x7F;
    }
    let sign = if val.is_sign_negative() { 0x80u8 } else { 0x00u8 };
    let abs_val = val.abs();

    if abs_val > FP8_E4M3_MAX_NORMAL {
        return sign | 0x7E; // Max finite normal in E4M3
    }
    if abs_val < (1.0 / 64.0) * (1.0 / 8.0) {
        return sign; // Zero / flush subnormal
    }

    // Binary search lookup or bit conversion
    let bits = abs_val.to_bits();
    let exp_f32 = ((bits >> 23) & 0xFF) as i32 - 127;
    let mant_f32 = bits & 0x007FFFFF;

    let target_exp = exp_f32 + 7;
    if target_exp <= 0 {
        // Subnormal in E4M3
        let shift = 1 - target_exp;
        if shift > 3 {
            return sign;
        }
        let mant = (1 << (3 - shift)) + ((mant_f32 >> (20 + shift)) & 0x07);
        sign | (mant as u8 & 0x07)
    } else if target_exp >= 15 {
        sign | 0x7E
    } else {
        let mant = (mant_f32 >> 20) & 0x07;
        sign | ((target_exp as u8) << 3) | (mant as u8)
    }
}

// ---------------------------------------------------------------------------
// Binary Frame Header & Dynamic Per-Row Activation Frame
// ---------------------------------------------------------------------------

/// Binary frame header for high-speed P2P activation streaming.
/// Fixed size: 42 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationHeader {
    pub magic: [u8; 4],          // b"AERO"
    pub version: u16,           // 2
    pub session_id: u64,        // Unique multi-turn chat session ID
    pub sequence_id: u64,       // Token generation sequence counter
    pub sequence_length: u32,   // S: Number of tokens in sequence (1 for decode, >1 for prefill)
    pub token_position: u32,    // Context position offset in KV-cache (pos)
    pub layer_index: u16,       // Boundary layer index (e.g. 24)
    pub hidden_dim: u32,        // Hidden dimension (e.g. 4096 / 5120)
    pub dtype: u8,              // 0=FP32, 1=FP16, 2=BF16, 3=INT8_PER_ROW, 4=FP8_E4M3
    pub flags: u8,              // Bitmask: FLAG_CLEAR_KV (0x01), FLAG_IS_PROMPT (0x02), etc.
    pub payload_bytes: u32,     // Length of payload following header
}

impl ActivationHeader {
    pub const SIZE: usize = 42;

    pub fn encode(&self, out: &mut [u8]) {
        assert!(out.len() >= Self::SIZE);
        out[0..4].copy_from_slice(&self.magic);
        BigEndian::write_u16(&mut out[4..6], self.version);
        BigEndian::write_u64(&mut out[6..14], self.session_id);
        BigEndian::write_u64(&mut out[14..22], self.sequence_id);
        BigEndian::write_u32(&mut out[22..26], self.sequence_length);
        BigEndian::write_u32(&mut out[26..30], self.token_position);
        BigEndian::write_u16(&mut out[30..32], self.layer_index);
        BigEndian::write_u32(&mut out[32..36], self.hidden_dim);
        out[36] = self.dtype;
        out[37] = self.flags;
        BigEndian::write_u32(&mut out[38..42], self.payload_bytes);
    }

    pub fn decode(src: &[u8]) -> Result<Self> {
        ensure!(src.len() >= Self::SIZE, "Buffer too short for ActivationHeader");
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&src[0..4]);
        ensure!(magic == ACTIVATION_MAGIC, "Invalid activation magic: {:?}", magic);

        let version = BigEndian::read_u16(&src[4..6]);
        ensure!(version == PROTOCOL_VERSION, "Unsupported activation protocol version: {}", version);

        let session_id = BigEndian::read_u64(&src[6..14]);
        let sequence_id = BigEndian::read_u64(&src[14..22]);
        let sequence_length = BigEndian::read_u32(&src[22..26]);
        let token_position = BigEndian::read_u32(&src[26..30]);
        let layer_index = BigEndian::read_u16(&src[30..32]);
        let hidden_dim = BigEndian::read_u32(&src[32..36]);
        let dtype = src[36];
        let flags = src[37];
        let payload_bytes = BigEndian::read_u32(&src[38..42]);

        Ok(Self {
            magic,
            version,
            session_id,
            sequence_id,
            sequence_length,
            token_position,
            layer_index,
            hidden_dim,
            dtype,
            flags,
            payload_bytes,
        })
    }

    pub fn has_flag(&self, flag: u8) -> bool {
        (self.flags & flag) != 0
    }
}

/// In-memory activation frame supporting both uncompressed float arrays and per-row dynamic quantization.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationFrame {
    pub header: ActivationHeader,
    pub payload: Bytes,
}

impl ActivationFrame {
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
        let header = ActivationHeader {
            magic: ACTIVATION_MAGIC,
            version: PROTOCOL_VERSION,
            session_id,
            sequence_id,
            sequence_length,
            token_position,
            layer_index,
            hidden_dim,
            dtype: dtype as u8,
            flags,
            payload_bytes: payload.len() as u32,
        };

        Ok(Self { header, payload })
    }

    pub fn from_f32_slice(
        session_id: u64,
        sequence_id: u64,
        token_position: u32,
        layer_index: u16,
        activations: &[f32],
        flags: u8,
    ) -> Self {
        Self::from_f32_matrix(
            session_id,
            sequence_id,
            1,
            token_position,
            layer_index,
            activations.len() as u32,
            activations,
            flags,
        )
    }

    pub fn from_f32_matrix(
        session_id: u64,
        sequence_id: u64,
        sequence_length: u32,
        token_position: u32,
        layer_index: u16,
        hidden_dim: u32,
        activations: &[f32],
        flags: u8,
    ) -> Self {
        let mut bytes = BytesMut::with_capacity(activations.len() * 4);
        for &val in activations {
            let b = val.to_bits();
            let mut buf = [0u8; 4];
            BigEndian::write_u32(&mut buf, b);
            bytes.extend_from_slice(&buf);
        }

        let header = ActivationHeader {
            magic: ACTIVATION_MAGIC,
            version: PROTOCOL_VERSION,
            session_id,
            sequence_id,
            sequence_length,
            token_position,
            layer_index,
            hidden_dim,
            dtype: ActivationDtype::Fp32 as u8,
            flags,
            payload_bytes: bytes.len() as u32,
        };

        Self {
            header,
            payload: bytes.freeze(),
        }
    }

    /// Dynamic Per-Row (Per-Token) Quantization.
    /// Computes an independent scale factor for each token row s in [0..S) along hidden_dim D.
    /// Memory layout: [Row Scales: S * 4 Bytes (f32 LE)] [Quantized Tensor: S * D Bytes (i8 / u8)]
    pub fn from_f32_matrix_quantized(
        session_id: u64,
        sequence_id: u64,
        sequence_length: u32,
        token_position: u32,
        layer_index: u16,
        hidden_dim: u32,
        activations: &[f32],
        target_dtype: ActivationDtype,
        flags: u8,
    ) -> Self {
        let s_len = sequence_length as usize;
        let d_dim = hidden_dim as usize;
        assert_eq!(activations.len(), s_len * d_dim);

        // Fallback to FP32 if full precision requested or small intermediate shape
        if flags & FLAG_FULL_PRECISION != 0 || activations.len() < 128 {
            return Self::from_f32_matrix(
                session_id,
                sequence_id,
                sequence_length,
                token_position,
                layer_index,
                hidden_dim,
                activations,
                flags,
            );
        }

        let scale_bytes_len = s_len * 4;
        let quant_bytes_len = s_len * d_dim;
        let total_payload_bytes = scale_bytes_len + quant_bytes_len;
        let mut payload = BytesMut::with_capacity(total_payload_bytes);

        match target_dtype {
            ActivationDtype::Int8PerRow => {
                let mut scales = Vec::with_capacity(s_len);
                let mut quant_data = Vec::with_capacity(quant_bytes_len);

                for row_idx in 0..s_len {
                    let row_start = row_idx * d_dim;
                    let row = &activations[row_start..row_start + d_dim];

                    // 1. Compute max abs for this token row
                    let mut max_abs = 0.0f32;
                    for &val in row {
                        let abs = val.abs();
                        if abs > max_abs {
                            max_abs = abs;
                        }
                    }

                    let scale = if max_abs > 1e-12 {
                        max_abs / 127.0
                    } else {
                        1.0
                    };
                    let inv_scale = 1.0 / scale;
                    scales.push(scale);

                    // 2. Quantize row elements into INT8
                    for &val in row {
                        let q = (val * inv_scale).round();
                        let q_clamped = q.clamp(-127.0, 127.0) as i8;
                        quant_data.push(q_clamped as u8);
                    }
                }

                // Pack scales (S * 4 bytes f32 Little-Endian)
                for scale in scales {
                    let mut s_buf = [0u8; 4];
                    LittleEndian::write_f32(&mut s_buf, scale);
                    payload.extend_from_slice(&s_buf);
                }
                // Pack quantized data (S * D bytes)
                payload.extend_from_slice(&quant_data);
            }
            ActivationDtype::Fp8E4M3 => {
                let mut scales = Vec::with_capacity(s_len);
                let mut quant_data = Vec::with_capacity(quant_bytes_len);

                for row_idx in 0..s_len {
                    let row_start = row_idx * d_dim;
                    let row = &activations[row_start..row_start + d_dim];

                    let mut max_abs = 0.0f32;
                    for &val in row {
                        let abs = val.abs();
                        if abs > max_abs {
                            max_abs = abs;
                        }
                    }

                    let scale = if max_abs > 1e-12 {
                        max_abs / FP8_E4M3_MAX_NORMAL
                    } else {
                        1.0
                    };
                    let inv_scale = 1.0 / scale;
                    scales.push(scale);

                    for &val in row {
                        let scaled_val = val * inv_scale;
                        let fp8_byte = f32_to_fp8_e4m3(scaled_val);
                        quant_data.push(fp8_byte);
                    }
                }

                for scale in scales {
                    let mut s_buf = [0u8; 4];
                    LittleEndian::write_f32(&mut s_buf, scale);
                    payload.extend_from_slice(&s_buf);
                }
                payload.extend_from_slice(&quant_data);
            }
            _ => {
                return Self::from_f32_matrix(
                    session_id,
                    sequence_id,
                    sequence_length,
                    token_position,
                    layer_index,
                    hidden_dim,
                    activations,
                    flags,
                );
            }
        }

        let header = ActivationHeader {
            magic: ACTIVATION_MAGIC,
            version: PROTOCOL_VERSION,
            session_id,
            sequence_id,
            sequence_length,
            token_position,
            layer_index,
            hidden_dim,
            dtype: target_dtype as u8,
            flags,
            payload_bytes: payload.len() as u32,
        };

        Self {
            header,
            payload: payload.freeze(),
        }
    }

    /// High-performance zero-allocation in-place dequantization directly into pre-allocated memory.
    pub fn dequantize_into(&self, out: &mut [f32]) -> Result<()> {
        let s_len = self.header.sequence_length as usize;
        let d_dim = self.header.hidden_dim as usize;
        let total_elements = s_len * d_dim;
        ensure!(
            out.len() >= total_elements,
            "Target buffer size {} < expected {}",
            out.len(),
            total_elements
        );

        let dtype = ActivationDtype::from_u8(self.header.dtype)?;
        match dtype {
            ActivationDtype::Fp32 => {
                ensure!(self.payload.len() >= total_elements * 4, "Corrupted FP32 payload");
                for i in 0..total_elements {
                    let bits = BigEndian::read_u32(&self.payload[i * 4..(i + 1) * 4]);
                    out[i] = f32::from_bits(bits);
                }
            }
            ActivationDtype::Fp16 => {
                ensure!(self.payload.len() >= total_elements * 2, "Corrupted FP16 payload");
                for i in 0..total_elements {
                    let half_bits = BigEndian::read_u16(&self.payload[i * 2..(i + 1) * 2]);
                    out[i] = half_to_float(half_bits);
                }
            }
            ActivationDtype::Bf16 => {
                ensure!(self.payload.len() >= total_elements * 2, "Corrupted BF16 payload");
                for i in 0..total_elements {
                    let bf_bits = BigEndian::read_u16(&self.payload[i * 2..(i + 1) * 2]);
                    out[i] = f32::from_bits((bf_bits as u32) << 16);
                }
            }
            ActivationDtype::Int8PerRow => {
                let scale_bytes_len = s_len * 4;
                ensure!(
                    self.payload.len() >= scale_bytes_len + total_elements,
                    "Corrupted INT8 per-row payload: len {} < {}",
                    self.payload.len(),
                    scale_bytes_len + total_elements
                );

                let scales_slice = &self.payload[0..scale_bytes_len];
                let quant_slice = &self.payload[scale_bytes_len..scale_bytes_len + total_elements];

                for row_idx in 0..s_len {
                    let scale = LittleEndian::read_f32(&scales_slice[row_idx * 4..(row_idx + 1) * 4]);
                    let row_start = row_idx * d_dim;
                    for col in 0..d_dim {
                        let q = quant_slice[row_start + col] as i8;
                        out[row_start + col] = (q as f32) * scale;
                    }
                }
            }
            ActivationDtype::Fp8E4M3 => {
                let scale_bytes_len = s_len * 4;
                ensure!(
                    self.payload.len() >= scale_bytes_len + total_elements,
                    "Corrupted FP8 payload"
                );

                let scales_slice = &self.payload[0..scale_bytes_len];
                let quant_slice = &self.payload[scale_bytes_len..scale_bytes_len + total_elements];

                for row_idx in 0..s_len {
                    let scale = LittleEndian::read_f32(&scales_slice[row_idx * 4..(row_idx + 1) * 4]);
                    let row_start = row_idx * d_dim;
                    for col in 0..d_dim {
                        let fp8_byte = quant_slice[row_start + col] as usize;
                        let unscaled = FP8_E4M3_LUT[fp8_byte];
                        out[row_start + col] = unscaled * scale;
                    }
                }
            }
        }

        Ok(())
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        let total_elements = (self.header.sequence_length as usize) * (self.header.hidden_dim as usize);
        let mut out = vec![0.0f32; total_elements];
        self.dequantize_into(&mut out)?;
        Ok(out)
    }

    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(ActivationHeader::SIZE + self.payload.len());
        let mut header_bytes = [0u8; ActivationHeader::SIZE];
        self.header.encode(&mut header_bytes);
        buf.extend_from_slice(&header_bytes);
        buf.extend_from_slice(&self.payload);
        buf.freeze()
    }

    pub async fn decode_async<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Self> {
        let mut header_buf = [0u8; ActivationHeader::SIZE];
        reader.read_exact(&mut header_buf).await?;
        let header = ActivationHeader::decode(&header_buf)?;

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

        let mut payload_buf = vec![0u8; header.payload_bytes as usize];
        reader.read_exact(&mut payload_buf)?;

        Ok(Self {
            header,
            payload: Bytes::from(payload_buf),
        })
    }
}

/// Initial Handshake Request sent from Coordinator to Worker to verify matching tensor shapes and layer slices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandshakeRequest {
    pub version: u16,
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
        let mut buf = BytesMut::with_capacity(32 + arch_bytes.len() + csum_bytes.len());

        buf.extend_from_slice(&HANDSHAKE_REQ_MAGIC);
        buf.extend_from_slice(&self.version.to_be_bytes());
        buf.extend_from_slice(&self.hidden_dim.to_be_bytes());
        buf.extend_from_slice(&self.total_layers.to_be_bytes());
        buf.extend_from_slice(&self.worker_layer_start.to_be_bytes());
        buf.extend_from_slice(&self.worker_layer_end.to_be_bytes());

        buf.extend_from_slice(&(arch_bytes.len() as u32).to_be_bytes());
        buf.extend_from_slice(arch_bytes);

        buf.extend_from_slice(&(csum_bytes.len() as u32).to_be_bytes());
        buf.extend_from_slice(csum_bytes);

        buf.freeze()
    }

    pub async fn decode_async<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).await?;
        ensure!(magic == HANDSHAKE_REQ_MAGIC, "Invalid handshake request magic: {:?}", magic);

        let version = reader.read_u16().await?;
        ensure!(version == PROTOCOL_VERSION, "Protocol version mismatch in handshake");

        let hidden_dim = reader.read_u32().await?;
        let total_layers = reader.read_u32().await?;
        let worker_layer_start = reader.read_u32().await?;
        let worker_layer_end = reader.read_u32().await?;

        let arch_len = reader.read_u32().await? as usize;
        let mut arch_buf = vec![0u8; arch_len];
        reader.read_exact(&mut arch_buf).await?;
        let model_architecture = String::from_utf8(arch_buf)?;

        let csum_len = reader.read_u32().await? as usize;
        let mut csum_buf = vec![0u8; csum_len];
        reader.read_exact(&mut csum_buf).await?;
        let checksum_prefix = String::from_utf8(csum_buf)?;

        Ok(Self {
            version,
            model_architecture,
            hidden_dim,
            total_layers,
            worker_layer_start,
            worker_layer_end,
            checksum_prefix,
        })
    }
}

/// Handshake Response returned by Worker node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandshakeResponse {
    pub version: u16,
    pub accepted: bool,
    pub error_message: String,
}

impl HandshakeResponse {
    pub fn ok() -> Self {
        Self {
            version: PROTOCOL_VERSION,
            accepted: true,
            error_message: String::new(),
        }
    }

    pub fn err<S: Into<String>>(msg: S) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            accepted: false,
            error_message: msg.into(),
        }
    }

    pub fn encode(&self) -> Bytes {
        let msg_bytes = self.error_message.as_bytes();
        let mut buf = BytesMut::with_capacity(16 + msg_bytes.len());
        buf.extend_from_slice(&HANDSHAKE_RESP_MAGIC);
        buf.extend_from_slice(&self.version.to_be_bytes());
        buf.extend_from_slice(&[if self.accepted { 1 } else { 0 }]);
        buf.extend_from_slice(&(msg_bytes.len() as u32).to_be_bytes());
        buf.extend_from_slice(msg_bytes);
        buf.freeze()
    }

    pub async fn decode_async<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).await?;
        ensure!(magic == HANDSHAKE_RESP_MAGIC, "Invalid handshake response magic: {:?}", magic);

        let version = reader.read_u16().await?;
        let accepted = reader.read_u8().await? != 0;
        let msg_len = reader.read_u32().await? as usize;

        let mut msg_buf = vec![0u8; msg_len];
        reader.read_exact(&mut msg_buf).await?;
        let error_message = String::from_utf8(msg_buf)?;

        Ok(Self {
            version,
            accepted,
            error_message,
        })
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
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).await?;
        ensure!(magic == TOKEN_RESP_MAGIC, "Invalid token response magic: {:?}", magic);

        let version = reader.read_u16().await?;
        ensure!(version == PROTOCOL_VERSION, "Unsupported token response protocol version: {}", version);

        let session_id = reader.read_u64().await?;
        let sequence_id = reader.read_u64().await?;
        let token_id = reader.read_i32().await?;
        let is_eos = reader.read_u8().await? != 0;
        let eval_time_bits = reader.read_u32().await?;
        let eval_time_ms = f32::from_bits(eval_time_bits);
        let text_len = reader.read_u32().await? as usize;

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
}

/// Helper function to convert IEEE 754 half-precision float to single-precision float.
fn half_to_float(h: u16) -> f32 {
    let sign = ((h >> 15) & 0x0001) as u32;
    let exp = ((h >> 10) & 0x001f) as u32;
    let mant = (h & 0x03ff) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut m = mant;
            let mut e = 0;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x03ff;
            let f_exp = (127 - 15 + 1 - e) << 23;
            let f_mant = m << 13;
            f32::from_bits((sign << 31) | f_exp | f_mant)
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (mant << 13))
        }
    } else {
        let f_exp = (exp + (127 - 15)) << 23;
        let f_mant = mant << 13;
        f32::from_bits((sign << 31) | f_exp | f_mant)
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_per_row_quantization_accuracy_and_outlier_resilience() {
        let s_len = 4usize;
        let d_dim = 2560usize;
        let mut activations = Vec::with_capacity(s_len * d_dim);

        // Generate synthetic LLM activation matrix with severe channel outliers
        for row in 0..s_len {
            for col in 0..d_dim {
                let base = ((row * d_dim + col) as f32 * 0.013).sin() * 2.5;
                // Add extreme outlier on col 42 for token 0 and 2
                let val = if col == 42 && (row == 0 || row == 2) {
                    base * 80.0 // Outlier feature of magnitude ~200.0
                } else {
                    base
                };
                activations.push(val);
            }
        }

        let frame = ActivationFrame::from_f32_matrix_quantized(
            1234,
            1,
            s_len as u32,
            0,
            24,
            d_dim as u32,
            &activations,
            ActivationDtype::Int8PerRow,
            0,
        );

        assert_eq!(frame.header.dtype, ActivationDtype::Int8PerRow as u8);
        // Payload size: 4 scales * 4 bytes + 4 * 2560 bytes = 16 + 10240 = 10256 bytes
        let expected_payload = s_len * 4 + s_len * d_dim;
        assert_eq!(frame.payload.len(), expected_payload);

        let mut reconstructed = vec![0.0f32; activations.len()];
        frame.dequantize_into(&mut reconstructed).unwrap();

        // 1. Check Cosine Similarity across all elements
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for i in 0..activations.len() {
            let a = activations[i] as f64;
            let b = reconstructed[i] as f64;
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        let cosine_sim = dot / (norm_a.sqrt() * norm_b.sqrt());
        assert!(cosine_sim > 0.995, "Cosine similarity too low: {}", cosine_sim);

        // 2. Check token 1 (without outlier) was NOT distorted by token 0 outlier
        let row1_orig = &activations[d_dim..2 * d_dim];
        let row1_rec = &reconstructed[d_dim..2 * d_dim];
        let mut row1_max_err = 0.0f32;
        for i in 0..d_dim {
            let err = (row1_orig[i] - row1_rec[i]).abs();
            if err > row1_max_err {
                row1_max_err = err;
            }
        }
        assert!(row1_max_err < 0.05, "Non-outlier row corrupted: max err = {}", row1_max_err);
    }

    #[test]
    fn test_fp8_e4m3_roundtrip() {
        let test_vals = vec![0.0f32, 1.0, -1.0, 15.5, -200.0, 440.0, 0.05];
        for &val in &test_vals {
            let encoded = f32_to_fp8_e4m3(val);
            let decoded = FP8_E4M3_LUT[encoded as usize];
            let rel_err = (val - decoded).abs() / (val.abs() + 1e-4);
            assert!(rel_err < 0.15, "FP8 roundtrip error too large for {}: got {} (err {})", val, decoded, rel_err);
        }
    }

    #[test]
    fn test_activation_frame_binary_fidelity_v2() {
        let original_data = vec![0.1234f32, -45.67f32, 100.0f32, 0.0001f32];
        let frame = ActivationFrame::from_f32_slice(
            999,
            42,
            128,
            24,
            &original_data,
            FLAG_CLEAR_KV | FLAG_IS_PROMPT,
        );

        let encoded = frame.encode();
        let mut cursor = std::io::Cursor::new(encoded);
        let decoded = ActivationFrame::decode_sync(&mut cursor).unwrap();

        assert_eq!(decoded.header, frame.header);
        let decoded_f32 = decoded.to_f32_vec().unwrap();
        assert_eq!(decoded_f32, original_data);
    }
}
