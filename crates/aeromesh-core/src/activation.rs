use std::io::Read;
use anyhow::{bail, ensure, Result};
use byteorder::{BigEndian, ByteOrder};
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

/// Binary frame header for high-speed P2P activation vector streaming with session & KV tracking.
/// Fixed size: 42 bytes (endian-safe BigEndian serialization).
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
    pub dtype: u8,              // 0 = FP32, 1 = FP16, 2 = BF16, 3 = INT8-Row, 4 = FP8-E4M3
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
        let payload_bytes = payload.len() as u32;
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
            payload_bytes,
        };

        Ok(Self { header, payload })
    }

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
        let seq_len = sequence_length as usize;
        let d = hidden_dim as usize;
        assert_eq!(matrix.len(), seq_len * d, "Matrix size mismatch");

        match dtype {
            ActivationDtype::Int8PerRow => {
                // Per-row layout: For each token s in [0, S): [scale_s: f32 (4B)][quant_data: i8 * d]
                let row_bytes = 4 + d;
                let mut payload = BytesMut::with_capacity(seq_len * row_bytes);

                for s in 0..seq_len {
                    let row = &matrix[s * d..(s + 1) * d];
                    let mut max_abs = 0.0f32;
                    for &val in row {
                        let abs = val.abs();
                        if abs > max_abs {
                            max_abs = abs;
                        }
                    }

                    let scale = max_abs / 127.0 + 1e-8;
                    let inv_scale = 1.0 / scale;

                    payload.extend_from_slice(&scale.to_le_bytes());
                    for &val in row {
                        let q = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;
                        payload.extend_from_slice(&(q as u8).to_le_bytes());
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
                    layer_index,
                    hidden_dim,
                    dtype: ActivationDtype::Fp8E4M3 as u8,
                    flags,
                    payload_bytes: payload.len() as u32,
                };

                Self { header, payload }
            }
            _ => {
                // Default uncompressed FP32
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
                    layer_index,
                    hidden_dim,
                    dtype: ActivationDtype::Fp32 as u8,
                    flags,
                    payload_bytes: payload.len() as u32,
                };
                Self { header, payload }
            }
        }
    }

    pub fn dequantize_into(&self, out: &mut [f32]) -> Result<()> {
        let seq_len = self.header.sequence_length as usize;
        let d = self.header.hidden_dim as usize;
        let total_elements = seq_len * d;
        ensure!(out.len() >= total_elements, "Output buffer too small for dequantization");

        match ActivationDtype::from_u8(self.header.dtype)? {
            ActivationDtype::Int8PerRow => {
                let row_bytes = 4 + d;
                ensure!(self.payload.len() == seq_len * row_bytes, "Invalid INT8-per-row payload length");

                let mut offset = 0;
                for s in 0..seq_len {
                    let scale_bytes: [u8; 4] = self.payload[offset..offset + 4].try_into()?;
                    let scale = f32::from_le_bytes(scale_bytes);
                    offset += 4;

                    let row_quant = &self.payload[offset..offset + d];
                    offset += d;

                    let out_row = &mut out[s * d..(s + 1) * d];
                    for i in 0..d {
                        out_row[i] = (row_quant[i] as i8 as f32) * scale;
                    }
                }
            }
            ActivationDtype::Fp8E4M3 => {
                ensure!(self.payload.len() == total_elements, "Invalid FP8 payload length");
                for i in 0..total_elements {
                    out[i] = fp8_e4m3_to_f32(self.payload[i]);
                }
            }
            ActivationDtype::Fp32 => {
                ensure!(self.payload.len() == total_elements * 4, "Invalid FP32 payload length");
                for i in 0..total_elements {
                    let chunk: [u8; 4] = self.payload[i * 4..(i + 1) * 4].try_into()?;
                    out[i] = f32::from_le_bytes(chunk);
                }
            }
            ActivationDtype::Fp16 => {
                ensure!(self.payload.len() == total_elements * 2, "Invalid FP16 payload length");
                for i in 0..total_elements {
                    let chunk: [u8; 2] = self.payload[i * 2..(i + 1) * 2].try_into()?;
                    let half = u16::from_le_bytes(chunk);
                    out[i] = half_to_float(half);
                }
            }
            ActivationDtype::Bf16 => {
                ensure!(self.payload.len() == total_elements * 2, "Invalid BF16 payload length");
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
        let total_elements = (self.header.sequence_length as usize) * (self.header.hidden_dim as usize);
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

// ---------------------------------------------------------------------------
// FP8 (E4M3) Conversion & Lookup Table
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
        // Subnormal
        let shift = 1 - target_exp;
        if shift > 3 {
            return (sign << 7) as u8;
        }
        let full_mant = (1 << 3) | (mant >> 20);
        let q_mant = (full_mant >> shift) & 0x07;
        return ((sign << 7) | q_mant) as u8;
    }

    if target_exp >= 15 {
        return ((sign << 7) | 0x7E) as u8; // Clamp to max normal
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
        // Subnormal
        let val = (mant as f32) / 8.0 * (2.0f32.powi(-6));
        return if sign == 1 { -val } else { val };
    }

    let mant_f = 1.0 + (mant as f32) / 8.0;
    let exp_f = 2.0f32.powi(exp as i32 - 7);
    let val = mant_f * exp_f;
    if sign == 1 { -val } else { val }
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

    pub fn decode_sync<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        ensure!(magic == TOKEN_RESP_MAGIC, "Invalid token response magic: {:?}", magic);

        let mut ver_buf = [0u8; 2];
        reader.read_exact(&mut ver_buf)?;
        let version = u16::from_be_bytes(ver_buf);
        ensure!(version == PROTOCOL_VERSION, "Unsupported token response protocol version: {}", version);

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

        buf.extend_from_slice(&(arch_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(arch_bytes);

        buf.extend_from_slice(&(csum_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(csum_bytes);

        buf.freeze()
    }

    pub async fn decode_async<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).await?;
        ensure!(magic == HANDSHAKE_REQ_MAGIC, "Invalid handshake request magic: {:?}", magic);

        let version = reader.read_u16().await?;
        ensure!(version == PROTOCOL_VERSION, "Unsupported handshake protocol version: {}", version);

        let hidden_dim = reader.read_u32().await?;
        let total_layers = reader.read_u32().await?;
        let worker_layer_start = reader.read_u32().await?;
        let worker_layer_end = reader.read_u32().await?;

        let arch_len = reader.read_u16().await? as usize;
        let mut arch_buf = vec![0u8; arch_len];
        reader.read_exact(&mut arch_buf).await?;
        let model_architecture = String::from_utf8(arch_buf)?;

        let csum_len = reader.read_u16().await? as usize;
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

/// Handshake Response returned from Worker to Coordinator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandshakeResponse {
    pub version: u16,
    pub accepted: bool,
    pub worker_layer_count: u32,
    pub error_message: String,
}

impl HandshakeResponse {
    pub fn ok() -> Self {
        Self {
            version: PROTOCOL_VERSION,
            accepted: true,
            worker_layer_count: 0,
            error_message: String::new(),
        }
    }

    pub fn encode(&self) -> Bytes {
        let err_bytes = self.error_message.as_bytes();
        let mut buf = BytesMut::with_capacity(16 + err_bytes.len());

        buf.extend_from_slice(&HANDSHAKE_RESP_MAGIC);
        buf.extend_from_slice(&self.version.to_be_bytes());
        buf.extend_from_slice(&[if self.accepted { 1 } else { 0 }]);
        buf.extend_from_slice(&self.worker_layer_count.to_be_bytes());

        buf.extend_from_slice(&(err_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(err_bytes);

        buf.freeze()
    }

    pub async fn decode_async<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).await?;
        ensure!(magic == HANDSHAKE_RESP_MAGIC, "Invalid handshake response magic: {:?}", magic);

        let version = reader.read_u16().await?;
        ensure!(version == PROTOCOL_VERSION, "Unsupported handshake response protocol version: {}", version);

        let accepted = reader.read_u8().await? != 0;
        let worker_layer_count = reader.read_u32().await?;

        let err_len = reader.read_u16().await? as usize;
        let mut err_buf = vec![0u8; err_len];
        reader.read_exact(&mut err_buf).await?;
        let error_message = String::from_utf8(err_buf)?;

        Ok(Self {
            version,
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
        let exp_f = (exp + 127 - 15) as u32;
        let mant_f = mant << 13;
        f32::from_bits((sign << 31) | (exp_f << 23) | mant_f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_per_row_quantization_accuracy_and_outlier_resilience() {
        let hidden_dim = 1536;
        let seq_len = 4;
        let mut data = vec![0.0f32; seq_len * hidden_dim];

        // Fill with synthetic activations + deliberate outlier at row 2, col 50
        for i in 0..data.len() {
            data[i] = ((i % 100) as f32 - 50.0) / 50.0; // [-1.0, 1.0]
        }
        data[2 * hidden_dim + 50] = 42.5; // Outlier

        let frame = ActivationFrame::from_f32_matrix_quantized(
            1234, 1, seq_len as u32, 0, 13, hidden_dim as u32,
            &data, ActivationDtype::Int8PerRow, 0,
        );

        let mut recovered = vec![0.0f32; data.len()];
        frame.dequantize_into(&mut recovered).unwrap();

        // Check outlier recovery accuracy
        let outlier_orig = data[2 * hidden_dim + 50];
        let outlier_rec = recovered[2 * hidden_dim + 50];
        assert!((outlier_orig - outlier_rec).abs() < 0.5, "Outlier should be preserved accurately");

        // Check average row error
        let mut max_err = 0.0f32;
        for i in 0..data.len() {
            let diff = (data[i] - recovered[i]).abs();
            if diff > max_err {
                max_err = diff;
            }
        }
        assert!(max_err < 0.6, "Max error under INT8 scaling should be minimal");
    }

    #[test]
    fn test_fp8_e4m3_roundtrip() {
        let values = [-5.0f32, -1.0, -0.25, 0.0, 0.25, 1.0, 3.5, 15.0];
        for &v in &values {
            let fp8 = f32_to_fp8_e4m3(v);
            let rec = fp8_e4m3_to_f32(fp8);
            let diff = (v - rec).abs();
            assert!(diff <= 0.5 * v.abs().max(0.1), "FP8 roundtrip error too large for {}", v);
        }
    }
}
