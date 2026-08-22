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
}

impl ActivationDtype {
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(Self::Fp32),
            1 => Ok(Self::Fp16),
            2 => Ok(Self::Bf16),
            _ => bail!("Unsupported activation dtype: {}", val),
        }
    }

    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::Fp32 => 4,
            Self::Fp16 | Self::Bf16 => 2,
        }
    }
}

/// Binary frame header for high-speed P2P activation vector streaming with session & KV tracking.
/// Fixed size: 38 bytes (endian-safe BigEndian serialization).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationHeader {
    pub magic: [u8; 4],          // b"AERO"
    pub version: u16,           // 2
    pub session_id: u64,        // Unique multi-turn chat session ID
    pub sequence_id: u64,       // Token generation sequence counter
    pub token_position: u32,    // Context position offset in KV-cache (pos)
    pub layer_index: u16,       // Boundary layer index (e.g. 24)
    pub hidden_dim: u32,        // Hidden dimension (e.g. 4096)
    pub dtype: u8,              // 0 = FP32, 1 = FP16, 2 = BF16
    pub flags: u8,              // Bitmask: FLAG_CLEAR_KV (0x01), FLAG_IS_PROMPT (0x02), etc.
    pub payload_bytes: u32,     // Length of payload following header
}

impl ActivationHeader {
    pub const SIZE: usize = 38;

    pub fn encode(&self, out: &mut [u8]) {
        assert!(out.len() >= Self::SIZE);
        out[0..4].copy_from_slice(&self.magic);
        BigEndian::write_u16(&mut out[4..6], self.version);
        BigEndian::write_u64(&mut out[6..14], self.session_id);
        BigEndian::write_u64(&mut out[14..22], self.sequence_id);
        BigEndian::write_u32(&mut out[22..26], self.token_position);
        BigEndian::write_u16(&mut out[26..28], self.layer_index);
        BigEndian::write_u32(&mut out[28..32], self.hidden_dim);
        out[32] = self.dtype;
        out[33] = self.flags;
        BigEndian::write_u32(&mut out[34..38], self.payload_bytes);
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
        let token_position = BigEndian::read_u32(&src[22..26]);
        let layer_index = BigEndian::read_u16(&src[26..28]);
        let hidden_dim = BigEndian::read_u32(&src[28..32]);
        let dtype = src[32];
        let flags = src[33];
        let payload_bytes = BigEndian::read_u32(&src[34..38]);

        Ok(Self {
            magic,
            version,
            session_id,
            sequence_id,
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

/// An in-memory activation frame ready for P2P network transport.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationFrame {
    pub header: ActivationHeader,
    pub payload: Bytes,
}

impl ActivationFrame {
    pub fn new(
        session_id: u64,
        sequence_id: u64,
        token_position: u32,
        layer_index: u16,
        hidden_dim: u32,
        dtype: ActivationDtype,
        flags: u8,
        payload: Bytes,
    ) -> Result<Self> {
        let expected_size = (hidden_dim as usize) * dtype.bytes_per_element();
        ensure!(
            payload.len() == expected_size,
            "Payload length mismatch: expected {} bytes, got {}",
            expected_size,
            payload.len()
        );

        let header = ActivationHeader {
            magic: ACTIVATION_MAGIC,
            version: PROTOCOL_VERSION,
            session_id,
            sequence_id,
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
            token_position,
            layer_index,
            hidden_dim: activations.len() as u32,
            dtype: ActivationDtype::Fp32 as u8,
            flags,
            payload_bytes: bytes.len() as u32,
        };

        Self {
            header,
            payload: bytes.freeze(),
        }
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        let dtype = ActivationDtype::from_u8(self.header.dtype)?;
        match dtype {
            ActivationDtype::Fp32 => {
                let count = self.payload.len() / 4;
                let mut result = Vec::with_capacity(count);
                for i in 0..count {
                    let bits = BigEndian::read_u32(&self.payload[i * 4..(i + 1) * 4]);
                    result.push(f32::from_bits(bits));
                }
                Ok(result)
            }
            ActivationDtype::Fp16 => {
                let count = self.payload.len() / 2;
                let mut result = Vec::with_capacity(count);
                for i in 0..count {
                    let half_bits = BigEndian::read_u16(&self.payload[i * 2..(i + 1) * 2]);
                    let f = half_to_float(half_bits);
                    result.push(f);
                }
                Ok(result)
            }
            ActivationDtype::Bf16 => {
                let count = self.payload.len() / 2;
                let mut result = Vec::with_capacity(count);
                for i in 0..count {
                    let bf_bits = BigEndian::read_u16(&self.payload[i * 2..(i + 1) * 2]);
                    let f = f32::from_bits((bf_bits as u32) << 16);
                    result.push(f);
                }
                Ok(result)
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_frame_binary_fidelity_v2() {
        let original_data = vec![0.1234f32, -45.67f32, 100.0f32, 0.0001f32];
        let frame = ActivationFrame::from_f32_slice(
            999, // session_id
            42,  // sequence_id
            128, // token_position
            24,  // layer_index
            &original_data,
            FLAG_CLEAR_KV | FLAG_IS_PROMPT,
        );

        assert_eq!(frame.header.session_id, 999);
        assert_eq!(frame.header.sequence_id, 42);
        assert_eq!(frame.header.token_position, 128);
        assert_eq!(frame.header.layer_index, 24);
        assert_eq!(frame.header.hidden_dim, 4);
        assert_eq!(frame.header.flags, FLAG_CLEAR_KV | FLAG_IS_PROMPT);
        assert!(frame.header.has_flag(FLAG_CLEAR_KV));

        let encoded = frame.encode();
        assert_eq!(encoded.len(), ActivationHeader::SIZE + 16);

        let mut cursor = std::io::Cursor::new(encoded);
        let decoded = ActivationFrame::decode_sync(&mut cursor).unwrap();

        assert_eq!(decoded.header, frame.header);
        let decoded_f32 = decoded.to_f32_vec().unwrap();
        assert_eq!(decoded_f32, original_data);
    }

    #[tokio::test]
    async fn test_handshake_roundtrip() {
        let req = HandshakeRequest {
            version: PROTOCOL_VERSION,
            model_architecture: "llama".to_string(),
            hidden_dim: 4096,
            total_layers: 48,
            worker_layer_start: 25,
            worker_layer_end: 47,
            checksum_prefix: "a1b2c3d4".to_string(),
        };

        let encoded = req.encode();
        let mut cursor = std::io::Cursor::new(encoded);
        let decoded = HandshakeRequest::decode_async(&mut cursor).await.unwrap();
        assert_eq!(decoded, req);

        let resp = HandshakeResponse::ok();
        let resp_enc = resp.encode();
        let mut resp_cursor = std::io::Cursor::new(resp_enc);
        let resp_dec = HandshakeResponse::decode_async(&mut resp_cursor).await.unwrap();
        assert_eq!(resp_dec, resp);
    }

    #[tokio::test]
    async fn test_token_response_roundtrip() {
        let token_resp = TokenResponseFrame {
            session_id: 12345,
            sequence_id: 101,
            token_id: 4892,
            is_eos: false,
            token_text: "distributed".to_string(),
            eval_time_ms: 12.5,
        };

        let encoded = token_resp.encode();
        let mut cursor = std::io::Cursor::new(encoded);
        let decoded = TokenResponseFrame::decode_async(&mut cursor).await.unwrap();

        assert_eq!(decoded, token_resp);
    }
}
