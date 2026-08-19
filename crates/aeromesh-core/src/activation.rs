use std::io::Read;
use anyhow::{bail, ensure, Result};
use byteorder::{BigEndian, ByteOrder};
use bytes::{Bytes, BytesMut};
use tokio::io::AsyncReadExt;

pub const ACTIVATION_MAGIC: [u8; 4] = *b"AERO";
pub const TOKEN_RESP_MAGIC: [u8; 4] = *b"ATOK";
pub const PROTOCOL_VERSION: u16 = 1;

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

/// Binary frame header for high-speed P2P activation vector streaming.
/// Fixed size: 26 bytes (endian-safe BigEndian serialization).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationHeader {
    pub magic: [u8; 4],      // b"AERO"
    pub version: u16,       // 1
    pub sequence_id: u64,   // Token generation sequence counter
    pub layer_index: u16,   // Boundary layer index (e.g. 24)
    pub hidden_dim: u32,    // Hidden dimension (e.g. 4096)
    pub dtype: u8,          // 0 = FP32, 1 = FP16, 2 = BF16
    pub payload_bytes: u32, // Length of payload following header
}

impl ActivationHeader {
    pub const SIZE: usize = 26;

    pub fn encode(&self, out: &mut [u8]) {
        assert!(out.len() >= Self::SIZE);
        out[0..4].copy_from_slice(&self.magic);
        BigEndian::write_u16(&mut out[4..6], self.version);
        BigEndian::write_u64(&mut out[6..14], self.sequence_id);
        BigEndian::write_u16(&mut out[14..16], self.layer_index);
        BigEndian::write_u32(&mut out[16..20], self.hidden_dim);
        out[20] = self.dtype;
        out[21] = 0; // reserved padding byte
        BigEndian::write_u32(&mut out[22..26], self.payload_bytes);
    }

    pub fn decode(src: &[u8]) -> Result<Self> {
        ensure!(src.len() >= Self::SIZE, "Buffer too short for ActivationHeader");
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&src[0..4]);
        ensure!(magic == ACTIVATION_MAGIC, "Invalid activation magic: {:?}", magic);

        let version = BigEndian::read_u16(&src[4..6]);
        ensure!(version == PROTOCOL_VERSION, "Unsupported activation protocol version: {}", version);

        let sequence_id = BigEndian::read_u64(&src[6..14]);
        let layer_index = BigEndian::read_u16(&src[14..16]);
        let hidden_dim = BigEndian::read_u32(&src[16..20]);
        let dtype = src[20];
        let payload_bytes = BigEndian::read_u32(&src[22..26]);

        Ok(Self {
            magic,
            version,
            sequence_id,
            layer_index,
            hidden_dim,
            dtype,
            payload_bytes,
        })
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
        sequence_id: u64,
        layer_index: u16,
        hidden_dim: u32,
        dtype: ActivationDtype,
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
            sequence_id,
            layer_index,
            hidden_dim,
            dtype: dtype as u8,
            payload_bytes: payload.len() as u32,
        };

        Ok(Self { header, payload })
    }

    pub fn from_f32_slice(
        sequence_id: u64,
        layer_index: u16,
        activations: &[f32],
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
            sequence_id,
            layer_index,
            hidden_dim: activations.len() as u32,
            dtype: ActivationDtype::Fp32 as u8,
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
                    // Standard FP16 to FP32 conversion
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

/// Token response frame streamed from final stage worker back to coordinator.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenResponseFrame {
    pub sequence_id: u64,
    pub token_id: i32,
    pub is_eos: bool,
    pub token_text: String,
    pub eval_time_ms: f32,
}

impl TokenResponseFrame {
    pub const HEADER_SIZE: usize = 4 + 2 + 8 + 4 + 1 + 4 + 4; // 27 bytes

    pub fn encode(&self) -> Bytes {
        let text_bytes = self.token_text.as_bytes();
        let mut buf = BytesMut::with_capacity(Self::HEADER_SIZE + text_bytes.len());
        buf.extend_from_slice(&TOKEN_RESP_MAGIC);
        buf.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
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
            // Subnormal
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
            f32::from_bits((sign << 31) | (0xff << 23)) // Inf
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (mant << 13)) // NaN
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
    fn test_activation_frame_binary_fidelity() {
        let original_data = vec![0.1234f32, -45.67f32, 100.0f32, 0.0001f32];
        let frame = ActivationFrame::from_f32_slice(42, 24, &original_data);

        assert_eq!(frame.header.sequence_id, 42);
        assert_eq!(frame.header.layer_index, 24);
        assert_eq!(frame.header.hidden_dim, 4);
        assert_eq!(frame.header.dtype, ActivationDtype::Fp32 as u8);
        assert_eq!(frame.header.payload_bytes, 16);

        let encoded = frame.encode();
        assert_eq!(encoded.len(), ActivationHeader::SIZE + 16);

        let mut cursor = std::io::Cursor::new(encoded);
        let decoded = ActivationFrame::decode_sync(&mut cursor).unwrap();

        assert_eq!(decoded.header, frame.header);
        let decoded_f32 = decoded.to_f32_vec().unwrap();
        assert_eq!(decoded_f32, original_data);
    }

    #[tokio::test]
    async fn test_token_response_roundtrip() {
        let token_resp = TokenResponseFrame {
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
