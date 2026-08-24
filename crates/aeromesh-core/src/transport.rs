use std::io::Cursor;
use std::net::SocketAddr;
use std::time::Duration;
use anyhow::{bail, Result};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

use crate::activation::{ActivationFrame, TokenResponseFrame};
use crate::shm::{SharedMemoryRingBuffer, DEFAULT_SHM_BUFFER_SIZE};

// ---------------------------------------------------------------------------
// TCP Transport (Low-Latency Tuned Sockets for Cross-Node Clusters)
// ---------------------------------------------------------------------------

pub struct TcpPipelineTransport {
    pub stream: TcpStream,
}

impl TcpPipelineTransport {
    pub fn new(stream: TcpStream) -> Result<Self> {
        stream.set_nodelay(true)?;
        Ok(Self { stream })
    }

    pub async fn connect(addr: SocketAddr) -> Result<Self> {
        let stream = tokio::time::timeout(Duration::from_secs(10), TcpStream::connect(addr)).await??;
        stream.set_nodelay(true)?;
        Ok(Self { stream })
    }

    pub async fn send_activation(&mut self, frame: &ActivationFrame) -> Result<()> {
        let bytes = frame.encode();
        self.stream.write_all(&bytes).await?;
        self.stream.flush().await?;
        Ok(())
    }

    pub async fn recv_activation(&mut self) -> Result<ActivationFrame> {
        ActivationFrame::decode_async(&mut self.stream).await
    }

    pub async fn send_response(&mut self, frame: &TokenResponseFrame) -> Result<()> {
        let bytes = frame.encode();
        self.stream.write_all(&bytes).await?;
        self.stream.flush().await?;
        Ok(())
    }

    pub async fn recv_response(&mut self) -> Result<TokenResponseFrame> {
        TokenResponseFrame::decode_async(&mut self.stream).await
    }
}

// ---------------------------------------------------------------------------
// SHM Transport (Intra-Host Zero-Kernel-Copy Shared Memory)
// ---------------------------------------------------------------------------

pub struct ShmPipelineTransport {
    pub send_rb: SharedMemoryRingBuffer,
    pub recv_rb: SharedMemoryRingBuffer,
}

impl ShmPipelineTransport {
    pub fn new_coordinator(channel_id: u16) -> Result<Self> {
        let base_dir = std::env::temp_dir().join("aeromesh_shm");
        let forward_path = base_dir.join(format!("channel_{}_fwd.bin", channel_id));
        let backward_path = base_dir.join(format!("channel_{}_bwd.bin", channel_id));

        let send_rb = SharedMemoryRingBuffer::create_or_open(&forward_path, DEFAULT_SHM_BUFFER_SIZE)?;
        let recv_rb = SharedMemoryRingBuffer::create_or_open(&backward_path, DEFAULT_SHM_BUFFER_SIZE)?;

        Ok(Self { send_rb, recv_rb })
    }

    pub fn open_worker(channel_id: u16) -> Result<Self> {
        let base_dir = std::env::temp_dir().join("aeromesh_shm");
        let forward_path = base_dir.join(format!("channel_{}_fwd.bin", channel_id));
        let backward_path = base_dir.join(format!("channel_{}_bwd.bin", channel_id));

        let recv_rb = SharedMemoryRingBuffer::create_or_open(&forward_path, DEFAULT_SHM_BUFFER_SIZE)?;
        let send_rb = SharedMemoryRingBuffer::create_or_open(&backward_path, DEFAULT_SHM_BUFFER_SIZE)?;

        Ok(Self { send_rb, recv_rb })
    }

    pub async fn send_activation(&mut self, frame: &ActivationFrame) -> Result<()> {
        let bytes = frame.encode();
        self.send_rb.write_slice(&bytes)?;
        Ok(())
    }

    pub async fn recv_activation(&mut self) -> Result<ActivationFrame> {
        let start = std::time::Instant::now();
        loop {
            if let Some(bytes) = self.recv_rb.read_slice()? {
                let mut cursor = Cursor::new(bytes);
                return ActivationFrame::decode_sync(&mut cursor);
            }
            if start.elapsed() > Duration::from_secs(60) {
                bail!("SHM activation receive timeout");
            }
            tokio::task::yield_now().await;
        }
    }

    pub async fn send_response(&mut self, frame: &TokenResponseFrame) -> Result<()> {
        let bytes = frame.encode();
        self.send_rb.write_slice(&bytes)?;
        Ok(())
    }

    pub async fn recv_response(&mut self) -> Result<TokenResponseFrame> {
        let start = std::time::Instant::now();
        loop {
            if let Some(bytes) = self.recv_rb.read_slice()? {
                let mut cursor = Cursor::new(bytes);
                return TokenResponseFrame::decode_sync(&mut cursor);
            }
            if start.elapsed() > Duration::from_secs(60) {
                bail!("SHM response receive timeout");
            }
            tokio::task::yield_now().await;
        }
    }
}

// ---------------------------------------------------------------------------
// Unified PipelineTransport Enum (Zero Vtable Overhead & High Speed)
// ---------------------------------------------------------------------------

pub enum PipelineTransport {
    Tcp(TcpPipelineTransport),
    Shm(ShmPipelineTransport),
}

impl PipelineTransport {
    pub async fn send_activation(&mut self, frame: &ActivationFrame) -> Result<()> {
        match self {
            Self::Tcp(t) => t.send_activation(frame).await,
            Self::Shm(s) => s.send_activation(frame).await,
        }
    }

    pub async fn recv_activation(&mut self) -> Result<ActivationFrame> {
        match self {
            Self::Tcp(t) => t.recv_activation().await,
            Self::Shm(s) => s.recv_activation().await,
        }
    }

    pub async fn send_response(&mut self, frame: &TokenResponseFrame) -> Result<()> {
        match self {
            Self::Tcp(t) => t.send_response(frame).await,
            Self::Shm(s) => s.send_response(frame).await,
        }
    }

    pub async fn recv_response(&mut self) -> Result<TokenResponseFrame> {
        match self {
            Self::Tcp(t) => t.recv_response().await,
            Self::Shm(s) => s.recv_response().await,
        }
    }

    pub fn is_shm(&self) -> bool {
        matches!(self, Self::Shm(_))
    }
}

// ---------------------------------------------------------------------------
// Factory for Automatic Transport Selection
// ---------------------------------------------------------------------------

pub struct PipelineTransportFactory;

impl PipelineTransportFactory {
    pub fn is_local_address(addr: &SocketAddr) -> bool {
        let ip = addr.ip();
        ip.is_loopback() || ip.to_string() == "0.0.0.0" || ip.to_string() == "127.0.0.1"
    }

    pub async fn connect_auto(target: SocketAddr) -> Result<PipelineTransport> {
        if Self::is_local_address(&target) {
            let channel_id = target.port();
            if let Ok(shm_transport) = ShmPipelineTransport::new_coordinator(channel_id) {
                return Ok(PipelineTransport::Shm(shm_transport));
            }
        }
        let tcp_transport = TcpPipelineTransport::connect(target).await?;
        Ok(PipelineTransport::Tcp(tcp_transport))
    }
}
