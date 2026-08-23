use std::net::SocketAddr;
use std::time::Duration;
use anyhow::{Context, Result};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

use crate::activation::{ActivationFrame, TokenResponseFrame};
use crate::shm::{SharedMemoryRingBuffer, DEFAULT_SHM_CAPACITY};

pub const DEFAULT_TRANSPORT_TIMEOUT: Duration = Duration::from_secs(60);

/// Unified async transport trait abstracting over Memory-Mapped SHM and TCP sockets.
pub trait PipelineTransport: Send + Sync {
    fn send_activation<'a>(
        &'a mut self,
        frame: &'a ActivationFrame,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>>;

    fn recv_activation<'a>(
        &'a mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ActivationFrame>> + Send + 'a>>;

    fn send_response<'a>(
        &'a mut self,
        resp: &'a TokenResponseFrame,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>>;

    fn recv_response<'a>(
        &'a mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TokenResponseFrame>> + Send + 'a>>;

    fn is_shm(&self) -> bool;
}

// ---------------------------------------------------------------------------
// 1. Memory-Mapped Shared Memory Transport (Zero-Kernel-Copy)
// ---------------------------------------------------------------------------

pub struct ShmPipelineTransport {
    fwd_ring: SharedMemoryRingBuffer,
    ret_ring: SharedMemoryRingBuffer,
    is_coordinator: bool,
}

impl ShmPipelineTransport {
    /// Creates a coordinator SHM transport endpoint (writes forward activations, reads return tokens).
    pub fn new_coordinator(channel_id: u16) -> Result<Self> {
        let fwd_name = format!("coord_to_worker_{}", channel_id);
        let ret_name = format!("worker_to_coord_{}", channel_id);

        let fwd_ring = SharedMemoryRingBuffer::create(&fwd_name, DEFAULT_SHM_CAPACITY)?;
        let ret_ring = SharedMemoryRingBuffer::create(&ret_name, DEFAULT_SHM_CAPACITY)?;

        Ok(Self {
            fwd_ring,
            ret_ring,
            is_coordinator: true,
        })
    }

    /// Connects to a worker SHM transport endpoint (reads forward activations, writes return tokens).
    pub fn open_worker(channel_id: u16) -> Result<Self> {
        let fwd_name = format!("coord_to_worker_{}", channel_id);
        let ret_name = format!("worker_to_coord_{}", channel_id);

        let fwd_ring = SharedMemoryRingBuffer::open(&fwd_name)?;
        let ret_ring = SharedMemoryRingBuffer::open(&ret_name)?;

        Ok(Self {
            fwd_ring,
            ret_ring,
            is_coordinator: false,
        })
    }
}

impl PipelineTransport for ShmPipelineTransport {
    fn send_activation<'a>(
        &'a mut self,
        frame: &'a ActivationFrame,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let encoded = frame.encode();
            self.fwd_ring.write_message_async(&encoded, DEFAULT_TRANSPORT_TIMEOUT).await
        })
    }

    fn recv_activation<'a>(
        &'a mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ActivationFrame>> + Send + 'a>> {
        Box::pin(async move {
            let mut buf = Vec::new();
            self.fwd_ring.read_message_async(&mut buf, DEFAULT_TRANSPORT_TIMEOUT).await?;
            let mut cursor = std::io::Cursor::new(buf);
            ActivationFrame::decode_sync(&mut cursor)
        })
    }

    fn send_response<'a>(
        &'a mut self,
        resp: &'a TokenResponseFrame,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let encoded = resp.encode();
            self.ret_ring.write_message_async(&encoded, DEFAULT_TRANSPORT_TIMEOUT).await
        })
    }

    fn recv_response<'a>(
        &'a mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TokenResponseFrame>> + Send + 'a>> {
        Box::pin(async move {
            let mut buf = Vec::new();
            self.ret_ring.read_message_async(&mut buf, DEFAULT_TRANSPORT_TIMEOUT).await?;
            let mut cursor = std::io::Cursor::new(buf);
            TokenResponseFrame::decode_async(&mut cursor).await
        })
    }

    fn is_shm(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// 2. Low-Latency Tuned TCP Pipeline Transport
// ---------------------------------------------------------------------------

pub struct TcpPipelineTransport {
    pub stream: TcpStream,
}

impl TcpPipelineTransport {
    pub fn new(stream: TcpStream) -> Result<Self> {
        stream.set_nodelay(true)?;

        // Tune low-level socket parameters
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::io::AsRawFd;
            let fd = stream.as_raw_fd();
            unsafe {
                let quickack: libc::c_int = 1;
                libc::setsockopt(
                    fd,
                    libc::IPPROTO_TCP,
                    libc::TCP_QUICKACK,
                    &quickack as *const _ as *const libc::c_void,
                    std::mem::size_of_val(&quickack) as libc::socklen_t,
                );
            }
        }

        Ok(Self { stream })
    }

    pub async fn connect(addr: SocketAddr) -> Result<Self> {
        let stream = TcpStream::connect(addr).await
            .with_context(|| format!("Failed to connect to worker TCP {}", addr))?;
        Self::new(stream)
    }
}

impl PipelineTransport for TcpPipelineTransport {
    fn send_activation<'a>(
        &'a mut self,
        frame: &'a ActivationFrame,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let encoded = frame.encode();
            self.stream.write_all(&encoded).await?;
            self.stream.flush().await?;
            Ok(())
        })
    }

    fn recv_activation<'a>(
        &'a mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ActivationFrame>> + Send + 'a>> {
        Box::pin(async move {
            ActivationFrame::decode_async(&mut self.stream).await
        })
    }

    fn send_response<'a>(
        &'a mut self,
        resp: &'a TokenResponseFrame,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let encoded = resp.encode();
            self.stream.write_all(&encoded).await?;
            self.stream.flush().await?;
            Ok(())
        })
    }

    fn recv_response<'a>(
        &'a mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TokenResponseFrame>> + Send + 'a>> {
        Box::pin(async move {
            TokenResponseFrame::decode_async(&mut self.stream).await
        })
    }

    fn is_shm(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// 3. Pipeline Transport Factory (Auto Local Host Detection)
// ---------------------------------------------------------------------------

pub struct PipelineTransportFactory;

impl PipelineTransportFactory {
    /// Determines whether the target address represents the local host/machine.
    pub fn is_local_address(addr: &SocketAddr) -> bool {
        let ip = addr.ip();
        ip.is_loopback() || ip == std::net::Ipv4Addr::new(0, 0, 0, 0) || ip == std::net::Ipv6Addr::UNSPECIFIED
    }

    /// Automatically selects Zero-Copy Shared Memory (SHM) for local peers or TCP for remote nodes.
    pub async fn connect_auto(target_addr: SocketAddr) -> Result<Box<dyn PipelineTransport>> {
        if Self::is_local_address(&target_addr) {
            let channel_id = target_addr.port();
            // Try connecting via Shared Memory first
            if let Ok(shm_transport) = ShmPipelineTransport::new_coordinator(channel_id) {
                return Ok(Box::new(shm_transport));
            }
        }

        // Connect via Tuned TCP Stream
        let tcp_transport = TcpPipelineTransport::connect(target_addr).await?;
        Ok(Box::new(tcp_transport))
    }
}
