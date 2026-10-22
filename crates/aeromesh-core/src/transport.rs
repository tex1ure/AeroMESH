use anyhow::{bail, Result};
use std::io::Cursor;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

use crate::activation::{ActivationFrame, TokenResponseFrame};
use crate::shm::{SharedMemoryRingBuffer, DEFAULT_SHM_BUFFER_SIZE};

// ---------------------------------------------------------------------------
// TCP Transport (Low-Latency Tuned Sockets for Cross-Node Clusters)
// ---------------------------------------------------------------------------

pub struct TcpPipelineTransport {
    pub stream: TcpStream,
    write_buf: bytes::BytesMut,
}

impl TcpPipelineTransport {
    pub fn new(stream: TcpStream) -> Result<Self> {
        stream.set_nodelay(true)?;
        Self::configure_low_latency_stream(&stream);
        Ok(Self {
            stream,
            write_buf: bytes::BytesMut::with_capacity(64 * 1024),
        })
    }

    pub async fn connect(addr: SocketAddr) -> Result<Self> {
        let stream =
            tokio::time::timeout(Duration::from_secs(10), TcpStream::connect(addr)).await??;
        stream.set_nodelay(true)?;
        Self::configure_low_latency_stream(&stream);
        Ok(Self {
            stream,
            write_buf: bytes::BytesMut::with_capacity(64 * 1024),
        })
    }

    #[cfg(windows)]
    pub fn configure_low_latency_stream(stream: &TcpStream) {
        use std::os::windows::io::AsRawSocket;
        const SIO_TCP_SET_ACK_FREQUENCY: u32 = 0x98000017;
        const SIO_LOOPBACK_FAST_PATH: u32 = 0x98000010;
        const SOL_SOCKET: i32 = 0xffff;
        const SO_SNDBUF: i32 = 0x1001;
        const SO_RCVBUF: i32 = 0x1002;
        #[link(name = "ws2_32")]
        extern "system" {
            fn WSAIoctl(
                s: usize,
                io_control_code: u32,
                in_buf: *const std::ffi::c_void,
                in_buf_len: u32,
                out_buf: *mut std::ffi::c_void,
                out_buf_len: u32,
                bytes_returned: *mut u32,
                overlapped: *mut std::ffi::c_void,
                completion_routine: *mut std::ffi::c_void,
            ) -> i32;
            fn setsockopt(
                s: usize,
                level: i32,
                optname: i32,
                optval: *const std::ffi::c_void,
                optlen: i32,
            ) -> i32;
        }
        let _ = stream.set_nodelay(true);
        let raw = stream.as_raw_socket() as usize;
        let enable: u32 = 1;
        let mut bytes_returned: u32 = 0;
        let buf_size: i32 = 512 * 1024;
        unsafe {
            let _ = WSAIoctl(
                raw,
                SIO_TCP_SET_ACK_FREQUENCY,
                (&enable as *const u32).cast::<std::ffi::c_void>(),
                std::mem::size_of::<u32>() as u32,
                std::ptr::null_mut(),
                0,
                &mut bytes_returned,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            let _ = WSAIoctl(
                raw,
                SIO_LOOPBACK_FAST_PATH,
                (&enable as *const u32).cast::<std::ffi::c_void>(),
                std::mem::size_of::<u32>() as u32,
                std::ptr::null_mut(),
                0,
                &mut bytes_returned,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            let _ = setsockopt(
                raw,
                SOL_SOCKET,
                SO_SNDBUF,
                (&buf_size as *const i32).cast::<std::ffi::c_void>(),
                std::mem::size_of::<i32>() as i32,
            );
            let _ = setsockopt(
                raw,
                SOL_SOCKET,
                SO_RCVBUF,
                (&buf_size as *const i32).cast::<std::ffi::c_void>(),
                std::mem::size_of::<i32>() as i32,
            );
        }
    }

    #[cfg(not(windows))]
    pub fn configure_low_latency_stream(stream: &TcpStream) {
        let _ = stream.set_nodelay(true);
    }

    pub async fn send_activation(&mut self, frame: &ActivationFrame) -> Result<()> {
        self.write_buf.clear();
        self.write_buf
            .reserve(crate::activation::ActivationHeader::SIZE + frame.payload.len());
        self.write_buf
            .resize(crate::activation::ActivationHeader::SIZE, 0);
        frame
            .header
            .encode(&mut self.write_buf[..crate::activation::ActivationHeader::SIZE]);
        self.write_buf.extend_from_slice(&frame.payload);
        self.stream.write_all(&self.write_buf).await?;
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

        let send_rb =
            SharedMemoryRingBuffer::create_or_open(&forward_path, DEFAULT_SHM_BUFFER_SIZE)?;
        let recv_rb =
            SharedMemoryRingBuffer::create_or_open(&backward_path, DEFAULT_SHM_BUFFER_SIZE)?;

        Ok(Self { send_rb, recv_rb })
    }

    pub fn open_worker(channel_id: u16) -> Result<Self> {
        let base_dir = std::env::temp_dir().join("aeromesh_shm");
        let forward_path = base_dir.join(format!("channel_{}_fwd.bin", channel_id));
        let backward_path = base_dir.join(format!("channel_{}_bwd.bin", channel_id));

        let recv_rb =
            SharedMemoryRingBuffer::create_or_open(&forward_path, DEFAULT_SHM_BUFFER_SIZE)?;
        let send_rb =
            SharedMemoryRingBuffer::create_or_open(&backward_path, DEFAULT_SHM_BUFFER_SIZE)?;

        Ok(Self { send_rb, recv_rb })
    }

    pub async fn send_activation(&mut self, frame: &ActivationFrame) -> Result<()> {
        let bytes = frame.encode();
        self.send_rb.write_slice(&bytes)?;
        Ok(())
    }

    pub async fn recv_activation(&mut self) -> Result<ActivationFrame> {
        let start = std::time::Instant::now();
        let mut spins = 0u32;
        loop {
            if let Some(bytes) = self.recv_rb.read_slice()? {
                let mut cursor = Cursor::new(bytes);
                return ActivationFrame::decode_sync(&mut cursor);
            }
            if spins < 64 {
                spins += 1;
                tokio::task::yield_now().await;
                continue;
            }
            if start.elapsed() > Duration::from_secs(60) {
                bail!("SHM activation receive timeout");
            }
            tokio::time::sleep(Duration::from_micros(200)).await;
        }
    }

    pub async fn send_response(&mut self, frame: &TokenResponseFrame) -> Result<()> {
        let bytes = frame.encode();
        self.send_rb.write_slice(&bytes)?;
        Ok(())
    }

    pub async fn recv_response(&mut self) -> Result<TokenResponseFrame> {
        let start = std::time::Instant::now();
        let mut spins = 0u32;
        loop {
            if let Some(bytes) = self.recv_rb.read_slice()? {
                let mut cursor = Cursor::new(bytes);
                return TokenResponseFrame::decode_sync(&mut cursor);
            }
            if spins < 64 {
                spins += 1;
                tokio::task::yield_now().await;
                continue;
            }
            if start.elapsed() > Duration::from_secs(60) {
                bail!("SHM response receive timeout");
            }
            tokio::time::sleep(Duration::from_micros(200)).await;
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
        ip.is_loopback() || ip.is_unspecified()
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
