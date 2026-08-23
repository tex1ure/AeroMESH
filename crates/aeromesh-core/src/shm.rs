use std::fs::OpenOptions;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use anyhow::{bail, ensure, Result};
use byteorder::{ByteOrder, LittleEndian};
use memmap2::MmapMut;

pub const SHM_MAGIC: [u8; 4] = *b"ASHM";
pub const SHM_VERSION: u32 = 1;
pub const SHM_HEADER_SIZE: usize = 192;
pub const DEFAULT_SHM_CAPACITY: usize = 16 * 1024 * 1024; // 16 MB ring buffer

/// Lock-free Single-Producer Single-Consumer (SPSC) Ring Buffer over Memory-Mapped Shared Memory.
/// Provides sub-microsecond true zero-kernel-copy intra-host IPC.
pub struct SharedMemoryRingBuffer {
    name: String,
    path: PathBuf,
    mmap: MmapMut,
    capacity: usize,
}

unsafe impl Send for SharedMemoryRingBuffer {}
unsafe impl Sync for SharedMemoryRingBuffer {}

impl SharedMemoryRingBuffer {
    fn shm_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("aeromesh_shm_{}.dat", name))
    }

    /// Creates and initializes a new Shared Memory Ring Buffer file.
    pub fn create(name: &str, capacity: usize) -> Result<Self> {
        let path = Self::shm_path(name);
        let total_size = SHM_HEADER_SIZE + capacity;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        file.set_len(total_size as u64)?;
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Zero header
        mmap[0..SHM_HEADER_SIZE].fill(0);

        // Write Magic, Version, and Capacity
        mmap[0..4].copy_from_slice(&SHM_MAGIC);
        LittleEndian::write_u32(&mut mmap[4..8], SHM_VERSION);
        LittleEndian::write_u64(&mut mmap[8..16], capacity as u64);

        // Write initial atomic head and tail pointers
        let head_ptr = unsafe { &*(mmap.as_ptr().add(64) as *const AtomicU64) };
        let tail_ptr = unsafe { &*(mmap.as_ptr().add(128) as *const AtomicU64) };
        head_ptr.store(0, Ordering::Release);
        tail_ptr.store(0, Ordering::Release);

        Ok(Self {
            name: name.to_string(),
            path,
            mmap,
            capacity,
        })
    }

    /// Opens an existing Shared Memory Ring Buffer.
    pub fn open(name: &str) -> Result<Self> {
        let path = Self::shm_path(name);
        if !path.exists() {
            bail!("SHM file not found: {:?}", path);
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;

        let mmap = unsafe { MmapMut::map_mut(&file)? };
        ensure!(mmap.len() >= SHM_HEADER_SIZE, "SHM file too small");

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&mmap[0..4]);
        ensure!(magic == SHM_MAGIC, "Invalid SHM magic: {:?}", magic);

        let version = LittleEndian::read_u32(&mmap[4..8]);
        ensure!(version == SHM_VERSION, "Unsupported SHM version: {}", version);

        let capacity = LittleEndian::read_u64(&mmap[8..16]) as usize;
        ensure!(mmap.len() >= SHM_HEADER_SIZE + capacity, "SHM file truncated");

        Ok(Self {
            name: name.to_string(),
            path,
            mmap,
            capacity,
        })
    }

    #[inline(always)]
    fn head_atomic(&self) -> &AtomicU64 {
        unsafe { &*(self.mmap.as_ptr().add(64) as *const AtomicU64) }
    }

    #[inline(always)]
    fn tail_atomic(&self) -> &AtomicU64 {
        unsafe { &*(self.mmap.as_ptr().add(128) as *const AtomicU64) }
    }

    #[inline(always)]
    fn data_slice_mut(&mut self) -> &mut [u8] {
        &mut self.mmap[SHM_HEADER_SIZE..SHM_HEADER_SIZE + self.capacity]
    }

    #[inline(always)]
    fn data_slice(&self) -> &[u8] {
        &self.mmap[SHM_HEADER_SIZE..SHM_HEADER_SIZE + self.capacity]
    }

    /// Non-blocking write of a framed message into the circular ring buffer.
    pub fn write_message(&mut self, message: &[u8]) -> Result<bool> {
        let msg_len = message.len();
        let total_frame_len = msg_len + 4;
        ensure!(total_frame_len < self.capacity, "Message size exceeds ring buffer capacity");

        let head = self.head_atomic().load(Ordering::Relaxed);
        let tail = self.tail_atomic().load(Ordering::Acquire);

        let used_bytes = head.wrapping_sub(tail) as usize;
        let available_bytes = self.capacity.saturating_sub(used_bytes);

        if available_bytes < total_frame_len {
            return Ok(false); // Ring buffer is full
        }

        let cap = self.capacity;
        let start_pos = (head as usize) % cap;

        // 1. Write 4-byte message length
        let mut len_buf = [0u8; 4];
        LittleEndian::write_u32(&mut len_buf, msg_len as u32);
        self.write_circular(start_pos, &len_buf);

        // 2. Write payload
        let payload_start = (start_pos + 4) % cap;
        self.write_circular(payload_start, message);

        // 3. Advance head pointer atomically
        self.head_atomic().store(head.wrapping_add(total_frame_len as u64), Ordering::Release);

        Ok(true)
    }

    #[inline(always)]
    fn write_circular(&mut self, start_pos: usize, src: &[u8]) {
        let cap = self.capacity;
        let end_pos = start_pos + src.len();
        let buffer = self.data_slice_mut();

        if end_pos <= cap {
            buffer[start_pos..end_pos].copy_from_slice(src);
        } else {
            let first_chunk = cap - start_pos;
            let second_chunk = src.len() - first_chunk;
            buffer[start_pos..cap].copy_from_slice(&src[0..first_chunk]);
            buffer[0..second_chunk].copy_from_slice(&src[first_chunk..]);
        }
    }

    /// Non-blocking read of the next framed message from the circular ring buffer.
    pub fn read_message(&self, out: &mut Vec<u8>) -> Result<bool> {
        let head = self.head_atomic().load(Ordering::Acquire);
        let tail = self.tail_atomic().load(Ordering::Relaxed);

        if tail == head {
            return Ok(false); // Ring buffer empty
        }

        let cap = self.capacity;
        let start_pos = (tail as usize) % cap;

        // 1. Read 4-byte message length
        let mut len_buf = [0u8; 4];
        self.read_circular(start_pos, &mut len_buf);
        let msg_len = LittleEndian::read_u32(&len_buf) as usize;
        ensure!(msg_len < cap, "Corrupted message length in SHM ring");

        let total_frame_len = msg_len + 4;

        // 2. Read payload
        out.resize(msg_len, 0);
        let payload_start = (start_pos + 4) % cap;
        self.read_circular(payload_start, out);

        // 3. Advance tail pointer atomically
        self.tail_atomic().store(tail.wrapping_add(total_frame_len as u64), Ordering::Release);

        Ok(true)
    }

    #[inline(always)]
    fn read_circular(&self, start_pos: usize, dst: &mut [u8]) {
        let cap = self.capacity;
        let end_pos = start_pos + dst.len();
        let buffer = self.data_slice();

        if end_pos <= cap {
            dst.copy_from_slice(&buffer[start_pos..end_pos]);
        } else {
            let first_chunk = cap - start_pos;
            let second_chunk = dst.len() - first_chunk;
            dst[0..first_chunk].copy_from_slice(&buffer[start_pos..cap]);
            dst[first_chunk..].copy_from_slice(&buffer[0..second_chunk]);
        }
    }

    /// Async write with exponential backoff spinning and timeout.
    pub async fn write_message_async(&mut self, message: &[u8], timeout: Duration) -> Result<()> {
        let start = Instant::now();
        let mut spins = 0;

        loop {
            if self.write_message(message)? {
                return Ok(());
            }

            if start.elapsed() > timeout {
                bail!("SHM write timed out after {:?}", timeout);
            }

            spins += 1;
            if spins < 32 {
                std::hint::spin_loop();
            } else if spins < 128 {
                tokio::task::yield_now().await;
            } else {
                tokio::time::sleep(Duration::from_micros(50)).await;
            }
        }
    }

    /// Async read with exponential backoff spinning and timeout.
    pub async fn read_message_async(&self, out: &mut Vec<u8>, timeout: Duration) -> Result<()> {
        let start = Instant::now();
        let mut spins = 0;

        loop {
            if self.read_message(out)? {
                return Ok(());
            }

            if start.elapsed() > timeout {
                bail!("SHM read timed out after {:?}", timeout);
            }

            spins += 1;
            if spins < 32 {
                std::hint::spin_loop();
            } else if spins < 128 {
                tokio::task::yield_now().await;
            } else {
                tokio::time::sleep(Duration::from_micros(50)).await;
            }
        }
    }
}

impl Drop for SharedMemoryRingBuffer {
    fn drop(&mut self) {
        // Shared memory remains accessible for other process until unlinked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shm_ring_buffer_spsc() {
        let channel_name = format!("test_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());
        let mut producer = SharedMemoryRingBuffer::create(&channel_name, 1024 * 1024).unwrap();
        let consumer = SharedMemoryRingBuffer::open(&channel_name).unwrap();

        let message1 = b"Hello Zero-Copy Distributed AeroMesh Shared Memory!";
        let message2 = vec![42u8; 8192];

        assert!(producer.write_message(message1).unwrap());
        assert!(producer.write_message(&message2).unwrap());

        let mut read_buf = Vec::new();
        assert!(consumer.read_message(&mut read_buf).unwrap());
        assert_eq!(read_buf, message1);

        assert!(consumer.read_message(&mut read_buf).unwrap());
        assert_eq!(read_buf, message2);

        // Buffer should now be empty
        assert!(!consumer.read_message(&mut read_buf).unwrap());
    }
}
