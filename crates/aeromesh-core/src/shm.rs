use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use anyhow::{ensure, Result};
use memmap2::{MmapMut, MmapOptions};

pub const SHM_MAGIC: &[u8; 4] = b"ASHM";
pub const SHM_HEADER_SIZE: usize = 192; // Cache-line aligned 64B * 3
pub const DEFAULT_SHM_BUFFER_SIZE: usize = 16 * 1024 * 1024; // 16MB ring buffer

/// SPSC Memory-Mapped Lock-Free Ring Buffer
pub struct SharedMemoryRingBuffer {
    mmap: MmapMut,
    capacity: usize,
    header_offset: usize,
}

impl SharedMemoryRingBuffer {
    pub fn create_or_open<P: AsRef<Path>>(path: P, capacity: usize) -> Result<Self> {
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let total_file_size = (SHM_HEADER_SIZE + capacity) as u64;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path_ref)?;

        file.set_len(total_file_size)?;

        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        // Initialize header if new
        if &mmap[0..4] != SHM_MAGIC {
            mmap[0..4].copy_from_slice(SHM_MAGIC);
            // Head atomic (offset 64)
            let head_atomic = unsafe { &*(&mmap[64] as *const u8 as *const AtomicUsize) };
            head_atomic.store(0, Ordering::SeqCst);
            // Tail atomic (offset 128)
            let tail_atomic = unsafe { &*(&mmap[128] as *const u8 as *const AtomicUsize) };
            tail_atomic.store(0, Ordering::SeqCst);
        }

        Ok(Self {
            mmap,
            capacity,
            header_offset: SHM_HEADER_SIZE,
        })
    }

    #[inline]
    fn head(&self) -> &AtomicUsize {
        unsafe { &*(&self.mmap[64] as *const u8 as *const AtomicUsize) }
    }

    #[inline]
    fn tail(&self) -> &AtomicUsize {
        unsafe { &*(&self.mmap[128] as *const u8 as *const AtomicUsize) }
    }

    /// Write a contiguous slice into the ring buffer (non-blocking)
    pub fn write_slice(&mut self, data: &[u8]) -> Result<()> {
        let len = data.len();
        let head = self.head().load(Ordering::Acquire);
        let tail = self.tail().load(Ordering::Acquire);

        let available = if head >= tail {
            self.capacity - (head - tail) - 1
        } else {
            tail - head - 1
        };

        ensure!(available >= len + 4, "SHM Ring Buffer Overflow (Available: {}, Requested: {})", available, len + 4);

        // Write 4-byte length prefix
        let len_bytes = (len as u32).to_le_bytes();
        self.write_raw(head, &len_bytes);
        let next_head = (head + 4) % self.capacity;

        // Write data
        self.write_raw(next_head, data);
        let final_head = (next_head + len) % self.capacity;

        self.head().store(final_head, Ordering::Release);
        Ok(())
    }

    fn write_raw(&mut self, offset: usize, data: &[u8]) {
        let base = self.header_offset;
        let end = offset + data.len();
        if end <= self.capacity {
            self.mmap[base + offset..base + end].copy_from_slice(data);
        } else {
            let first_part = self.capacity - offset;
            let second_part = data.len() - first_part;
            self.mmap[base + offset..base + self.capacity].copy_from_slice(&data[..first_part]);
            self.mmap[base..base + second_part].copy_from_slice(&data[first_part..]);
        }
    }

    /// Read next message from ring buffer into output buffer (non-blocking)
    pub fn read_slice(&mut self) -> Result<Option<Vec<u8>>> {
        let head = self.head().load(Ordering::Acquire);
        let tail = self.tail().load(Ordering::Acquire);

        if head == tail {
            return Ok(None); // Buffer empty
        }

        let available = if head >= tail {
            head - tail
        } else {
            self.capacity - (tail - head)
        };
        ensure!(
            available >= 4,
            "Corrupted ring buffer: available bytes ({}) less than length prefix header (4)",
            available
        );

        // Read 4-byte length prefix
        let mut len_bytes = [0u8; 4];
        self.read_raw(tail, &mut len_bytes);
        let len = u32::from_le_bytes(len_bytes) as usize;

        let max_allowed = (self.capacity.saturating_sub(4)).min(crate::activation::MAX_ACTIVATION_PAYLOAD_BYTES as usize);
        ensure!(
            len <= max_allowed && len <= available - 4,
            "Shared memory length prefix ({}) exceeds valid bounds (available payload: {}, max allowed: {})",
            len,
            available - 4,
            max_allowed
        );

        let next_tail = (tail + 4) % self.capacity;
        let mut data = vec![0u8; len];
        self.read_raw(next_tail, &mut data);
        let final_tail = (next_tail + len) % self.capacity;

        self.tail().store(final_tail, Ordering::Release);
        Ok(Some(data))
    }

    fn read_raw(&self, offset: usize, out: &mut [u8]) {
        let base = self.header_offset;
        let end = offset + out.len();
        if end <= self.capacity {
            out.copy_from_slice(&self.mmap[base + offset..base + end]);
        } else {
            let first_part = self.capacity - offset;
            let second_part = out.len() - first_part;
            out[..first_part].copy_from_slice(&self.mmap[base + offset..base + self.capacity]);
            out[first_part..].copy_from_slice(&self.mmap[base..base + second_part]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shm_ring_buffer_spsc() {
        let path = std::env::temp_dir().join("aeromesh_test_shm.bin");
        let _ = std::fs::remove_file(&path);

        let mut rb1 = SharedMemoryRingBuffer::create_or_open(&path, 1024 * 1024).unwrap();
        let mut rb2 = SharedMemoryRingBuffer::create_or_open(&path, 1024 * 1024).unwrap();

        let msg = b"AEROMESH_ZERO_COPY_SHM_TENSOR_ACTIVATION";
        rb1.write_slice(msg).unwrap();

        let recv = rb2.read_slice().unwrap().expect("should receive msg");
        assert_eq!(recv.as_slice(), msg);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_shm_ring_buffer_rejects_oversized_len() {
        let path = std::env::temp_dir().join("aeromesh_test_shm_corrupt.bin");
        let _ = std::fs::remove_file(&path);

        let mut rb = SharedMemoryRingBuffer::create_or_open(&path, 1024).unwrap();
        // Artificially corrupt the buffer: advance head to 10, but write a 1GB length prefix at tail (0)
        let malicious_len: u32 = 1024 * 1024 * 1024; // 1 GB
        rb.write_raw(0, &malicious_len.to_le_bytes());
        rb.head().store(10, Ordering::Release);
        rb.tail().store(0, Ordering::Release);

        let res = rb.read_slice();
        assert!(res.is_err(), "Must reject oversized length prefix without allocating");
        assert!(res.unwrap_err().to_string().contains("exceeds valid bounds"));

        let _ = std::fs::remove_file(&path);
    }
}
