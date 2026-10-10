use anyhow::Result;
use std::os::windows::io::AsRawHandle;
use std::process::Child;
use tracing::info;
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

/// A Win32 Job Object wrapper ensuring all spawned CUDA/RPC worker processes
/// and their child processes are immediately terminated and their GPU VRAM
/// is freed by the Windows kernel upon handle drop.
pub struct SafeProcessJob {
    job_handle: HANDLE,
}

unsafe impl Send for SafeProcessJob {}
unsafe impl Sync for SafeProcessJob {}

impl SafeProcessJob {
    pub fn new() -> Result<Self> {
        let job_handle = unsafe { CreateJobObjectW(std::ptr::null_mut(), std::ptr::null()) };
        if job_handle.is_null() {
            return Err(anyhow::anyhow!("Failed to create Windows Job Object"));
        }

        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

        let res = unsafe {
            SetInformationJobObject(
                job_handle,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const _,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };

        if res == 0 {
            unsafe { CloseHandle(job_handle) };
            return Err(anyhow::anyhow!(
                "Failed to configure JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE"
            ));
        }

        info!("🛡️ Windows Job Object initialized (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE active)");
        Ok(Self { job_handle })
    }

    pub fn assign_child(&self, child: &Child) -> Result<()> {
        let raw_handle = child.as_raw_handle() as HANDLE;
        let res = unsafe { AssignProcessToJobObject(self.job_handle, raw_handle) };
        if res == 0 {
            return Err(anyhow::anyhow!(
                "Failed to assign child process to Windows Job Object"
            ));
        }
        info!(
            pid = child.id(),
            "Assigned child process to Windows Job Object"
        );
        Ok(())
    }
}

impl Drop for SafeProcessJob {
    fn drop(&mut self) {
        if !self.job_handle.is_null() {
            unsafe { CloseHandle(self.job_handle) };
            info!(
                "🛡️ Windows Job Object handle closed (kernel reclaimed all child processes & VRAM)"
            );
        }
    }
}
