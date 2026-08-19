use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use anyhow::{bail, Context, Result};
use tracing::{error, info};
use crate::job_object::SafeProcessJob;

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub output_text: String,
    pub rpc_devices_detected: Vec<String>,
    pub performance_summary: Vec<String>,
}

pub struct EngineSupervisor {
    bin_dir: PathBuf,
    job: SafeProcessJob,
    active_children: Vec<Child>,
}

impl EngineSupervisor {
    pub fn new<P: AsRef<Path>>(workspace_root: P) -> Result<Self> {
        let ws = workspace_root.as_ref();
        let bin_candidate_1 = ws.join("bin");
        let bin_candidate_2 = ws.join("llama.cpp").join("build").join("bin").join("Release");

        let bin_dir = if bin_candidate_1.join("ggml-rpc-server.exe").exists() {
            bin_candidate_1
        } else if bin_candidate_2.join("ggml-rpc-server.exe").exists() {
            bin_candidate_2
        } else {
            bail!("Could not find llama.cpp binary directory in {:?}", ws);
        };

        info!(bin_dir = %bin_dir.display(), "Engine supervisor initialized with binary directory");
        let job = SafeProcessJob::new()?;

        Ok(Self {
            bin_dir,
            job,
            active_children: Vec::new(),
        })
    }

    pub fn spawn_rpc_worker(&mut self, host: &str, port: u16, use_cache: bool) -> Result<()> {
        let exe = self.bin_dir.join("ggml-rpc-server.exe");
        if !exe.exists() {
            bail!("ggml-rpc-server.exe not found at {:?}", exe);
        }

        info!(host = %host, port = port, cache = use_cache, "🚀 Spawning CUDA RPC backend worker");
        let mut cmd = Command::new(&exe);
        cmd.args(["--host", host, "--port", &port.to_string()]);
        if use_cache {
            cmd.arg("--cache");
        }

        let child = cmd
            .current_dir(&self.bin_dir)
            .spawn()
            .context(format!("Failed to start ggml-rpc-server at {}:{}", host, port))?;

        self.job.assign_child(&child)?;
        self.active_children.push(child);
        info!("✅ RPC Worker process running in Windows Job Object");
        Ok(())
    }

    pub fn run_completion(
        &mut self,
        model_path: &Path,
        peers: &[String],
        n_gpu_layers: i32,
        prompt: &str,
    ) -> Result<InferenceResult> {
        // Resolve absolute model path so working directory does not break file access
        let abs_model_path = if model_path.is_absolute() {
            model_path.to_path_buf()
        } else {
            std::env::current_dir()?.join(model_path)
        };

        if !abs_model_path.exists() {
            bail!("Model file does not exist at {:?}", abs_model_path);
        }

        let exe_completion = self.bin_dir.join("llama-completion.exe");
        let exe_cli = self.bin_dir.join("llama-cli.exe");

        let exe = if exe_completion.exists() {
            exe_completion
        } else if exe_cli.exists() {
            exe_cli
        } else {
            bail!("llama-completion.exe / llama-cli.exe not found in {:?}", self.bin_dir);
        };

        let mut cmd = Command::new(&exe);
        cmd.args([
            "-no-cnv",
            "-m",
            &abs_model_path.to_string_lossy(),
            "-ngl",
            &n_gpu_layers.to_string(),
            "-p",
            prompt,
            "-n",
            "64",
            "--no-warmup",
        ]);

        if !peers.is_empty() {
            let rpc_arg = peers.join(",");
            info!(rpc_peers = %rpc_arg, "Configuring multi-node RPC pipeline distribution");
            cmd.args(["--rpc", &rpc_arg]);
        }

        cmd.current_dir(&self.bin_dir);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        info!(model = %abs_model_path.display(), "🚀 Launching inference run");
        let output = cmd.output().context("Failed to execute llama binary")?;

        let stdout_str = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr_str = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() {
            error!(stderr = %stderr_str, "llama execution failed");
            bail!("llama execution failed with status: {}", output.status);
        }

        // Parse stderr to extract cluster offloading breakdown and performance metrics
        let mut rpc_devices = Vec::new();
        let mut perf_metrics = Vec::new();

        for line in stderr_str.lines() {
            let trimmed = line.trim();
            if trimmed.contains("RPC") || trimmed.contains("CUDA") || trimmed.contains("offload") || trimmed.contains("device") || trimmed.contains("remote") || trimmed.contains("backend") {
                if !trimmed.contains("DEBUG") && !trimmed.is_empty() {
                    rpc_devices.push(trimmed.to_string());
                }
            }
            if trimmed.contains("eval time") || trimmed.contains("prompt eval time") || trimmed.contains("total time") || trimmed.contains("tokens per second") {
                perf_metrics.push(trimmed.to_string());
            }
        }

        Ok(InferenceResult {
            output_text: stdout_str,
            rpc_devices_detected: rpc_devices,
            performance_summary: perf_metrics,
        })
    }

    pub fn shutdown_all(&mut self) {
        info!("Tearing down all active worker child processes");
        for mut child in self.active_children.drain(..) {
            let _ = child.kill();
        }
    }
}

impl Drop for EngineSupervisor {
    fn drop(&mut self) {
        self.shutdown_all();
    }
}
