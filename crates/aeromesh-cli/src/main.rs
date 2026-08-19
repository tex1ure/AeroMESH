use std::net::SocketAddr;
use std::path::PathBuf;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use aeromesh_core::tailscale::TailscaleInspector;
use aeromesh_engine::{
    inspect_gguf_file, resolve_model_path, EngineSupervisor, GgufSliceLoader,
    LayerSliceConfig, PipelineCoordinatorClient, PipelineWorkerService,
};

#[derive(Parser, Debug)]
#[command(name = "aeromesh")]
#[command(about = "AeroMesh: Distributed Fault-Tolerant LLM Cluster Engine in Rust", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// View real-time cluster connectivity and node status over Tailscale
    Status {
        /// RPC Port to check on worker nodes
        #[arg(long, default_value_t = 50052)]
        port: u16,
    },

    /// Verify GGUF model headers, layer count, and cross-node hash integrity
    ModelCheck {
        /// Path to the .gguf model file (optional, auto-discovers if omitted)
        #[arg(value_name = "FILE")]
        path: Option<PathBuf>,
    },

    /// Inspect GGUF layer ranges, tensor boundaries, and VRAM memory-mapping footprint
    SliceInfo {
        /// Path to the .gguf model file
        #[arg(long)]
        model: Option<PathBuf>,

        /// Specific layer range (e.g. "0..24" or "25..48")
        #[arg(long)]
        layers: Option<String>,
    },

    /// Probe Tailscale link quality (Direct WireGuard vs DERP, RTT latency)
    Probe {
        /// Target Tailscale address (e.g. 100.122.125.95:50052)
        #[arg(value_name = "TARGET")]
        target: String,
    },

    /// Start a worker node (Zero-Weight Pipeline or Supervised CUDA RPC)
    Worker {
        /// Host IP to bind
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Port to bind
        #[arg(long, default_value_t = 50052)]
        port: u16,

        /// Path to local GGUF model file for Zero-Weight Pipeline mode
        #[arg(long)]
        model: Option<PathBuf>,

        /// Layer range for this worker stage (e.g. "16..32" or "auto")
        #[arg(long)]
        layers: Option<String>,

        /// Enable local tensor disk caching (for RPC backend mode)
        #[arg(long, default_value_t = true)]
        cache: bool,
    },

    /// Run the multi-node cluster coordinator
    Coordinator {
        /// Path to GGUF model file (e.g. models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf)
        #[arg(long)]
        model: Option<PathBuf>,

        /// Comma-separated list of remote RPC peer addresses (e.g. 100.122.125.95:50052)
        #[arg(long)]
        peers: Option<String>,

        /// Execution mode: "rpc" (CUDA RPC backend across cluster GPUs) or "pipeline" (P2P activation streaming)
        #[arg(long, default_value = "rpc")]
        mode: String,

        /// Number of layers to offload to GPU (-1 for all, RPC mode)
        #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
        ngl: i32,

        /// Prompt text to execute
        #[arg(long, default_value = "Write a short sentence about distributed GPU clusters.")]
        prompt: String,

        /// Maximum tokens to generate (pipeline mode)
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let cli = Cli::parse();
    let current_dir = std::env::current_dir()?;

    match cli.command {
        Commands::Status { port } => {
            info!("🛰️ Scanning Tailscale cluster mesh on port {}...", port);
            let nodes = TailscaleInspector::discover_cluster_nodes(port).await?;

            println!("\n==========================================================================================");
            println!("                           AEROMESH CLUSTER NODE STATUS                                   ");
            println!("==========================================================================================");
            println!("{:<24} {:<16} {:<12} {:<18} {:<12} {:<16}", "NODE / HOSTNAME", "TAILSCALE IP", "TS STATUS", "LINK TYPE", "LATENCY", "RPC DAEMON");
            println!("------------------------------------------------------------------------------------------");

            let mut ready_count = 0;
            for node in &nodes {
                let ts_status = if node.is_tailscale_active { "ONLINE" } else { "OFFLINE" };
                let link_type = if node.is_self {
                    "Local Host"
                } else if node.is_direct_wireguard {
                    "Direct WireGuard"
                } else if let Some(relay) = &node.relay_region {
                    &format!("DERP ({})", relay)
                } else {
                    "DERP Relay"
                };

                let latency_str = if let Some(rtt) = node.rtt_ms {
                    format!("{:.1} ms", rtt)
                } else {
                    "N/A".into()
                };

                let rpc_status = if node.is_self {
                    "Coordinator"
                } else if node.rpc_online {
                    if node.cluster_ready { "✅ READY" } else { "⚠️ HIGH PING" }
                } else {
                    "❌ STOPPED"
                };

                if node.is_self || node.cluster_ready {
                    ready_count += 1;
                }

                println!("{:<24} {:<16} {:<12} {:<18} {:<12} {:<16}", node.host_name, node.ip, ts_status, link_type, latency_str, rpc_status);
            }

            println!("==========================================================================================");
            println!("  Total Nodes Discovered: {} | Active Compute Nodes Ready: {}", nodes.len(), ready_count);
            println!("==========================================================================================\n");
        }

        Commands::ModelCheck { path } => {
            let model_path = resolve_model_path(path.as_ref())?;
            info!("🔍 Inspecting GGUF model integrity at {:?}", model_path);
            let meta = inspect_gguf_file(&model_path)?;
            println!("\n========================================================");
            println!("   AEROMESH GGUF MODEL INTEGRITY REPORT");
            println!("========================================================");
            println!("  File Path:      {}", meta.file_path);
            println!("  GGUF Version:   v{}", meta.version);
            println!("  Tensor Count:   {}", meta.tensor_count);
            println!("  KV Metadata:    {}", meta.kv_count);
            println!("  File Size:      {:.2} GB", (meta.file_size_bytes as f64) / 1024.0 / 1024.0 / 1024.0);
            println!("  Fast Checksum:  {}", meta.fast_checksum);
            println!("  Status:         ✅ VALID (Ready for Zero-Copy mmap)");
            println!("========================================================\n");
        }

        Commands::SliceInfo { model, layers } => {
            let model_path = resolve_model_path(model.as_ref())?;
            let loader = GgufSliceLoader::open(&model_path)?;

            let slice_config = if let Some(l_str) = layers {
                parse_layer_range(&l_str, loader.total_layers)?
            } else {
                LayerSliceConfig::new(0, loader.total_layers.saturating_sub(1) / 2, loader.total_layers)?
            };

            let report = loader.get_slice_report(&slice_config);

            println!("\n========================================================");
            println!("   AEROMESH GGUF ZERO-COPY SLICE INSPECTOR");
            println!("========================================================");
            println!("  Model Path:          {}", report.model_path.display());
            println!("  Architecture:        {}", report.architecture);
            println!("  Total Model Layers:  {}", report.total_layers);
            println!("  Assigned Slice:      Layers {}..={} ({} layers)", 
                slice_config.layer_start, slice_config.layer_end, slice_config.layer_count());
            println!("  Stage Type:          {}", 
                if slice_config.is_first_stage { "Stage 1 (Embeddings + Lower Layers)" } 
                else if slice_config.is_last_stage { "Final Stage (Upper Layers + LM Head)" } 
                else { "Intermediate Transformer Block" });
            println!("  Slice Tensors:       {} tensors", report.tensor_count);
            println!("  Slice VRAM Required: {:.2} MB ({:.2} GB)", 
                (report.slice_bytes as f64) / 1024.0 / 1024.0,
                (report.slice_bytes as f64) / 1024.0 / 1024.0 / 1024.0);
            println!("  Full Model Size:     {:.2} GB", 
                (report.total_model_bytes as f64) / 1024.0 / 1024.0 / 1024.0);
            println!("  VRAM Savings:        {:.1}% less VRAM than full model", 
                report.memory_reduction_ratio * 100.0);
            println!("========================================================\n");
        }

        Commands::Probe { target } => {
            info!("🛰️ Probing Tailscale target: {}", target);
            let addr: SocketAddr = target.parse().context("Invalid target socket address")?;
            let report = TailscaleInspector::probe_socket_link(addr).await?;
            println!("\n========================================================");
            println!("   AEROMESH TAILSCALE LINK QUALITY REPORT");
            println!("========================================================");
            println!("  Target Host:    {}", report.host_name);
            println!("  IP Address:     {}", report.ip);
            println!("  Direct WireGuard: {}", if report.is_direct_wireguard { "✅ YES" } else { "❌ NO (DERP Relay)" });
            if let Some(relay) = report.relay_region {
                println!("  Relay Region:   {}", relay);
            }
            println!("  TCP RTT Ping:   {:.2} ms", report.tcp_rtt_ms);
            println!("  Cluster Status: {}", if report.is_acceptable { "✅ ELIGIBLE (Direct / Low Latency)" } else { "❌ REJECTED (DERP Relay / RTT > 150ms)" });
            println!("========================================================\n");
        }

        Commands::Worker { host, port, model, layers, cache } => {
            if let Some(m_path) = model {
                // Native Zero-Weight Pipeline Worker Mode
                let resolved_path = resolve_model_path(Some(&m_path))?;
                let loader = GgufSliceLoader::open(&resolved_path)?;
                let total_layers = loader.total_layers;

                let slice_config = if let Some(l_str) = layers {
                    parse_layer_range(&l_str, total_layers)?
                } else {
                    // Default to second half
                    let mid = total_layers / 2;
                    LayerSliceConfig::new(mid, total_layers.saturating_sub(1), total_layers)?
                };

                let service = PipelineWorkerService::new(&resolved_path, slice_config)?;
                service.run_server(&host, port).await?;
            } else {
                // Supervised CUDA RPC Backend Worker
                info!("🛡️ Starting AeroMesh Worker Daemon on {}:{} (cache: {})", host, port, cache);
                let mut supervisor = EngineSupervisor::new(&current_dir)?;
                supervisor.spawn_rpc_worker(&host, port, cache)?;

                println!("\n========================================================");
                println!("   AEROMESH CUDA RPC WORKER RUNNING");
                println!("========================================================");
                println!("  Bound Address:  {}:{}", host, port);
                println!("  Protection:     Windows Job Object Active (Leak-Proof VRAM)");
                println!("  Tensor Caching: {}", if cache { "✅ ENABLED (Zero-Network reloads from disk cache)" } else { "❌ DISABLED" });
                println!("  Status:         Listening for Coordinator Tensor Offloads...");
                println!("  Press Ctrl+C to terminate worker safely.");
                println!("========================================================\n");

                tokio::signal::ctrl_c().await?;
                info!("Received shutdown signal. Reclaiming all resources...");
                supervisor.shutdown_all();
            }
        }

        Commands::Coordinator {
            model,
            peers,
            mode,
            ngl,
            prompt,
            max_tokens,
        } => {
            println!("\n========================================================");
            println!("   AEROMESH DISTRIBUTED COORDINATOR INITIALIZING");
            println!("========================================================");

            // Step 1: Check model
            info!("Step 1/3: Verifying model file...");
            let model_path = resolve_model_path(model.as_ref())?;
            let meta = inspect_gguf_file(&model_path)?;
            info!(
                model = %model_path.display(),
                tensors = meta.tensor_count,
                checksum = %meta.fast_checksum,
                "Model integrity confirmed"
            );

            // Step 2: Probe Tailscale peers
            let peer_list: Vec<String> = peers
                .as_ref()
                .map(|p| p.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
                .unwrap_or_default();

            let mut approved_peers = Vec::new();
            if !peer_list.is_empty() {
                info!("Step 2/3: Probing Tailscale peer quality for {} nodes...", peer_list.len());
                for peer in &peer_list {
                    if let Ok(addr) = peer.parse::<SocketAddr>() {
                        match TailscaleInspector::probe_socket_link(addr).await {
                            Ok(quality) => {
                                if quality.is_acceptable {
                                    approved_peers.push((peer.clone(), quality.tcp_rtt_ms, addr));
                                    info!(peer = %peer, rtt_ms = quality.tcp_rtt_ms, "Peer approved for cluster");
                                } else {
                                    warn!(
                                        peer = %peer,
                                        rtt_ms = quality.tcp_rtt_ms,
                                        is_direct = quality.is_direct_wireguard,
                                        "Peer rejected: violates RTT <= 150ms or DERP rule"
                                    );
                                }
                            }
                            Err(e) => {
                                warn!(peer = %peer, error = %e, "Peer unreachable; skipping");
                            }
                        }
                    } else {
                        warn!(peer = %peer, "Invalid socket address format");
                    }
                }
            } else {
                info!("No remote peers configured. Running on local node.");
            }

            println!("\n========================================================");
            println!("   CLUSTER TOPOLOGY & PIPELINE ASSIGNMENT");
            println!("========================================================");
            println!("  [Node 1] Local Machine (Coordinator GPU): ACTIVE");
            if approved_peers.is_empty() {
                println!("  [Remote] No approved remote workers active (running standalone)");
            } else {
                for (idx, (peer_str, rtt, _)) in approved_peers.iter().enumerate() {
                    println!("  [Node {}] Remote Worker ({}) - RTT: {:.1}ms [OFFLOAD ACTIVE]", idx + 2, peer_str, rtt);
                }
            }
            println!("========================================================\n");

            // Step 3: Launch inference execution
            if mode.eq_ignore_ascii_case("pipeline") && !approved_peers.is_empty() {
                // Native Zero-Weight P2P Activation Streaming Mode
                info!("Step 3/3: Dispatching via Native Zero-Weight Pipeline Engine...");
                let worker_sock_addrs: Vec<SocketAddr> = approved_peers.iter().map(|(_, _, addr)| *addr).collect();
                let mut client = PipelineCoordinatorClient::new(&model_path, worker_sock_addrs)?;
                let (_output_text, perf_metrics) = client.run_pipeline_completion(&prompt, max_tokens).await?;

                println!("\n--- CLUSTER PERFORMANCE METRICS ---");
                for metric in &perf_metrics {
                    println!("  {}", metric);
                }
                println!("------------------------------------\n");

                info!("🎉 Zero-Weight P2P Pipeline generation completed successfully!");
            } else {
                // Supervised CUDA RPC Backend Mode
                info!("Step 3/3: Dispatching prompt across active RPC cluster backend...");
                let active_peer_addrs: Vec<String> = approved_peers.iter().map(|(p, _, _)| p.clone()).collect();
                let mut supervisor = EngineSupervisor::new(&current_dir)?;
                let result = supervisor.run_completion(&model_path, &active_peer_addrs, ngl, &prompt)?;

                println!("\n========================================================");
                println!("   AEROMESH INFERENCE GENERATION OUTPUT");
                println!("========================================================");
                println!("{}", result.output_text.trim());
                println!("========================================================");

                if !result.performance_summary.is_empty() {
                    println!("\n--- CLUSTER PERFORMANCE METRICS ---");
                    for metric in &result.performance_summary {
                        println!("  {}", metric);
                    }
                    println!("------------------------------------\n");
                }

                info!("🎉 Multi-node token generation completed successfully!");
            }
        }
    }

    Ok(())
}

fn parse_layer_range(s: &str, total_layers: usize) -> Result<LayerSliceConfig> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("auto") {
        let mid = total_layers / 2;
        return LayerSliceConfig::new(mid, total_layers.saturating_sub(1), total_layers);
    }

    if let Some((start_str, end_str)) = s.split_once("..=") {
        let start: usize = start_str.trim().parse()?;
        let end: usize = end_str.trim().parse()?;
        LayerSliceConfig::new(start, end, total_layers)
    } else if let Some((start_str, end_str)) = s.split_once("..") {
        let start: usize = start_str.trim().parse()?;
        let end: usize = end_str.trim().parse()?;
        let inclusive_end = if end >= total_layers {
            total_layers.saturating_sub(1)
        } else if end > start {
            end - 1
        } else {
            start
        };
        LayerSliceConfig::new(start, inclusive_end, total_layers)
    } else if let Ok(single) = s.parse::<usize>() {
        LayerSliceConfig::new(single, single, total_layers)
    } else {
        anyhow::bail!("Invalid layer range format: '{}'. Expected e.g. '16..32' or '16..=31'", s);
    }
}
