use std::net::SocketAddr;
use std::path::PathBuf;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use aeromesh_core::tailscale::TailscaleInspector;
use aeromesh_engine::{inspect_gguf_file, resolve_model_path, EngineSupervisor};

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

    /// Probe Tailscale link quality (Direct WireGuard vs DERP, RTT latency)
    Probe {
        /// Target Tailscale address (e.g. 100.122.125.95:50052)
        #[arg(value_name = "TARGET")]
        target: String,
    },

    /// Start a supervised CUDA RPC backend worker node
    Worker {
        /// Host IP to bind
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Port to bind
        #[arg(long, default_value_t = 50052)]
        port: u16,

        /// Enable local tensor disk caching (avoids re-sending GBs over network)
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

        /// Number of layers to offload to GPU (-1 for all)
        #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
        ngl: i32,

        /// Prompt text to execute
        #[arg(long, default_value = "Write a short sentence about distributed GPU clusters.")]
        prompt: String,
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

        Commands::Worker { host, port, cache } => {
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

        Commands::Coordinator {
            model,
            peers,
            ngl,
            prompt,
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
                                    approved_peers.push((peer.clone(), quality.tcp_rtt_ms));
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
                println!("  [Remote] No approved remote RPC workers active (running standalone)");
            } else {
                for (idx, (peer_addr, rtt)) in approved_peers.iter().enumerate() {
                    println!("  [Node {}] Remote Worker ({}) - RTT: {:.1}ms [OFFLOAD ACTIVE]", idx + 2, peer_addr, rtt);
                }
            }
            println!("========================================================\n");

            // Step 3: Launch inference execution
            info!("Step 3/3: Dispatching prompt across active cluster pipeline...");
            let active_peer_addrs: Vec<String> = approved_peers.iter().map(|(p, _)| p.clone()).collect();
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

    Ok(())
}
