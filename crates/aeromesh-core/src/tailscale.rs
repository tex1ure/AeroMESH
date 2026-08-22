use std::collections::HashMap;
use std::net::SocketAddr;
use std::process::Command;
use std::time::{Duration, Instant};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TailscalePeerInfo {
    #[serde(rename = "HostName")]
    pub host_name: String,
    #[serde(rename = "TailscaleIPs")]
    pub tailscale_ips: Vec<String>,
    #[serde(rename = "Relay")]
    pub relay: Option<String>,
    #[serde(rename = "CurAddr")]
    pub cur_addr: Option<String>,
    #[serde(rename = "Active")]
    pub active: Option<bool>,
    #[serde(rename = "RxBytes")]
    pub rx_bytes: Option<u64>,
    #[serde(rename = "TxBytes")]
    pub tx_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TailscaleStatusOutput {
    #[serde(rename = "Self")]
    pub self_node: Option<TailscalePeerInfo>,
    #[serde(rename = "Peer")]
    pub peer: Option<HashMap<String, TailscalePeerInfo>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerLinkQuality {
    pub host_name: String,
    pub ip: String,
    pub is_direct_wireguard: bool,
    pub relay_region: Option<String>,
    pub tcp_rtt_ms: f64,
    pub is_acceptable: bool, // true if direct or RTT <= 150ms
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredNodeStatus {
    pub host_name: String,
    pub ip: String,
    pub is_self: bool,
    pub is_tailscale_active: bool,
    pub is_direct_wireguard: bool,
    pub relay_region: Option<String>,
    pub rpc_online: bool,
    pub rtt_ms: Option<f64>,
    pub cluster_ready: bool,
}

pub struct TailscaleInspector;

impl TailscaleInspector {
    pub fn get_status() -> Result<TailscaleStatusOutput> {
        let output = Command::new("tailscale")
            .args(["status", "--json"])
            .output()
            .context("Failed to execute 'tailscale status --json'")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("tailscale status failed: {}", stderr);
        }

        let parsed: TailscaleStatusOutput = serde_json::from_slice(&output.stdout)
            .context("Failed to parse tailscale JSON output")?;
        Ok(parsed)
    }

    pub async fn probe_socket_link(target: SocketAddr) -> Result<PeerLinkQuality> {
        let start = Instant::now();
        let stream = tokio::time::timeout(
            Duration::from_millis(3000),
            TcpStream::connect(target),
        )
        .await
        .context(format!("Timeout connecting to Tailscale target {}", target))??;

        // Ensure TCP_NODELAY is active for minimum latency
        stream.set_nodelay(true)?;
        let rtt = start.elapsed();
        let rtt_ms = (rtt.as_micros() as f64) / 1000.0;

        // Cross-reference with tailscale status to detect DERP
        let mut is_direct = true;
        let mut relay_region = None;
        let ip_str = target.ip().to_string();
        let mut host_name = target.to_string();

        if let Ok(status) = Self::get_status() {
            if let Some(peers) = status.peer {
                for (_key, p) in peers {
                    if p.tailscale_ips.iter().any(|ip| ip == &ip_str) {
                        host_name = p.host_name;
                        if let Some(relay) = &p.relay {
                            if p.cur_addr.is_none() {
                                is_direct = false;
                                relay_region = Some(relay.clone());
                            }
                        }
                        break;
                    }
                }
            }
        }

        // CRUCIAL RULE: Must not route via DERP or exceed 150ms RTT
        let is_acceptable = is_direct && (rtt_ms <= 150.0);

        if !is_acceptable {
            warn!(
                target = %target,
                rtt_ms = rtt_ms,
                is_direct = is_direct,
                relay = ?relay_region,
                "⚠️ Tailscale link degraded: high latency or DERP relay active"
            );
        } else {
            info!(
                target = %target,
                rtt_ms = rtt_ms,
                is_direct = is_direct,
                "✅ Tailscale Direct WireGuard link established"
            );
        }

        Ok(PeerLinkQuality {
            host_name,
            ip: ip_str,
            is_direct_wireguard: is_direct,
            relay_region,
            tcp_rtt_ms: rtt_ms,
            is_acceptable,
        })
    }

    pub async fn discover_cluster_nodes(port: u16) -> Result<Vec<DiscoveredNodeStatus>> {
        let status = Self::get_status()?;
        let mut nodes = Vec::new();

        // 1. Check self node
        if let Some(self_node) = status.self_node {
            let self_ip = self_node.tailscale_ips.first().cloned().unwrap_or_else(|| "127.0.0.1".into());
            let socket: SocketAddr = format!("{}:{}", self_ip, port).parse().unwrap_or_else(|_| "127.0.0.1:50052".parse().unwrap());
            let rpc_online = tokio::time::timeout(Duration::from_millis(500), TcpStream::connect(socket)).await.is_ok();

            nodes.push(DiscoveredNodeStatus {
                host_name: format!("{} (Coordinator)", self_node.host_name),
                ip: self_ip,
                is_self: true,
                is_tailscale_active: true,
                is_direct_wireguard: true,
                relay_region: None,
                rpc_online,
                rtt_ms: Some(0.5),
                cluster_ready: true,
            });
        }

        // 2. Check peer nodes
        if let Some(peers) = status.peer {
            for (_id, peer) in peers {
                if let Some(ip) = peer.tailscale_ips.first() {
                    let addr_str = format!("{}:{}", ip, port);
                    let mut is_direct = true;
                    let mut relay_region = None;

                    if let Some(relay) = &peer.relay {
                        if peer.cur_addr.is_none() {
                            is_direct = false;
                            relay_region = Some(relay.clone());
                        }
                    }

                    let active = peer.active.unwrap_or(false);
                    let mut rpc_online = false;
                    let mut rtt_ms = None;

                    if let Ok(addr) = addr_str.parse::<SocketAddr>() {
                        let start = Instant::now();
                        if let Ok(Ok(stream)) = tokio::time::timeout(Duration::from_millis(1500), TcpStream::connect(addr)).await {
                            let _ = stream.set_nodelay(true);
                            rpc_online = true;
                            rtt_ms = Some((start.elapsed().as_micros() as f64) / 1000.0);
                        }
                    }

                    let cluster_ready = rpc_online && is_direct && rtt_ms.is_some_and(|r| r <= 150.0);

                    nodes.push(DiscoveredNodeStatus {
                        host_name: peer.host_name,
                        ip: ip.clone(),
                        is_self: false,
                        is_tailscale_active: active,
                        is_direct_wireguard: is_direct,
                        relay_region,
                        rpc_online,
                        rtt_ms,
                        cluster_ready,
                    });
                }
            }
        }

        Ok(nodes)
    }
}
