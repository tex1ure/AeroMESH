use thiserror::Error;

#[derive(Error, Debug)]
pub enum AeroMeshError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Network connection failed to {target}: {reason}")]
    Network { target: String, reason: String },

    #[error(
        "Tailscale DERP relay detected for node {node_id} (RTT {rtt_ms:.1}ms > 150ms threshold)"
    )]
    TailscaleDerpRelay { node_id: String, rtt_ms: f64 },

    #[error("Invalid GGUF file: {0}")]
    InvalidGguf(String),

    #[error("Process supervisor error: {0}")]
    Supervisor(String),

    #[error("Cluster configuration error: {0}")]
    Config(String),
}
