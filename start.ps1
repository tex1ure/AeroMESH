# ==============================================================================
# AeroMesh 1-Click All-in-One Cluster Launcher
# Usage:
#   .\start.ps1                         -> Interactive Menu
#   .\start.ps1 coordinator             -> Starts Coordinator + Claymorphic Web UI
#   .\start.ps1 worker                  -> Starts Zero-Weight Worker Node
#   .\start.ps1 all                     -> Starts All-in-One Local Test Mesh (Worker+Coord+UI)
#   .\start.ps1 status                  -> Scans Tailscale cluster mesh
# ==============================================================================

[CmdletBinding()]
param (
    [Parameter(Position = 0)]
    [ValidateSet("coordinator", "worker", "all", "status", "")]
    [string]$Role = "",

    [Parameter(Mandatory = $false)]
    [string]$Model = "",

    [Parameter(Mandatory = $false)]
    [string]$Layers = "",

    [Parameter(Mandatory = $false)]
    [string]$Peers = "",

    [Parameter(Mandatory = $false)]
    [int]$Port = 0,

    [Parameter(Mandatory = $false)]
    [switch]$NoUI,

    [Parameter(Mandatory = $false)]
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 1. Ensure Compiler & Toolchain PATH and DLL directories
$binDir = Join-Path $PSScriptRoot "bin"
$env:PATH = "C:\w64devkit\bin;C:\Users\" + $env:USERNAME + "\.cargo\bin;$binDir;" + $env:PATH

# Ensure release DLLs are present
if (Test-Path "target\release") {
    Get-ChildItem -Path "bin\*.dll" -ErrorAction SilentlyContinue | Copy-Item -Destination "target\release" -Force -ErrorAction SilentlyContinue
}

function Get-AeroMeshExe {
    if (Test-Path "target\release\aeromesh.exe") {
        return (Resolve-Path "target\release\aeromesh.exe").Path
    }
    return ""
}

function Stop-PortProcess {
    param ([int]$TargetPort)
    try {
        $conns = Get-NetTCPConnection -LocalPort $TargetPort -State Listen -ErrorAction SilentlyContinue
        foreach ($conn in $conns) {
            if ($conn.OwningProcess -and $conn.OwningProcess -ne $PID) {
                Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {}
}

function Show-AeroMeshBanner {
    Write-Host ""
    Write-Host "========================================================================" -ForegroundColor Cyan
    Write-Host "   [+] AEROMESH: DISTRIBUTED ZERO-WEIGHT LLM CLUSTER ENGINE             " -ForegroundColor Cyan
    Write-Host "   Windows + NVIDIA CUDA + Tailscale WireGuard (0.0 MB Wire Transfer)  " -ForegroundColor DarkCyan
    Write-Host "========================================================================" -ForegroundColor Cyan
}

function Find-ModelPath {
    param ([string]$ExplicitPath)
    if ($ExplicitPath -and (Test-Path $ExplicitPath)) {
        return $ExplicitPath
    }
    
    # Check for primary DeepSeek model first if present
    if (Test-Path "models\DS.gguf") {
        return "models/DS.gguf"
    }

    # Dynamically scan models/ or current directory for any other .gguf files
    $allGgufs = Get-ChildItem -Path @("models", ".") -Filter "*.gguf" -File -ErrorAction SilentlyContinue
    if ($allGgufs -and $allGgufs.Count -gt 0) {
        $first = $allGgufs[0]
        if ($first.Directory.Name -eq "models") {
            return ("models/" + $first.Name)
        }
        return $first.Name
    }

    return ""
}

function Get-PythonCommand {
    if (Test-Path ".\.venv\Scripts\python.exe") {
        return (Resolve-Path ".\.venv\Scripts\python.exe").Path
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    return ""
}

function Get-TailscaleIPv4 {
    if (Get-Command tailscale -ErrorAction SilentlyContinue) {
        try {
            $ip = (tailscale ip -4 2>$null).Trim()
            if ($ip) { return $ip }
        } catch {}
    }
    return "127.0.0.1"
}

# ------------------------------------------------------------------------------
# INTERACTIVE SELECTION (IF NO ROLE PROVIDED)
# ------------------------------------------------------------------------------
Show-AeroMeshBanner

$detectedModel = Find-ModelPath -ExplicitPath $Model
$modelName = "No .gguf found in models/"
if ($detectedModel) {
    $modelName = [System.IO.Path]::GetFileName($detectedModel)
}

$localTailscaleIP = Get-TailscaleIPv4

Write-Host ""
Write-Host "  [-] Local Tailscale IP:  " -NoNewline -ForegroundColor Gray
Write-Host "$localTailscaleIP" -ForegroundColor Green
Write-Host "  [-] Detected Model:      " -NoNewline -ForegroundColor Gray
Write-Host "$modelName" -ForegroundColor Yellow
Write-Host "  [-] Wire Weight Transfer:" -NoNewline -ForegroundColor Gray
Write-Host " 0.0 MB (P2P Activation Streaming)" -ForegroundColor Cyan
Write-Host "------------------------------------------------------------------------" -ForegroundColor DarkGray

if (-not $Role) {
    Write-Host ""
    Write-Host "Select your node role for this machine:" -ForegroundColor White
    Write-Host "  [1] " -NoNewline -ForegroundColor Cyan
    Write-Host "Coordinator Node + Web UI " -NoNewline -ForegroundColor Green
    Write-Host "(Laptop A: Runs Layers 0..24, Hosts API + Web Interface)" -ForegroundColor Gray

    Write-Host "  [2] " -NoNewline -ForegroundColor Cyan
    Write-Host "Worker Node               " -NoNewline -ForegroundColor Yellow
    Write-Host "(Laptop B: Runs Layers 25..48 + LM Head on Port 50052)" -ForegroundColor Gray

    Write-Host "  [3] " -NoNewline -ForegroundColor Cyan
    Write-Host "Full Local Test Mesh      " -NoNewline -ForegroundColor Magenta
    Write-Host "(Starts Worker + Coordinator + Web UI together locally)" -ForegroundColor Gray

    Write-Host "  [4] " -NoNewline -ForegroundColor Cyan
    Write-Host "Cluster Status / Probe    " -NoNewline -ForegroundColor White
    Write-Host "(Scans Tailscale Mesh for active worker nodes)" -ForegroundColor Gray

    Write-Host "  [Q] Quit" -ForegroundColor DarkGray
    Write-Host ""

    $choice = Read-Host "Enter choice [1, 2, 3, 4, Q]"
    switch ($choice.Trim().ToUpper()) {
        "1" { $Role = "coordinator" }
        "2" { $Role = "worker" }
        "3" { $Role = "all" }
        "4" { $Role = "status" }
        default {
            Write-Host ""
            Write-Host "Exiting AeroMesh Launcher." -ForegroundColor DarkGray
            Write-Host ""
            exit 0
        }
    }
}

# ------------------------------------------------------------------------------
# ROLE EXECUTION
# ------------------------------------------------------------------------------

# 1. STATUS
if ($Role -eq "status") {
    Write-Host ""
    Write-Host "[+] Scanning Tailscale Cluster Nodes..." -ForegroundColor Cyan
    cargo run --bin aeromesh -- status
    exit 0
}

# Check Model Existence for Compute Roles
if (-not $detectedModel) {
    Write-Host ""
    Write-Host "[-] ERROR: No .gguf model found in 'models/' directory!" -ForegroundColor Red
    Write-Host "Please place a model (e.g. models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf) and retry." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# 2. WORKER NODE (Laptop B)
if ($Role -eq "worker") {
    $workerPort = 50052
    if ($Port -gt 0) { $workerPort = $Port }

    $workerLayers = "auto"
    if ($Layers) { $workerLayers = $Layers }

    $bindStr = "0.0.0.0:" + $workerPort

    Write-Host ""
    Write-Host "[+] Starting AeroMesh Zero-Weight Worker Stage (Laptop B)..." -ForegroundColor Cyan
    Write-Host "  Model:   $modelName" -ForegroundColor Gray
    Write-Host "  Layers:  $workerLayers" -ForegroundColor Gray
    Write-Host "  Port:    $workerPort" -ForegroundColor Gray
    Write-Host "  Binding: $bindStr" -ForegroundColor Green
    Write-Host ""

    $aeroExe = Get-AeroMeshExe
    if ($aeroExe) {
        & $aeroExe worker --model "$detectedModel" --layers "$workerLayers" --port $workerPort
    } else {
        cargo run --release --bin aeromesh -- worker --model "$detectedModel" --layers "$workerLayers" --port $workerPort
    }
    exit 0
}

# 3. FULL LOCAL TEST MESH (Worker + Coordinator + Web UI)
if ($Role -eq "all") {
    $apiPort = 8080
    if ($Port -gt 0) { $apiPort = $Port }
    $uiPort = 7860
    $pythonExe = Get-PythonCommand

    if (-not $pythonExe) {
        Write-Host ""
        Write-Host "[-] ERROR: Python executable not found!" -ForegroundColor Red
        exit 1
    }

    $coordUrl = "http://127.0.0.1:" + $apiPort
    $uiUrl = "http://127.0.0.1:" + $uiPort

    # Clean up any lingering process holding the ports
    Stop-PortProcess $uiPort
    Stop-PortProcess $apiPort

    Write-Host ""
    Write-Host "[+] Launching Full Local Demo Mesh (Worker + Coordinator + Web UI)..." -ForegroundColor Cyan
    Write-Host "  Model:       $modelName" -ForegroundColor Gray
    Write-Host "  Coordinator: $coordUrl" -ForegroundColor Green
    Write-Host "  Worker:      127.0.0.1:50052 (Stage 2 Loopback)" -ForegroundColor Yellow
    Write-Host "  Web UI:      $uiUrl" -ForegroundColor Cyan
    Write-Host ""

    $aeroExe = Get-AeroMeshExe
    if ($aeroExe) {
        $rustProc = Start-Process -FilePath $aeroExe -ArgumentList @("start", "--role", "all", "--model", "$detectedModel", "--port", "$apiPort") -PassThru -NoNewWindow
    } else {
        $cargoArgs = @("run", "--release", "--bin", "aeromesh", "--", "start", "--role", "all", "--model", "$detectedModel", "--port", "$apiPort")
        $rustProc = Start-Process -FilePath "cargo" -ArgumentList $cargoArgs -PassThru -NoNewWindow
    }

    Start-Sleep -Milliseconds 2500

    # Open Browser
    if (-not $NoBrowser) {
        Start-Process $uiUrl
    }

    # Run Python Web UI in foreground
    try {
        $env:AEROMESH_ENDPOINT = $coordUrl
        $env:PORT = "7860"
        if ($pythonExe -eq "python") {
            python app.py
        } else {
            & $pythonExe app.py
        }
    } finally {
        Write-Host ""
        Write-Host "Shutting down local cluster mesh..." -ForegroundColor Yellow
        if ($rustProc -and -not $rustProc.HasExited) {
            Stop-Process -Id $rustProc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    exit 0
}

# 4. COORDINATOR NODE (Laptop A)
if ($Role -eq "coordinator") {
    $apiPort = 8080
    if ($Port -gt 0) { $apiPort = $Port }

    $coordLayers = "auto"
    if ($Layers) { $coordLayers = $Layers }

    $pythonExe = Get-PythonCommand

    # Check for Peer IP if not supplied
    if (-not $Peers) {
        Write-Host ""
        Write-Host "Enter the Worker Tailscale Address (example: 100.101.147.24:50052):" -ForegroundColor Yellow
        $enteredPeer = Read-Host "Worker Peer IP:Port"
        if ($enteredPeer) {
            $Peers = $enteredPeer.Trim()
        }
    }

    $coordUrl = "http://127.0.0.1:" + $apiPort
    $uiUrl = "http://127.0.0.1:7860"

    $workerStatusStr = "Local Standalone"
    if ($Peers) { $workerStatusStr = $Peers }

    Write-Host ""
    Write-Host "[+] Launching AeroMesh Coordinator (Laptop A)..." -ForegroundColor Cyan
    Write-Host "  Model:       $modelName" -ForegroundColor Gray
    Write-Host "  Layers:      $coordLayers" -ForegroundColor Gray
    Write-Host "  Worker Peer: $workerStatusStr" -ForegroundColor Yellow
    Write-Host "  API Server:  $coordUrl" -ForegroundColor Green
    Write-Host "  Web UI:      $uiUrl" -ForegroundColor Cyan
    Write-Host ""

    $peerArg = @()
    if ($Peers) { $peerArg = @("--peers", "$Peers") }
    $aeroExe = Get-AeroMeshExe

    if ($NoUI) {
        if ($aeroExe) {
            & $aeroExe start --role coordinator --model "$detectedModel" --layers "$coordLayers" @peerArg --port $apiPort
        } else {
            cargo run --release --bin aeromesh -- start --role coordinator --model "$detectedModel" --layers "$coordLayers" @peerArg --port $apiPort
        }
        exit 0
    }

    # Clean up any lingering process holding the ports
    Stop-PortProcess 7860
    Stop-PortProcess $apiPort

    # Start Coordinator API in background process
    if ($aeroExe) {
        $exeArgs = @("start", "--role", "coordinator", "--model", "$detectedModel", "--layers", "$coordLayers") + $peerArg + @("--port", "$apiPort")
        $rustProc = Start-Process -FilePath $aeroExe -ArgumentList $exeArgs -PassThru -NoNewWindow
    } else {
        $cargoArgs = @("run", "--release", "--bin", "aeromesh", "--", "start", "--role", "coordinator", "--model", "$detectedModel", "--layers", "$coordLayers") + $peerArg + @("--port", "$apiPort")
        $rustProc = Start-Process -FilePath "cargo" -ArgumentList $cargoArgs -PassThru -NoNewWindow
    }

    Start-Sleep -Milliseconds 2500

    if (-not $NoBrowser) {
        Start-Process $uiUrl
    }

    # Run Python Web UI in foreground
    try {
        $env:AEROMESH_ENDPOINT = $coordUrl
        $env:PORT = "7860"
        if ($pythonExe -eq "python") {
            python app.py
        } else {
            & $pythonExe app.py
        }
    } finally {
        Write-Host ""
        Write-Host "Shutting down coordinator API server..." -ForegroundColor Yellow
        if ($rustProc -and -not $rustProc.HasExited) {
            Stop-Process -Id $rustProc.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
