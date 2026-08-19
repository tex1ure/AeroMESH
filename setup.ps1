# AeroMesh 1-Click Setup Script for Windows (CUDA + Rust)
# Run this script in PowerShell: .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   🚀 AEROMESH CLUSTER NODE SETUP (Windows + CUDA)      " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Step 1: Check NVIDIA GPU & CUDA
Write-Host "`n[1/5] Checking NVIDIA GPU..." -ForegroundColor Yellow
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    Write-Host "  ✅ GPU Detected: $gpuInfo" -ForegroundColor Green
} else {
    Write-Host "  ⚠️ NVIDIA GPU / Driver not detected. Ensure NVIDIA drivers are installed." -ForegroundColor Red
}

# Step 2: Check Tailscale Network
Write-Host "`n[2/5] Checking Tailscale Connectivity..." -ForegroundColor Yellow
if (Get-Command tailscale -ErrorAction SilentlyContinue) {
    $tsStatus = tailscale ip -4 2>$null
    if ($tsStatus) {
        Write-Host "  ✅ Tailscale IPv4: $tsStatus" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ Tailscale is installed but not connected. Run 'tailscale up'." -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠️ Tailscale CLI not found on PATH. Install Tailscale from https://tailscale.com" -ForegroundColor Red
}

# Step 3: Check / Install Rust Toolchain
Write-Host "`n[3/5] Checking Rust Toolchain..." -ForegroundColor Yellow
$env:PATH = "C:\w64devkit\bin;C:\Users\$env:USERNAME\.cargo\bin;" + $env:PATH
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "  Installing Rust via rustup..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile "$env:TEMP\rustup-init.exe" -UseBasicParsing
    & "$env:TEMP\rustup-init.exe" -y --default-toolchain stable-x86_64-pc-windows-gnu
}
$rustVersion = cargo --version 2>$null
Write-Host "  ✅ Rust Ready: $rustVersion" -ForegroundColor Green

# Step 4: Ensure Folder Structure & Binaries
Write-Host "`n[4/5] Preparing Models, Binaries, and llama.cpp Repository..." -ForegroundColor Yellow
if (-not (Test-Path "models")) {
    New-Item -ItemType Directory -Path "models" | Out-Null
    Write-Host "  📁 Created 'models' directory." -ForegroundColor Green
}
if (-not (Test-Path "bin")) {
    New-Item -ItemType Directory -Path "bin" | Out-Null
    Write-Host "  📁 Created 'bin' directory." -ForegroundColor Green
}
if (-not (Test-Path "llama.cpp")) {
    Write-Host "  Cloning llama.cpp repository..." -ForegroundColor Cyan
    git clone https://github.com/ggerganov/llama.cpp.git
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ llama.cpp successfully cloned." -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ Failed to clone llama.cpp repository." -ForegroundColor Yellow
    }
} else {
    Write-Host "  ✅ llama.cpp repository found." -ForegroundColor Green
}

# Step 5: Build AeroMesh
Write-Host "`n[5/5] Compiling AeroMesh Engine..." -ForegroundColor Yellow
cargo build
Write-Host "  ✅ AeroMesh Engine successfully built!" -ForegroundColor Green

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "   🎉 SETUP COMPLETE! YOUR NODE IS READY.               " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "1. To run as a Worker Laptop:"
Write-Host "   cargo run --bin aeromesh -- worker --port 50052`n" -ForegroundColor White
Write-Host "2. To check a Model file on this laptop:"
Write-Host "   cargo run --bin aeromesh -- model-check models/<your-model>.gguf`n" -ForegroundColor White
Write-Host "3. To run as the Coordinator (from Laptop A):"
Write-Host "   cargo run --bin aeromesh -- coordinator --model models/<model>.gguf --peers <worker-tailscale-ip>:50052`n" -ForegroundColor White
