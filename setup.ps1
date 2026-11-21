# AeroMesh Developer Setup Script for Windows (CUDA / Vulkan / CPU + Rust)
#
# Run from the repository root:  pwsh -File .\setup.ps1
#
# What this does:
#   1. Reports GPU / Tailscale / Rust / Python readiness (advisory)
#   2. Creates models\ and bin\
#   3. Stages the native llama.cpp runtime (pinned to b10499) into bin\
#   4. Optionally fetches verified offline UI vendor assets into static\vendor\
#   5. Builds BOTH aeromesh.exe and AeroMesh-Desktop.exe in RELEASE mode
#   6. Verifies the compiled CLI binary starts cleanly
#
# End users can instead install the pre-built Windows wizard (AeroMesh-Setup-x64.exe).

[CmdletBinding()]
param(
    # Clone pinned llama.cpp (b10499) if it is missing (requires git + network).
    [switch]$CloneLlamaCpp,
    # Pinned llama.cpp release tag that matches crates/aeromesh-engine/src/llama_ffi.rs ABI.
    [string]$LlamaCppTag = "b10499",
    # Allow installing Rust via rustup if cargo is missing.
    [switch]$InstallRust,
    # Skip fetching offline UI vendor libraries into static/vendor/.
    [switch]$SkipVendorAssets,
    # Skip the cargo build step.
    [switch]$SkipBuild,
    # Only report readiness; change nothing.
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-Location $PSScriptRoot

function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$Command,
        [string[]]$Arguments = @(),
        [switch]$IgnoreErrors
    )
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Command @Arguments 2>&1
        $code = $LASTEXITCODE
    } catch {
        $output = $_.Exception.Message
        $code = 1
    } finally {
        $ErrorActionPreference = $prev
    }
    if ($code -ne 0 -and -not $IgnoreErrors) {
        return [pscustomobject]@{ Ok = $false; Code = $code; Output = ($output | Out-String).Trim() }
    }
    return [pscustomobject]@{ Ok = ($code -eq 0); Code = $code; Output = ($output | Out-String).Trim() }
}

$script:Warnings = New-Object System.Collections.Generic.List[string]

function Write-Step($n, $total, $text) {
    Write-Host "`n[$n/$total] $text" -ForegroundColor Yellow
}
function Write-Ok($text)   { Write-Host "  [OK] $text" -ForegroundColor Green }
function Write-Warn($text) { Write-Host "  [!]  $text" -ForegroundColor Yellow; $script:Warnings.Add($text) }
function Write-Bad($text)  { Write-Host "  [X]  $text" -ForegroundColor Red;    $script:Warnings.Add($text) }

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   AEROMESH DEVELOPER NODE SETUP (Windows)              " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$TOTAL_STEPS = 6

# ---------------------------------------------------------------------------
# 1. GPU
# ---------------------------------------------------------------------------
Write-Step 1 $TOTAL_STEPS "Checking GPU / CUDA driver..."
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $r = Invoke-Native -Command "nvidia-smi" -Arguments @("--query-gpu=name,memory.total,driver_version", "--format=csv,noheader") -IgnoreErrors
    if ($r.Ok -and $r.Output) { Write-Ok "NVIDIA GPU: $($r.Output -replace "`r?`n", '; ')" }
    else { Write-Warn "nvidia-smi returned no usable devices." }
} else {
    Write-Warn "nvidia-smi not found. Vulkan or CPU inference will be used unless NVIDIA drivers are installed."
}

# ---------------------------------------------------------------------------
# 2. Tailscale
# ---------------------------------------------------------------------------
Write-Step 2 $TOTAL_STEPS "Checking Tailscale connectivity..."
if (Get-Command tailscale -ErrorAction SilentlyContinue) {
    $r = Invoke-Native -Command "tailscale" -Arguments @("ip", "-4") -IgnoreErrors
    if ($r.Ok -and $r.Output) { Write-Ok "Tailscale IPv4: $($r.Output -replace "`r?`n", ', ')" }
    else { Write-Warn "Tailscale installed but not connected. Run 'tailscale up'." }
} else {
    Write-Warn "Tailscale CLI not on PATH. Required for cross-network clusters; LAN works without it."
}

# ---------------------------------------------------------------------------
# 3. Rust toolchain
# ---------------------------------------------------------------------------
Write-Step 3 $TOTAL_STEPS "Checking Rust toolchain..."
$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if (Test-Path $cargoBin) { $env:PATH = "$cargoBin;$env:PATH" }

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    if ($InstallRust -and -not $CheckOnly) {
        Write-Host "  Installing Rust toolchain via rustup (https://rustup.rs)..." -ForegroundColor Cyan
        $rustup = Join-Path $env:TEMP "rustup-init.exe"
        try {
            Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustup -UseBasicParsing
            & $rustup -y
            $env:PATH = "$cargoBin;$env:PATH"
        } catch {
            throw "rustup install failed: $($_.Exception.Message). Install Rust manually from https://rustup.rs"
        }
    } else {
        Write-Warn "cargo not found. Install Rust from https://rustup.rs or re-run with -InstallRust."
    }
}
if (Get-Command cargo -ErrorAction SilentlyContinue) {
    $r = Invoke-Native -Command "cargo" -Arguments @("--version") -IgnoreErrors
    Write-Ok "Rust: $($r.Output)"
} else {
    Write-Bad "Rust is not available; the build step will be skipped."
}

# ---------------------------------------------------------------------------
# 4. Python runtime (optional — only needed for legacy app.py dev server)
# ---------------------------------------------------------------------------
Write-Step 4 $TOTAL_STEPS "Checking Python (optional web gateway)..."
$pythonExe = $null
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
    Write-Ok "Using existing virtualenv: $venvPython"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = (Get-Command python).Source
    Write-Ok "Python: $(& $pythonExe --version 2>&1)"
} else {
    Write-Warn "Python 3.11+ not found. AeroMesh-Desktop.exe runs standalone without Python."
}

if ($pythonExe -and -not $CheckOnly -and -not (Test-Path $venvPython)) {
    Write-Host "  Creating .venv and installing optional FastAPI gateway dependencies..." -ForegroundColor Cyan
    try {
        Invoke-Native -Command $pythonExe -Arguments @("-m", "venv", ".venv") | Out-Null
        $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            Invoke-Native -Command "uv" -Arguments @("sync", "--extra", "dev") | Out-Null
        } else {
            Invoke-Native -Command $venvPython -Arguments @("-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-cache-dir", "fastapi>=0.100.0", "uvicorn>=0.22.0", "httpx>=0.24.0", "pytest>=8.0.0") | Out-Null
        }
        Write-Ok "Python environment ready."
    } catch {
        Write-Warn "Python environment setup failed: $($_.Exception.Message)"
    }
}

# ---------------------------------------------------------------------------
# 5. Native llama.cpp runtime (pinned ABI tag) + offline UI vendor assets
# ---------------------------------------------------------------------------
Write-Step 5 $TOTAL_STEPS "Preparing native llama.cpp runtime ($LlamaCppTag) in bin\..."
if (-not $CheckOnly) {
    foreach ($dir in @("models", "bin")) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir | Out-Null
            Write-Ok "Created '$dir' directory."
        }
    }
}

if (-not (Test-Path "llama.cpp")) {
    if ($CloneLlamaCpp -and -not $CheckOnly) {
        Write-Host "  Cloning llama.cpp pinned to tag $LlamaCppTag..." -ForegroundColor Cyan
        $cloneRes = Invoke-Native -Command "git" -Arguments @("clone", "--depth", "1", "--branch", $LlamaCppTag, "https://github.com/ggml-org/llama.cpp.git") -IgnoreErrors
        if ($cloneRes.Ok) {
            Write-Ok "Cloned llama.cpp@$LlamaCppTag."
        } else {
            Write-Warn "git clone of llama.cpp@$LlamaCppTag failed: $($cloneRes.Output)"
        }
    } else {
        Write-Warn "llama.cpp/ is absent. Re-run with -CloneLlamaCpp to fetch tag $LlamaCppTag, or copy runtime DLLs into bin\."
    }
} else {
    Write-Ok "llama.cpp/ present."
}

$candidateBuildDirs = @(
    "llama.cpp\build\bin\Release",
    "llama.cpp\build\bin",
    "llama.cpp\build\bin\Debug"
)
$copiedFrom = $null
foreach ($dir in $candidateBuildDirs) {
    if (Test-Path $dir) {
        Copy-Item -Path "$dir\*.dll" -Destination "bin" -Force -ErrorAction SilentlyContinue
        Copy-Item -Path "$dir\*.exe" -Destination "bin" -Force -ErrorAction SilentlyContinue
        $copiedFrom = $dir
        break
    }
}
if ($copiedFrom) {
    Write-Ok "Copied native runtime from '$copiedFrom' into bin\."
} elseif (Test-Path "bin\llama.dll") {
    Write-Ok "bin\llama.dll already present; using it (loaded dynamically via LoadLibraryW)."
} else {
    Write-Warn "No llama.cpp build output found and bin\llama.dll is missing."
    Write-Host "       Build llama.cpp ($LlamaCppTag) with (from llama.cpp\):" -ForegroundColor DarkGray
    Write-Host "         cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release" -ForegroundColor DarkGray
    Write-Host "         cmake --build build --config Release -j" -ForegroundColor DarkGray
    Write-Host "       then re-run this script." -ForegroundColor DarkGray
}

# Fetch offline web UI assets if helper script is available
$vendorScript = Join-Path $PSScriptRoot "scripts\fetch-vendor-assets.ps1"
if (-not $SkipVendorAssets -and -not $CheckOnly -and (Test-Path $vendorScript)) {
    Write-Host "  Ensuring offline UI vendor assets (static\vendor\)..." -ForegroundColor Cyan
    try {
        & $vendorScript
        Write-Ok "Offline UI vendor assets verified."
    } catch {
        Write-Warn "Could not fetch offline UI vendor assets (CDN fallback still active): $($_.Exception.Message)"
    }
}

# ---------------------------------------------------------------------------
# 6. Build (RELEASE — both CLI engine and native Desktop Studio)
# ---------------------------------------------------------------------------
Write-Step 6 $TOTAL_STEPS "Compiling AeroMesh (release: aeromesh + AeroMesh-Desktop)..."
if ($SkipBuild -or $CheckOnly) {
    Write-Host "  Skipped (-SkipBuild / -CheckOnly)." -ForegroundColor DarkGray
} elseif (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Warn "cargo unavailable; skipping build."
} else {
    $r = Invoke-Native -Command "cargo" -Arguments @("build", "--release", "--bin", "aeromesh", "--bin", "AeroMesh-Desktop") -IgnoreErrors
    if ($r.Ok) {
        Write-Ok "Built target\release\aeromesh.exe and target\release\AeroMesh-Desktop.exe"
    } else {
        Write-Bad "cargo build failed (exit $($r.Code))."
        if ($r.Output) { Write-Host ($r.Output -split "`n" | Select-Object -Last 15 | Out-String) -ForegroundColor DarkGray }
    }
}

$exe = Join-Path $PSScriptRoot "target\release\aeromesh.exe"
if ((Test-Path $exe) -and -not $CheckOnly) {
    Write-Host "  Verifying aeromesh.exe starts cleanly..." -ForegroundColor Cyan
    $env:PATH = "$(Join-Path $PSScriptRoot 'bin');$env:PATH"
    $r = Invoke-Native -Command $exe -Arguments @("--help") -IgnoreErrors
    if ($r.Output -match 'Usage: aeromesh') {
        Write-Ok "aeromesh.exe runs cleanly."
    } else {
        Write-Warn "aeromesh.exe did not report expected usage output."
    }
}

# ---------------------------------------------------------------------------
Write-Host "`n========================================================" -ForegroundColor Cyan
if ($script:Warnings.Count -eq 0) {
    Write-Host "   SETUP COMPLETE - NODE READY                          " -ForegroundColor Green
} else {
    Write-Host "   SETUP FINISHED WITH $($script:Warnings.Count) WARNING(S)              " -ForegroundColor Yellow
}
Write-Host "========================================================" -ForegroundColor Cyan
if ($script:Warnings.Count -gt 0) {
    Write-Host "`nWarnings:" -ForegroundColor Yellow
    $script:Warnings | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
}

Write-Host "`nNext steps:" -ForegroundColor White
Write-Host "  Launch native Desktop Studio:      .\target\release\AeroMesh-Desktop.exe" -ForegroundColor White
Write-Host "  Launch interactive cluster script: .\start.ps1" -ForegroundColor White
Write-Host "  Build Windows .exe installer:      .\installer\build-installer.ps1" -ForegroundColor White
Write-Host "  Run the tests:                     cargo test --workspace" -ForegroundColor DarkGray

