# AeroMesh 1-Click Setup Script for Windows (CUDA + Rust)
#
# Run from the repository root:  pwsh -File .\setup.ps1
#
# What this does:
#   1. Reports GPU / Tailscale / Rust / Python readiness (advisory)
#   2. Creates models\ and bin\
#   3. Makes the native llama.cpp runtime available in bin\
#   4. Generates MinGW import libraries (libllama.dll.a, libggml.dll.a)
#   5. Builds the Rust workspace in RELEASE mode (what start.ps1 expects)
#   6. Verifies the binary actually runs
#
# Nothing here writes outside the repository except a temporary rustup download.

[CmdletBinding()]
param(
    # Clone llama.cpp if it is missing (requires git + network).
    [switch]$CloneLlamaCpp,
    # Skip the cargo build step.
    [switch]$SkipBuild,
    # Only report readiness; change nothing.
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-Location $PSScriptRoot

# External tools (nvidia-smi, tailscale, rustc, cargo, dlltool) write diagnostics
# to stderr even on success. Under $ErrorActionPreference = "Stop" that would
# abort the script, so every native invocation is wrapped in a helper that
# tolerates it and reports the exit code instead.
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
Write-Host "   AEROMESH CLUSTER NODE SETUP (Windows + CUDA)         " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$TOTAL_STEPS = 6

# ---------------------------------------------------------------------------
# 1. GPU
# ---------------------------------------------------------------------------
Write-Step 1 $TOTAL_STEPS "Checking NVIDIA GPU..."
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $r = Invoke-Native -Command "nvidia-smi" -Arguments @("--query-gpu=name,memory.total,driver_version", "--format=csv,noheader") -IgnoreErrors
    if ($r.Ok -and $r.Output) { Write-Ok "GPU: $($r.Output -replace "`r?`n", '; ')" }
    else { Write-Warn "nvidia-smi returned no usable devices." }
} else {
    Write-Warn "nvidia-smi not found. CPU-only operation is possible but very slow."
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
if (Test-Path "C:\w64devkit\bin") { $env:PATH = "C:\w64devkit\bin;$env:PATH" }
if (Test-Path $cargoBin)         { $env:PATH = "$cargoBin;$env:PATH" }

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Warn "cargo not found. Installing rustup (GNU toolchain)..."
    if (-not $CheckOnly) {
        $rustup = Join-Path $env:TEMP "rustup-init.exe"
        try {
            Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustup -UseBasicParsing
            & $rustup -y --default-toolchain stable-x86_64-pc-windows-gnu
            $env:PATH = "$cargoBin;$env:PATH"
        } catch {
            throw "rustup install failed: $($_.Exception.Message). Install Rust manually from https://rustup.rs"
        }
    }
}
if (Get-Command cargo -ErrorAction SilentlyContinue) {
    $r = Invoke-Native -Command "cargo" -Arguments @("--version") -IgnoreErrors
    Write-Ok "Rust: $($r.Output)"
} else {
    Write-Bad "Rust is not available; the build step will be skipped."
}

# ---------------------------------------------------------------------------
# 4. Python runtime (the web gateway needs it)
# ---------------------------------------------------------------------------
Write-Step 4 $TOTAL_STEPS "Checking Python..."
$pythonExe = $null
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
    Write-Ok "Using existing virtualenv: $venvPython"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = (Get-Command python).Source
    Write-Ok "Python: $(& $pythonExe --version 2>&1)"
} else {
    Write-Warn "Python 3.11+ not found. The web UI (app.py) will not run; the Rust CLI still works."
}

if ($pythonExe -and -not $CheckOnly -and -not (Test-Path $venvPython)) {
    Write-Host "  Creating .venv and installing gateway dependencies..." -ForegroundColor Cyan
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
# 5. Native llama.cpp runtime into bin\
# ---------------------------------------------------------------------------
Write-Step 5 $TOTAL_STEPS "Preparing native llama.cpp runtime in bin\..."
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
        Write-Host "  Cloning llama.cpp..." -ForegroundColor Cyan
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
    } else {
        Write-Warn "llama.cpp/ is absent. Re-run with -CloneLlamaCpp to fetch it, or copy the runtime DLLs into bin\ manually."
    }
} else {
    Write-Ok "llama.cpp/ present."
}

# Search repo-relative build outputs only -- no paths from other machines.
$candidateBuildDirs = @(
    "llama.cpp\build\bin\Release",
    "llama.cpp\build\bin",
    "llama.cpp\build\bin\Debug"
)
$copiedFrom = $null
foreach ($dir in $candidateBuildDirs) {
    if (Test-Path $dir) {
        Copy-Item -Path "$dir\*" -Destination "bin" -Recurse -Force -ErrorAction SilentlyContinue
        $copiedFrom = $dir
        break
    }
}
if ($copiedFrom) {
    Write-Ok "Copied native runtime from '$copiedFrom' into bin\."
} elseif (Test-Path "bin\llama.dll") {
    Write-Ok "bin\llama.dll already present; using it."
} else {
    Write-Warn "No llama.cpp build output found and bin\llama.dll is missing."
    Write-Host "       Build it with (from llama.cpp\):" -ForegroundColor DarkGray
    Write-Host "         cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release" -ForegroundColor DarkGray
    Write-Host "         cmake --build build --config Release -j" -ForegroundColor DarkGray
    Write-Host "       then re-run this script." -ForegroundColor DarkGray
}

# MinGW/GNU toolchains link against an *import library*, not the DLL. The Rust
# build fails with "ld: cannot find -llama" without this step.
if ((Test-Path "bin\llama.dll") -and -not $CheckOnly) {
    $hostTriple = (& rustc -vV 2>$null | Select-String -Pattern '^host:').ToString()
    $isGnu = $hostTriple -match 'gnu'
    if ($isGnu) {
        $dlltool = Get-Command dlltool -ErrorAction SilentlyContinue
        if (-not $dlltool -and (Test-Path "C:\w64devkit\bin\dlltool.exe")) {
            $env:PATH = "C:\w64devkit\bin;$env:PATH"
            $dlltool = Get-Command dlltool -ErrorAction SilentlyContinue
        }
        if ($dlltool) {
            $symbolSets = @{
                "llama" = @(
                    'llama_backend_init','llama_backend_free','llama_log_set',
                    'llama_model_default_params','llama_model_load_from_file','llama_model_free',
                    'llama_model_desc','llama_model_n_embd','llama_model_n_layer','llama_model_get_vocab',
                    'llama_context_default_params','llama_init_from_model','llama_free','llama_synchronize',
                    'llama_tokenize','llama_token_to_piece','llama_detokenize',
                    'llama_vocab_is_eog','llama_vocab_is_control','llama_vocab_bos','llama_vocab_eos','llama_vocab_n_tokens',
                    'llama_batch_init','llama_batch_free','llama_decode','llama_get_logits_ith','llama_get_embeddings_ith',
                    'llama_sampler_chain_default_params','llama_sampler_chain_init','llama_sampler_chain_add',
                    'llama_sampler_init_greedy','llama_sampler_init_temp','llama_sampler_init_top_p',
                    'llama_sampler_init_top_k','llama_sampler_init_min_p','llama_sampler_init_penalties',
                    'llama_sampler_init_dist','llama_sampler_accept','llama_sampler_reset','llama_sampler_sample',
                    'llama_sampler_free','llama_get_memory','llama_memory_clear'
                )
                # ggml_backend_tensor_get is exported by ggml-base.dll, not ggml.dll.
                "ggml"  = @(
                    'ggml_backend_load_all','ggml_backend_load_all_from_path',
                    'ggml_backend_reg_count','ggml_backend_tensor_get'
                )
            }
            $defSource = @{ "llama" = "llama.dll"; "ggml" = "ggml-base.dll" }
            foreach ($stem in $symbolSets.Keys) {
                $srcDll = $defSource[$stem]
                if (-not (Test-Path "bin\$srcDll")) { continue }
                $defPath = "bin\$stem-import.def"
                $lines = @("LIBRARY $stem", "EXPORTS") + ($symbolSets[$stem] | ForEach-Object { "    $_" })
                [System.IO.File]::WriteAllLines((Join-Path $PSScriptRoot $defPath), $lines)
                Push-Location "bin"
                try {
                    & dlltool -d "$stem-import.def" -D $srcDll -l "lib$stem.dll.a" 2>&1 | Out-Null
                } finally {
                    Pop-Location
                }
                if (Test-Path "bin\lib$stem.dll.a") {
                    Write-Ok "Generated bin\lib$stem.dll.a for the GNU toolchain."
                } else {
                    Write-Warn "dlltool could not produce bin\lib$stem.dll.a; link will fail."
                }
            }
        } else {
            Write-Warn "dlltool not found (install w64devkit). A GNU-toolchain build will fail with 'cannot find -llama'."
        }
    } else {
        Write-Ok "MSVC toolchain detected; linking against bin\*.lib directly."
    }
}

# ---------------------------------------------------------------------------
# 6. Build (RELEASE -- this is what start.ps1 executes)
# ---------------------------------------------------------------------------
Write-Step 6 $TOTAL_STEPS "Compiling AeroMesh (release)..."
$built = $false
if ($SkipBuild -or $CheckOnly) {
    Write-Host "  Skipped (-SkipBuild / -CheckOnly)." -ForegroundColor DarkGray
} elseif (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Warn "cargo unavailable; skipping build."
} else {
    $r = Invoke-Native -Command "cargo" -Arguments @("build", "--release", "--bin", "aeromesh") -IgnoreErrors
    if ($r.Ok) {
        $built = $true
        Write-Ok "Built target\release\aeromesh.exe"
    } else {
        Write-Bad "cargo build failed (exit $($r.Code))."
        if ($r.Output) { Write-Host ($r.Output -split "`n" | Select-Object -Last 12 | Out-String) -ForegroundColor DarkGray }
    }
}

# Verify the binary actually starts. Because the native runtime is now resolved
# lazily, even `--help` proves the load path works.
$exe = Join-Path $PSScriptRoot "target\release\aeromesh.exe"
if ((Test-Path $exe) -and -not $CheckOnly) {
    Write-Host "  Verifying the binary starts..." -ForegroundColor Cyan
    $env:PATH = "$(Join-Path $PSScriptRoot 'bin');$env:PATH"
    $r = Invoke-Native -Command $exe -Arguments @("--help") -IgnoreErrors
    if ($r.Output -match 'Usage: aeromesh') {
        Write-Ok "aeromesh.exe runs and resolves its native runtime."
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
Write-Host "  Launch everything interactively:   .\start.ps1" -ForegroundColor White
Write-Host "  Worker node (Laptop B):            .\start.ps1 worker -Port 50052" -ForegroundColor White
Write-Host "  Coordinator (Laptop A):            .\start.ps1 coordinator -Peers '<ip>:50052'" -ForegroundColor White
Write-Host "  Inspect a model:                   .\target\release\aeromesh.exe model-check models\<file>.gguf" -ForegroundColor White
Write-Host "  Run the tests:                     .\target\release\aeromesh.exe --help   # then: cargo test --workspace" -ForegroundColor DarkGray
