# ==============================================================================
# AeroMESH Windows Installer Build & Staging Script
#
# Builds the Rust release binaries (aeromesh.exe + AeroMesh-Desktop.exe),
# verifies offline UI vendor assets, stages llama.cpp (pinned b10499) backend
# DLLs (Core, CPU, NVIDIA CUDA 12 + redistributable CUDA 12 runtime DLLs,
# Vulkan) and optional Diffusion tools, then compiles AeroMesh-Setup-x64.exe
# with Inno Setup 6 and emits dist\SHA256SUMS.txt.
#
# Usage:
#   pwsh -File .\installer\build-installer.ps1
#   pwsh -File .\installer\build-installer.ps1 -Version 0.3.0 -IncludeCuda -IncludeVulkan
# ==============================================================================

[CmdletBinding()]
param(
    [string]$Version = "0.3.0",
    [string]$LlamaCppTag = "b10499",
    [switch]$SkipRustBuild,
    [switch]$SkipVendorAssets,
    [switch]$FetchLlamaPrebuilt,
    [switch]$IncludeCuda = $true,
    [switch]$IncludeVulkan = $true,
    [switch]$IncludeDiffusion,
    [string]$CudaBinDir = "",
    [string]$VcRedistPath = "",
    [string]$SignThumbprint = "",
    [string]$TimestampUrl = "http://timestamp.digicert.com"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = Split-Path -Parent $PSScriptRoot
$distDir = Join-Path $repoRoot "dist"
$stagingDir = Join-Path $distDir "staging"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "   AEROMESH WINDOWS INSTALLER BUILDER (v$Version)               " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# 1. Clean & create staging tree
if (Test-Path $stagingDir) {
    Remove-Item -Path $stagingDir -Recurse -Force
}
$stageSubDirs = @(
    $stagingDir,
    (Join-Path $stagingDir "bin"),
    (Join-Path $stagingDir "backends\cpu"),
    (Join-Path $stagingDir "backends\cuda"),
    (Join-Path $stagingDir "backends\vulkan"),
    (Join-Path $stagingDir "diffusion"),
    (Join-Path $stagingDir "redist"),
    (Join-Path $stagingDir "static")
)
foreach ($d in $stageSubDirs) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}

# 2. Fetch & SHA-384 verify offline frontend vendor libraries
if (-not $SkipVendorAssets) {
    Write-Host "`n[1/5] Preparing offline UI vendor assets..." -ForegroundColor Yellow
    $vendorScript = Join-Path $repoRoot "scripts\fetch-vendor-assets.ps1"
    & $vendorScript
}

# 3. Compile Rust release binaries
if (-not $SkipRustBuild) {
    Write-Host "`n[2/5] Compiling Rust release binaries (aeromesh + AeroMesh-Desktop)..." -ForegroundColor Yellow
    Push-Location $repoRoot
    try {
        & cargo build --release --bin aeromesh --bin AeroMesh-Desktop
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
}

$cliExe = Join-Path $repoRoot "target\release\aeromesh.exe"
$desktopExe = Join-Path $repoRoot "target\release\AeroMesh-Desktop.exe"
if (-not (Test-Path $cliExe) -or -not (Test-Path $desktopExe)) {
    throw "Expected compiled binaries missing: $cliExe and/or $desktopExe"
}
Copy-Item $cliExe -Destination $stagingDir -Force
Copy-Item $desktopExe -Destination $stagingDir -Force
Write-Host "  [OK] Staged aeromesh.exe and AeroMesh-Desktop.exe" -ForegroundColor Green

# Copy static web UI + offline vendor assets
Copy-Item -Path (Join-Path $repoRoot "static\*") -Destination (Join-Path $stagingDir "static") -Recurse -Force
Write-Host "  [OK] Staged static\ web interface and offline vendor assets" -ForegroundColor Green

# 4. Stage llama.cpp core + CPU / CUDA 12 / Vulkan backend DLLs
Write-Host "`n[3/5] Staging llama.cpp ($LlamaCppTag) runtime & compute backends..." -ForegroundColor Yellow

function Copy-MatchingDlls {
    param([string]$SourceDir, [string[]]$Patterns, [string]$DestDir)
    if (-not (Test-Path $SourceDir)) { return 0 }
    $count = 0
    foreach ($pat in $Patterns) {
        Get-ChildItem -Path $SourceDir -Filter $pat -File -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item $_.FullName -Destination $DestDir -Force
            $count++
        }
    }
    return $count
}

if ($FetchLlamaPrebuilt) {
    Write-Host "  Fetching official llama.cpp release $LlamaCppTag Windows archives..." -ForegroundColor Cyan
    $dlTmp = Join-Path $distDir "llama_dl_$LlamaCppTag"
    if (Test-Path $dlTmp) { Remove-Item $dlTmp -Recurse -Force }
    New-Item -ItemType Directory -Path $dlTmp -Force | Out-Null

    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$LlamaCppTag"
    $targets = @(
        @{ Regex = 'llama-.*-bin-win-cpu-x64\.zip$';             Label = "CPU";     OutSub = "cpu" },
        @{ Regex = 'llama-.*-bin-win-cuda-12\.\d+-x64\.zip$';    Label = "CUDA 12"; OutSub = "cuda";   Enabled = $IncludeCuda },
        @{ Regex = 'cudart-llama-bin-win-cuda-12\.\d+-x64\.zip$';Label = "CUDA RT"; OutSub = "cudart"; Enabled = $IncludeCuda },
        @{ Regex = 'llama-.*-bin-win-vulkan-x64\.zip$';          Label = "Vulkan";  OutSub = "vulkan"; Enabled = $IncludeVulkan }
    )

    foreach ($t in $targets) {
        if ($t.ContainsKey("Enabled") -and -not $t.Enabled) { continue }
        $asset = $release.assets | Where-Object { $_.name -match $t.Regex } | Select-Object -First 1
        if (-not $asset) {
            Write-Host "  [!] No matching archive for $($t.Label) in llama.cpp $LlamaCppTag" -ForegroundColor Yellow
            continue
        }
        $zipPath = Join-Path $dlTmp $asset.name
        $extractDir = Join-Path $dlTmp $t.OutSub
        New-Item -ItemType Directory -Path $extractDir -Force | Out-Null
        Write-Host "  Downloading $($asset.name)..." -ForegroundColor DarkGray
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zipPath -UseBasicParsing
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

        # Core shared DLLs (only once from the first archive that has them)
        Copy-MatchingDlls -SourceDir $extractDir -Patterns @("llama.dll", "ggml.dll", "ggml-base.dll", "libomp*.dll") -DestDir (Join-Path $stagingDir "bin") | Out-Null

        if ($t.OutSub -eq "cpu") {
            Copy-MatchingDlls -SourceDir $extractDir -Patterns @("ggml-cpu*.dll") -DestDir (Join-Path $stagingDir "backends\cpu") | Out-Null
        } elseif ($t.OutSub -eq "cuda") {
            Copy-MatchingDlls -SourceDir $extractDir -Patterns @("ggml-cuda*.dll") -DestDir (Join-Path $stagingDir "backends\cuda") | Out-Null
        } elseif ($t.OutSub -eq "cudart") {
            Copy-MatchingDlls -SourceDir $extractDir -Patterns @("cudart64_12*.dll", "cublas64_12*.dll", "cublasLt64_12*.dll") -DestDir (Join-Path $stagingDir "backends\cuda") | Out-Null
        } elseif ($t.OutSub -eq "vulkan") {
            Copy-MatchingDlls -SourceDir $extractDir -Patterns @("ggml-vulkan*.dll") -DestDir (Join-Path $stagingDir "backends\vulkan") | Out-Null
        }
    }
} else {
    # Stage from local repo bin\ directory
    $localBin = Join-Path $repoRoot "bin"
    $coreCount = Copy-MatchingDlls -SourceDir $localBin -Patterns @("llama.dll", "ggml.dll", "ggml-base.dll", "libomp*.dll") -DestDir (Join-Path $stagingDir "bin")
    $cpuCount = Copy-MatchingDlls -SourceDir $localBin -Patterns @("ggml-cpu*.dll") -DestDir (Join-Path $stagingDir "backends\cpu")
    Write-Host "  Staged $coreCount core DLL(s) and $cpuCount CPU backend DLL(s) from bin\" -ForegroundColor Green

    if ($IncludeCuda) {
        $cudaCount = Copy-MatchingDlls -SourceDir $localBin -Patterns @("ggml-cuda*.dll", "cudart64_12*.dll", "cublas64_12*.dll", "cublasLt64_12*.dll") -DestDir (Join-Path $stagingDir "backends\cuda")
        # Also search installed CUDA Toolkit bin directory if CUDA runtime DLLs were not in bin\
        $resolvedCudaBin = $CudaBinDir
        if (-not $resolvedCudaBin -and $env:CUDA_PATH) {
            $resolvedCudaBin = Join-Path $env:CUDA_PATH "bin"
        }
        if ($resolvedCudaBin -and (Test-Path $resolvedCudaBin)) {
            $rtCount = Copy-MatchingDlls -SourceDir $resolvedCudaBin -Patterns @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll") -DestDir (Join-Path $stagingDir "backends\cuda")
            $cudaCount += $rtCount
        }
        Write-Host "  Staged $cudaCount NVIDIA CUDA 12 DLL(s)" -ForegroundColor Green
    }

    if ($IncludeVulkan) {
        $vkCount = Copy-MatchingDlls -SourceDir $localBin -Patterns @("ggml-vulkan*.dll") -DestDir (Join-Path $stagingDir "backends\vulkan")
        Write-Host "  Staged $vkCount Vulkan backend DLL(s)" -ForegroundColor Green
    }
}

if ($IncludeDiffusion) {
    $localDiff = Join-Path $repoRoot "bin\diffusion"
    $sdCount = Copy-MatchingDlls -SourceDir $localDiff -Patterns @("sd-cli.exe", "stable-diffusion*.dll") -DestDir (Join-Path $stagingDir "diffusion")
    Write-Host "  Staged $sdCount Stable Diffusion file(s)" -ForegroundColor Green
}

if ($VcRedistPath -and (Test-Path $VcRedistPath)) {
    Copy-Item $VcRedistPath -Destination (Join-Path $stagingDir "redist\vc_redist.x64.exe") -Force
    Write-Host "  [OK] Staged vc_redist.x64.exe" -ForegroundColor Green
}

# 5. Locate Inno Setup 6 (ISCC.exe) and compile AeroMesh-Setup-x64.exe
Write-Host "`n[4/5] Compiling Windows Setup Wizard (Inno Setup 6)..." -ForegroundColor Yellow
$iscc = Get-Command iscc -ErrorAction SilentlyContinue
$isccPath = if ($iscc) { $iscc.Source } else { "" }
if (-not $isccPath) {
    foreach ($cand in @(
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe",
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
    )) {
        if (Test-Path $cand) { $isccPath = $cand; break }
    }
}
if (-not $isccPath) {
    throw "Inno Setup 6 (ISCC.exe) not found. Install it from https://jrsoftware.org/isinfo.php or via 'winget install JRSoftware.InnoSetup'."
}

$issFile = Join-Path $PSScriptRoot "AeroMesh.iss"
& $isccPath "/DAppVersion=$Version" "/DStagingDir=$stagingDir" "/DOutputDirPath=$distDir" $issFile
if ($LASTEXITCODE -ne 0) { throw "ISCC.exe failed with exit code $LASTEXITCODE" }

$installerExe = Join-Path $distDir "AeroMesh-Setup-x64.exe"
if (-not (Test-Path $installerExe)) {
    throw "Installer was not produced at $installerExe"
}

# Optional Authenticode signing
if ($SignThumbprint) {
    Write-Host "  Signing AeroMesh-Setup-x64.exe with certificate $SignThumbprint..." -ForegroundColor Cyan
    $signtool = Get-Command signtool -ErrorAction SilentlyContinue
    if (-not $signtool) { throw "signtool.exe not found on PATH (install Windows SDK)." }
    & $signtool.Source sign /sha1 $SignThumbprint /fd SHA256 /tr $TimestampUrl /td SHA256 $installerExe
    if ($LASTEXITCODE -ne 0) { throw "signtool failed with exit code $LASTEXITCODE" }
}

# 6. Emit SHA256SUMS.txt
Write-Host "`n[5/5] Generating dist\SHA256SUMS.txt..." -ForegroundColor Yellow
$hash = (Get-FileHash -Path $installerExe -Algorithm SHA256).Hash.ToLower()
$checksumFile = Join-Path $distDir "SHA256SUMS.txt"
Set-Content -Path $checksumFile -Value "$hash  AeroMesh-Setup-x64.exe" -Encoding ASCII

$sizeMb = [math]::Round((Get-Item $installerExe).Length / 1MB, 2)
Write-Host "`n================================================================" -ForegroundColor Green
Write-Host "   BUILD COMPLETE: dist\AeroMesh-Setup-x64.exe ($sizeMb MB)" -ForegroundColor Green
Write-Host "   SHA-256: $hash" -ForegroundColor Gray
Write-Host "================================================================" -ForegroundColor Green
