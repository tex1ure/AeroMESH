# ==============================================================================
# AeroMesh 1-Click All-in-One Cluster Launcher
#
# Usage:
#   .\start.ps1                         -> Interactive menu
#   .\start.ps1 coordinator             -> Coordinator API + web UI (Laptop A)
#   .\start.ps1 worker                  -> Zero-weight pipeline worker (Laptop B)
#   .\start.ps1 all                     -> Full local mesh: worker + coordinator + UI
#   .\start.ps1 status                  -> Scan the cluster mesh
#
# Common options:
#   -Model <path>      Explicit .gguf (otherwise models\ is scanned)
#   -Layers "0..24"    Layer range for this node (worker / coordinator / all)
#   -Peers "<ip:port>" Comma-separated worker endpoints (coordinator / all)
#   -Port <n>          Worker TCP port, or API port for coordinator/all
#   -Mode pipeline|rpc pipeline = zero-weight activation streaming (default)
#   -NoUI              Do not start the Python web gateway
#   -NoBrowser         Do not open a browser window
#   -Force             Kill conflicting listeners without asking
# ==============================================================================

[CmdletBinding()]
param (
    [Parameter(Position = 0)]
    [ValidateSet("coordinator", "worker", "all", "status", "auto", "doctor", "")]
    [string]$Role = "",

    [Parameter(Mandatory = $false)]
    [string]$Model = "",

    [Parameter(Mandatory = $false)]
    [string]$Layers = "",

    [Parameter(Mandatory = $false)]
    [string]$Peers = "",

    [Parameter(Mandatory = $false)]
    [ValidateSet("pipeline", "rpc", "")]
    [string]$Mode = "pipeline",

    [Parameter(Mandatory = $false)]
    [int]$Port = 0,

    [Parameter(Mandatory = $false)]
    [switch]$NoUI,

    [Parameter(Mandatory = $false)]
    [switch]$NoBrowser,

    [Parameter(Mandatory = $false)]
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$script:DefaultWorkerPort = 50052
$script:DefaultApiPort = 8080
$script:DefaultUiPort = 7860

# ------------------------------------------------------------------------------
# Toolchain / PATH
# ------------------------------------------------------------------------------
$binDir = Join-Path $PSScriptRoot "bin"
$userPythonDir = Join-Path $env:USERPROFILE "AppData\Local\Programs\Python"
$pyPathAdditions = ""
if (Test-Path $userPythonDir) {
    $latestPy = Get-ChildItem -Path $userPythonDir -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
    if ($latestPy) { $pyPathAdditions = "$($latestPy.FullName);$($latestPy.FullName)\Scripts;" }
}
$env:PATH = "C:\w64devkit\bin;$(Join-Path $env:USERPROFILE '.cargo\bin');$binDir;$pyPathAdditions$env:PATH"

# Native runtime next to the release binary, so it is always resolvable.
if (Test-Path "target\release") {
    Get-ChildItem -Path "bin\*.dll" -ErrorAction SilentlyContinue |
        Copy-Item -Destination "target\release" -Force -ErrorAction SilentlyContinue
}

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
    [pscustomobject]@{ Ok = ($code -eq 0); Code = $code; Output = ($output | Out-String).Trim() }
}

# ------------------------------------------------------------------------------
# Secrets
# ------------------------------------------------------------------------------
function Initialize-AeroMeshSecrets {
    $envPath = Join-Path $PSScriptRoot ".env"
    $apiKey = ""
    $workerSecret = ""
    $fileKeys = @{}

    if (Test-Path $envPath) {
        Get-Content $envPath | ForEach-Object {
            $line = $_.Trim()
            if (-not $line -or $line.StartsWith("#")) { return }
            $idx = $line.IndexOf("=")
            if ($idx -gt 0) {
                $k = $line.Substring(0, $idx).Trim()
                $v = $line.Substring($idx + 1).Trim().Trim('"').Trim("'")
                $fileKeys[$k] = $v
                [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
                Set-Item -Path "env:$k" -Value $v
                if ($k -eq "AEROMESH_API_KEY") { $apiKey = $v }
                if ($k -eq "AEROMESH_WORKER_SECRET") { $workerSecret = $v }
            }
        }
    }

    function New-Secret {
        $bytes = New-Object byte[] 32
        [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        return ([System.BitConverter]::ToString($bytes)).Replace("-", "").ToLower()
    }

    # A key that is present but empty is treated as missing, and the file is
    # rewritten so the on-disk value always matches the in-process value.
    $apiMissing = [string]::IsNullOrWhiteSpace($apiKey)
    $secretMissing = [string]::IsNullOrWhiteSpace($workerSecret)
    if ($apiMissing) { $apiKey = New-Secret }
    if ($secretMissing) { $workerSecret = New-Secret }

    if ($apiMissing -or $secretMissing -or -not (Test-Path $envPath)) {
        $lines = @(
            "# AeroMesh Cluster Authentication Secrets (SEC-01 / generated by start.ps1)",
            "# Keep this file secure and never commit it to source control.",
            "AEROMESH_API_KEY=$apiKey",
            "AEROMESH_WORKER_SECRET=$workerSecret",
            "AEROMESH_ACTIVATION_DTYPE=$($fileKeys['AEROMESH_ACTIVATION_DTYPE'])"
        )
        if ([string]::IsNullOrWhiteSpace($fileKeys['AEROMESH_ACTIVATION_DTYPE'])) {
            $lines[4] = "AEROMESH_ACTIVATION_DTYPE=int8"
        }
        Set-Content -Path $envPath -Value $lines -Encoding UTF8
        Write-Host "  [*] Generated cluster secrets in .env" -ForegroundColor DarkGray
    }

    $env:AEROMESH_API_KEY = $apiKey
    $env:AEROMESH_WORKER_SECRET = $workerSecret
    $env:LLAMA_API_KEY = $apiKey
    if (-not $env:AEROMESH_ACTIVATION_DTYPE) {
        $env:AEROMESH_ACTIVATION_DTYPE = "int8"
        [System.Environment]::SetEnvironmentVariable("AEROMESH_ACTIVATION_DTYPE", "int8", "Process")
    }
}

# ------------------------------------------------------------------------------
# Port handling
# ------------------------------------------------------------------------------
function Stop-PortConflict {
    [CmdletBinding()]
    param ([int[]]$Ports, [switch]$Always)

    foreach ($p in $Ports) {
        $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
        if (-not $conns) { continue }

        foreach ($c in $conns) {
            $ownerPid = $c.OwningProcess
            if (-not $ownerPid -or $ownerPid -eq $PID) { continue }

            $procName = "unknown"
            $proc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
            if ($proc) { $procName = $proc.ProcessName }

            $isOurs = $procName -match '^(aeromesh|llama-server|ggml-rpc-server|python|pythonw)$'
            if (-not $isOurs -and -not $Force -and -not $Always) {
                Write-Host "  [!] Port $p is held by '$procName' (PID $ownerPid), which is not an AeroMesh process." -ForegroundColor Yellow
                Write-Host "      Re-run with -Force to terminate it, or choose another port with -Port." -ForegroundColor Yellow
                throw "Port $p is in use by '$procName' (PID $ownerPid)."
            }

            Write-Host "  [*] Freeing port $p (stopping $procName, PID $ownerPid)..." -ForegroundColor DarkGray
            Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
        }
    }
}

function Test-PortFree {
    param ([int]$Port)
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return (-not $conns)
}

function Test-PeerReachable {
    param ([string]$PeerStr)
    if (-not $PeerStr) { return $false }
    try {
        $idx = $PeerStr.LastIndexOf(":")
        if ($idx -le 0) { return $false }
        $h = $PeerStr.Substring(0, $idx).Trim()
        $p = [int]$PeerStr.Substring($idx + 1).Trim()
        $client = New-Object System.Net.Sockets.TcpClient
        try {
            $iar = $client.BeginConnect($h, $p, $null, $null)
            return $iar.AsyncWaitHandle.WaitOne(2000, $false)
        } finally {
            $client.Close()
        }
    } catch {
        return $false
    }
}

function Wait-ServerReady {
    param ([string]$Url, [int]$TimeoutSeconds = 45, [string]$Label = "Engine")
    Write-Host "  [*] Waiting for $Label to become ready (up to ${TimeoutSeconds}s)..." -ForegroundColor DarkGray
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri "$Url/health" -UseBasicParsing -TimeoutSec 1 -ErrorAction SilentlyContinue
            if ($resp -and $resp.StatusCode -eq 200) {
                Write-Host "  [+] $Label ready." -ForegroundColor Green
                return $true
            }
        } catch {}
        Start-Sleep -Milliseconds 600
    }
    Write-Host "  [!] $Label did not report healthy within ${TimeoutSeconds}s." -ForegroundColor Yellow
    return $false
}

function Get-AeroMeshExe {
    # Prefer the release binary (what setup.ps1 builds); fall back to debug so a
    # plain `cargo build` checkout still works.
    foreach ($candidate in @("target\release\aeromesh.exe", "target\debug\aeromesh.exe")) {
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }
    return ""
}

function Find-AeroMeshWorkers {
    param ([int]$TargetPort = 50052)
    Write-Host "  [*] Auto-scanning Tailscale Mesh & Wi-Fi /24 Subnet for active Worker nodes (:$TargetPort)..." -ForegroundColor Cyan
    $candidateIPs = New-Object System.Collections.Generic.List[string]
    $localIPs = @("127.0.0.1", "0.0.0.0")
    try {
        $localIPs += @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty IPAddress)
    } catch {}

    # 1. Tailscale peers
    if (Get-Command tailscale -ErrorAction SilentlyContinue) {
        try {
            $tsJson = (& tailscale status --json 2>$null | Out-String)
            if ($tsJson) {
                $tsObj = $tsJson | ConvertFrom-Json
                if ($tsObj.Peer) {
                    foreach ($prop in $tsObj.Peer.PSObject.Properties) {
                        $peer = $prop.Value
                        if ($peer.Online -and $peer.TailscaleIPs) {
                            foreach ($ip in $peer.TailscaleIPs) {
                                if ($ip -match '^\d+\.\d+\.\d+\.\d+$' -and $ip -notin $localIPs -and -not $candidateIPs.Contains($ip)) {
                                    $candidateIPs.Add($ip)
                                }
                            }
                        }
                    }
                }
            }
        } catch {}
    }

    # 2. Local Wi-Fi / LAN ARP neighbors + active /24 subnet sweep
    try {
        $arpLines = (& arp -a 2>$null)
        foreach ($line in $arpLines) {
            if ($line -match '(\d+\.\d+\.\d+\.\d+)\s+[0-9a-fA-F-]+\s+dynamic') {
                $lip = $Matches[1]
                if ($lip -notin $localIPs -and $lip -notmatch '\.255$' -and $lip -notmatch '^224\.' -and -not $candidateIPs.Contains($lip)) {
                    $candidateIPs.Add($lip)
                }
            }
        }
        # Also sweep the primary Wi-Fi / LAN /24 subnet so even un-pinged laptops are discovered
        foreach ($myIp in $localIPs) {
            if ($myIp -match '^(192\.168\.\d+|10\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+)\.(\d+)$') {
                $prefix = $Matches[1]
                for ($h = 1; $h -le 254; $h++) {
                    $subIp = "$prefix.$h"
                    if ($subIp -notin $localIPs -and -not $candidateIPs.Contains($subIp)) {
                        $candidateIPs.Add($subIp)
                    }
                }
                break
            }
        }
    } catch {}

    $found = New-Object System.Collections.Generic.List[string]
    $tasks = @{}
    foreach ($cip in $candidateIPs) {
        try {
            $sock = New-Object System.Net.Sockets.TcpClient
            $tasks[$cip] = @{ Socket = $sock; Async = $sock.BeginConnect($cip, $TargetPort, $null, $null) }
        } catch {}
    }
    foreach ($cip in $tasks.Keys) {
        $entry = $tasks[$cip]
        try {
            if ($entry.Async.AsyncWaitHandle.WaitOne(380, $false) -and $entry.Socket.Connected) {
                $ep = "${cip}:${TargetPort}"
                $found.Add($ep)
                $netType = if ($cip.StartsWith("100.")) { "Tailscale WireGuard" } else { "Wi-Fi / LAN Direct" }
                Write-Host "      [+] Auto-Discovered Worker: $ep ($netType)" -ForegroundColor Green
            }
        } catch {} finally {
            $entry.Socket.Close()
        }
    }
    return ,$found
}

function Invoke-AeroMeshBootstrap {
    Write-Host "`n========================================================================" -ForegroundColor Cyan
    Write-Host "   [*] AEROMESH BRAND-NEW PC DOCTOR & AUTO-BOOTSTRAPPER                 " -ForegroundColor Cyan
    Write-Host "========================================================================" -ForegroundColor Cyan

    foreach ($dir in @("models", "bin", "bin\diffusion")) {
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    }

    # 1. Check / Build Rust aeromesh.exe if missing
    $exe = Get-AeroMeshExe
    if (-not $exe) {
        Write-Host "  [*] aeromesh.exe not built yet -> Running setup.ps1 to compile release binary..." -ForegroundColor Yellow
        & .\setup.ps1
        $exe = Get-AeroMeshExe
    } else {
        Write-Host "  [OK] Rust Engine Binary:      $exe" -ForegroundColor Green
    }

    # 2. Check / Auto-Download CUDA Diffusion Engine (bin\diffusion\sd-cli.exe + cublas64_12.dll)
    $sdCli = Join-Path $PSScriptRoot "bin\diffusion\sd-cli.exe"
    $cublasDll = Join-Path $PSScriptRoot "bin\diffusion\cublas64_12.dll"
    if (-not (Test-Path $sdCli) -or -not (Test-Path $cublasDll)) {
        Write-Host "  [*] Auto-installing CUDA 12 Diffusion Engine (sd-cli.exe + cuBLAS DLLs)..." -ForegroundColor Cyan
        try {
            $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
            foreach ($asset in $rel.assets) {
                $needSd = ($asset.name -match "win-cuda12-x64\.zip$" -and $asset.name -notmatch "^cudart" -and -not (Test-Path $sdCli))
                $needCu = ($asset.name -match "^cudart.*cu12.*\.zip$" -and -not (Test-Path $cublasDll))
                if ($needSd -or $needCu) {
                    $zPath = Join-Path $env:TEMP $asset.name
                    $xDir = Join-Path $env:TEMP ("sd_auto_" + [System.IO.Path]::GetFileNameWithoutExtension($asset.name))
                    curl.exe -L -C - $asset.browser_download_url -o $zPath
                    if (Test-Path $xDir) { Remove-Item $xDir -Recurse -Force }
                    Expand-Archive -Path $zPath -DestinationPath $xDir -Force
                    Get-ChildItem -Path $xDir -Recurse -Include "*.dll", "*.exe" | ForEach-Object {
                        Copy-Item $_.FullName -Destination "bin\diffusion\" -Force
                    }
                }
            }
        } catch {
            Write-Host "  [!] Could not auto-download diffusion binaries: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }

    Get-ChildItem -Path "bin\*.dll" -File -ErrorAction SilentlyContinue | ForEach-Object {
        if (-not (Test-Path (Join-Path "bin\diffusion" $_.Name))) {
            Copy-Item $_.FullName -Destination "bin\diffusion\" -Force
        }
    }

    if (Test-Path $sdCli) {
        $r = Invoke-Native -Command $sdCli -Arguments @("--help") -IgnoreErrors
        if ($r.Ok) {
            Write-Host "  [OK] CUDA Diffusion Engine:   bin\diffusion\sd-cli.exe (DLLs Verified)" -ForegroundColor Green
        } else {
            Write-Host "  [!] CUDA Diffusion Engine:   Exited with code $($r.Code)" -ForegroundColor Yellow
        }
    }

    # 3. Check Diffusion Model in models\
    $sdModels = @(Get-ChildItem -Path "models" -Filter "*.safetensors" -File -ErrorAction SilentlyContinue)
    if ($sdModels.Count -eq 0) {
        Write-Host "  [*] No .safetensors image model found in models\ -> Auto-downloading DreamShaper 8 (1.99 GB FP16)..." -ForegroundColor Cyan
        curl.exe -L -C - "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors" -o "models\DreamShaper_8_pruned.safetensors"
        $sdModels = @(Get-ChildItem -Path "models" -Filter "*.safetensors" -File -ErrorAction SilentlyContinue)
    }
    if ($sdModels.Count -gt 0) {
        Write-Host "  [OK] Active Image Model:      models\$($sdModels[0].Name)" -ForegroundColor Green
    }
}

function Show-AeroMeshBanner {
    Write-Host ""
    Write-Host "========================================================================" -ForegroundColor Cyan
    Write-Host "   [+] AEROMESH: DISTRIBUTED ZERO-WEIGHT LLM CLUSTER ENGINE             " -ForegroundColor Cyan
    Write-Host "   Windows + NVIDIA CUDA + Tailscale WireGuard (0.0 MB Wire Transfer)   " -ForegroundColor DarkCyan
    Write-Host "========================================================================" -ForegroundColor Cyan
}

function Find-ModelPath {
    param ([string]$ExplicitPath)

    if ($ExplicitPath) {
        if (Test-Path $ExplicitPath) { return $ExplicitPath }
        $inModels = Join-Path "models" $ExplicitPath
        if (Test-Path $inModels) { return $inModels }
        throw "Model '$ExplicitPath' not found (also checked models\$ExplicitPath)."
    }

    $all = @(Get-ChildItem -Path @("models", ".") -Filter "*.gguf" -File -ErrorAction SilentlyContinue |
             Where-Object { $_.Name -notmatch '_stage\d' })
    if ($all.Count -eq 0) { return "" }

    $preferred = $all | Where-Object { $_.Name -eq "test.gguf" } | Select-Object -First 1
    if (-not $preferred) {
        $preferred = $all | Sort-Object Length -Descending | Select-Object -First 1
    }
    if ($preferred.Directory.Name -eq "models") { return ("models/" + $preferred.Name) }
    return $preferred.Name
}

function Get-PythonCommand {
    $venv = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path $venv) { return $venv }

    foreach ($cand in @(@("python", @("--version")), @("py", @("-3", "--version")))) {
        $cmd = $cand[0]
        if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { continue }
        $r = Invoke-Native -Command $cmd -Arguments $cand[1] -IgnoreErrors
        if ($r.Ok) { return $cmd }
    }

    $userPy = Join-Path $env:USERPROFILE "AppData\Local\Programs\Python"
    if (Test-Path $userPy) {
        $found = Get-ChildItem -Path $userPy -Filter "python.exe" -Recurse -Depth 2 -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    return ""
}

function Get-TailscaleIPv4 {
    if (Get-Command tailscale -ErrorAction SilentlyContinue) {
        $r = Invoke-Native -Command "tailscale" -Arguments @("ip", "-4") -IgnoreErrors
        if ($r.Ok -and $r.Output) { return ($r.Output -split "`r?`n")[0].Trim() }
    }
    return "127.0.0.1"
}

function Start-WebGateway {
    <#
      Launches the FastAPI gateway in the foreground and returns when it exits.
      -NoUI skips it entirely.
    #>
    param (
        [string]$CoordUrl,
        [int]$UiPort,
        [string]$PythonExe
    )
    if ($NoUI) {
        Write-Host "  [*] -NoUI set: skipping the web gateway." -ForegroundColor DarkGray
        Write-Host "      The OpenAI-compatible API is at $CoordUrl" -ForegroundColor DarkGray
        return
    }
    if (-not $PythonExe) {
        Write-Host "  [!] Python not found; skipping the web gateway." -ForegroundColor Yellow
        Write-Host "      Install Python 3.11+ then run: python app.py" -ForegroundColor Yellow
        Write-Host "      The API is still available at $CoordUrl" -ForegroundColor DarkGray
        return
    }

    $env:AEROMESH_ENDPOINT = $CoordUrl
    $env:AEROMESH_UI_PORT = "$UiPort"
    $env:PORT = "$UiPort"
    Write-Host "  [*] Web UI: http://127.0.0.1:$UiPort" -ForegroundColor Cyan
    Push-Location $PSScriptRoot
    try {
        if ($PythonExe -eq "python" -or $PythonExe -eq "py") {
            & $PythonExe app.py
        } else {
            & $PythonExe app.py
        }
    } finally {
        Pop-Location
    }
}

# ------------------------------------------------------------------------------
# Banner / secrets / interactive role selection
# ------------------------------------------------------------------------------
Show-AeroMeshBanner
Initialize-AeroMeshSecrets

$detectedModel = ""
try {
    $detectedModel = Find-ModelPath -ExplicitPath $Model
} catch {
    Write-Host "  [!] $($_.Exception.Message)" -ForegroundColor Yellow
}
$modelName = if ($detectedModel) { [System.IO.Path]::GetFileName($detectedModel) } else { "No .gguf found in models/" }
$localTailscaleIP = Get-TailscaleIPv4

Write-Host ""
Write-Host "  [-] Local Tailscale IP:  " -NoNewline -ForegroundColor Gray
Write-Host "$localTailscaleIP" -ForegroundColor Green
Write-Host "  [-] Detected Model:      " -NoNewline -ForegroundColor Gray
Write-Host "$modelName" -ForegroundColor Yellow
Write-Host "  [-] Wire Weight Transfer:" -NoNewline -ForegroundColor Gray
Write-Host " 0.0 MB (P2P activation streaming)" -ForegroundColor Cyan
$activeCodec = if ($env:AEROMESH_ACTIVATION_DTYPE) { $env:AEROMESH_ACTIVATION_DTYPE.ToUpper() } else { "INT8" }
Write-Host "  [-] Activation Codec:    " -NoNewline -ForegroundColor Gray
Write-Host "$activeCodec" -ForegroundColor Yellow
Write-Host "  [-] Control Surface Auth:" -NoNewline -ForegroundColor Gray
Write-Host " Enabled (API key + worker handshake secret)" -ForegroundColor Green
Write-Host "------------------------------------------------------------------------" -ForegroundColor DarkGray

if (-not $Role) {
    Write-Host ""
    Write-Host "Select your node role for this machine:" -ForegroundColor White
    Write-Host "  [A] " -NoNewline -ForegroundColor Cyan
    Write-Host "Auto-Pilot (1-Click)      " -NoNewline -ForegroundColor Cyan
    Write-Host "(Checks PC, scans Wi-Fi & Tailscale for workers, launches full cluster)" -ForegroundColor Gray
    Write-Host "  [1] " -NoNewline -ForegroundColor Cyan
    Write-Host "Coordinator Node + Web UI " -NoNewline -ForegroundColor Green
    Write-Host "(Laptop A: runs the lower layers, hosts the API + web interface)" -ForegroundColor Gray
    Write-Host "  [2] " -NoNewline -ForegroundColor Cyan
    Write-Host "Worker Node               " -NoNewline -ForegroundColor Yellow
    Write-Host "(Laptop B: runs the upper layers + LM head)" -ForegroundColor Gray
    Write-Host "  [3] " -NoNewline -ForegroundColor Cyan
    Write-Host "Full Local Mesh           " -NoNewline -ForegroundColor Magenta
    Write-Host "(worker + coordinator + web UI on this machine, via shared memory)" -ForegroundColor Gray
    Write-Host "  [4] " -NoNewline -ForegroundColor Cyan
    Write-Host "Cluster Status / Probe    " -NoNewline -ForegroundColor White
    Write-Host "(scans the mesh for reachable workers)" -ForegroundColor Gray
    Write-Host "  [Q] Quit" -ForegroundColor DarkGray
    Write-Host ""

    $choice = Read-Host "Enter choice [A, 1, 2, 3, 4, Q] (Default: A)"
    switch ($choice.Trim().ToUpper()) {
        ""  { $Role = "auto" }
        "A" { $Role = "auto" }
        "1" { $Role = "coordinator" }
        "2" { $Role = "worker" }
        "3" { $Role = "all" }
        "4" { $Role = "status" }
        default {
            Write-Host ""
            Write-Host "Exiting AeroMesh Launcher." -ForegroundColor DarkGray
            exit 0
        }
    }
}

# ------------------------------------------------------------------------------
# 0. AUTO-PILOT & DOCTOR ROLES (.\start.ps1 auto | .\start.ps1 doctor)
# ------------------------------------------------------------------------------
if ($Role -eq "doctor") {
    Invoke-AeroMeshBootstrap
    $discovered = Find-AeroMeshWorkers -TargetPort $script:DefaultWorkerPort
    Write-Host "`n========================================================================" -ForegroundColor Cyan
    Write-Host "   DOCTOR SUMMARY: Discovered $($discovered.Count) remote worker(s) on Wi-Fi / Tailscale" -ForegroundColor Green
    Write-Host "========================================================================`n" -ForegroundColor Cyan
    exit 0
}

if ($Role -eq "auto") {
    Invoke-AeroMeshBootstrap
    $discovered = Find-AeroMeshWorkers -TargetPort $script:DefaultWorkerPort
    if ($discovered.Count -gt 0) {
        $Peers = ($discovered -join ",")
        Write-Host "`n  [+] AUTO-PILOT: Found $($discovered.Count) remote worker(s) ($Peers) -> Starting as Coordinator + Web UI!" -ForegroundColor Green
        $Role = "coordinator"
    } else {
        Write-Host "`n  [+] AUTO-PILOT: No external workers on Wi-Fi/Tailscale -> Starting Full 2-Node Local Mesh (Worker + Coordinator + Web UI)!" -ForegroundColor Green
        $Role = "all"
    }
}

# ------------------------------------------------------------------------------
# 1. STATUS
# ------------------------------------------------------------------------------
if ($Role -eq "status") {
    Write-Host ""
    Write-Host "[+] Scanning cluster nodes..." -ForegroundColor Cyan
    $exe = Get-AeroMeshExe
    if ($exe) {
        & $exe status
    } else {
        Write-Host "  [!] aeromesh.exe not built yet. Run .\setup.ps1 first." -ForegroundColor Yellow
    }
    exit 0
}

# Compute roles need a model.
if (-not $detectedModel) {
    Write-Host ""
    Write-Host "[-] ERROR: no .gguf model found." -ForegroundColor Red
    Write-Host "    Place one in models\ (both cluster nodes need the same file), or pass -Model <path>." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

$exe = Get-AeroMeshExe
$layerArgs = @()
if ($Layers) { $layerArgs = @("--layers", $Layers) }

# ------------------------------------------------------------------------------
# 2. WORKER NODE (Laptop B)
# ------------------------------------------------------------------------------
if ($Role -eq "worker") {
    $workerPort = if ($Port -gt 0) { $Port } else { $script:DefaultWorkerPort }

    Stop-PortConflict @($workerPort)

    Write-Host ""
    Write-Host "[+] Starting AeroMesh zero-weight worker node..." -ForegroundColor Cyan
    Write-Host "  Model:   $modelName" -ForegroundColor Gray
    Write-Host "  Port:    $workerPort" -ForegroundColor Gray
    Write-Host "  Layers:  $(if ($Layers) { $Layers } else { 'upper half (auto)' })" -ForegroundColor Gray
    Write-Host "  Mode:    $($Mode.ToUpper())" -ForegroundColor Cyan
    if ($Mode -eq "rpc") {
        Write-Host "  RPC cache: primed from local SSD (see 'aeromesh prime-cache')" -ForegroundColor Yellow
    }
    Write-Host ""

    if ($Mode -eq "rpc") {
        $rpcServerExe = Join-Path $PSScriptRoot "bin\ggml-rpc-server.exe"
        if (Test-Path $rpcServerExe) {
            & $rpcServerExe -H 0.0.0.0 -p $workerPort -c
        } elseif ($exe) {
            & $exe worker --port $workerPort
        } else {
            Write-Host "  [-] Neither bin\ggml-rpc-server.exe nor aeromesh.exe is available." -ForegroundColor Red
            exit 1
        }
    } else {
        $workerArgs = @("worker", "--port", "$workerPort", "--model", "$detectedModel") + $layerArgs
        if ($exe) {
            & $exe @workerArgs
        } else {
            Write-Host "  [!] aeromesh.exe not built; falling back to 'cargo run' (slow)." -ForegroundColor Yellow
            Invoke-Native -Command "cargo" -Arguments (@("run", "--release", "--bin", "aeromesh", "--") + $workerArgs) | Out-Null
        }
    }
    exit 0
}

# ------------------------------------------------------------------------------
# 3. FULL LOCAL MESH (worker + coordinator + UI, one machine)
# ------------------------------------------------------------------------------
if ($Role -eq "all") {
    $apiPort = if ($Port -gt 0) { $Port } else { $script:DefaultApiPort }
    $uiPort = $script:DefaultUiPort
    $pythonExe = Get-PythonCommand
    $coordUrl = "http://127.0.0.1:$apiPort"

    Stop-PortConflict @($apiPort, $uiPort, $script:DefaultWorkerPort)

    Write-Host ""
    Write-Host "[+] Launching full local mesh (worker + coordinator + web UI)..." -ForegroundColor Cyan
    Write-Host "  Model:       $modelName" -ForegroundColor Gray
    Write-Host "  Coordinator: $coordUrl" -ForegroundColor Green
    Write-Host "  Worker:      127.0.0.1:$($script:DefaultWorkerPort) (in-process, shared memory)" -ForegroundColor Yellow
    Write-Host ""

    if (-not $exe) {
        Write-Host "  [-] aeromesh.exe not found. Run .\setup.ps1 first." -ForegroundColor Red
        exit 1
    }

    # `start --role all` spawns the stage-2 worker in-process and connects over
    # the shared-memory ring buffer. The previous implementation launched stock
    # llama.cpp (ggml-rpc-server + llama-server) instead, which does not expose
    # /api/cluster/status, /api/model/switch or /api/chat/abort at all.
    $allArgs = @("start", "--role", "all", "--model", "$detectedModel", "--port", "$apiPort") + $layerArgs
    $coordProc = Start-Process -FilePath $exe -ArgumentList $allArgs -PassThru -NoNewWindow

    try {
        $isReady = Wait-ServerReady $coordUrl 90 "Coordinator"
        if ($isReady -and -not $NoBrowser -and -not $NoUI) { Start-Process "http://127.0.0.1:$uiPort" }
        Start-WebGateway -CoordUrl $coordUrl -UiPort $uiPort -PythonExe $pythonExe
    } finally {
        Write-Host ""
        Write-Host "Shutting down local mesh..." -ForegroundColor Yellow
        if ($coordProc -and -not $coordProc.HasExited) {
            Stop-Process -Id $coordProc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    exit 0
}

# ------------------------------------------------------------------------------
# 4. COORDINATOR NODE (Laptop A)
# ------------------------------------------------------------------------------
if ($Role -eq "coordinator") {
    $apiPort = if ($Port -gt 0) { $Port } else { $script:DefaultApiPort }
    $uiPort = $script:DefaultUiPort
    $pythonExe = Get-PythonCommand
    $coordUrl = "http://127.0.0.1:$apiPort"

    if (-not $Peers) {
        $autoPeers = Find-AeroMeshWorkers -TargetPort $script:DefaultWorkerPort
        if ($autoPeers.Count -gt 0) {
            $Peers = ($autoPeers -join ",")
            Write-Host "  [+] Automatically connected to discovered worker(s): $Peers" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "  [!] No remote workers auto-detected on Wi-Fi or Tailscale." -ForegroundColor Yellow
            Write-Host "      Enter worker address(es) as host:port, or press Enter for standalone:" -ForegroundColor DarkGray
            $entered = Read-Host "Worker peer(s)"
            if ($entered) { $Peers = $entered.Trim() }
        }
    }

    $peerList = @()
    if ($Peers) {
        foreach ($rp in ($Peers -split ',')) {
            $p = $rp.Trim()
            if (-not $p) { continue }
            if (-not $p.Contains(":")) { $p = "${p}:$($script:DefaultWorkerPort)" }
            $peerList += $p
        }
    }

    $onlinePeers = @()
    $offlinePeers = @()
    foreach ($p in $peerList) {
        if (Test-PeerReachable $p) { $onlinePeers += $p } else { $offlinePeers += $p }
    }

    Stop-PortConflict @($apiPort, $uiPort)

    Write-Host ""
    Write-Host "[+] Launching AeroMesh coordinator..." -ForegroundColor Cyan
    Write-Host "  Model:     $modelName" -ForegroundColor Gray
    Write-Host "  Mode:      $($Mode.ToUpper())" -ForegroundColor Cyan
    Write-Host "  Layers:    $(if ($Layers) { $Layers } else { 'lower half (auto)' })" -ForegroundColor Gray
    Write-Host "  API:       $coordUrl" -ForegroundColor Green
    if ($onlinePeers.Count -gt 0) {
        Write-Host "  Workers:   $($onlinePeers -join ', ')" -ForegroundColor Green
    } else {
        Write-Host "  Workers:   none reachable (running standalone)" -ForegroundColor Yellow
    }
    if ($offlinePeers.Count -gt 0) {
        Write-Host "  [!] Unreachable and excluded: $($offlinePeers -join ', ')" -ForegroundColor Yellow
        Write-Host "      Start 'aeromesh worker' on those hosts first." -ForegroundColor DarkGray
    }
    Write-Host ""

    $peerArg = if ($onlinePeers.Count -gt 0) { $onlinePeers -join "," } else { "" }

    if ($Mode -eq "rpc") {
        $serverExe = Join-Path $PSScriptRoot "bin\llama-server.exe"
        if (-not (Test-Path $serverExe)) {
            Write-Host "  [-] bin\llama-server.exe not found; run .\setup.ps1 or use -Mode pipeline." -ForegroundColor Red
            exit 1
        }
        $serverArgs = @("-m", "$detectedModel", "-ngl", "99", "--port", "$apiPort", "--host", "127.0.0.1")
        if ($env:AEROMESH_API_KEY) { $serverArgs += @("--api-key", "$env:AEROMESH_API_KEY") }
        if ($onlinePeers.Count -gt 0) {
            $tsArg = ((1..($onlinePeers.Count + 1)) | ForEach-Object { "1" }) -join ","
            $serverArgs += @("-ts", $tsArg, "--rpc", $peerArg)
        }
        $coordProc = Start-Process -FilePath $serverExe -ArgumentList $serverArgs -PassThru -NoNewWindow
    } else {
        if (-not $exe) {
            Write-Host "  [-] aeromesh.exe not found. Run .\setup.ps1 first." -ForegroundColor Red
            exit 1
        }
        $coordArgs = @("start", "--role", "coordinator", "--model", "$detectedModel", "--port", "$apiPort") + $layerArgs
        if ($peerArg) { $coordArgs += @("--peers", "$peerArg") }
        $coordProc = Start-Process -FilePath $exe -ArgumentList $coordArgs -PassThru -NoNewWindow
    }

    try {
        $timeout = if ($Mode -eq "rpc") { 120 } else { 60 }
        $isReady = Wait-ServerReady $coordUrl $timeout "Coordinator"
        if ($isReady -and -not $NoBrowser -and -not $NoUI) { Start-Process "http://127.0.0.1:$uiPort" }
        Start-WebGateway -CoordUrl $coordUrl -UiPort $uiPort -PythonExe $pythonExe
    } finally {
        Write-Host ""
        Write-Host "Shutting down coordinator..." -ForegroundColor Yellow
        if ($coordProc -and -not $coordProc.HasExited) {
            Stop-Process -Id $coordProc.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
