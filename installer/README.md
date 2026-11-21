# AeroMESH Windows Setup Wizard (`AeroMesh-Setup-x64.exe`)

AeroMESH packages into a single VS Code–style Windows Setup Wizard built with **Inno Setup 6**. End users do **not** need to install Rust, Python, Git, MinGW, or the 4 GB NVIDIA CUDA Toolkit.

## What the Installer Gives End Users

1. **Per-User or All-Users Mode**: Installs to `%LOCALAPPDATA%\Programs\AeroMesh` by default (no UAC required), or `C:\Program Files\AeroMesh` when installing for all users.
2. **Clean Read-Only vs. Writable Separation**:
   - Binaries, runtime DLLs, and offline UI assets live in `{app}`.
   - Models (`.gguf` / `.safetensors`), ephemeral stage caches, `.env` secrets, and logs live in `%LOCALAPPDATA%\AeroMesh\`.
3. **Automatic Hardware Detection & Selectable Components**:
   - **Core Engine & Desktop Studio** (fixed): `AeroMesh-Desktop.exe`, `aeromesh.exe`, `llama.dll`, `ggml*.dll`, and SHA-384 verified offline web UI assets (`static/vendor/`).
   - **NVIDIA CUDA 12 GPU Runtime** (auto-checked when an NVIDIA CUDA 12 driver is detected): bundles `ggml-cuda.dll` plus the redistributable CUDA 12 runtime DLLs (`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`).
   - **Vulkan GPU Acceleration** (auto-checked on AMD/Intel systems without NVIDIA CUDA): bundles `ggml-vulkan.dll`.
   - **CPU Inference Fallback** (checked by default): bundles `ggml-cpu*.dll`.
   - **Stable Diffusion Image Studio** (optional): bundles `bin\diffusion\sd-cli.exe`.
   - **Tailscale P2P Mesh Helper** (optional): opens the official Windows Tailscale installer if not already installed.
4. **VS Code–Style Additional Tasks**:
   - Create a Desktop shortcut.
   - Add `aeromesh` CLI to user or system `PATH` (cleanly removed on uninstall).
   - Add Windows Defender Firewall inbound rule for AeroMESH Worker (`TCP 50052`, cleanly removed on uninstall).
   - Optionally auto-start the AeroMESH Worker node on Windows sign-in.
5. **Safe Uninstaller**: Removes binaries, shortcuts, `PATH`, firewall rule, and autostart entry, then asks whether to keep or delete `%LOCALAPPDATA%\AeroMesh` so downloaded models are never silently wiped.

---

## Building the `.exe` Installer Locally on Windows

### Prerequisites (Build Machine Only)
- Windows 10/11 x64
- [Rust](https://rustup.rs) (`stable-x86_64-pc-windows-msvc`)
- [Inno Setup 6](https://jrsoftware.org/isinfo.php) (`winget install JRSoftware.InnoSetup`)

### One-Command Build (Recommended — Auto-Fetches Pinned `llama.cpp` `b10499` Runtimes)

```powershell
.\installer\build-installer.ps1 -FetchLlamaPrebuilt -IncludeCuda -IncludeVulkan
```

### Build Using Your Locally Compiled `bin\` DLLs

```powershell
# Stages DLLs from bin\ (and optional %CUDA_PATH%\bin for CUDA 12 redistributables)
.\installer\build-installer.ps1 -IncludeCuda -IncludeVulkan
```

Outputs:
- `dist\AeroMesh-Setup-x64.exe`
- `dist\SHA256SUMS.txt`

### Optional Authenticode Code Signing

To avoid Windows SmartScreen "Unknown Publisher" warnings on end-user PCs, sign the installer using an EV or OV code-signing certificate:

```powershell
.\installer\build-installer.ps1 `
  -FetchLlamaPrebuilt `
  -SignThumbprint "<YOUR_CERT_SHA1_THUMBPRINT>"
```

### Silent / Unattended Enterprise Deployment

Standard Inno Setup command-line switches are supported:

```powershell
# Silent install with CUDA 12 + CPU fallback + PATH integration
.\AeroMesh-Setup-x64.exe /VERYSILENT /NORESTART /COMPONENTS="core,backend\cuda,backend\cpu" /TASKS="desktopicon,addtopath"
```

## Automated CI Releases

Pushing a version tag (for example `git tag v0.3.0 && git push origin v0.3.0`) triggers `.github/workflows/windows-release.yml`, which runs all Rust and Python tests, builds `AeroMesh-Setup-x64.exe`, generates `SHA256SUMS.txt`, and publishes both to the GitHub Release.
