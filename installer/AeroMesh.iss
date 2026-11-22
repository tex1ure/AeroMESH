; ==============================================================================
; AeroMESH Windows Setup Wizard (Inno Setup 6)
;
; Produces a VS Code-style Windows installer (AeroMesh-Setup-x64.exe) featuring:
;   - Per-user (%LOCALAPPDATA%\Programs\AeroMesh) OR All-Users (Program Files) mode
;   - Automatic NVIDIA CUDA 12 / Vulkan GPU detection on the Components screen
;   - Selectable Compute Backends (NVIDIA CUDA 12 Runtime, Vulkan, CPU Fallback)
;   - Optional Stable Diffusion Studio (sd-cli.exe) & Tailscale P2P Mesh setup
;   - Additional Tasks: Desktop icon, Add CLI to PATH, Windows Firewall TCP 50052,
;     and optional Worker auto-start on Windows sign-in
;   - Clean uninstaller that removes PATH/Firewall/Autostart and asks whether to
;     keep or delete %LOCALAPPDATA%\AeroMesh (models & configuration)
;
; Build via:
;   .\installer\build-installer.ps1
; Or directly with Inno Setup 6:
;   ISCC.exe /DAppVersion=0.3.0 /DStagingDir=..\dist\staging installer\AeroMesh.iss
; ==============================================================================

#ifndef AppVersion
  #define AppVersion "0.3.0"
#endif

#ifndef StagingDir
  #define StagingDir "..\dist\staging"
#endif

#ifndef OutputDirPath
  #define OutputDirPath "..\dist"
#endif

#define AppName "AeroMESH Studio"
#define AppShortName "AeroMesh"
#define AppPublisher "AeroMESH Contributors"
#define AppExeName "AeroMesh-Desktop.exe"
#define AppCliName "aeromesh.exe"
#define AppIdGuid "{{8D49E9C2-4F7A-4A56-9E31-9A2B8F41C7E2}"

[Setup]
AppId={#AppIdGuid}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppShortName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir={#OutputDirPath}
OutputBaseFilename=AeroMesh-Setup-x64
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.18362
WizardStyle=modern
WizardResizable=yes
; VS Code-style per-user by default, with interactive dialog option for All-Users
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ChangesEnvironment=yes
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
CloseApplications=yes
RestartApplications=no
; Ensure existing AeroMESH processes are gracefully detected before upgrading
AppMutex=AeroMeshDesktopStudioMutex

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "recommended"; Description: "Recommended (Auto-detected GPU Acceleration + Desktop Studio)"
Name: "full";        Description: "Full Installation (CUDA 12 + Vulkan + CPU + Image Studio)"
Name: "minimal";     Description: "Minimal Installation (Core Engine + CPU Fallback Only)"
Name: "custom";      Description: "Custom Installation (Choose Components)"; Flags: iscustom

[Components]
Name: "core";           Description: "AeroMESH Core Engine & Desktop Studio (CLI + Offline Web UI)"; Types: recommended full minimal custom; Flags: fixed
Name: "backend";        Description: "Inference Compute Backends"; Types: recommended full minimal custom
Name: "backend\cuda";   Description: "NVIDIA CUDA 12 GPU Runtime (Bundled cuBLAS/cuDARt DLLs - no 4 GB CUDA Toolkit needed)"; Types: full
Name: "backend\vulkan"; Description: "Vulkan GPU Acceleration (AMD Radeon / Intel Arc / Generic Vulkan GPU)"; Types: full
Name: "backend\cpu";    Description: "CPU Inference Backend (Multi-core AVX2 Fallback)"; Types: recommended full minimal custom
Name: "diffusion";      Description: "Stable Diffusion Image Studio (sd-cli.exe)"; Types: full
Name: "tailscale";      Description: "Tailscale P2P Mesh Helper (Opens official Tailscale download if not installed)"; Types: custom

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"
Name: "addtopath";   Description: "Add 'aeromesh' CLI to &PATH (available in new terminals)"; GroupDescription: "System integration:"
Name: "firewall";    Description: "Allow AeroMESH Worker (TCP port 50052) through Windows Defender &Firewall"; GroupDescription: "Networking & Cluster:"; Check: IsAdminInstallMode
Name: "autostart";   Description: "Start AeroMESH Worker node on Windows sign-in (background)"; GroupDescription: "Networking & Cluster:"; Flags: unchecked

[Files]
; Core binaries (required)
Source: "{#StagingDir}\AeroMesh-Desktop.exe"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#StagingDir}\aeromesh.exe";         DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "..\LICENSE";                         DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "..\.env.example";                    DestDir: "{app}"; Components: core; Flags: ignoreversion

; Core llama.cpp & ggml runtime DLLs
Source: "{#StagingDir}\bin\llama.dll";        DestDir: "{app}\bin"; Components: core; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\bin\ggml.dll";         DestDir: "{app}\bin"; Components: core; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\bin\ggml-base.dll";    DestDir: "{app}\bin"; Components: core; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\bin\libomp*.dll";      DestDir: "{app}\bin"; Components: core; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\bin\mtmd*.dll";        DestDir: "{app}\bin"; Components: core; Flags: ignoreversion skipifsourcedoesntexist

; Offline Web UI + verified vendor libraries (Lucide, Marked, DOMPurify, Highlight.js, KaTeX + fonts)
Source: "{#StagingDir}\static\*";             DestDir: "{app}\static"; Components: core; Flags: ignoreversion recursesubdirs createallsubdirs

; Compute Backend: CPU Fallback
Source: "{#StagingDir}\backends\cpu\*";       DestDir: "{app}\bin"; Components: backend\cpu; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

; Compute Backend: NVIDIA CUDA 12 Runtime (ggml-cuda.dll + cudart64_12.dll + cublas64_12.dll + cublasLt64_12.dll)
Source: "{#StagingDir}\backends\cuda\*";      DestDir: "{app}\bin"; Components: backend\cuda; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

; Compute Backend: Vulkan GPU Runtime (ggml-vulkan.dll)
Source: "{#StagingDir}\backends\vulkan\*";    DestDir: "{app}\bin"; Components: backend\vulkan; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

; Optional: Stable Diffusion Image Studio (sd-cli.exe + companion DLLs)
Source: "{#StagingDir}\diffusion\*";          DestDir: "{app}\bin\diffusion"; Components: diffusion; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

; Optional: Microsoft Visual C++ x64 Redistributable (bundled if staged)
Source: "{#StagingDir}\redist\vc_redist.x64.exe"; DestDir: "{tmp}"; Components: core; Flags: deleteafterinstall skipifsourcedoesntexist

[Dirs]
; Writable per-user data directories (kept outside Program Files)
Name: "{localappdata}\{#AppShortName}"
Name: "{localappdata}\{#AppShortName}\models"
Name: "{localappdata}\{#AppShortName}\cache"
Name: "{localappdata}\{#AppShortName}\logs"

[Icons]
Name: "{autoprograms}\{#AppName}";                                 Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{autoprograms}\{#AppShortName} Models Folder";              Filename: "{localappdata}\{#AppShortName}\models"
Name: "{autoprograms}\Uninstall {#AppName}";                       Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";                                  Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Registry]
; Optional Worker autostart on sign-in (current user)
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "AeroMeshWorker"; ValueData: """{app}\{#AppCliName}"" worker --port 50052"; Tasks: autostart; Flags: uninsdeletevalue

[Run]
; Silently install Visual C++ Runtime if bundled and needed
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing Microsoft Visual C++ Runtime..."; Check: FileExists(ExpandConstant('{tmp}\vc_redist.x64.exe')); Flags: waituntilterminated

; Configure Windows Defender Firewall rules for TCP 50052 (Worker), TCP 50053 (Cluster Pairing), and TCP 50054 (Peer Model Sync)
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""AeroMESH Worker (TCP 50052)"""; StatusMsg: "Updating Windows Firewall rules..."; Tasks: firewall; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""AeroMESH Worker (TCP 50052)"" dir=in action=allow protocol=TCP localport=50052 program=""{app}\{#AppCliName}"""; StatusMsg: "Allowing AeroMESH Worker through Windows Firewall..."; Tasks: firewall; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""AeroMESH Cluster Pairing (TCP 50053)"""; StatusMsg: "Updating Windows Firewall rules..."; Tasks: firewall; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""AeroMESH Cluster Pairing (TCP 50053)"" dir=in action=allow protocol=TCP localport=50053 program=""{app}\{#AppExeName}"""; StatusMsg: "Allowing AeroMESH Cluster Pairing through Windows Firewall..."; Tasks: firewall; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""AeroMESH Peer Model Sync (TCP 50054)"""; StatusMsg: "Updating Windows Firewall rules..."; Tasks: firewall; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""AeroMESH Peer Model Sync (TCP 50054)"" dir=in action=allow protocol=TCP localport=50054 program=""{app}\{#AppExeName}"""; StatusMsg: "Allowing AeroMESH Peer Model Sync through Windows Firewall..."; Tasks: firewall; Flags: runhidden waituntilterminated

; Optional Tailscale download page if user checked Tailscale and it is not already installed
Filename: "https://tailscale.com/download/windows"; Description: "Download Tailscale for cross-network P2P clusters"; Components: tailscale; Check: not IsTailscaleInstalled; Flags: shellexec postinstall skipifsilent

; Launch Desktop Studio on finish
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""AeroMESH Worker (TCP 50052)"""; Check: IsAdminInstallMode; Flags: runhidden waituntilterminated; RunOnceId: "RemoveAeroMeshFirewallRule"
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""AeroMESH Cluster Pairing (TCP 50053)"""; Check: IsAdminInstallMode; Flags: runhidden waituntilterminated; RunOnceId: "RemoveAeroMeshPairingFirewallRule"
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""AeroMESH Peer Model Sync (TCP 50054)"""; Check: IsAdminInstallMode; Flags: runhidden waituntilterminated; RunOnceId: "RemoveAeroMeshSyncFirewallRule"

[Code]
var
  GpuSummaryLabel: TNewStaticText;
  DetectedNvidiaCuda12: Boolean;
  DetectedVulkan: Boolean;
  DetectedDriverVersion: String;
  ComponentsInitialized: Boolean;

function HasNvidiaCuda12Driver(var DriverVer: String): Boolean;
var
  NvCudaPath: String;
  MS, LS: Cardinal;
  MajorVer: Cardinal;
begin
  Result := False;
  DriverVer := '';
  NvCudaPath := ExpandConstant('{sys}\nvcuda.dll');
  if not FileExists(NvCudaPath) then
    Exit;

  if GetVersionNumbers(NvCudaPath, MS, LS) then
  begin
    // NVIDIA nvcuda.dll version format is typically 31.0.15.xxxx or 32.0.15.xxxx
    // where the last two segments encode the driver branch (e.g. 525.60 -> 15.2560).
    // Any modern RTX/GTX driver shipping nvcuda.dll on Win10/11 >= 525 supports CUDA 12.
    GetVersionNumbersString(NvCudaPath, DriverVer);
    MajorVer := MS shr 16;
    Result := (MajorVer >= 26);
  end
  else
  begin
    DriverVer := 'detected';
    Result := True;
  end;
end;

function HasVulkanRuntime(): Boolean;
begin
  Result := FileExists(ExpandConstant('{sys}\vulkan-1.dll'));
end;

function IsTailscaleInstalled(): Boolean;
begin
  Result :=
    FileExists(ExpandConstant('{pf64}\Tailscale\tailscale.exe')) or
    FileExists(ExpandConstant('{pf}\Tailscale\tailscale.exe')) or
    FileExists(ExpandConstant('{localappdata}\Tailscale\tailscale.exe'));
end;

procedure SetComponentChecked(const ComponentName: String; Checked: Boolean);
var
  Idx: Integer;
begin
  Idx := WizardForm.ComponentsList.Items.IndexOf(ComponentName);
  if Idx >= 0 then
    WizardForm.ComponentsList.Checked[Idx] := Checked;
end;

procedure ApplyAutoDetectedBackends();
var
  I: Integer;
  ItemText: String;
begin
  // Match components by description prefix so we reliably toggle the child items
  for I := 0 to WizardForm.ComponentsList.Items.Count - 1 do
  begin
    ItemText := WizardForm.ComponentsList.Items[I];
    if Pos('NVIDIA CUDA 12', ItemText) > 0 then
      WizardForm.ComponentsList.Checked[I] := DetectedNvidiaCuda12
    else if Pos('Vulkan GPU Acceleration', ItemText) > 0 then
      WizardForm.ComponentsList.Checked[I] := (not DetectedNvidiaCuda12) and DetectedVulkan
    else if Pos('CPU Inference Backend', ItemText) > 0 then
      WizardForm.ComponentsList.Checked[I] := True;
  end;
end;

procedure InitializeWizard();
var
  SummaryText: String;
begin
  ComponentsInitialized := False;
  DetectedNvidiaCuda12 := HasNvidiaCuda12Driver(DetectedDriverVersion);
  DetectedVulkan := HasVulkanRuntime();

  GpuSummaryLabel := TNewStaticText.Create(WizardForm.SelectComponentsPage);
  GpuSummaryLabel.Parent := WizardForm.SelectComponentsPage;
  GpuSummaryLabel.Left := WizardForm.ComponentsList.Left;
  GpuSummaryLabel.Width := WizardForm.ComponentsList.Width;
  GpuSummaryLabel.Top := WizardForm.TypesCombo.Top + WizardForm.TypesCombo.Height + 6;
  GpuSummaryLabel.AutoSize := False;
  GpuSummaryLabel.Height := 32;
  GpuSummaryLabel.WordWrap := True;
  GpuSummaryLabel.Font.Style := [fsBold];

  if DetectedNvidiaCuda12 then
    SummaryText := 'Hardware detected: NVIDIA CUDA GPU driver (' + DetectedDriverVersion + '). NVIDIA CUDA 12 runtime has been pre-selected.'
  else if DetectedVulkan then
    SummaryText := 'Hardware detected: Vulkan-compatible GPU (no NVIDIA CUDA driver found). Vulkan GPU acceleration has been pre-selected.'
  else
    SummaryText := 'Hardware detected: No NVIDIA CUDA or Vulkan driver found. CPU inference fallback has been pre-selected.';

  GpuSummaryLabel.Caption := SummaryText;

  // Shift the component checklist down cleanly so it does not overlap the hardware banner
  WizardForm.ComponentsList.Top := GpuSummaryLabel.Top + GpuSummaryLabel.Height + 6;
  WizardForm.ComponentsList.Height := WizardForm.ComponentsDiskSpaceLabel.Top - WizardForm.ComponentsList.Top - 8;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = wpSelectComponents) and (not ComponentsInitialized) then
  begin
    ComponentsInitialized := True;
    if WizardSetupType(False) = 'recommended' then
      ApplyAutoDetectedBackends();
  end;
end;

// -----------------------------------------------------------------------------
// Safe PATH management (User HKCU or Machine HKLM depending on install mode)
// -----------------------------------------------------------------------------
function EnvRootKey(): Integer;
begin
  if IsAdminInstallMode then
    Result := HKLM
  else
    Result := HKCU;
end;

function EnvSubKey(): String;
begin
  if IsAdminInstallMode then
    Result := 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment'
  else
    Result := 'Environment';
end;

procedure AddAppToPath(const DirToAdd: String);
var
  OrigPath: String;
  CheckPath: String;
  NewPath: String;
begin
  if not RegQueryStringValue(EnvRootKey(), EnvSubKey(), 'Path', OrigPath) then
    OrigPath := '';

  CheckPath := ';' + Uppercase(OrigPath) + ';';
  if Pos(';' + Uppercase(DirToAdd) + ';', CheckPath) > 0 then
    Exit;

  if (Length(OrigPath) > 0) and (OrigPath[Length(OrigPath)] <> ';') then
    NewPath := OrigPath + ';' + DirToAdd
  else
    NewPath := OrigPath + DirToAdd;

  RegWriteExpandStringValue(EnvRootKey(), EnvSubKey(), 'Path', NewPath);
end;

procedure RemoveAppFromPath(const DirToRemove: String);
var
  OrigPath: String;
  Remaining: String;
  Segment: String;
  CleanPath: String;
  SemiPos: Integer;
begin
  if not RegQueryStringValue(EnvRootKey(), EnvSubKey(), 'Path', OrigPath) then
    Exit;

  Remaining := OrigPath;
  CleanPath := '';
  while Length(Remaining) > 0 do
  begin
    SemiPos := Pos(';', Remaining);
    if SemiPos > 0 then
    begin
      Segment := Copy(Remaining, 1, SemiPos - 1);
      Delete(Remaining, 1, SemiPos);
    end
    else
    begin
      Segment := Remaining;
      Remaining := '';
    end;

    if (Trim(Segment) <> '') and (Uppercase(Trim(Segment)) <> Uppercase(DirToRemove)) then
    begin
      if CleanPath <> '' then
        CleanPath := CleanPath + ';' + Segment
      else
        CleanPath := Segment;
    end;
  end;

  if CleanPath <> OrigPath then
    RegWriteExpandStringValue(EnvRootKey(), EnvSubKey(), 'Path', CleanPath);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if WizardIsTaskSelected('addtopath') then
      AddAppToPath(ExpandConstant('{app}'));
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    RemoveAppFromPath(ExpandConstant('{app}'));

    DataDir := ExpandConstant('{localappdata}\{#AppShortName}');
    if DirExists(DataDir) and (not UninstallSilent) then
    begin
      if MsgBox(
        'Do you also want to delete your AeroMESH user data folder?' + #13#10 + #13#10 +
        DataDir + #13#10 + #13#10 +
        'Select NO to keep your downloaded .gguf models and cluster settings.',
        mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      begin
        DelTree(DataDir, True, True, True);
      end;
    end;
  end;
end;
