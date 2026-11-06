; AeroMESH Distributed LLM Cluster Studio - Production Inno Setup Script
; Generates a signed, single-file Windows installer for standard or administrator accounts.

#define MyAppName "AeroMESH Desktop Studio"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "AeroMesh Engineering Team"
#define MyAppURL "https://github.com/aeromesh/aeromesh"
#define MyAppExeName "AeroMesh-Desktop.exe"

[Setup]
AppId={{9F82A4D1-5C2B-48E3-A1D7-77DF09C5E401}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\AeroMESH
DisableProgramGroupPage=yes
OutputBaseFilename=AeroMESH-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupIconFile=..\static\img\aeromesh_favicon.png

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Primary Desktop Executable
Source: "..\target\release\AeroMesh-Desktop.exe"; DestDir: "{app}"; Flags: ignoreversion
; Core Distributed Engine Binary
Source: "..\target\release\aeromesh.exe"; DestDir: "{app}"; Flags: ignoreversion
; Runtime DLLs & CUDA Dependencies
Source: "..\bin\*"; DestDir: "{app}\bin"; Flags: ignoreversion recursesubdirs createallsubdirs
; Static Web Assets & Offline Vendor Bundles
Source: "..\static\*"; DestDir: "{app}\static"; Flags: ignoreversion recursesubdirs createallsubdirs
; Documentation
Source: "..\README.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\LICENSE*"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean ephemeral files inside application folder
Type: filesandordirs; Name: "{app}\target"
Type: filesandordirs; Name: "{app}\models\.aeromesh_*"

[Code]
// Optional cleanup of %LOCALAPPDATA%\AeroMESH during uninstallation
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  LocalAppDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    LocalAppDir := ExpandConstant('{localappdata}\AeroMESH');
    if DirExists(LocalAppDir) then
    begin
      if MsgBox('Do you want to delete downloaded models and cached data in ' + LocalAppDir + '?', mbConfirmation, MB_YESNO) = IDYES then
      begin
        DelTree(LocalAppDir, True, True, True);
      end;
    end;
  end;
end;
