; Inno Setup script for the Assist native Windows installer (Phase 2).
; Version is passed in by build-installer.ps1 via /DMyAppVersion=...; the
; fallback keeps a manual `ISCC installer\Assist.iss` working.
#ifndef MyAppVersion
  #define MyAppVersion "1.0.1"
#endif
#define MyAppName "Assist"
#define MyAppExe "Assist.exe"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\Assist
DefaultGroupName=Assist
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExe}
OutputBaseFilename=Assist-Setup
OutputDir=Output
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

; Source is resolved relative to THIS .iss file's directory (installer\), so
; reach the repo-root build output with `..\`.
[Files]
Source: "..\dist\Assist\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Assist"; Filename: "{app}\{#MyAppExe}"
Name: "{autodesktop}\Assist"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "Launch Assist"; Flags: nowait postinstall skipifsilent
