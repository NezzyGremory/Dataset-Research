#define AppName "Dataset Research"
#define AppVersion "3.0.0"
#define AppPublisher "Dataset Research"
#define AppExeName "Dataset Research.exe"

[Setup]
AppId={{6A358E27-5C04-49D8-B3D7-7CC448928AE1}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\Dataset Research
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\hasil_compile
OutputBaseFilename=DatasetResearch-v3.0.0-Setup
SetupIconFile=..\app\ui\assets\dataset_research.ico
UninstallDisplayIcon={app}\{#AppExeName}
ArchitecturesInstallIn64BitMode=x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
Uninstallable=yes
CreateAppDir=yes

[Files]
Source: "..\hasil_compile\payload\Dataset Research\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: postinstall nowait skipifsilent