# Windows Installer

The installer build creates a per-user Windows setup executable. It creates the install directory under `%LOCALAPPDATA%\Programs\Dataset Research`, always creates Desktop and Start Menu shortcuts, and includes an uninstaller. Administrator permission is not required.

## Prerequisites

- Windows 10 or 11, 64-bit
- Python and the project virtual environment at `.venv`
- Inno Setup 6: <https://jrsoftware.org/isdl.php>

## Build

Open PowerShell in the repository folder and run:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
& .\build_windows_installer.bat
```

All compiled output is kept in the new `hasil_compile` directory. The installer is written to:

```text
hasil_compile\DatasetResearch-v3.0.0-Setup.exe
```

The standalone application folder is also available at `hasil_compile\payload\Dataset Research\`. The build generates `app\ui\assets\dataset_research.ico` from the project logo, bundles the application with PyInstaller, then compiles the setup program with Inno Setup.

## Existing Data

Development runs continue using the repository's `data` directory. An installed build uses `%LOCALAPPDATA%\Dataset Research` for its database, configuration, downloaded datasets, and dataset versions. If an older portable copy has a `data` folder beside its executable, the new build copies missing files into the per-user data folder on first launch and leaves the original files untouched.

To carry data from the development checkout into the installed copy, close the app and copy the contents of the repository's `data` folder to `%LOCALAPPDATA%\Dataset Research` before launching the installed app. Do not distribute `data\config.json` or `.env`; they may contain API credentials.

## Distribution Notes

The installer is not code-signed, so Windows SmartScreen may show an unrecognized publisher warning. Signing requires a code-signing certificate. Build output and generated icons are local artifacts and are not included in source control by default.