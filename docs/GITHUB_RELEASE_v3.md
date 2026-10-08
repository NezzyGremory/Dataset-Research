# GitHub Release: v3.0.0

## Release title

Dataset Research v3.0.0

## Tag

`v3.0.0`

Create this tag from the commit that contains the finalized v3 code and installer configuration.

## Release description

Dataset Research v3.0.0 brings a more adaptable desktop workspace and a proper Windows installer for the dataset analysis and academic research workflow.

### Highlights

- Responsive PySide6 interface that adapts the sidebar and dashboard layout to different laptop screen sizes.
- Dataset analysis workspace with interactive charts for exploring distributions, categories, correlations, and missing values.
- Windows setup installer that creates the application directory, Start Menu shortcut, and Desktop shortcut automatically.
- Per-user storage for the installed app's database, downloaded datasets, API configuration, and dataset versions.
- Existing data is copied during migration when found beside a portable executable; original files are not deleted.

### Install on Windows

1. Download `DatasetResearch-v3.0.0-Setup.exe` from the Assets section below.
2. Run the installer. It installs for the current Windows user and does not require administrator access.
3. Launch Dataset Research from the Desktop or Start Menu shortcut.

The app stores runtime data under `%LOCALAPPDATA%\Dataset Research`. The installer is currently unsigned, so Windows may display a SmartScreen warning.

### Requirements

- Windows 10 or 11, 64-bit
- Internet access for online dataset search and academic literature discovery
- API credentials are optional; do not upload `.env` or `data/config.json` as release assets

### Checks

- Automated test suite: 72 passed
- PyInstaller Windows payload built successfully
- Inno Setup installer compiled successfully

## Release asset

Attach only:

- `hasil_compile\DatasetResearch-v3.0.0-Setup.exe`

Do not upload `.env`, `data/config.json`, the repository's dataset folders, `.venv`, or the raw `payload` directory. The installer is the distribution artifact.