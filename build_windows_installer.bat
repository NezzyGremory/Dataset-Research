@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
set "APP_NAME=Dataset Research"
set "ISCC="

if not exist "%PYTHON%" (
    echo [ERROR] Project virtual environment not found: %PYTHON%
    echo Create it and install requirements.txt before building.
    exit /b 1
)

if not exist "app\ui\assets\logo.jpg" (
    echo [ERROR] App logo not found: app\ui\assets\logo.jpg
    exit /b 1
)

if not exist "installer\DatasetResearch.iss" (
    echo [ERROR] Installer definition not found: installer\DatasetResearch.iss
    exit /b 1
)

echo [0/4] Cleaning previous installer output...
if exist build\installer rmdir /s /q build\installer
if exist hasil_compile rmdir /s /q hasil_compile
if not exist build mkdir build
if not exist hasil_compile mkdir hasil_compile

echo [1/4] Creating Windows application icon...
"%PYTHON%" scripts\create_app_icon.py
if errorlevel 1 exit /b 1

echo [2/4] Building application with PyInstaller...
"%PYTHON%" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onedir ^
    --windowed ^
    --name "%APP_NAME%" ^
    --distpath "hasil_compile\payload" ^
    --workpath "build\installer" ^
    --specpath "build\installer" ^
    --icon "%CD%\app\ui\assets\dataset_research.ico" ^
    --paths "%CD%" ^
    --add-data "%CD%\app\ui\assets\logo.jpg;app\ui\assets" ^
    --add-data "%CD%\app\ui\assets\dataset_research.ico;app\ui\assets" ^
    --add-data "%CD%\app\ui\app.qss;app\ui" ^
    "app\main.py"
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed.
    exit /b 1
)

echo [3/4] Locating Inno Setup compiler...
where ISCC.exe >nul 2>nul
if not errorlevel 1 set "ISCC=ISCC.exe"
if not defined ISCC if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"

if not defined ISCC (
    echo [ERROR] Inno Setup 6 compiler ISCC.exe was not found.
    echo Install Inno Setup 6 from https://jrsoftware.org/isdl.php
    echo The application files are already available in hasil_compile\payload\%APP_NAME%\
    exit /b 1
)

echo [4/4] Compiling the Windows installer...
"%ISCC%" "installer\DatasetResearch.iss"
if errorlevel 1 (
    echo [ERROR] Inno Setup compilation failed.
    exit /b 1
)

echo.
echo SUCCESS: installer created in hasil_compile\DatasetResearch-v3.0.0-Setup.exe
exit /b 0