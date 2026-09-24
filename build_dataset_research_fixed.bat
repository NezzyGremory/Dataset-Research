@echo off
setlocal

REM =============================================
REM Dataset Research - Windows Build Script
REM =============================================

cd /d "%~dp0"

echo.
echo ============================================
echo   Dataset Research - PyInstaller Build
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found:
    echo         .venv\Scripts\python.exe
    echo.
    pause
    exit /b 1
)

if not exist "app\main.py" (
    echo [ERROR] Entry point not found: app\main.py
    echo.
    pause
    exit /b 1
)

if not exist "app\ui\main_window.py" (
    echo [ERROR] UI module not found: app\ui\main_window.py
    echo.
    pause
    exit /b 1
)

if not exist "app\ui\assets\logo.jpg" (
    echo [WARNING] Logo not found: app\ui\assets\logo.jpg
    echo The application can still be built, but the logo may be missing.
    echo.
)

echo [1/4] Cleaning previous build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist "Dataset Research.spec" del /q "Dataset Research.spec"

echo [2/4] Building executable...
.venv\Scripts\python.exe -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --name "Dataset Research" ^
    --paths "%CD%" ^
    --add-data "app\ui\assets\logo.jpg;app\ui\assets" ^
    --add-data "app\ui\app.qss;app\ui" ^
    "app\main.py"

if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller build failed.
    echo.
    pause
    exit /b 1
)

echo.
echo [3/4] Build completed successfully.
echo.

echo [4/4] Output:
echo     dist\Dataset Research.exe
if exist "dist\Dataset Research.exe" (
    echo.
    echo SUCCESS: executable created.
) else (
    echo.
    echo [ERROR] Expected executable was not found.
    pause
    exit /b 1
)

echo.
pause
endlocal
