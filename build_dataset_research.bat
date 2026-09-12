@echo off
setlocal

REM Build Dataset Research as a single EXE and bundle the application logo.
REM Run this BAT from the project root.

pyinstaller --onefile --noconsole --name DatasetResearch ^
  --add-data "app/ui/assets/logo.jpg;app/ui/assets" ^
  app/main.py

endlocal
