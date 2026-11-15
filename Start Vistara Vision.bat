@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "apps\launcher\vistara_launcher.py"
) else (
    start "" pythonw "apps\launcher\vistara_launcher.py"
)
