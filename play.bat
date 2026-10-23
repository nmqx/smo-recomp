@echo off
rem Opens the launcher (picture, window and interpolation options, then Play). Run setup.py first (README: Build).
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo .venv not found: build first, see README.md
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" scripts\launcher.py
