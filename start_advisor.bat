@echo off
cd /d "%~dp0"
title MTGA Voice Advisor Launcher

echo ===================================================
echo Starting MTGA Voice Advisor...
echo ===================================================

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo Error: Virtual environment not found at .\venv
    echo Please ensure you have set up the project correctly.
    pause
    exit /b 1
)

REM Run the application
.\venv\Scripts\python.exe main.py %*

REM Pause only if there was an error (exit code not 0)
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ===================================================
    echo Application exited with error code %ERRORLEVEL%
    echo ===================================================
    pause
)
