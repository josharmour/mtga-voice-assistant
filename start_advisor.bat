@echo off
cd /d "%~dp0"
title MTGA Voice Advisor Launcher

echo ===================================================
echo Starting MTGA Voice Advisor...
echo ===================================================

REM Check for Python installation
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in your PATH.
    echo Please install Python 3.10 or higher from python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check if venv exists
if exist "venv\Scripts\python.exe" goto :launch

echo.
echo Virtual environment not found. Setting up for the first time...
echo ---------------------------------------------------
echo 1. Creating virtual environment...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo Error creating virtual environment.
    pause
    exit /b 1
)

echo 2. Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip

echo 3. Installing dependencies (this may take a few minutes)...
venv\Scripts\pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error installing dependencies.
    pause
    exit /b 1
)
echo ---------------------------------------------------
echo Setup complete!
echo.

:launch

REM Run the application
echo Launching...
.\venv\Scripts\python.exe main.py %*

REM Pause only if there was an error (exit code not 0)
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ===================================================
    echo Application exited with error code %ERRORLEVEL%
    echo ===================================================
    pause
)
