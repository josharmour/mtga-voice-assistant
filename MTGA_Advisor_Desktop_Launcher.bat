@echo off
title MTGA Voice Assistant Launcher
echo Connecting to repository on network share...

REM Use PUSHD to handle UNC paths correctly if Z: is not mapped
pushd "Z:\mtga-voice-assistant" 2>nul || pushd "\10.0.0.2epos\mtga-voice-assistant"

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not find the repository at Z:\mtga-voice-assistant or \10.0.0.2epos\mtga-voice-assistant
    echo Please ensure your network drive is connected.
    pause
    exit /b 1
)

echo Starting Advisor...
call start_advisor.bat

popd
