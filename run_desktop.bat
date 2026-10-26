@echo off
setlocal
cd /d "%~dp0"
title Virtual Media Lab - Native Desktop GUI (Low Memory Mode)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo ============================================================================
echo   Virtual Media Lab - Native Low-Memory Desktop Workstation
echo   - Zero Browser RAM Overhead / Zero HTTP Polling / Zero PNG-JPEG Loopback
echo   - Auto Frame Cache Cap (max_cache_frames=2) + On-Demand Scopes and Audio
echo ============================================================================

python desktop.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] Desktop GUI exited with error code %errorlevel%.
    pause
)
endlocal
