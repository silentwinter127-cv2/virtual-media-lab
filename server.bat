@echo off
REM Convenience wrapper: double-click to start, or drag a command onto it.
REM   server.bat            -> start
REM   server.bat stop       -> stop
REM   server.bat status     -> status
REM   server.bat restart    -> restart
cd /d "%~dp0"
if "%~1"=="" (
  python server.py start --open
) else (
  python server.py %*
)
if errorlevel 1 (
  echo.
  echo (exit code %errorlevel%)
)
if "%~1"=="" pause
