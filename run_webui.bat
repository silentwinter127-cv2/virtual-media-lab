@echo off
REM analog-tape-sim launcher (Windows)
REM Starts the WebUI and opens a browser. Pass a clip path as the first argument
REM to load it at startup:  run_webui.bat "D:\clips\tape.mp4"
setlocal
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
if "%~1"=="" (
  python -m atsim.web.server --port 8931 --open
) else (
  python -m atsim.web.server --port 8931 --open --clip "%~1"
)
endlocal
