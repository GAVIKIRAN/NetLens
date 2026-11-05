@echo off
rem Starts the dashboard ready for live capture. Double-click it, or run it from a terminal.
rem   scripts\start_live.bat           check for the driver, then start
rem   scripts\start_live.bat --check   only report whether capture is ready
rem If capture reports "permission denied", run this file as Administrator.
setlocal
cd /d "%~dp0.."

if exist "%SystemRoot%\System32\Npcap\wpcap.dll" goto driver_ok
if exist "%SystemRoot%\System32\wpcap.dll" goto driver_ok
echo.
echo   Npcap is not installed, so live capture cannot start.
echo     1. Install it from https://npcap.com
echo        and tick "Install Npcap in WinPcap API-compatible Mode".
echo     2. Restart your PC.
echo     3. Run this file again.
echo.
if /i "%~1"=="--check" exit /b 1
start "" "https://npcap.com/#download"
pause
exit /b 1

:driver_ok
if /i "%~1"=="--check" (
    echo   Npcap is installed.
    exit /b 0
)

".venv\Scripts\python.exe" scripts\check_environment.py
echo   Starting the dashboard. In Live Monitor choose LIVE, then press Start.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address localhost --server.headless false
pause
