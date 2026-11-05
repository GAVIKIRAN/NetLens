@echo off
rem Kills every running dashboard server, then starts exactly ONE fresh server.
rem Use this instead of start_live.bat whenever you have changed code or are
rem seeing stale errors.  For live capture, right-click -> Run as administrator.
setlocal
cd /d "%~dp0.."

echo.
echo   Stopping any running dashboard servers...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*streamlit run app.py*' } | ForEach-Object { Write-Host ('   killed PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force }"
timeout /t 2 /nobreak >nul

echo   Starting ONE fresh dashboard at http://localhost:8501 ...
echo   (In Live Monitor choose LIVE, then press Start. Close any OLD browser tabs.)
echo.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address localhost --server.port 8501 --server.headless false
pause
