@echo off
rem ============================================================
rem  Start Whisper Live Transcriber (Windows).
rem  1. create .venv (if missing)
rem  2. install requirements
rem  3. start the app
rem  4. open the page in the default browser once the server answers
rem ============================================================
setlocal EnableExtensions
cd /d "%~dp0.."

set "HOST=127.0.0.1"
set "PORT=8080"
set "URL=http://%HOST%:%PORT%"

rem --- 1. virtual environment ------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
  echo error: python not found on PATH ^(Python 3.10+ required^)
  goto :fail
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  python -m venv .venv || goto :fail
)
call ".venv\Scripts\activate.bat"

rem --- 2. dependencies ---------------------------------------------------
echo Installing requirements ^(safe to re-run^)...
python -m pip install --quiet --upgrade pip || goto :fail
python -m pip install --quiet -r requirements.txt || goto :fail

rem --- 4. open the browser once the server answers ^(background^) --------
start "open-browser" /min powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok = $false; for ($i = 0; $i -lt 240; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 '%URL%' | Out-Null; $ok = $true; break } catch { Start-Sleep -Milliseconds 500 } }; if ($ok) { Start-Process '%URL%' }"

rem --- 3. start the app ---------------------------------------------------
echo Starting Whisper Live Transcriber at %URL%
python -m uvicorn app.main:app --host %HOST% --port %PORT%
goto :end

:fail
echo.
echo Startup failed.
:end
endlocal
exit /b %errorlevel%
