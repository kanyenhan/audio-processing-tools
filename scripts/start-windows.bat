@echo off
rem ============================================================
rem  Start Whisper Live Transcriber (Windows).
rem  1. create .venv (if missing)
rem  2. install requirements
rem  3. start the app
rem  4. open the page in a new browser window once the server answers
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
rem Open in a NEW window instead of a tab: if the default browser (http
rem protocol association) is Chrome/Edge/Brave/Firefox, launch it directly
rem with its new-window flag; otherwise fall back to the usual tab behavior.
rem Note: the whole -Command string may not contain double quotes (cmd.exe
rem would break it), so the quote character is built via [char]34.
start "open-browser" /min powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok = $false; for ($i = 0; $i -lt 240; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 '%URL%' | Out-Null; $ok = $true; break } catch { Start-Sleep -Milliseconds 500 } }; if ($ok) { $assoc = $null; try { $assoc = (Get-ItemProperty 'Registry::HKCR\http\shell\open\command' -ErrorAction Stop).'(default)' } catch {}; $q = [char]34; $exe = $null; $flag = $null; if ($assoc -and $assoc -match '(?i)(chrome|msedge|brave|firefox)\.exe') { $i1 = $assoc.IndexOf($q); if ($i1 -ge 0) { $i2 = $assoc.IndexOf($q, $i1 + 1); if ($i2 -gt $i1) { $exe = $assoc.Substring($i1 + 1, $i2 - $i1 - 1) } } else { $exe = $assoc.Split(' ')[0] }; $flag = '-new-window'; if ($assoc -match '(?i)(chrome|msedge|brave)\.exe') { $flag = '--new-window' } }; if ($exe) { Start-Process -FilePath $exe -ArgumentList $flag, '%URL%' } else { Start-Process '%URL%' } }"

rem --- 3. start the app ---------------------------------------------------
echo Starting Whisper Live Transcriber at %URL%
python -m uvicorn app.main:app --host %HOST% --port %PORT%

rem A double-clicked console window closes on its own once the bat reaches
rem its end (there is no pause) — print a farewell so the moment is visible.
echo.
echo App stopped. Closing this window...
timeout /t 3 /nobreak >nul
goto :end

:fail
echo.
echo Startup failed.
:end
endlocal
exit /b %errorlevel%
