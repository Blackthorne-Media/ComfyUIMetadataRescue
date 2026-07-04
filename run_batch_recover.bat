@echo off
setlocal
if "%~1"=="" (
  echo Drag a folder onto this file to recover metadata from every supported image inside it.
  echo.
  pause
  exit /b 1
)

where py >nul 2>nul
if %errorlevel%==0 (
  py "%~dp0krea_metadata_recover.py" "%~1" --batch
) else (
  python "%~dp0krea_metadata_recover.py" "%~1" --batch
)

echo.
pause
