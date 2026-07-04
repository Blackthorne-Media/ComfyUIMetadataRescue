@echo off
setlocal
if "%~1"=="" (
  echo Drag an image or ExifTool metadata dump onto this file.
  echo.
  pause
  exit /b 1
)

where py >nul 2>nul
if %errorlevel%==0 (
  py "%~dp0krea_metadata_recover.py" "%~1"
) else (
  python "%~dp0krea_metadata_recover.py" "%~1"
)

echo.
pause
