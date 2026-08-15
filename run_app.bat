@echo off
title LungXpert Hospital PACS Web Server
cd /d "%~dp0"

echo ======================================================================
echo              LUNGXPERT: MULTIMODAL PACS WEB SERVER
echo ======================================================================
echo.

:: 1. Verify Python availability
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please install Python 3.10 or 3.11 and check "Add Python to PATH".
    echo Download: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: 2. Check and activate virtual environment (.venv or venv)
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] No virtual environment detected. Running with system Python.
    echo [HINT] To create an isolated environment, run: python -m venv .venv
)

:: 3. Launch Flask application
echo.
echo ======================================================================
echo [INFO] Starting LungXpert Web Application...
echo [INFO] Access the Clinical PACS Dashboard in your web browser at:
echo.
echo         http://127.0.0.1:5000
echo.
echo Press Ctrl+C in this terminal window to stop the server.
echo ======================================================================
echo.

python src/web/app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    echo Please verify that dependencies are installed: pip install -r requirements.txt
)

echo.
pause
