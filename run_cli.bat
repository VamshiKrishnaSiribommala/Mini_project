@echo off
title LungXpert CLI Manual Analysis Tool
cd /d "%~dp0"

echo ======================================================================
echo              LUNGXPERT: CLI CHEST X-RAY ANALYSIS TOOL
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
)

:: 3. Run CLI Analysis
python run_cli_analysis.py %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] CLI execution encountered an error (code: %errorlevel%).
    echo Ensure required dependencies are installed: pip install -r requirements.txt
)

echo.
pause
