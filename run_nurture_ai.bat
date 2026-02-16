@REM --- ULTRA COMPATIBLE START SCRIPT ---
@echo off
cd /d "%~dp0"

echo [1/3] Activating Environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo Error: .venv folder not found.
    pause
    exit /b
)

echo [2/3] Checking dependencies...
pip install -r requirements.txt --quiet

echo [3/3] Starting App...
echo ---------------------------------------
python app.py

echo.
echo App stopped.
pause
