@echo off
echo ========================================================
echo   Dental Caries Detection using UNet++ - Setup
echo ========================================================
echo.

python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create Python virtual environment.
    exit /b %errorlevel%
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing project requirements...
pip install -r requirements.txt

echo.
echo ========================================================
echo Setup complete! To launch the web diagnostic dashboard:
echo   call venv\Scripts\activate.bat
echo   python web_app\app.py
echo ========================================================
pause
