@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Could not find .venv\Scripts\python.exe
    echo Make sure this file is sitting inside the pharmacy-desert-app folder
    echo and that the .venv folder hasn't been moved or deleted.
    pause
    exit /b 1
)

echo Starting Pharmacy Desert Planner...
echo Your browser will open automatically in a few seconds.
echo To stop the app, just close this window.
echo.

".venv\Scripts\python.exe" -m streamlit run "app\streamlit_app.py" --browser.gatherUsageStats false

pause
