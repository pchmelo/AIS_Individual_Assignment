@echo off
REM Quick Launch Script for Dataset Fairness Evaluation System

echo ========================================
echo Dataset Fairness Evaluation System
echo ========================================
echo.
echo Select mode:
echo 1. Terminal Mode (CLI)
echo 2. GUI Mode (Web Interface)
echo.
set /p mode="Enter choice (1 or 2): "

if "%mode%"=="1" (
    echo.
    echo Launching Terminal Mode...
    python src\main.py
) else if "%mode%"=="2" (
    echo.
    echo Launching GUI Mode...
    echo Opening web browser...
    python -m streamlit run src\gui_app.py
) else (
    echo Invalid choice. Please run the script again.
)

pause
