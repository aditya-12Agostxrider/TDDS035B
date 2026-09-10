@echo off
title Hotel & Restaurant Review Analyzer API - FastAPI Server
color 0A

echo =====================================================================
echo    HOTEL & RESTAURANT REVIEW ANALYSIS & GRAPH SUMMARIZATION API
echo =====================================================================
echo.

cd /d "%~dp0"

:: Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo [*] Virtual environment not found. Creating venv...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Python not found in system PATH. Please install Python 3.11+.
        pause
        exit /b %errorlevel%
    )
    echo [*] Installing required packages from requirements.txt...
    .\venv\Scripts\pip.exe install -r requirements.txt
    echo [*] Downloading NLTK VADER lexicon...
    .\venv\Scripts\python.exe -c "import nltk; nltk.download('vader_lexicon')"
)

echo [*] Starting FastAPI application server on http://127.0.0.1:8000 ...
echo [*] Interactive Web App: http://127.0.0.1:8000/
echo [*] Swagger API Docs:   http://127.0.0.1:8000/docs
echo.
echo =====================================================================
echo    Opening Dashboard in your web browser... Press CTRL+C to stop.
echo =====================================================================
echo.

:: Automatically open browser after 2 seconds
start "" http://127.0.0.1:8000/

:: Start Uvicorn Server
.\venv\Scripts\uvicorn.exe main:app --host 127.0.0.1 --port 8000 --reload

pause
