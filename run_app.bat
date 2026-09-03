@echo off
title FedFairChain

cd /d "%~dp0"

echo ==========================================
echo              FEDFAIRCHAIN
echo ==========================================
echo.
echo Starting FedFairChain...
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo Run setup first and install requirements.
    pause
    exit /b 1
)

if not exist "models\fair_federated_classifier_chain.joblib" (
    echo ERROR: Final FedFairChain model not found.
    pause
    exit /b 1
)

start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo FedFairChain has stopped.
pause