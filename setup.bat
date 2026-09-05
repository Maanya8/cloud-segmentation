@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment. Ensure Python is installed and available as ^"py^".
        exit /b 1
    )
)

echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo Installing project dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo Setup complete.
echo Activate the environment with: .venv\Scripts\activate
endlocal