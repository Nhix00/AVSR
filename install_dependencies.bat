@echo off
REM Batch Installation Script
REM Multimodal Interaction Project - Windows Setup
REM Requires: Python 3.8+

echo ===========================================================
echo 🔧 Installation Script - Multimodal Interaction (Windows)
echo ===========================================================
echo.

REM Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo Make sure to check 'Add Python to PATH' during installation
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ Python found: %PYTHON_VERSION%
echo.

echo ===========================================================
echo Step 1/3: Installing PyAudio (Audio Capture)
echo ===========================================================
echo.

echo Attempting to install PyAudio...
echo (This may fail if you don't have Microsoft C++ Build Tools)
echo.

python -m pip install --upgrade pip
python -m pip install pyaudio >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Standard PyAudio installation failed
    echo.
    echo Trying alternative method: pipwin...
    python -m pip install pipwin >nul 2>&1
    pipwin install pyaudio >nul 2>&1
    if errorlevel 1 (
        echo ❌ Automatic PyAudio installation failed
        echo.
        echo MANUAL INSTALLATION REQUIRED:
        echo ------------------------------------------------
        echo Option 1: Download precompiled wheel
        echo   1. Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
        echo   2. Download the .whl file matching your Python version
        echo      Example: PyAudio-0.2.13-cp310-cp310-win_amd64.whl for Python 3.10
        echo   3. Run: python -m pip install [downloaded-file.whl]
        echo.
        echo Option 2: Install Microsoft C++ Build Tools (6GB+)
        echo   https://visualstudio.microsoft.com/visual-cpp-build-tools/
        echo.
        echo Option 3: Use Anaconda instead of pip
        echo   conda install pyaudio
        echo ------------------------------------------------
        echo.
    ) else (
        echo ✅ PyAudio installed via pipwin
    )
) else (
    echo ✅ PyAudio installed successfully
)

echo.
echo ===========================================================
echo Step 2/3: Installing Core Dependencies
echo ===========================================================
echo.

echo Installing Librosa (audio processing)...
python -m pip install --default-timeout=100 librosa
if errorlevel 1 (
    echo ❌ Librosa installation failed
) else (
    echo ✅ Librosa installed
)

echo.
echo Installing TensorFlow (deep learning)...
echo (This is a large package, ~400MB, please wait...)
python -m pip install --default-timeout=200 tensorflow
if errorlevel 1 (
    echo ⚠️  TensorFlow installation failed
    echo You can use PyTorch instead: pip install torch
) else (
    echo ✅ TensorFlow installed
)

echo.
echo ===========================================================
echo Step 3/3: Installing Remaining Packages
echo ===========================================================
echo.

if exist requirements.txt (
    echo Installing packages from requirements.txt...
    python -m pip install --default-timeout=100 -r requirements.txt
    if errorlevel 1 (
        echo ⚠️  Some packages may have failed to install
    ) else (
        echo ✅ All requirements installed
    )
) else (
    echo ⚠️  requirements.txt not found
)

echo.
echo ===========================================================
echo 🧪 Testing Environment
echo ===========================================================
echo.

if exist test_environment.py (
    python test_environment.py
) else (
    echo ⚠️  test_environment.py not found, skipping tests
)

echo.
echo ===========================================================
echo ✅ Installation Complete!
echo ===========================================================
echo.
echo Next Steps:
echo   1. Review the test results above
echo   2. If PyAudio failed, follow manual installation instructions
echo   3. Allow Python to access Camera/Microphone in Windows Settings
echo      (Settings → Privacy → Camera/Microphone → Allow desktop apps)
echo   4. Run the data collection script:
echo      python data_collection.py
echo.
echo 📖 For more information, see README.md
echo.
pause
