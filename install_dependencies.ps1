# PowerShell Installation Script
# Multimodal Interaction Project - Windows Setup
# Requires: PowerShell 5.1+ and Python 3.8+

# Enable strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "🔧 Installation Script - Multimodal Interaction (Windows)" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""

# Function to test command existence
function Test-CommandExists {
    param($Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
if (-not (Test-CommandExists python)) {
    Write-Host "❌ Python not found in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    exit 1
}

$pythonVersion = python --version 2>&1
Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green

# Verify Python version (3.8+)
$versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
if ($versionMatch) {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
        Write-Host "❌ Python version too old. Requires 3.8+" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "Step 1/3: Installing PyAudio (Audio Capture)" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""

# PyAudio is tricky on Windows - try multiple approaches
Write-Host "Attempting to install PyAudio..." -ForegroundColor Yellow
Write-Host "(This may fail if you don't have Microsoft C++ Build Tools)" -ForegroundColor Gray

$pyaudioInstalled = $false

try {
    python -m pip install --upgrade pip | Out-Null
    python -m pip install pyaudio 2>&1 | Out-Null
    $pyaudioInstalled = $true
    Write-Host "✅ PyAudio installed successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Standard PyAudio installation failed" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Trying alternative method: pipwin..." -ForegroundColor Yellow
    
    try {
        python -m pip install pipwin 2>&1 | Out-Null
        pipwin install pyaudio 2>&1 | Out-Null
        $pyaudioInstalled = $true
        Write-Host "✅ PyAudio installed via pipwin" -ForegroundColor Green
    } catch {
        Write-Host "❌ Automatic PyAudio installation failed" -ForegroundColor Red
        Write-Host ""
        Write-Host "MANUAL INSTALLATION REQUIRED:" -ForegroundColor Yellow
        Write-Host "------------------------------------------------" -ForegroundColor Yellow
        Write-Host "Option 1: Download precompiled wheel" -ForegroundColor White
        Write-Host "  1. Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio" -ForegroundColor Gray
        Write-Host "  2. Download the .whl file matching your Python version" -ForegroundColor Gray
        Write-Host "     Example: PyAudio-0.2.13-cp310-cp310-win_amd64.whl for Python 3.10" -ForegroundColor Gray
        Write-Host "  3. Run: python -m pip install [downloaded-file.whl]" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Option 2: Install Microsoft C++ Build Tools (6GB+)" -ForegroundColor White
        Write-Host "  https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Option 3: Use Anaconda instead of pip" -ForegroundColor White
        Write-Host "  conda install pyaudio" -ForegroundColor Gray
        Write-Host "------------------------------------------------" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "Step 2/3: Installing Core Dependencies" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""

# Install large packages individually with progress
Write-Host "Installing Librosa (audio processing)..." -ForegroundColor Yellow
try {
    python -m pip install --default-timeout=100 librosa
    Write-Host "✅ Librosa installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Librosa installation failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Installing TensorFlow (deep learning)..." -ForegroundColor Yellow
Write-Host "(This is a large package, ~400MB, please wait...)" -ForegroundColor Gray
try {
    python -m pip install --default-timeout=200 tensorflow
    Write-Host "✅ TensorFlow installed" -ForegroundColor Green
} catch {
    Write-Host "⚠️  TensorFlow installation failed" -ForegroundColor Yellow
    Write-Host "You can use PyTorch instead: pip install torch" -ForegroundColor Gray
}

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "Step 3/3: Installing Remaining Packages" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""

# Install from requirements.txt
if (Test-Path "requirements.txt") {
    Write-Host "Installing packages from requirements.txt..." -ForegroundColor Yellow
    try {
        python -m pip install --default-timeout=100 -r requirements.txt
        Write-Host "✅ All requirements installed" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Some packages may have failed to install" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  requirements.txt not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "🧪 Testing Environment" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path "test_environment.py") {
    python test_environment.py
} else {
    Write-Host "⚠️  test_environment.py not found, skipping tests" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "✅ Installation Complete!" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review the test results above" -ForegroundColor White
Write-Host "  2. If PyAudio failed, follow manual installation instructions" -ForegroundColor White
Write-Host "  3. Allow Python to access Camera/Microphone in Windows Settings" -ForegroundColor White
Write-Host "     (Settings → Privacy → Camera/Microphone → Allow desktop apps)" -ForegroundColor Gray
Write-Host "  4. Run the data collection script:" -ForegroundColor White
Write-Host "     python data_collection.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "📖 For more information, see README.md" -ForegroundColor Gray
Write-Host ""
