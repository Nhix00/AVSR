#!/bin/bash
# Dependency installation script for the Multimodal Interaction project
# Run with: bash install_dependencies.sh

echo "=================================================="
echo "🔧 Installation Script - Multimodal Interaction"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python 3 found: $(python3 --version)${NC}"

# Step 1: Install system dependencies for PyAudio
echo ""
echo "Step 1/3: Installing system dependencies for PyAudio..."
echo "--------------------------------------------------"

if command -v apt-get &> /dev/null; then
    echo "Detected Debian/Ubuntu system"
    echo "This step requires sudo privileges for: apt-get install portaudio19-dev"
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev
elif command -v brew &> /dev/null; then
    echo "Detected macOS with Homebrew"
    brew install portaudio
elif command -v yum &> /dev/null; then
    echo "Detected RHEL/CentOS system"
    sudo yum install -y portaudio-devel
else
    echo -e "${YELLOW}⚠️  Could not detect package manager. Please install PortAudio manually.${NC}"
fi

# Step 2: Install PyAudio
echo ""
echo "Step 2/3: Installing PyAudio..."
echo "--------------------------------------------------"
pip install pyaudio

# Step 3: Install other Python packages
echo ""
echo "Step 3/3: Installing Python packages..."
echo "--------------------------------------------------"

# Install packages one at a time with a longer timeout
echo "Installing librosa..."
pip install --default-timeout=100 librosa

echo "Installing TensorFlow..."
pip install --default-timeout=200 tensorflow

echo ""
echo "Installing remaining packages from requirements.txt..."
pip install --default-timeout=100 -r requirements.txt

# Verify installation
echo ""
echo "=================================================="
echo "🧪 Verifying installation..."
echo "=================================================="

python3 test_environment.py

echo ""
echo "=================================================="
echo "✅ Installation complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Check the test results above"
echo "  2. If all tests pass, run: python3 data_collection.py"
echo "  3. Read README_DataCollection.md for usage instructions"
echo ""
