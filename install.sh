#!/bin/bash

echo "Installing WorldMachina dependencies..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip is required but not installed. Please install pip and try again."
    exit 1
fi

echo "Installing Python dependencies..."
pip install pygame pyopengl numpy matplotlib pillow

echo "Installing PyPlatec (tectonic plate simulation library)..."
# Set C++14 flag for PyPlatec compilation
CXXFLAGS="-std=c++14" pip install PyPlatec

echo ""
echo "Installation complete! You can now run WorldMachina with:"
echo "python src/main.py"