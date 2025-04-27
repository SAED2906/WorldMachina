#!/bin/bash

# WorldMachina launcher script

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    # If venv directory exists, activate it
    if [ -d "venv" ]; then
        echo "Activating virtual environment..."
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
        elif [ -f "venv/Scripts/activate" ]; then
            source venv/Scripts/activate
        else
            echo "Warning: Virtual environment found but activation script not detected."
        fi
    else
        echo "No virtual environment detected. It's recommended to run WorldMachina in a virtual environment."
        echo "Would you like to create one now? (y/n)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            echo "Creating virtual environment..."
            python3 -m venv venv
            
            if [ -f "venv/bin/activate" ]; then
                source venv/bin/activate
            elif [ -f "venv/Scripts/activate" ]; then
                source venv/Scripts/activate
            fi
            
            echo "Installing dependencies..."
            pip install pygame pyopengl numpy matplotlib pillow
            CXXFLAGS="-std=c++14" pip install PyPlatec
        fi
    fi
fi

# Check if src/main.py exists
if [ ! -f "src/main.py" ]; then
    echo "Error: src/main.py not found. Please make sure you're running this script from the WorldMachina root directory."
    exit 1
fi

# Run the application
echo "Starting WorldMachina..."
python3 src/main.py

# Deactivate virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi