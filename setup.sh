#!/bin/bash

# APGI Framework Automated Setup Script
# This script automates installation and setup process for APGI Framework

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        PYTHON_CMD="python3"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PYTHON_CMD="python3"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        PYTHON_CMD="python"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    print_status "Detected OS: $OS"
}

# Function to check Python installation
check_python() {
    print_header "Checking Python Installation"
    
    if ! command_exists $PYTHON_CMD; then
        print_error "Python is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d" " -f2)
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    print_status "Found Python version: $PYTHON_VERSION"

    # Check if Python version is 3.8 or higher
    if [[ "$PYTHON_MAJOR" -lt 3 ]] || ([[ "$PYTHON_MAJOR" -eq 3 ]] && [[ "$PYTHON_MINOR" -lt 8 ]]); then
        print_error "Python 3.8 or higher is required. Found version: $PYTHON_VERSION"
        exit 1
    fi
    
    print_status "Python version check passed"
}

# Function to create virtual environment
create_venv() {
    print_header "Creating Virtual Environment"
    
    VENV_DIR="apgi_venv"
    
    if [[ -d "$VENV_DIR" ]]; then
        print_warning "Virtual environment '$VENV_DIR' already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Using existing virtual environment"
            return
        fi
        rm -rf "$VENV_DIR"
    fi
    
    print_status "Creating virtual environment: $VENV_DIR"
    $PYTHON_CMD -m venv "$VENV_DIR"
    
    if [[ ! -d "$VENV_DIR" ]]; then
        print_error "Failed to create virtual environment"
        exit 1
    fi
    
    print_status "Virtual environment created successfully"
}

# Function to activate virtual environment
activate_venv() {
    print_header "Activating Virtual Environment"
    
    VENV_DIR="apgi_venv"
    
    if [[ ! -d "$VENV_DIR" ]]; then
        print_error "Virtual environment not found. Please run setup again."
        exit 1
    fi
    
    # Determine activation script based on OS
    if [[ "$OS" == "windows" ]]; then
        ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
    else
        ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
    fi
    
    if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
        print_error "Virtual environment activation script not found"
        exit 1
    fi
    
    print_status "Activating virtual environment"
    source "$ACTIVATE_SCRIPT"
    
    # Verify activation
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_status "Virtual environment activated successfully"
    else
        print_error "Failed to activate virtual environment"
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    print_header "Installing Dependencies"
    
    # Upgrade pip first
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        print_status "Installing requirements from requirements.txt..."
        pip install -r requirements.txt
    else
        print_warning "requirements.txt not found, installing basic dependencies..."
        pip install numpy pandas matplotlib seaborn scipy scikit-learn customtkinter
    fi
    
    # Install in development mode
    print_status "Installing APGI Framework in development mode..."
    pip install -e .
    
    print_status "Dependencies installed successfully"
}

# Function to verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    # Test imports
    print_status "Testing core imports..."
    $PYTHON_CMD -c "
import sys
try:
    import apgi_framework
    print('Core imports successful')
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'Unexpected error: {e}')
    sys.exit(1)
"
    
    if [[ $? -eq 0 ]]; then
        # APGIGui import removed
        print_status "Import verification passed"
    else
        print_error "Import verification failed"
        exit 1
    fi

}

# Function to create desktop shortcut
create_shortcut() {
    print_header "Creating Desktop Shortcut"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    LAUNCH_SCRIPT="$SCRIPT_DIR/launch_apgi.sh"
    
    # Create launch script
    cat > "$LAUNCH_SCRIPT" << EOF
#!/bin/bash
# APGI Framework Launcher
cd "$SCRIPT_DIR"

# Activate virtual environment
if [[ -d "apgi_venv" ]]; then
    if [[ "\$(uname)" == "Darwin" ]]; then
        source "apgi_venv/bin/activate"
    elif [[ "\$(uname)" == "Linux" ]]; then
        source "apgi_venv/bin/activate"
    else
        # Windows
        call "apgi_venv\\Scripts\\activate.bat"
    fi
fi

# Launch the GUI
python GUI-Launcher.py

EOF
    
    chmod +x "$LAUNCH_SCRIPT"
    
    # Create desktop shortcut based on OS
    if [[ "$OS" == "linux" ]]; then
        DESKTOP_DIR="$HOME/Desktop"
        SHORTCUT="$DESKTOP_DIR/APGI-Framework.desktop"
        
        cat > "$SHORTCUT" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=APGI Framework
Comment=Active Passive Inference Framework
Exec=bash $LAUNCH_SCRIPT
Icon=$SCRIPT_DIR/docs/apgi_icon.png
Terminal=false
Categories=Science;
EOF
        
        chmod +x "$SHORTCUT"
        print_status "Created desktop shortcut: $SHORTCUT"
        
    elif [[ "$OS" == "macos" ]]; then
        # macOS doesn't use traditional desktop shortcuts
        print_status "macOS detected. Use the launch script: $LAUNCH_SCRIPT"
        
    elif [[ "$OS" == "windows" ]]; then
        DESKTOP_DIR="$USERPROFILE/Desktop"
        SHORTCUT="$DESKTOP_DIR/APGI-Framework.bat"
        
        cat > "$SHORTCUT" << EOF
@echo off
cd /d "$SCRIPT_DIR"
call "apgi_venv\\Scripts\\activate.bat"
python GUI-Launcher.py
EOF

        
        print_status "Created desktop shortcut: $SHORTCUT"
    fi
}

# Function to show next steps
show_next_steps() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}APGI Framework has been successfully installed and configured!${NC}"
    echo
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Launch the application using one of the following methods:"
    echo "   - Run: bash launch_apgi.sh"
    if [[ "$OS" == "linux" ]]; then
        echo "   - Double-click the APGI-Framework.desktop file on your desktop"
    elif [[ "$OS" == "macos" ]]; then
        echo "   - Run: open launch_apgi.sh"
    elif [[ "$OS" == "windows" ]]; then
        echo "   - Double-click the APGI-Framework.bat file on your desktop"
    fi
    echo
    echo "2. For development:"
    echo "   - Activate the environment: source apgi_venv/bin/activate"
    echo "   - Run tests: python -m pytest tests/"
    echo "   - View documentation: open docs/README.md"
    echo
    echo -e "${YELLOW}Note: The virtual environment must be activated before running the application.${NC}"
}

# Main setup function
main() {
    print_header "APGI Framework Automated Setup"
    
    # Check if we're in the right directory
    if [[ ! -f "apgi_setup.py" ]] && [[ ! -f "requirements.txt" ]]; then
        print_error "Please run this script from the APGI Framework root directory"
        print_error "This directory should contain apgi_setup.py and requirements.txt files"
        exit 1
    fi
    
    # Run setup steps
    detect_os
    check_python
    create_venv
    activate_venv
    install_dependencies
    verify_installation
    create_shortcut
    show_next_steps
}

# Handle command line arguments
case "${1:-}" in
    "--help"|"-h")
        echo "APGI Framework Setup Script"
        echo
        echo "Usage: $0 [OPTIONS]"
        echo
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --clean        Clean up virtual environment before setup"
        echo "  --dev          Install in development mode (default)"
        echo "  --no-gui       Skip GUI verification"
        echo
        exit 0
        ;;
    "--clean")
        print_header "Cleaning Up"
        if [[ -d "apgi_venv" ]]; then
            print_status "Removing existing virtual environment..."
            rm -rf apgi_venv
            print_status "Cleanup complete"
        else
            print_status "No virtual environment to clean"
        fi
        exit 0
        ;;
esac

# Run main setup
main "$@"
