# APGI Framework Troubleshooting Guide

This guide provides solutions for common issues when using the APGI Framework.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Import Errors](#import-errors)
3. [Configuration Problems](#configuration-problems)
4. [Test Execution Issues](#test-execution-issues)
5. [GUI Problems](#gui-problems)
6. [Performance Issues](#performance-issues)
7. [Data Processing Errors](#data-processing-errors)
8. [Statistical Analysis Problems](#statistical-analysis-problems)
9. [Workflow Execution Issues](#workflow-execution-issues)
10. [Getting Help](#getting-help)

## Installation Issues

### Python Version Compatibility

**Problem**: Import errors or compatibility issues with Python version.

**Symptoms**:

- `SyntaxError` or `ImportError` on basic imports
- Type annotation errors
- Async/await syntax errors

**Solutions**:

```bash
# Check Python version (requires 3.8+)
python --version

# Use compatible Python version
python3.9 -m pip install apgi-framework

# Create virtual environment with correct Python
python3.9 -m venv apgi_env
source apgi_env/bin/activate
pip install apgi-framework
```

### Dependency Installation Failures

**Problem**: Package installation fails due to missing system dependencies.

**Symptoms**:

- `pip install` fails with compilation errors
- Import errors for compiled packages (numpy, scipy, h5py)

**Solutions**:

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-dev build-essential
sudo apt-get install libhdf5-dev liblapack-dev libblas-dev

# Install system dependencies (macOS with Homebrew)
brew install hdf5 lapack blas

# Install in virtual environment
python -m venv apgi_env
source apgi_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install apgi-framework
```

### Permission Errors

**Problem**: Installation fails due to insufficient permissions.

**Solutions**:

```bash
# Install for user only
pip install --user apgi-framework

# Use virtual environment (recommended)
python -m venv apgi_env
source apgi_env/bin/activate
pip install apgi-framework

# System-wide installation (requires sudo)
sudo pip install apgi-framework
```

## Import Errors

### Module Not Found

**Problem**: `ImportError: No module named 'apgi_framework'`

**Solutions**:

```bash
# Check installation
pip list | grep apgi

# Reinstall package
pip uninstall apgi-framework
pip install apgi-framework

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Install in development mode
git clone https://github.com/apgiframework/apgi-system.git
cd apgi-system
pip install -e .
```

### Tkinter Missing

**Problem**: GUI components fail with tkinter import errors.

**Solutions**:

```bash
# Install tkinter (Ubuntu/Debian)
sudo apt-get install python3-tk

# Install tkinter (macOS) - usually included
# Check if available
python -c "import tkinter; print('Tkinter available')"

# Use CLI-only mode if GUI not needed
export APGI_GUI_MODE=cli
```

### Optional Dependencies

**Problem**: Features unavailable due to missing optional dependencies.

**Solutions**:

```bash
# Install all optional dependencies
pip install apgi-framework[all]

# Install specific optional groups
pip install apgi-framework[ml]        # Machine learning
pip install apgi-framework[neural]    # Neural data processing
pip install apgi-framework[gui]       # GUI components
pip install apgi-framework[web]       # Web interface
```

## Configuration Problems

### Configuration File Not Found

**Problem**: Framework cannot find configuration file.

**Symptoms**:

- `FileNotFoundError` on startup
- Using default configuration unexpectedly

**Solutions**:

```bash
# Check configuration file location
ls -la config/

# Create default configuration
apgi config create default

# Specify configuration file explicitly
apgi --config /path/to/config.yaml command

# Set environment variable
export APGI_CONFIG=/path/to/config.yaml
```

### Invalid Configuration

**Problem**: Configuration file has syntax or validation errors.

**Symptoms**:

- `ValidationError` on configuration load
- Unexpected behavior with default values

**Solutions**:

```bash
# Validate configuration
apgi config validate config.yaml

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Reset to defaults
apgi config reset

# Export template configuration
apgi config export template.yaml
```

### Parameter Range Errors

**Problem**: Configuration parameters outside valid ranges.

**Solutions**:

```bash
# Check parameter specifications
cat docs/APGI-Parameter-Specifications.md

# Use parameter validator
python -c "
from apgi_framework.utils.parameter_validator import ParameterValidator
validator = ParameterValidator()
result = validator.validate_range('tau_S', 0.5)
print(result)
"

# Reset problematic parameters
apgi config set model.tau_S 0.5
```

## Test Execution Issues

### Test Collection Errors

**Problem**: Tests fail to collect or import.

**Symptoms**:

- `pytest` reports collection errors
- Import errors during test discovery

**Solutions**:

```bash
# Check test directory structure
find tests/ -name "*.py" | head -10

# Run specific test file
python -m pytest tests/test_basic.py -v

# Check for import issues
python -c "import apgi_framework; print('Import successful')"

# Update test dependencies
pip install pytest hypothesis coverage
```

### Test Timeout

**Problem**: Tests hang or exceed timeout limits.

**Solutions**:

```bash
# Run with increased timeout
python -m pytest --timeout=600

# Run specific slow tests
python -m pytest -k "not slow" --maxfail=5

# Debug hanging test
python -m pytest --pdb --tb=long tests/test_specific.py::TestClass::test_method
```

### Memory Issues During Testing

**Problem**: Tests fail with memory errors.

**Solutions**:

```bash
# Run tests with reduced parallelism
python -m pytest -n 1

# Increase system memory limits
ulimit -v unlimited

# Run memory profiling
python -m pytest --profile-memory
```

## GUI Problems

### GUI Won't Start

**Problem**: Graphical interface fails to launch.

**Symptoms**:

- Import errors for tkinter or customtkinter
- Display/server connection errors

**Solutions**:

```bash
# Check GUI dependencies
pip install customtkinter pillow pyautogui

# Set display for headless systems
export DISPLAY=:0

# Use alternative GUI backend
export APGI_GUI_BACKEND=qt

# Launch in safe mode
apgi gui --safe-mode
```

### GUI Freezes or Becomes Unresponsive

**Problem**: Interface stops responding to input.

**Solutions**:

```bash
# Check system resources
top -p $(pgrep -f apgi)

# Restart with debugging
APGI_LOG_LEVEL=DEBUG apgi gui

# Force quit and restart
pkill -f apgi
apgi gui
```

### Display/Rendering Issues

**Problem**: GUI elements display incorrectly or not at all.

**Solutions**:

```bash
# Update graphics drivers
# Ubuntu/Debian
sudo apt-get install mesa-utils
glxinfo | grep "OpenGL version"

# Check DPI scaling
export QT_SCALE_FACTOR=1.0

# Use software rendering
export APGI_SOFTWARE_RENDER=1
```

## Performance Issues

### Slow Execution

**Problem**: Framework runs slower than expected.

**Solutions**:

```bash
# Profile performance
python -c "
import cProfile
cProfile.run('import apgi_framework; apgi_framework.run_quick_test()')
"

# Enable performance logging
export APGI_PERFORMANCE_LOG=1

# Optimize configuration
apgi config set simulation.max_steps 1000
apgi config set simulation.enable_plots false
```

### High Memory Usage

**Problem**: Excessive memory consumption.

**Solutions**:

```bash
# Monitor memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"

# Reduce batch sizes
apgi config set simulation.batch_size 100

# Enable garbage collection
python -c "import gc; gc.collect()"
```

### CPU Usage Issues

**Problem**: High CPU utilization or throttling.

**Solutions**:

```bash
# Check CPU usage
top -p $(pgrep -f apgi)

# Adjust parallelism
apgi config set parallel.max_workers 2

# Enable CPU affinity
taskset -c 0-3 python -m apgi_framework
```

## Data Processing Errors

### File Format Issues

**Problem**: Data files cannot be read or parsed.

**Solutions**:

```bash
# Check file format
file data/experiment.csv

# Validate data structure
python -c "
from apgi_framework.utils.data_validation import DataValidator
validator = DataValidator()
result = validator.validate_file_format('data/experiment.csv')
print(result)
"

# Convert file formats
pandas.read_csv('data.csv').to_json('data.json')
```

### Data Validation Failures

**Problem**: Data fails validation checks.

**Solutions**:

```bash
# Run data validation
apgi validate data data.json

# Check validation rules
python -c "
from apgi_framework.utils.data_validation import DataValidator
validator = DataValidator()
print('Validation rules:', validator.validation_rules)
"

# Fix common data issues
# Remove outliers
# Fill missing values
# Normalize ranges
```

### Large Dataset Handling

**Problem**: Performance issues with large datasets.

**Solutions**:

```bash
# Process in chunks
python -c "
import pandas as pd
chunk_size = 10000
for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    process_chunk(chunk)
"

# Use memory-efficient formats
# Convert CSV to HDF5
import pandas as pd
df = pd.read_csv('data.csv')
df.to_hdf('data.h5', key='data', mode='w')
```

## Statistical Analysis Problems

### Convergence Issues

**Problem**: Statistical models fail to converge.

**Solutions**:

```bash
# Increase iteration limits
apgi config set stats.max_iterations 10000

# Adjust convergence criteria
apgi config set stats.tolerance 1e-6

# Use different optimization method
apgi config set stats.optimizer lbfgs
```

### Numerical Stability Issues

**Problem**: Numerical errors in calculations.

**Solutions**:

```bash
# Check numerical precision
python -c "
import numpy as np
print('Machine epsilon:', np.finfo(float).eps)
print('Numerical precision OK')
"

# Use higher precision
apgi config set simulation.dtype float64

# Enable numerical checks
export APGI_NUMERICAL_CHECKS=1
```

### Statistical Test Failures

**Problem**: Hypothesis tests return unexpected results.

**Solutions**:

```bash
# Check test assumptions
# Normality: Shapiro-Wilk test
# Homoscedasticity: Levene's test
# Independence: Durbin-Watson test

# Use robust alternatives
# t-test -> Mann-Whitney U test
# ANOVA -> Kruskal-Wallis test

# Check sample sizes
apgi validate sample-sizes data.json
```

## Workflow Execution Issues

### Workflow Cancellation

**Problem**: Cannot cancel long-running workflows.

**Solutions**:

```bash
# Cancel from another terminal
apgi workflow cancel

# Send interrupt signal
kill -INT $(pgrep -f workflow)

# Force termination (last resort)
kill -9 $(pgrep -f workflow)
```

### Workflow State Corruption

**Problem**: Workflows fail due to corrupted state.

**Solutions**:

```bash
# Clean workflow state
apgi workflow clean

# Reset workflow configuration
apgi workflow reset

# Start fresh workflow
apgi workflow run-quick
```

### Parallel Execution Issues

**Problem**: Parallel workflows fail or hang.

**Solutions**:

```bash
# Reduce parallelism
apgi config set parallel.max_workers 2

# Disable parallel execution
apgi workflow run --no-parallel

# Check for race conditions
APGI_LOG_LEVEL=DEBUG apgi workflow run-parallel
```

## Getting Help

### Diagnostic Information

Collect system information for support:

```bash
# System information
uname -a
python --version
pip list | grep apgi

# Configuration
apgi config show

# Log files
ls -la logs/
tail -50 logs/*.log
```

### Support Channels

1. **Documentation**: Check all guides in `docs/` directory
2. **Issue Tracker**: GitHub issues for bugs and feature requests
3. **Community**: Discussion forums and user groups
4. **Professional Support**: Contact development team for enterprise support

### Before Contacting Support

1. Update to latest version
2. Try in clean environment
3. Collect diagnostic information
4. Reproduce issue with minimal example
5. Check existing issues and solutions

### Emergency Procedures

For critical system issues:

1. **Stop all processes**: `pkill -f apgi`
2. **Backup data**: Copy important files
3. **Reset configuration**: `apgi config reset`
4. **Clean reinstall**: Remove and reinstall package
5. **Contact support**: Provide complete diagnostic information

---

For detailed API documentation, see the [API Reference](api/index.md).
For configuration help, see the [User Guide](USER-GUIDE.md).
