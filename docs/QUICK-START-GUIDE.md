# APGI Framework Test Enhancement - Quick Start Guide

Welcome to the APGI Framework Test Enhancement system! This guide will help you get up and running quickly.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 1GB of free disk space
- 2GB of available RAM (recommended)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/apgiframework/apgi-system.git
cd apgi-system

# Install the framework
pip install -e .
```

### Installation with Optional Components

```bash
# Install with GUI support
pip install -e ".[gui]"

# Install with development tools
pip install -e ".[dev]"

# Install everything
pip install -e ".[all]"
```

### Verify Installation

```bash
# Run installation validation
python -m apgi_framework.installation_validator

# Or use the CLI command
apgi-test --help
```

## Quick Start

### 1. Basic Test Execution

#### Command Line Interface (CLI)

```bash
# Run all tests
apgi-test cli --run-all

# Run unit tests only
apgi-test cli --run-unit

# Run integration tests only
apgi-test cli --run-integration

# Run tests matching a pattern
apgi-test cli --test-pattern "test_core_*.py"

# Generate coverage report
apgi-test cli --coverage-report
```

#### Graphical User Interface (GUI)

```bash
# Launch GUI mode
apgi-test gui

# Or with specific theme
apgi-test gui --theme dark
```

### 2. Configuration

#### Using Configuration Files

Create a configuration file based on the template:

```bash
# Copy the template
cp config/test_config_template.json config/my_config.json

# Edit the configuration
# ... modify settings as needed ...

# Use the configuration
apgi-test cli --config config/my_config.json --run-all
```

#### Key Configuration Options

```json
{
  "test_configuration": {
    "execution": {
      "parallel_execution": true,
      "max_workers": 4,
      "timeout_seconds": 300
    },
    "coverage": {
      "line_threshold": 80.0,
      "branch_threshold": 75.0
    },
    "property_testing": {
      "default_iterations": 100
    }
  }
}
```

### 3. Understanding Test Results

#### CLI Output

```text
TEST EXECUTION SUMMARY
==================================================
Total Tests:  150
Passed:       142
Failed:       8
Skipped:      0
Pass Rate:    94.7%
Duration:     0:02:34

Coverage:
Lines:        87.3%
Branches:     82.1%
==================================================
```

#### GUI Interface

The GUI provides:

- **Test Tree**: Organized view of all tests by category
- **Progress Tracking**: Real-time execution progress
- **Results Viewer**: Detailed test results with filtering
- **Coverage Visualization**: Interactive coverage reports
- **Output Panel**: Detailed test output and logs

### 4. Common Workflows

#### Development Workflow

```bash
# 1. Run tests during development
apgi-test cli --run-unit

# 2. Check coverage
apgi-test cli --coverage-report

# 3. Run full test suite before commit
apgi-test cli --run-all
```

#### CI/CD Integration

```bash
# In your CI pipeline
apgi-test cli --run-all --config config/ci_config.json
```

#### Interactive Development

```bash
# Launch interactive CLI
apgi-test cli

# Then choose from menu:
# 1. Run all tests
# 2. Run unit tests
# 3. Run integration tests
# 4. Run tests by pattern
# 5. Generate coverage report
# 6. View test history
# 7. Exit
```

## Directory Structure

After installation, your project should have this structure:

```text
apgi-framework/
├── apgi_framework/          # Core framework modules
│   ├── testing/            # Test enhancement components
│   ├── core/               # Core analysis modules
│   ├── analysis/           # Analysis components
│   └── ...
├── tests/                  # Test files
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── ...
├── config/                 # Configuration files
├── logs/                   # Log files
├── test_reports/          # Test execution reports
├── coverage_reports/      # Coverage reports
└── docs/                  # Documentation
```

## Configuration Examples

### Development Environment

```json
{
  "test_configuration": {
    "execution": {
      "parallel_execution": false,
      "max_workers": 1,
      "timeout_seconds": 60
    },
    "coverage": {
      "line_threshold": 70.0
    },
    "logging": {
      "level": "DEBUG"
    }
  }
}
```

### CI Environment

```json
{
  "test_configuration": {
    "execution": {
      "parallel_execution": true,
      "max_workers": 2,
      "timeout_seconds": 600
    },
    "coverage": {
      "line_threshold": 85.0,
      "branch_threshold": 80.0
    },
    "notifications": {
      "enabled": true
    }
  }
}
```

### Production Environment

```json
{
  "test_configuration": {
    "execution": {
      "parallel_execution": true,
      "max_workers": 8
    },
    "coverage": {
      "line_threshold": 90.0,
      "branch_threshold": 85.0
    },
    "logging": {
      "level": "WARNING"
    }
  }
}
```

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# If you get import errors, ensure the project is in your Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

#### GUI Not Working

```bash
# Install GUI dependencies
pip install PySide6

# Or install with GUI extras
pip install -e ".[gui]"
```

#### Permission Errors

```bash
# On Unix systems, you might need to make scripts executable
chmod +x scripts/*.sh

# Or run with python explicitly
python -m apgi_framework.testing.main gui
```

#### Memory Issues

```bash
# Reduce parallel workers
apgi-test cli --max-workers 1 --run-all

# Or use configuration file with lower limits
```

### Getting Help

1. **Check the logs**: Look in `logs/test_enhancement.log` for detailed error messages
2. **Run validation**: Use `python -m apgi_framework.installation_validator` to check your setup
3. **Check configuration**: Ensure your configuration file is valid JSON
4. **Update dependencies**: Run `pip install --upgrade -e .` to update

### Performance Tips

1. **Use parallel execution** for large test suites
2. **Filter tests** during development to run only relevant tests
3. **Use caching** to avoid re-running unchanged tests
4. **Monitor memory usage** with large test suites

## Next Steps

1. **Explore the GUI**: Launch `apgi-test gui` to see the full interface
2. **Customize configuration**: Copy and modify `config/test_config_template.json`
3. **Set up CI integration**: Add test execution to your CI/CD pipeline
4. **Read the full documentation**: Check `docs/` for detailed guides
5. **Contribute**: See `CONTRIBUTING.md` for development guidelines

## Support

- **Documentation**: See `docs/` directory for detailed guides
- **Issues**: Report bugs and feature requests on GitHub
- **Community**: Join our discussion forums for help and tips
