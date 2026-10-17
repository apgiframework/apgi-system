# APGI Framework Test Enhancement - Comprehensive User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Configuration](#configuration)
4. [Command Line Interface](#command-line-interface)
5. [Graphical User Interface](#graphical-user-interface)
6. [Test Organization and Categories](#test-organization-and-categories)
7. [Coverage Analysis](#coverage-analysis)
8. [Property-Based Testing](#property-based-testing)
9. [Reporting and Visualization](#reporting-and-visualization)
10. [CI/CD Integration](#cicd-integration)
11. [Performance Optimization](#performance-optimization)
12. [Troubleshooting](#troubleshooting)
13. [Advanced Features](#advanced-features)

## Introduction

The APGI Framework Test Enhancement system provides comprehensive testing capabilities for the Allostatic Precision-Gated Ignition (APGI) framework. It offers:

- **Comprehensive Test Analysis**: Identify failures, quality issues, and coverage gaps
- **Advanced Coverage Measurement**: Line, branch, and function coverage with gap identification
- **Property-Based Testing**: Automated test generation using Hypothesis
- **Dual Interface**: Both GUI and CLI for different workflows
- **CI/CD Integration**: Automated testing and reporting
- **Performance Optimization**: Parallel execution and resource management

## Installation and Setup

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Disk Space**: 1GB free space minimum
- **Dependencies**: See `requirements.txt` for complete list

### Installation Methods

#### Standard Installation

```bash
# Clone repository
git clone https://github.com/apgi-research/apgi-framework.git
cd apgi-framework

# Install framework
pip install -e .
```

#### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

#### Complete Installation

```bash
# Install all optional components
pip install -e ".[all]"
```

### Post-Installation Validation

```bash
# Run comprehensive validation
python -m apgi_framework.installation_validator

# Quick validation
apgi-test cli --help
```

## Configuration

### Configuration File Structure

The system uses JSON configuration files with the following structure:

```json
{
  "test_configuration": {
    "project_root": ".",
    "execution": { ... },
    "coverage": { ... },
    "property_testing": { ... },
    "reporting": { ... },
    "gui": { ... },
    "logging": { ... }
  },
  "environment_specific": {
    "development": { ... },
    "ci": { ... },
    "production": { ... }
  }
}
```

### Key Configuration Sections

#### Execution Configuration

```json
"execution": {
  "parallel_execution": true,
  "max_workers": 4,
  "timeout_seconds": 300,
  "retry_failed_tests": false,
  "retry_count": 1
}
```

#### Coverage Configuration

```json
"coverage": {
  "enabled": true,
  "line_threshold": 80.0,
  "branch_threshold": 75.0,
  "function_threshold": 85.0,
  "module_thresholds": {
    "apgi_framework.core": 90.0,
    "apgi_framework.testing": 95.0
  }
}
```

#### Property Testing Configuration

```json
"property_testing": {
  "enabled": true,
  "default_iterations": 100,
  "max_examples": 1000,
  "deadline": 200
}
```

### Environment-Specific Configuration

Create different configurations for different environments:

- **Development**: Lower thresholds, debug logging, single-threaded
- **CI**: Moderate thresholds, parallel execution, notifications
- **Production**: High thresholds, optimized performance, minimal logging

### Configuration Management

```bash
# Use specific configuration
apgi-test cli --config config/development.json --run-all

# Validate configuration
python -c "import json; json.load(open('config/my_config.json'))"
```

## Command Line Interface

### Basic Commands

```bash
# Show help
apgi-test --help
apgi-test cli --help
apgi-test gui --help

# Run all tests
apgi-test cli --run-all

# Run specific test categories
apgi-test cli --run-unit
apgi-test cli --run-integration

# Run tests by pattern
apgi-test cli --test-pattern "test_core_*.py"

# Generate coverage report
apgi-test cli --coverage-report
```

### Advanced CLI Options

```bash
# Custom configuration
apgi-test cli --config config/my_config.json --run-all

# Performance tuning
apgi-test cli --max-workers 8 --run-all

# Coverage threshold
apgi-test cli --coverage-threshold 0.85 --run-all

# Logging level
apgi-test cli --log-level DEBUG --run-all
```

### Interactive CLI Mode

```bash
# Launch interactive mode
apgi-test cli

# Interactive menu provides:
# 1. Run all tests
# 2. Run unit tests
# 3. Run integration tests
# 4. Run tests by pattern
# 5. Generate coverage report
# 6. View test history
# 7. Exit
```

### CLI Output Formats

The CLI provides detailed output including:

- **Progress indicators**: Real-time test execution progress
- **Summary statistics**: Pass/fail counts, execution time
- **Coverage metrics**: Line, branch, and function coverage
- **Failed test details**: Error messages and stack traces
- **Performance metrics**: Execution time and resource usage

## Graphical User Interface

### Launching the GUI

```bash
# Basic GUI launch
apgi-test gui

# With theme selection
apgi-test gui --theme dark

# With custom configuration
apgi-test gui --config config/my_config.json
```

### GUI Components

#### Main Window Layout

- **Left Panel**: Test selection and configuration
- **Right Panel**: Results and visualization
- **Status Bar**: Current operation status
- **Menu Bar**: File operations and settings
- **Toolbar**: Quick access to common actions

#### Test Selection Panel

- **Test Tree**: Hierarchical view of all tests
- **Filter Options**: Filter by category, status, or pattern
- **Selection Controls**: Check/uncheck tests for execution
- **Configuration Panel**: Execution settings

#### Results Panel

- **Summary Tab**: Overall test statistics and failed tests
- **Detailed Results Tab**: Complete test results table
- **Coverage Tab**: Coverage metrics and module breakdown
- **Output Tab**: Detailed test output and logs

### GUI Features

#### Test Execution

1. **Select Tests**: Use checkboxes in test tree
2. **Configure Execution**: Set parallel execution, workers, thresholds
3. **Start Execution**: Click "Run Selected Tests" or "Run All Tests"
4. **Monitor Progress**: Watch real-time progress and current test
5. **View Results**: Examine results in various tabs

#### Coverage Visualization

- **Overall Metrics**: Line, branch, and function coverage percentages
- **Module Breakdown**: Per-module coverage with color coding
- **Trend Analysis**: Coverage changes over time
- **Gap Identification**: Uncovered code paths highlighted

#### Result Management

- **Save Results**: Export results to various formats
- **Filter Results**: Show only failed tests or specific categories
- **Navigate Failures**: Quick jump to failure details
- **Compare Runs**: Compare results between different executions

## Test Organization and Categories

### Test Discovery

The system automatically discovers tests using these patterns:

- **File Patterns**: `test_*.py`, `*_test.py`
- **Directory Scanning**: `tests/`, `apgi_framework/`
- **Exclusion Patterns**: `__pycache__`, `*.pyc`, `.pytest_cache`

### Test Categories

Tests are automatically categorized:

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test module interactions
3. **Property Tests**: Property-based tests using Hypothesis
4. **Performance Tests**: Benchmarking and performance validation
5. **End-to-End Tests**: Complete workflow validation

### Test Metadata

Each test includes metadata:

- **Module**: Source module being tested
- **Category**: Test type (unit, integration, etc.)
- **Requirements**: Linked requirements from design document
- **Execution Time**: Historical execution time
- **Dependencies**: Required fixtures or setup

### Custom Test Organization

```python
# Mark tests with custom categories
import pytest

@pytest.mark.unit
@pytest.mark.core
def test_core_functionality():
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_module_integration():
    pass
```

## Coverage Analysis

### Coverage Types

The system measures three types of coverage:

1. **Line Coverage**: Percentage of code lines executed
2. **Branch Coverage**: Percentage of code branches taken
3. **Function Coverage**: Percentage of functions called

### Coverage Collection

Coverage is collected automatically during test execution:

```bash
# Run tests with coverage
apgi-test cli --run-all  # Coverage included by default

# Generate coverage report only
apgi-test cli --coverage-report
```

### Coverage Thresholds

Set different thresholds for different modules:

```json
"coverage": {
  "line_threshold": 80.0,
  "branch_threshold": 75.0,
  "module_thresholds": {
    "apgi_framework.core": 90.0,
    "apgi_framework.testing": 95.0,
    "apgi_framework.analysis": 85.0
  }
}
```

### Coverage Gap Identification

The system identifies coverage gaps:

- **Uncovered Lines**: Specific lines not executed
- **Uncovered Branches**: Conditional branches not taken
- **Uncovered Functions**: Functions never called
- **Priority Scoring**: Gaps ranked by importance

### Coverage Reporting

Multiple report formats available:

- **HTML Reports**: Interactive web-based reports
- **JSON Reports**: Machine-readable data
- **XML Reports**: CI/CD integration format
- **Console Output**: Summary statistics

## Property-Based Testing

### Introduction to Property-Based Testing

Property-based testing automatically generates test cases to verify that properties hold across all valid inputs.

### Property Definition

Properties are defined as functions that should always return True:

```python
from hypothesis import given, strategies as st

@given(st.integers())
def test_addition_commutative(x):
    """Addition should be commutative."""
    y = 5
    assert x + y == y + x
```

### Hypothesis Integration

The system integrates with Hypothesis for property-based testing:

- **Strategy Generation**: Automatic test data generation
- **Shrinking**: Minimal failing examples
- **Stateful Testing**: Complex state machine testing
- **Custom Strategies**: Domain-specific data generators

### Property Test Configuration

```json
"property_testing": {
  "enabled": true,
  "default_iterations": 100,
  "max_examples": 1000,
  "deadline": 200,
  "strategies": {
    "integers": {"min_value": -1000, "max_value": 1000},
    "floats": {"min_value": -1000.0, "max_value": 1000.0}
  }
}
```

### Scientific Computing Properties

Special support for scientific computing properties:

- **Numerical Stability**: Properties for floating-point operations
- **Signal Processing**: Properties for EEG, pupillometry data
- **Statistical Properties**: Properties for analysis algorithms
- **Performance Properties**: Properties for execution time and memory

## Reporting and Visualization

### Report Types

The system generates multiple report types:

1. **Execution Reports**: Test run summaries and details
2. **Coverage Reports**: Coverage analysis and gaps
3. **Performance Reports**: Execution time and resource usage
4. **Trend Reports**: Historical analysis and trends
5. **Failure Reports**: Detailed failure analysis

### Report Formats

- **HTML**: Interactive web reports with drill-down
- **JSON**: Machine-readable structured data
- **XML**: Standard format for CI/CD integration
- **PDF**: Printable summary reports
- **CSV**: Data export for analysis

### Visualization Features

#### Advanced Coverage Features

- **Heat Maps**: Visual representation of coverage
- **Trend Charts**: Coverage changes over time
- **Module Breakdown**: Per-module coverage comparison
- **Gap Highlighting**: Visual identification of uncovered code

#### Test Result Visualization

- **Pass/Fail Charts**: Test result distribution
- **Execution Time Charts**: Performance trends
- **Failure Analysis**: Common failure patterns
- **Category Breakdown**: Results by test category

### Custom Reporting

```python
# Custom report generation
from apgi_framework.testing.reporting import ReportGenerator

generator = ReportGenerator()
report = generator.generate_custom_report(
    test_results=results,
    coverage_data=coverage,
    template="custom_template.html"
)
```

## CI/CD Integration

### Supported CI Systems

- **GitHub Actions**: Native integration
- **Jenkins**: Plugin support
- **GitLab CI**: Pipeline integration
- **Azure DevOps**: Task integration
- **Generic**: Standard exit codes and reports

### CI Configuration Examples

#### GitHub Actions

```yaml
name: APGI Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: pip install -e ".[dev]"
    - name: Run tests
      run: apgi-test cli --config config/ci.json --run-all
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

#### Jenkins Pipeline

```groovy
pipeline {
    agent any
    stages {
        stage('Test') {
            steps {
                sh 'pip install -e ".[dev]"'
                sh 'apgi-test cli --config config/ci.json --run-all'
            }
        }
    }
    post {
        always {
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'coverage_reports',
                reportFiles: 'index.html',
                reportName: 'Coverage Report'
            ])
        }
    }
}
```

### Change Impact Analysis

The system can analyze code changes and run only affected tests:

```bash
# Run tests affected by recent changes
apgi-test cli --changed-only

# Run tests affected by specific commit
apgi-test cli --since-commit abc123
```

### Notification Integration

Configure notifications for CI failures:

```json
"notifications": {
  "enabled": true,
  "channels": [
    {
      "type": "slack",
      "webhook_url": "https://hooks.slack.com/...",
      "on_failure": true,
      "on_success": false
    },
    {
      "type": "email",
      "smtp_server": "smtp.company.com",
      "recipients": ["team@company.com"],
      "on_failure": true
    }
  ]
}
```

## Performance Optimization

### Parallel Execution

Enable parallel test execution for better performance:

```json
"execution": {
  "parallel_execution": true,
  "max_workers": 4  // Adjust based on CPU cores
}
```

### Memory Management

For large test suites, optimize memory usage:

```json
"execution": {
  "memory_limit_mb": 2048,
  "cleanup_interval": 100  // Clean up every 100 tests
}
```

### Test Result Caching

Cache test results to avoid re-running unchanged tests:

```bash
# Enable caching
apgi-test cli --cache-results --run-all

# Clear cache
apgi-test cli --clear-cache
```

### Performance Monitoring

Monitor test execution performance:

- **Execution Time Tracking**: Per-test and overall timing
- **Memory Usage Monitoring**: Peak and average memory usage
- **Resource Utilization**: CPU and I/O usage
- **Performance Regression Detection**: Identify slow tests

### Optimization Strategies

1. **Test Parallelization**: Run independent tests in parallel
2. **Smart Test Selection**: Run only affected tests
3. **Resource Pooling**: Reuse expensive setup operations
4. **Incremental Coverage**: Only measure coverage for changed code
5. **Result Caching**: Cache results for unchanged tests

## Troubleshooting

### Common Issues

#### Installation Problems

```bash
# Python version issues
python --version  # Should be 3.8+

# Dependency conflicts
pip install --upgrade pip
pip install -e . --force-reinstall

# Permission issues (Unix)
sudo pip install -e .
# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install -e .
```

#### Import Errors

```bash
# Module not found
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .

# Check installation
python -c "import apgi_framework; print('OK')"
```

#### GUI Issues

```bash
# PySide6 not installed
pip install PySide6

# Display issues (Linux)
export DISPLAY=:0

# Qt platform issues
export QT_QPA_PLATFORM=xcb  # Linux
export QT_QPA_PLATFORM=windows  # Windows
```

#### Performance Issues

```bash
# Reduce parallel workers
apgi-test cli --max-workers 1 --run-all

# Increase timeout
apgi-test cli --timeout 600 --run-all

# Check system resources
python -c "import psutil; print(f'CPU: {psutil.cpu_count()}, RAM: {psutil.virtual_memory().total/1e9:.1f}GB')"
```

### Debugging

#### Enable Debug Logging

```bash
# CLI debug mode
apgi-test cli --log-level DEBUG --run-all

# Check log files
tail -f logs/test_enhancement.log
```

#### Verbose Output

```bash
# Verbose test execution
apgi-test cli --verbose --run-all

# Show all test output
apgi-test cli --capture=no --run-all
```

#### Configuration Validation

```bash
# Validate configuration file
python -c "import json; json.load(open('config/my_config.json')); print('Valid')"

# Run installation validator
python -m apgi_framework.installation_validator --output validation_report.json
```

### Getting Help

1. **Check Documentation**: Review relevant sections of this guide
2. **Examine Log Files**: Look in `logs/` directory for detailed error messages
3. **Run Validation**: Use installation validator to check setup
4. **Check Configuration**: Ensure configuration files are valid JSON
5. **Update Dependencies**: Run `pip install --upgrade -e .`
6. **Community Support**: Check GitHub issues and discussions

## Advanced Features

### Custom Test Generators

Create custom test generators for domain-specific testing:

```python
from apgi_framework.testing.generators import BaseTestGenerator

class NeuralSignalTestGenerator(BaseTestGenerator):
    def generate_eeg_tests(self, signal_properties):
        # Generate tests for EEG signal processing
        pass
    
    def generate_pupillometry_tests(self, measurement_properties):
        # Generate tests for pupillometry analysis
        pass
```

### Plugin System

Extend functionality with plugins:

```python
from apgi_framework.testing.plugins import TestPlugin

class CustomAnalysisPlugin(TestPlugin):
    def analyze_results(self, test_results):
        # Custom result analysis
        pass
    
    def generate_report(self, analysis):
        # Custom report generation
        pass
```

### API Integration

Use the programmatic API for custom workflows:

```python
from apgi_framework.testing.api import TestRunner, CoverageAnalyzer

# Programmatic test execution
runner = TestRunner(config=my_config)
results = runner.run_tests(test_files=['test_core.py'])

# Coverage analysis
analyzer = CoverageAnalyzer()
coverage = analyzer.analyze_coverage(results.coverage_data)
```

### Custom Strategies for Property Testing

Define domain-specific strategies:

```python
from hypothesis import strategies as st

# Custom strategy for neural signals
@st.composite
def neural_signal_strategy(draw):
    sampling_rate = draw(st.integers(min_value=100, max_value=1000))
    duration = draw(st.floats(min_value=1.0, max_value=10.0))
    noise_level = draw(st.floats(min_value=0.0, max_value=0.1))
    
    return NeuralSignal(
        sampling_rate=sampling_rate,
        duration=duration,
        noise_level=noise_level
    )
```

### Batch Processing

Process multiple test configurations:

```bash
# Run multiple configurations
for config in config/*.json; do
    apgi-test cli --config "$config" --run-all
done

# Batch report generation
python -m apgi_framework.testing.batch_processor --config-dir config/
```

### Integration with External Tools

#### Code Quality Tools

```bash
# Integration with black, flake8, mypy
apgi-test cli --run-all --code-quality-checks

# Custom quality checks
apgi-test cli --run-all --custom-checks "black,flake8,mypy"
```

#### Performance Profiling

```bash
# Profile test execution
apgi-test cli --profile --run-all

# Memory profiling
apgi-test cli --memory-profile --run-all
```

#### Database Integration

Store test results in database for historical analysis:

```python
from apgi_framework.testing.storage import DatabaseStorage

storage = DatabaseStorage(connection_string="sqlite:///test_results.db")
storage.store_results(test_results)
```
