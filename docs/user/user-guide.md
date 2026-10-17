# APGI Framework Testing System - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [GUI User Guide](#gui-user-guide)
4. [CLI User Guide](#cli-user-guide)
5. [Understanding Results](#understanding-results)
6. [Data Management](#data-management)
7. [Reporting Results](#reporting-results)
8. [Additional Resources](#additional-resources)

## Introduction

The APGI (Allostatic Precision-Gated Ignition) Framework Testing System is a comprehensive platform for testing the predictions of the APGI Framework through systematic falsification testing. This guide will help you:

- Run falsification tests using the GUI or CLI
- Configure experimental parameters
- Interpret test results
- Troubleshoot common issues
- Follow best practices for rigorous testing

### Quick Navigation

**New to the system?** Start with the [Quick Start Guide](quick-start.md) to get running in 5 minutes.
**Using the GUI?** See the [GUI Visual Guide](gui-guide.md) for a visual walkthrough with diagrams.
**Need help?** Check the [Troubleshooting Guide](troubleshooting.md) for common issues and solutions.

### What is Testing?

Falsification testing is a scientific methodology where we attempt to disprove a theory by identifying conditions under which its predictions fail. For the APGI Framework, we test four primary falsification criteria:

1. **Primary Falsification**: Full ignition signatures without consciousness
2. **Consciousness Without Ignition**: Conscious reports without ignition signatures
3. **Threshold Insensitivity**: Ignition threshold unaffected by neuromodulation
4. **Soma-Bias Absence**: No preferential weighting of interoceptive signals

## Getting Started

### Installation

Ensure you have Python 3.8+ installed, then install dependencies:

```bash
pip install -r requirements.txt
```

### Quick Start

#### Using Python API

```python
from apgi_framework.main_controller import MainApplicationController

# Initialize system
controller = MainApplicationController()
controller.initialize_system()

# Run primary falsification test
tests = controller.get_falsification_tests()
result = tests['primary'].run_test(n_trials=1000)

# View results
print(f"Falsified: {result.is_falsified}")
print(f"Confidence: {result.confidence_level:.3f}")

# Cleanup
controller.shutdown_system()
```

#### Using CLI

```bash
# Run primary falsification test
python -m apgi_framework.cli run-test primary --trials 1000

# View system status
python -m apgi_framework.cli status

# Validate system
python -m apgi_framework.cli validate-system
```

#### Using GUI

```bash
# Launch GUI application
python GUI-Launcher.py
```

## GUI User Guide

### Main Window Overview

The GUI provides an intuitive interface for running falsification tests and viewing results.

#### Components

1. **Test Selection Panel** (Left): Choose which falsification test to run
2. **Parameter Configuration Panel** (Center): Set APGI parameters and experimental settings
3. **Results Display Panel** (Right): View test results and visualizations
4. **Progress Panel** (Bottom): Monitor test execution progress
5. **Menu Bar** (Top): Access file operations, settings, and help

### Running a Test

#### Step 1: Select Test Type

Click on one of the test buttons in the left panel:

- **Primary Falsification Test**: Tests for full ignition without consciousness
- **Consciousness Without Ignition**: Tests for consciousness without ignition signatures
- **Threshold Insensitivity**: Tests neuromodulatory effects on threshold
- **Soma-Bias Test**: Tests interoceptive vs exteroceptive weighting

#### Step 2: Configure Parameters

In the Parameter Configuration Panel, set:

**APGI Parameters:**

- **Exteroceptive Precision** (0.1-10.0): Precision of external sensory signals
  - Default: 2.0
  - Higher values = more reliable external signals

- **Interoceptive Precision** (0.1-10.0): Precision of internal bodily signals
  - Default: 1.5
  - Higher values = more reliable internal signals

- **Threshold** (0.5-10.0): Ignition threshold for conscious access
  - Default: 3.5
  - Lower values = easier ignition

- **Steepness** (0.1-5.0): Sigmoid steepness for ignition probability
  - Default: 2.0
  - Higher values = sharper threshold

- **Somatic Gain** (0.1-5.0): Gain for somatic marker modulation
  - Default: 1.3
  - Higher values = stronger emotional modulation

**Experimental Settings:**

- **Number of Trials**: How many trials to simulate (100-10000)
  - Recommended: 1000 for standard tests
  - Use 2000+ for high-power analyses

- **Number of Participants**: For multi-participant tests (10-1000)
  - Recommended: 100 for soma-bias test

- **Random Seed**: For reproducible results (optional)
  - Leave empty for random seed
  - Use specific value (e.g., 42) for reproducibility

#### Step 3: Run Test

1. Click the **"Run Test"** button
2. Monitor progress in the Progress Panel

3. Wait for test completion (may take 30 seconds to several minutes)

#### Step 4: View Results

Results appear in the Results Display Panel:

**Summary Section:**

- Falsification status (FALSIFIED / NOT FALSIFIED)
- Confidence level (0-1)
- Statistical significance (p-value)
- Effect size (Cohen's d)
- Statistical power

**Detailed Metrics:**

- Trial-by-trial breakdown
- Neural signature statistics
- Consciousness assessment metrics

**Visualizations:**

- Distribution plots
- Time series (if applicable)
- Statistical summaries

### Saving and Loading Results

#### Saving Results

1. After test completion, click **File → Save Results**
2. Choose location and filename
3. Results saved in JSON format with metadata

#### Loading Previous Results

1. Click **File → Load Results**
2. Select result file (.json)
3. Results displayed in Results Panel

### Exporting Data

#### Export Options

1. Click **File → Export**
2. Choose format:
   - **JSON**: Full data with metadata
   - **CSV**: Tabular data for spreadsheet analysis
   - **HDF5**: Large datasets with compression
   - **PDF Report**: Publication-ready report

### Batch Processing

For running multiple configurations:

1. Click **Tools → Batch Processing**
2. Define parameter ranges:

   - Select parameters to vary
   - Set min, max, and step values
3. Click **"Start Batch"**
4. Monitor progress for all configurations
5. View comparative results when complete

## CLI User Guide

### Basic Commands

#### Run Individual Test

```bash
# Run primary falsification test
python -m apgi_framework.cli run-test primary --trials 1000

# With custom parameters
python -m apgi_framework.cli run-test primary \
  --trials 2000 \
  --seed 42 \
  --output-dir results/my_test
```

#### Run Batch Tests

```bash
# Run all tests
python -m apgi_framework.cli run-batch --all-tests

# Run specific tests
python -m apgi_framework.cli run-batch \
  --tests primary consciousness-without-ignition

# Parallel execution (experimental)
python -m apgi_framework.cli run-batch --all-tests --parallel
```

#### Configuration Management

```bash
# Generate default configuration
python -m apgi_framework.cli generate-config --output config.json

# With custom template
python -m apgi_framework.cli generate-config \
  --template comprehensive \
  --output comprehensive_config.json

# Minimal configuration
python -m apgi_framework.cli generate-config \
  --template minimal \
  --output minimal_config.json
```

#### System Management

```bash
python -m apgi_framework.cli status
python -m apgi_framework.cli validate-system
python -m apgi_framework.cli validate-system --detailed
```

#### Parameter Configuration

```bash
python -m apgi_framework.cli set-params \
  --threshold 4.0 \
  --intero_precision 2.0
```

### Using Configuration Files

#### Create Configuration File

```json
{
  "apgi_parameters": {
    "extero_precision": 2.0,
    "intero_precision": 1.5,
    "threshold": 3.5,
    "steepness": 2.0,
    "somatic_gain": 1.3
  },
  "experimental_config": {
    "n_trials": 1000,
    "n_participants": 100,
    "random_seed": 42,
    "output_directory": "results",
    "log_level": "INFO"
  }
}
```

#### Use Configuration File

```bash
python -m apgi_framework.cli run-test primary --config config.json
python -m apgi_framework.cli run-batch --all-tests --config config.json
```

### Advanced CLI Usage

#### Logging Control

```bash
python -m apgi_framework.cli run-test primary \
  --log-level DEBUG \
  --trials 100
```

#### Output Directory Management

```bash
python -m apgi_framework.cli run-test primary \
  --trials 2000 \
  --seed 42 \
  --output-dir results/high_power_test
```

## Understanding Results

### Falsification Result Structure

Every test returns a `FalsificationResult` with:

```python
FalsificationResult(
  test_type: str,              # Type of test run
  is_falsified: bool,          # Whether framework was falsified
  confidence_level: float,     # Confidence in result (0-1)
  effect_size: float,          # Cohen's d effect size
  p_value: float,              # Statistical significance
  statistical_power: float,    # Power of the test (0-1)
  replication_count: int,      # Number of replications
  detailed_results: dict       # Additional metrics
)
```

#### Issue: Results Not Saving

**Symptoms**: Results don't appear in output directory.

**Cause**: Permission issues or invalid output path.

**Solution**:

```bash
# Check permissions
ls -la results/

# Create directory if needed
mkdir -p results

# Specify absolute path
python -m apgi_framework.cli run-test primary --output-dir /absolute/path/to/results
```

#### Issue: Inconsistent Results

**Symptoms**: Different results on repeated runs.

**Cause**: Random seed not set.

**Solution**:

```python
# Set random seed for reproducibility
controller.config_manager.update_experimental_config(random_seed=42)
```

### Data Management

1. **Organize Results Systematically**

```markdown
results/
├── experiment_001/
│   ├── config.json
│   ├── primary_result_20250107.json
│   └── analysis_report.json
├── experiment_002/
│   └── ...
└── batch_processing/
    └── ...
```

1. **Use Descriptive Filenames**
   - Include date, test type, and key parameters
   - Example: `primary_test_threshold3.5_20250107.json`

2. **Backup Important Results**

   - Regularly backup results directory
   - Use version control for configurations
   - Archive completed experiments

3. **Document Experiments**
   - Keep lab notebook or README
   - Document rationale for parameter choices
   - Note any anomalies or issues

### Reporting Results

1. **Include All Relevant Metrics**
   - Falsification status
   - Confidence level
   - P-value and effect size
   - Statistical power
   - Sample size

2. **Provide Context**
   - Explain parameter choices
   - Describe experimental conditions
   - Relate to theoretical predictions

3. **Be Transparent**
   - Report all tests run, not just significant ones
   - Disclose any data exclusions
   - Share data and code when possible

4. **Interpret Cautiously**
   - Distinguish statistical from practical significance
   - Consider alternative explanations
   - Acknowledge limitations

---

## Additional Resources

- **Examples**: See `examples/` directory for working code examples
- **API Documentation**: Check module docstrings in `apgi_framework/`
- **Theoretical Background**: See `APGI-Falsification.md`
- **CLI Reference**: Run `python -m apgi_framework.cli --help`
- **Error Handling**: See `docs/ERROR_HANDLING_QUICK_REFERENCE.md`

## Support

For issues, questions, or contributions:

- Check existing documentation first
- Review examples for similar use cases
- Run system validation to diagnose issues
- Enable debug logging for detailed information

---

**Version**: 1.0
**Last Updated**: 2025-01-07
**Maintainer**: APGI Framework Development Team
