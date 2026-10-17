# APGI Framework Testing System - GUI Application

## Overview

The APGI Framework Testing System GUI provides a comprehensive graphical interface for conducting falsification tests on the Allostatic Precision-Gated Ignition (APGI) Framework. The application features a tabbed interface for different test types, parameter configuration, real-time progress tracking, and results visualization.

## Features

### 1. Configuration Management

- **APGI Parameters**: Configure core equation parameters (precision, errors, gains, thresholds)
- **Experimental Settings**: Set trial counts, participant numbers, statistical thresholds
- **Load/Save**: Import and export configuration files in JSON format
- **Reset to Defaults**: Quickly restore default parameter values

### 2. Falsification Test Tabs

#### Primary Falsification Test

- Tests for full ignition signatures without consciousness
- Validates P3b ERP, gamma synchrony, BOLD activation, and PCI signatures
- Checks for absence of consciousness reports and AI/ACC engagement

#### Consciousness Without Ignition Test

- Tests for consciousness reports without ignition signatures
- Validates framework boundaries for conscious access mechanisms
- Requires >20% occurrence rate for falsification

#### Threshold Insensitivity Test

- Tests neuromodulatory threshold dynamics
- Simulates pharmacological challenges (propranolol, L-DOPA, SSRIs, physostigmine)
- Validates threshold modulation predictions

#### Soma-Bias Test

- Tests interoceptive vs exteroceptive prediction error bias
- Compares matched precision conditions
- Calculates β values for bias assessment

### 3. Progress Tracking

- Real-time progress bars for running tests
- Status updates and completion notifications
- Start/stop controls for test execution
- Thread-safe operation to maintain GUI responsiveness

### 4. Results Visualization

- **Falsification Rates**: Bar charts showing rates by test type
- **Effect Size Distribution**: Histograms of statistical effect sizes
- **P-values vs Effect Sizes**: Scatter plots with significance thresholds
- **Statistical Power**: Distribution of power analysis results
- **Export Capabilities**: Save results in JSON or CSV formats

### 5. System Logging

- Real-time log display with configurable levels (DEBUG, INFO, WARNING, ERROR)
- System initialization and validation messages
- Test execution progress and completion status
- Error reporting and troubleshooting information
- Save logs to file functionality

### 6. Menu System

- **File Menu**: Configuration management and result export
- **System Menu**: Initialization, status checking, and validation
- **Help Menu**: About information and system details

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Required packages (install via `pip install -r requirements.txt`):
  - tkinter (usually included with Python)
  - matplotlib >= 3.4.0
  - numpy >= 1.21.0
  - pandas >= 1.3.0
  - scipy >= 1.7.0

### Running the Application

#### Option 1: Direct Launch

```bash
python GUI.py
```

#### Option 2: Using Launcher (Recommended)

```bash
python GUI-Launcher.py
```

The launcher script provides dependency checking and better error handling.

## Usage Guide

### 1. Initial Setup

1. Launch the application using one of the methods above
2. The system will automatically attempt to initialize
3. Check the "System Logs" tab for initialization status
4. Use "System → System Status" to verify all components are ready

### 2. Configuration

1. Go to the "Configuration" tab
2. Adjust APGI parameters and experimental settings as needed
3. Click "Apply Changes" to update the configuration
4. Optionally save your configuration using "File → Save Configuration"

### 3. Running Tests

1. Select the desired falsification test tab
2. Configure test-specific parameters (trials, participants)
3. Click "Run Test" to start execution
4. Monitor progress via the progress bar and status updates
5. View results in the test panel when complete

### 4. Analyzing Results

1. Go to the "Results & Visualization" tab
2. Generate sample data or view results from completed tests
3. Examine the four visualization plots:
   - Falsification rates by test type
   - Effect size distributions
   - P-values vs effect sizes
   - Statistical power distributions
4. Export results using the "Export Results" button

### 5. Monitoring System

1. Use the "System Logs" tab to monitor system activity
2. Adjust log level as needed (DEBUG for detailed information)
3. Clear logs or save them to file as required
4. Check system status via "System → System Status"

## Configuration File Format

Configuration files are saved in JSON format with the following structure:

```json
{
  "apgi_parameters": {
    "extero_precision": 2.0,
    "intero_precision": 1.5,
    "extero_error": 1.2,
    "intero_error": 0.8,
    "somatic_gain": 1.3,
    "threshold": 3.5,
    "steepness": 2.0
  },
  "experimental_config": {
    "n_trials": 1000,
    "n_participants": 100,
    "random_seed": null,
    "p3b_threshold": 5.0,
    "gamma_plv_threshold": 0.3,
    "bold_z_threshold": 3.1,
    "pci_threshold": 0.4,
    "alpha_level": 0.05,
    "effect_size_threshold": 0.5,
    "power_threshold": 0.8
  }
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required packages are installed

   ```bash
   pip install -r requirements.txt
   ```

2. **System Initialization Fails**: Check the System Logs tab for detailed error messages

3. **GUI Not Responsive**: Tests run in separate threads; check if a test is currently running

4. **Matplotlib Display Issues**: Ensure you have a proper display environment (X11 on Linux, etc.)

### Error Messages

- **"System not initialized"**: Use "System → Initialize System" from the menu
- **"Configuration validation failed"**: Check parameter values are within valid ranges
- **"Test execution failed"**: Review system logs for detailed error information

## Technical Details

### Architecture

- **Main Application**: `APGIFrameworkGUI` class manages the overall application
- **Configuration Panel**: `ParameterConfigPanel` handles parameter management
- **Test Panels**: `FalsificationTestPanel` instances for each test type
- **Visualization**: `ResultsVisualizationPanel` with matplotlib integration
- **Logging**: `LoggingPanel` with custom log handler

### Threading

- System initialization runs in background threads
- Test execution uses separate worker threads
- GUI remains responsive during long-running operations
- Thread-safe communication via queues and `after()` calls

### Data Management

- Configuration stored in JSON format
- Results can be exported as JSON or CSV
- Matplotlib figures support various export formats
- Logging supports file output for debugging

## Development Notes

The GUI application integrates with the existing APGI Framework components:

- `MainApplicationController` for system orchestration
- `ConfigManager` for parameter management
- Falsification test controllers for test execution
- Neural simulators for signature generation
- Statistical analysis engines for result validation

For extending the GUI, follow the established patterns:

- Use ttk widgets for consistent styling
- Implement proper error handling and user feedback
- Maintain thread safety for background operations
- Follow the tabbed interface design for new features
