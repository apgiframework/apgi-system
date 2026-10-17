# APGI System: Allostatic Precision-Gated Ignition Framework

A computational implementation of consciousness based on active inference, predictive processing, and allostatic regulation.

## Overview

The APGI framework integrates:

- **Active Inference**: Variational free energy minimization and Bayesian inference
- **Predictive Processing**: Hierarchical prediction error minimization
- **Allostatic Regulation**: Interoceptive predictive processing and homeostatic control
- **Ignition Dynamics**: Threshold-based global workspace broadcasting
- **Self-Modeling**: Minimal and narrative self-representations

### APGI System  ✅

The core APGI system with all subsystems:

- Active Inference Engine
- Hierarchical Predictor
- Precision Weighting
- Body Model
- Allostatic Regulator
- Somatic Marker System
- Ignition Threshold
- Global Workspace
- Self-Model (Minimal & Narrative)
- Metabolic Budget
- Neural Oscillations

### System Step Execution ✅

The system successfully executes simulation steps:

- Processes exteroceptive input (256-dimensional)
- Updates all subsystems
- Returns complete state dictionary
- Tracks ignition events
- Manages workspace broadcasting
- Updates precision and free energy
- Tracks system metrics

### Experimental Tasks ✅

All experimental paradigms are importable and functional:

- Attentional Blink Task
- Change Blindness Task
- Binocular Rivalry Task
- Iowa Gambling Task
- Masking Paradigm Task

### GUI Features Validation ✅

#### Control Panel

- ✓ Start/Pause/Stop/Reset buttons
- ✓ Speed control slider (0.1x - 10x)
- ✓ System status display
- ✓ Quick parameter adjustments (8 sliders)
- ✓ Event log with timestamps

#### Visualization Tabs (6 tabs)

1. ✓ Neural Activity - Ignition events, workspace, precision, free energy
2. ✓ Interoception - Heart rate, cortisol, allostatic load, metabolic reserves
3. ✓ System Metrics - Somatic markers, gamma/beta power
4. ✓ Self-Model - Coherence tracking with thresholds
5. ✓ Oscillations - Signal and power spectrum
6. ✓ State Space - 3D trajectory visualization

#### Menu System

- ✓ File Menu (8 items) - New, Load, Save, Export, Auto-save, Exit
- ✓ Edit Menu (4 items) - Parameters, Precision, Threshold, Reset
- ✓ Simulation Menu (5 items) - Start, Pause, Stop, Reset, Preset Tasks
- ✓ View Menu (7 items) - Panel toggles, Zoom controls
- ✓ Tools Menu (6 items) - Trigger ignition, Induce stressor, Modulate precision
- ✓ Analysis Menu (5 items) - Statistics, Reports, Analysis tools
- ✓ Help Menu (3 items) - Documentation, Shortcuts, About

#### Keyboard Shortcuts

- ✓ Ctrl+N - New Session
- ✓ Ctrl+O - Load Configuration
- ✓ Ctrl+S - Save Configuration
- ✓ Ctrl+E - Export Data
- ✓ Ctrl+Q - Exit
- ✓ F5 - Start Simulation
- ✓ F6 - Pause/Resume
- ✓ F7 - Stop Simulation
- ✓ F8 - Reset System

---

## Functional Testing

### Real-Time Simulation ✅

**Test:** Start simulation and observe updates

- ✓ Simulation thread starts successfully
- ✓ System steps execute at ~1000 Hz
- ✓ GUI updates at 10 Hz (100ms intervals)
- ✓ FPS counter displays correctly
- ✓ Status labels update in real-time

### Parameter Adjustment ✅

**Test:** Modify parameters during simulation

- ✓ Sliders respond immediately
- ✓ Values apply to running system
- ✓ Changes reflected in plots
- ✓ No crashes or errors

### Data Export ✅

**Test:** Export simulation data

- ✓ CSV export format available
- ✓ JSON export format available
- ✓ All metrics included in export
- ✓ Timestamps accurate

**Launch Command:**

```bash
python apgi_gui.py
```

## Architecture

```text
apgi_simulation/
├── core/               # Active inference and predictive processing
├── neural/             # Multi-scale neural networks
├── interoception/      # Body state modeling and allostasis
├── ignition/           # Global workspace and ignition dynamics
├── self_model/         # Minimal and narrative self
├── thermodynamic/      # Energy budget and entropy tracking
└── visualization/      # Real-time monitoring
```

## Installation

```bash
pip install -e .
```

## Quick Start

### GUI Applications (Recommended)

#### Main APGI System GUI

```bash
python APGI_GUI.py
```

**Features**: Complete real-time visualization, parameter control, and data export

#### Specialized GUIs

```bash
# Psychological states parameter exploration
python Psychological-States-GUI.py

# Run and monitor test scripts
python Tests-GUI.py

# Execute utility scripts with GUI interface
python Utils-GUI.py

# Interactive AI assistant for APGI system
python Assistant_GUI.py
```

### Programmatic Usage

```python
from apgi_simulation.system import APGISystem

# Initialize the system
system = APGISystem(config_path="config/default.yaml")

# Run a simulation
results = system.run(duration_ms=10000.0)

# Access results
print(f"Ignition events: {results['ignition_count']}")
print(f"Final state: {results['final_state']}")
```

### Command Line Examples

```bash
# Run basic simulation with visualization
python utils/basic_simulation.py --duration 1000 --input-size 256

# Check system dependencies
python utils/dependency_checker.py

# Generate documentation screenshots
python utils/take_screenshots.py

# Run experimental tasks
python apgi_simulation/experiments/test_iowa_gambling.py
python apgi_simulation/experiments/test_masking.py

# Run tests
pytest tests/test_core.py -v
```

## Key Features

- **Multi-scale neural dynamics** (micro, meso, macro)
- **Hierarchical predictive coding** (3-4 layers)
- **Precision-weighted prediction errors**
- **Somatic marker learning** and retrieval
- **Dynamic ignition thresholds**
- **Thermodynamic constraint tracking**
- **Neural oscillations** (gamma, beta, alpha, theta, delta)

## Success Criteria

### Minimal

- Ignition dynamics match 300-500ms timeline
- Metabolic costs align with 5-10% overhead
- Reproduces attentional blink and masking phenomena

### Target

- Human-comparable performance on benchmark tasks
- Neural signatures match empirical data
- Pathology models reproduce clinical phenotypes

## APGI System: Comprehensive Analysis

### System Overview

The **APGI (Allostatic Precision-Gated Ignition) System** is a sophisticated computational framework for modeling consciousness based on active inference, predictive processing, and allostatic regulation. It implements a biologically-inspired model of cognitive function with multiple interacting subsystems.

### Core Architecture

The system is organized into several key modules:

### **Core Systems** (`apgi_simulation/core/`)

- **Active Inference Engine**: Variational free energy minimization and Bayesian inference
- **Hierarchical Predictor**: Multi-level predictive processing (4 levels: sensory → feature → category → abstract)
- **Precision Weighting**: Dynamic precision modulation of prediction errors
- **Free Energy Calculator**: Tracks system-wide variational free energy

### **Neural Systems** (`apgi_simulation/neural/`)

- **Microscale**: Spiking neural networks
- **Mesoscale**: Neural column dynamics
- **Macroscale**: Large-scale brain networks
- **Oscillations**: Multi-band neural oscillations (delta, theta, alpha, beta, gamma)

### **Interoception** (`apgi_simulation/interoception/`)

- **Body Model**: Simulates physiological states (heart rate, respiration, temperature, glucose, cortisol)
- **Allostatic Regulator**: Maintains homeostatic balance
- **Somatic Marker System**: Emotion-like signals for decision-making

### **Ignition Dynamics** (`apgi_simulation/ignition/`)

- **Ignition Threshold**: Dynamic threshold for global broadcasting
- **Global Workspace**: Information sharing across subsystems
- **Temporal Dynamics**: Timing of ignition events (300-500ms timeline)

### **Self-Model** (`apgi_simulation/self_model/`)

- **Minimal Self**: Basic self-representation and coherence tracking
- **Narrative Self**: Episodic memory and autobiographical processing
- **Coherence Maintenance**: Self-model integrity monitoring

### **Thermodynamic** (`apgi_simulation/thermodynamic/`)

- **Metabolic Budget**: Energy constraints and consumption tracking
- **Entropy Tracker**: Information entropy and thermodynamic costs

## Entry Points

### **1. GUI Application** (Primary Interface)

**File**: [apgi_gui.py]
**Launch**:

```bash
python APGI_GUI.py
```

**Features**:

- **Real-time visualization** with 6 tabs:
  1. Neural Activity (ignition events, workspace, precision, free energy)
  2. Interoception (heart rate, cortisol, allostatic load, metabolic reserves)
  3. System Metrics (somatic markers, gamma/beta power)
  4. Self-Model (coherence tracking with thresholds)
  5. Oscillations (signal and power spectrum)
  6. State Space (3D trajectory visualization)
- **Complete menu system** (File, Edit, Simulation, View, Tools, Analysis, Help)
- **Interactive controls** (Start/Pause/Stop/Reset, speed control, parameter sliders)
- **Data export** (CSV/JSON formats)
- **Experimental task execution** with progress tracking
- **Keyboard shortcuts** (Ctrl+N/O/S/E/Q, F5-F8)

### **2. Psychological States GUI**

**File**: [Psychological-States-GUI.py](cci:7://file:///Users/apgiframework/Sites/PYTHON/apgi-system/Psychological-States-GUI.py:0:0-0:0) (3,137 bytes)
**Launch**:

```bash
python Psychological-States-GUI.py
```

**Features**:

- **Parameter Library**: Complete parameter mappings for 51 psychological states
- **Advanced Visualizations**: Interactive visualizations rendered directly in the GUI
- **State Management**: Comprehensive psychological state parameter exploration
- **No External Dependencies**: All visualizations displayed within the application

### **3. Tests GUI**

**File**: [Tests-GUI.py](cci:7://file:///Users/apgiframework/Sites/PYTHON/apgi-system/Tests-GUI.py:0:0-0:0) (167 bytes)
**Launch**:

```bash
python Tests-GUI.py
```

**Features**:

- **Test Script Runner**: GUI interface to run all test scripts from the tests folder
- **Real-time Output Display**: Live output display with error handling
- **Test Results Summary**: Statistics panel showing test results and pass/fail counts
- **Process Management**: Individual test execution and full test suite running
- **Script Selection**: Browse and select individual test scripts to run

### **4. Utils GUI**

**File**: [Utils-GUI.py](cci:7://file:///Users/apgiframework/Sites/PYTHON/apgi-system/Utils-GUI.py:0:0-0:0) (68 bytes)
**Launch**:

```bash
python Utils-GUI.py
```

**Features**:

- **Utility Script Runner**: GUI interface to run all utility scripts from the utils folder
- **Real-time Output Display**: Live output display with error handling
- **Process Management**: Individual script execution and batch running
- **Script Selection**: Browse and select individual utility scripts to run

### **5. Assistant GUI**

**File**: [Assistant_GUI.py](cci:7://file:///Users/apgiframework/Sites/PYTHON/apgi-system/Assistant_GUI.py:0:0-0:0) (8,647 bytes)
**Launch**:

```bash
python Assistant-GUI.py
```

**Features**:

- **AI Assistant Interface**: Interactive AI assistant for APGI system guidance
- **Multi-threaded Architecture**: Responsive UI with background processing
- **Queue-based Communication**: Asynchronous message handling
- **Comprehensive Logging**: Rotating file handler and structured logging
- **Dynamic Module Loading**: Runtime loading of APGI components

### **6. REST API** (Web Interface)

**File**: [api/main.py](cci:7://file:///Users/apgiframework/Sites/PYTHON/apgi-system/api/main.py:0:0-0:0)
**Framework**: FastAPI with comprehensive middleware
**Features**:

- RESTful endpoints for system control
- Authentication and rate limiting
- Database integration with Alembic migrations
- Prometheus metrics and structured logging
- Redis for session management
- CORS support for web clients

### **7. Programmatic Interface**

**File**: [apgi_simulation/system.py](cci:7://file:///Users/apgiframework/Sites/PYTHON/apgi-system/apgi_simulation/system.py:0:0-0:0)
**Usage**:

```python
from apgi_simulation.system import APGISystem

system = APGISystem(config_path="config/default.yaml")
results = system.run(duration_ms=10000.0)
```

### **8. Experimental Tasks**

**Location**: `apgi_simulation/experiments/tasks/`
**Available Tasks**:

- **Iowa Gambling Task**: Decision-making under uncertainty
- **Attentional Blink Task**: Temporal attention limitations
- **Change Blindness Task**: Visual perception failures
- **Binocular Rivalry Task**: Competitive perception
- **Masking Paradigm Task**: Subliminal processing

### **9. Utility Scripts**

**Location**: `utils/`
**Available Scripts**:

#### **basic_simulation.py** - APGI System Simulation

```bash
python utils/basic_simulation.py --duration 1000 --input-size 256
```

- Runs configurable APGI system simulations
- Generates visualizations of ignition events, free energy, and metabolic reserves
- Supports sinusoidal sensory input with adjustable parameters

#### **build_common.py** - Build Utilities

- Common build and development utilities for the APGI system
- Project root detection and command execution helpers

#### **circuit_breaker.py** - System Resilience

- Implements circuit breaker pattern for system fault tolerance
- Prevents cascading failures in distributed components

#### **datetime_utils.py** - Date/Time Utilities

- UTC time handling and elapsed time calculations
- Timestamp formatting for logging and data export

#### **dependency_checker.py** - System Validation

```bash
python utils/dependency_checker.py
```

- Checks for required Python packages and system dependencies
- Provides installation guidance for missing components
- Validates Python version compatibility

#### **demo_analysis.py** - Analysis Tools

- Demonstration analysis utilities for simulation results

#### **installer_utils.py** - Installation Helpers

- Automated installation utilities and setup scripts

#### **release.py** - Release Management

- Release management and deployment utilities

#### **script_runner_gui.py** - GUI Framework

- Shared GUI base class for running scripts from directories
- Used by Tests-GUI.py and Utils-GUI.py applications

#### **take_screenshots.py** - Documentation Tools

```bash
python utils/take_screenshots.py
```

- Automated screenshot capture for Python desktop applications
- Generates documentation screenshots with GUI interaction
- Requires pyautogui, pygetwindow, pillow, opencv-python

#### **test_platform_utils.py** - Platform Testing

- Platform utility testing and validation

#### **test_rate_limiter_debug.py** - Debug Tools

- Rate limiter debugging utilities

#### **validate_app.py** - Application Validation

- Comprehensive application validation and health checks

## Key Functionality

### **Real-Time Simulation**

- **Timestep**: 1ms microscale resolution
- **Update Rate**: ~1000 Hz system, 10 Hz GUI
- **Multi-threaded architecture** for responsive UI
- **State tracking** with comprehensive history

### **Hierarchical Processing**

- **4-level hierarchy**: 256→128→64→32 nodes
- **Different timescales**: 10ms→50ms→200ms→500ms
- **Predictive coding** with error minimization
- **Precision weighting** with neuromodulator effects

### **Physiological Modeling**

- **Body states**: Heart rate, respiration, temperature, glucose, cortisol
- **Allostatic ranges**: Tight (10%), moderate (20%), wide (30%)
- **Metabolic constraints**: 5-10% ignition overhead, 25-30% task overhead
- **Energy budgeting** with recovery dynamics

### **Consciousness Metrics**

- **Ignition events**: Global workspace broadcasting
- **Phi integration**: IIT consciousness measure
- **PCI complexity**: Perturbational complexity index
- **Coherence tracking**: Self-model integrity
- **Somatic markers**: Decision-making signals

## Configuration

**File**: [config/default.yaml](cci:7://file:///Users/apgiframework/Sites/PYTHON/apgi-system/config/default.yaml:0:0-0:0)
**Sections**:

- System parameters (timestep, random seed)
- Hierarchy structure (levels, nodes, timescales)
- Active inference (learning rates, thresholds)
- Precision weighting (baselines, neuromodulators)
- Ignition dynamics (thresholds, timing)
- Interoception (body states, allostatic ranges)
- Neural oscillations (frequency bands, coupling)
- Thermodynamic constraints (energy budgets)
- Validation metrics (phi, PCI, ERPs)

## Dependencies

**Core**: NumPy, SciPy, JAX, PyTorch, Matplotlib, Pandas
**GUI**: Tkinter (built-in), Matplotlib backend
**API**: FastAPI, Redis, Alembic, Prometheus
**Testing**: Pytest, Hypothesis
**Development**: Black, Flake8, MyPy

## Validation & Testing

The system includes comprehensive validation:

- **Unit tests** for individual components
- **Integration tests** for subsystem interactions
- **Property-based tests** using Hypothesis
- **Experimental paradigms** reproducing known phenomena
- **Performance monitoring** with FPS counters and metrics

## Research Applications

This system models:

- **Attentional blink** and perceptual limitations
- **Change blindness** and visual awareness
- **Decision-making** under uncertainty (Iowa Gambling)
- **Binocular rivalry** and competitive perception
- **Somatic marker hypothesis** in decision-making
- **Neural oscillations** and consciousness correlates
- **Metabolic constraints** on cognitive processing

The APGI framework provides a comprehensive platform for studying consciousness through computational modeling, integrating multiple theoretical perspectives into a unified, testable system.

## Next Steps

To explore the system further:

1. **Start the Main GUI**: Run `python APGI_GUI.py` to launch the comprehensive interactive interface
2. **Explore Specialized GUIs**:
   - `python Psychological-States-GUI.py` - Parameter library for 51 psychological states
   - `python Tests-GUI.py` - Run and monitor test scripts
   - `python Utils-GUI.py` - Execute utility scripts with GUI interface
   - `python Assistant_GUI.py` - Interactive AI assistant for guidance
3. **Use Utility Scripts**:
   - `python utils/basic_simulation.py` - Run configurable simulations
   - `python utils/dependency_checker.py` - Check system requirements
   - `python utils/take_screenshots.py` - Generate documentation screenshots
4. **Review the API**: Check `docs/api/REST-API.md` for web service endpoints
5. **Examine the core**: Look at `apgi_simulation/system.py` for the main simulation logic
6. **Check experiments**: Explore `apgi_simulation/experiments/` for task implementations
7. **Review configuration**: Study `config/default.yaml` for system parameters
