# APGI System - Complete How-To Guide

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation & Setup](#installation--setup)
3. [Getting Started](#getting-started)
4. [GUI Application Usage](#gui-application-usage)
5. [REST API Usage](#rest-api-usage)
6. [Programmatic Usage](#programmatic-usage)
7. [Experimental Tasks](#experimental-tasks)
8. [Data Analysis & Export](#data-analysis--export)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Features](#advanced-features)

---

## System Overview

The **APGI (Allostatic Precision-Gated Ignition) System** is a computational framework for modeling consciousness based on:

- **Active Inference**: Variational free energy minimization
- **Predictive Processing**: Hierarchical prediction error minimization
- **Allostatic Regulation**: Interoceptive predictive processing
- **Ignition Dynamics**: Threshold-based global workspace broadcasting
- **Self-Modeling**: Minimal and narrative self-representations

### Key Components

- **Active Inference Engine**: Bayesian inference and learning
- **Hierarchical Predictor**: 4-level predictive processing
- **Precision Weighting**: Dynamic precision modulation
- **Body Model**: Physiological state simulation
- **Ignition Threshold**: Consciousness event gating
- **Global Workspace**: Information sharing system
- **Self-Model**: Coherence tracking
- **Metabolic Budget**: Energy constraint management

---

## Installation & Setup

### Prerequisites

- Python 3.11 or higher
- Git (for cloning the repository)
- Optional: PostgreSQL 14+ (for full API functionality)
- Optional: Redis 7+ (for caching)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd apgi-system
```

### Step 2: Install Dependencies

```bash
# Install the package in development mode
pip install -e .

# Or install specific requirements
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Test core system
python -c "from apgi_simulation.system import APGISystem; print('✅ Core system OK')"

# Test GUI
python -c "import tkinter; print('✅ Tkinter available')"

# Test API dependencies
python -c "import fastapi; print('✅ FastAPI available')"
```

### Step 4: Environment Configuration (Optional)

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
# DATABASE_URL=postgresql://user:password@localhost:5432/apgi_api
# REDIS_URL=redis://localhost:6379/0
# JWT_SECRET_KEY=your-secret-key-here
```

---

## Getting Started

### Quick Start Options

#### Option 1: GUI Application (Recommended for Beginners)

```bash
python apgi_gui.py
```

#### Option 2: REST API (for Web Integration)

```bash
python -m api.main
```

#### Option 3: Programmatic Usage (for Developers)

```python
from apgi_simulation.system import APGISystem

system = APGISystem(config_path="config/default.yaml")
results = system.run(duration_ms=10000.0)
```

### First Run Checklist

- [ ] System installed without errors
- [ ] GUI launches successfully
- [ ] All visualization tabs load
- [ ] Simulation starts and runs smoothly
- [ ] Data export functionality works

---

## GUI Application Usage

### Launching the GUI

```bash
# Method 1: Direct execution
python apgi_gui.py

# Method 2: Using launcher script
python run_gui.py

# Method 3: As module
python -m apgi_gui
```

### Interface Overview

The GUI consists of four main sections:

1. **Control Panel** (Left): Simulation controls and parameters
2. **Visualization Tabs** (Right): 6 real-time monitoring tabs
3. **Menu Bar** (Top): File operations and advanced features
4. **Status Bar** (Bottom): System status and event log

### Basic Workflow

#### 1. Start a Simulation

```text
Click "▶ Start" button OR press F5
```

The system will:

- Initialize all subsystems
- Begin real-time simulation at ~1000 Hz
- Update GUI at 10 Hz (100ms intervals)
- Display live metrics in all tabs

#### 2. Monitor System State

Watch these key indicators:

- **Simulation Time**: Elapsed simulation time
- **Ignition Events**: Count of consciousness events
- **Workspace State**: Idle/Broadcasting status
- **Metabolic Reserves**: Energy level (0-100%)
- **Allostatic Load**: Cumulative stress (0-100%)
- **FPS**: GUI update rate

#### 3. Adjust Parameters

Use the Quick Parameters sliders:

- **Ignition Threshold** (1.0-5.0): Consciousness gating
- **Exteroceptive Precision** (0.1-10.0): Sensory confidence
- **Interoceptive Precision** (0.1-10.0): Body confidence
- **Arousal Level** (0.0-1.0): System activation
- **Stress Level** (0.0-1.0): Allostatic stress
- **Activity Level** (0.0-1.0): Behavioral activation
- **Learning Rate** (0.001-0.1): Adaptation speed
- **Attention Gain** (0.5-3.0): Attentional amplification

#### 4. Navigate Visualization Tabs

##### Tab 1: Neural Activity

- Ignition events (red dots)
- Global workspace activity (blue line)
- Precision dynamics (green/red lines)
- Free energy trajectory (purple line)

##### Tab 2: Interoception

- Heart rate monitoring
- Cortisol levels
- Allostatic load
- Metabolic reserves

##### Tab 3: System Metrics

- Somatic marker count
- Gamma band power (30-80 Hz)
- Beta band power (12-30 Hz)
- Performance metrics

##### Tab 4: Self-Model

- Minimal self coherence
- Depersonalization thresholds
- Narrative self integrity

##### Tab 5: Oscillations

- Real-time oscillation signals
- Multi-band power spectrum
- Delta, theta, alpha, beta, gamma bands

##### Tab 6: 3D State Space

- 3D trajectory visualization
- Free Energy × Precision × Allostatic Load
- Color-coded temporal evolution

### Advanced GUI Features

#### Menu System

##### File Menu (Ctrl+N/O/S/E/Q)

- New Session: Start fresh simulation
- Load/Save Configuration: YAML parameter files
- Export Data: CSV/JSON formats
- Export Plot: Save visualizations
- Auto-save: Toggle periodic saving
- Exit: Close application

##### Simulation Menu (F5-F8)

- Start/Pause/Stop/Reset controls
- Run Preset Task: Experimental paradigms

##### Tools Menu

- Trigger Ignition Event: Force consciousness
- Induce Stressor: Add allostatic load
- Modulate Precision: Adjust precision weights
- System Diagnostics: Health check

##### Analysis Menu

- Ignition Statistics: Detailed metrics
- Energy Budget Report: Metabolic analysis
- Generate Report: Comprehensive summary

#### Keyboard Shortcuts

| Shortcut | Action |
| ---------- | -------- |
| **F5** | Start Simulation |
| **F6** | Pause/Resume |
| **F7** | Stop Simulation |
| **F8** | Reset System |
| **Ctrl+N** | New Session |
| **Ctrl+O** | Load Configuration |
| **Ctrl+S** | Save Configuration |
| **Ctrl+E** | Export Data |
| **Ctrl+Q** | Exit |
| **Ctrl+Tab** | Cycle Tabs |
| **Ctrl++/-** | Zoom In/Out |
| **F1** | Show Help |

#### Running Experimental Tasks

1. **Simulation → Run Preset Task**

2. Select from:

   - Attentional Blink Task
   - Change Blindness Task
   - Binocular Rivalry Task
   - Masking Paradigm Task
   - Iowa Gambling Task

3. Click "Run Task"

4. Monitor progress and results

#### Manual Interventions

##### Trigger Ignition

- Tools → Trigger Ignition Event
- Forces high arousal/stress for ignition

##### Induce Stressor

- Tools → Induce Stressor
- Adds allostatic load spike

##### Modulate Precision

- Tools → Modulate Precision
- Apply multiplicative factor to precision weights

### Data Export

#### CSV Format

```csv
time,ignition,free_energy,extero_precision,intero_precision,...
0.00,0,1.234,1.0,0.8,...
0.01,0,1.245,1.0,0.8,...
0.02,1,2.456,1.5,0.9,...
```

#### JSON Format

```json
[
  {
    "time": 0.00,
    "ignition": 0,
    "free_energy": 1.234,
    "extero_precision": 1.0,
    ...
  },
  ...
]
```

---

## REST API Usage

### Starting the API Server

#### Method 1: Direct Launch

```bash
python -m api.main
```

#### Method 2: Uvicorn

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Method 3: Docker Compose

```bash
docker-compose up -d
```

### API Documentation

Once running, access:

- **Swagger UI**: <http://localhost:8000/docs>
- **ReDoc**: <http://localhost:8000/redoc>
- **Health Check**: <http://localhost:8000/health>

### Common API Endpoints

#### System Control

```bash
# Start simulation
curl -X POST "http://localhost:8000/simulation/start" \
     -H "Content-Type: application/json" \
     -d '{"duration_ms": 10000}'

# Get system status
curl "http://localhost:8000/system/status"

# Stop simulation
curl -X POST "http://localhost:8000/simulation/stop"
```

#### Parameter Management

```bash
# Get current parameters
curl "http://localhost:8000/parameters"

# Update parameters
curl -X PUT "http://localhost:8000/parameters" \
     -H "Content-Type: application/json" \
     -d '{"ignition_threshold": 2.5, "arousal_level": 0.8}'
```

#### API Data Export

```bash
# Export simulation data
curl "http://localhost:8000/export/csv?duration_ms=5000" \
     -o simulation_data.csv

# Get ignition statistics
curl "http://localhost:8000/analysis/ignition-stats"
```

### Authentication (Production)

For production use with authentication:

```bash
# Get token
curl -X POST "http://localhost:8000/auth/token" \
     -H "Content-Type: application/json" \
     -d '{"username": "user", "password": "pass"}'

# Use token in requests
curl "http://localhost:8000/system/status" \
     -H "Authorization: Bearer <token>"
```

---

## Programmatic Usage

### Basic System Initialization

```python
from apgi_simulation.system import APGISystem

# Initialize with default configuration
system = APGISystem()

# Initialize with custom configuration
system = APGISystem(config_path="config/default.yaml")
```

### Running Simulations

```python
# Run for specific duration
results = system.run(duration_ms=10000.0)

# Run step by step
for step in range(1000):
    state = system.step()
    print(f"Step {step}: Ignition={state['ignition']}")

# Run with input
input_data = {"sensory": np.random.randn(256)}
results = system.run(duration_ms=5000, input_data=input_data)
```

### Accessing System State

```python
# Get current state
state = system.get_state()

# Access specific subsystems
neural_state = system.neural_system.get_state()
interoception_state = system.interoception.get_state()
ignition_state = system.ignition_system.get_state()

# Get metrics
metrics = system.get_metrics()
print(f"Ignition count: {metrics['ignition_count']}")
print(f"Average free energy: {metrics['avg_free_energy']}")
```

### Parameter Modification

```python
# Modify parameters
system.set_parameter("ignition_threshold", 2.5)
system.set_parameter("arousal_level", 0.8)
system.set_parameter("exteroceptive_precision", 1.5)

# Batch update
params = {
    "ignition_threshold": 2.0,
    "arousal_level": 0.7,
    "stress_level": 0.3
}
system.set_parameters(params)
```

### Data Collection

```python
# Collect time series data
data = []
for step in range(1000):
    state = system.step()
    data.append({
        "time": step,
        "ignition": state["ignition"],
        "free_energy": state["free_energy"],
        "workspace": state["workspace_activity"]
    })

# Convert to pandas DataFrame
import pandas as pd
df = pd.DataFrame(data)

# Save to file
df.to_csv("simulation_results.csv", index=False)
```

---

## Experimental Tasks

### Available Tasks

1. **Attentional Blink Task**
   - Temporal attention limitations
   - Target detection under rapid presentation

2. **Change Blindness Task**
   - Visual perception failures
   - Scene change detection

3. **Binocular Rivalry Task**
   - Competitive perception
   - Alternating dominance

4. **Masking Paradigm Task**
   - Subliminal processing
   - Conscious/unconscious perception

5. **Iowa Gambling Task**
   - Decision-making under uncertainty
   - Risk/reward learning

### Running Tasks via GUI

1. **Simulation → Run Preset Task**
2. Select desired task
3. Configure task parameters
4. Click "Run Task"
5. Monitor progress in real-time

### Running Tasks Programmatically

```python
from apgi_simulation.experiments.tasks import AttentionalBlinkTask

# Initialize task
task = AttentionalBlinkTask(system=system)

# Run task
results = task.run()

# Access results
performance = results["performance"]
metrics = results["metrics"]
```

### Custom Task Creation

```python
from apgi_simulation.experiments.base import BaseTask

class CustomTask(BaseTask):
    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)
        # Initialize task-specific parameters
    
    def run(self):
        # Implement task logic
        pass
    
    def analyze_results(self, results):
        # Analyze task performance
        pass

# Use custom task
task = CustomTask(system=system)
results = task.run()
```

---

## Data Analysis & Export

### Export Formats

#### CSV Export

```python
# Export simulation data
system.export_csv("results.csv", duration_ms=10000)

# Export specific metrics
system.export_csv("metrics.csv", 
                  metrics=["ignition", "free_energy", "precision"])
```

#### JSON Export

```python
# Export full state
system.export_json("full_state.json", duration_ms=5000)

# Export configuration
system.export_config("current_config.yaml")
```

### Analysis Tools

#### Ignition Analysis

```python
# Get ignition statistics
stats = system.analyze_ignitions()
print(f"Total ignitions: {stats['count']}")
print(f"Mean interval: {stats['mean_interval']} ms")
print(f"Ignition rate: {stats['rate']} Hz")
```

#### Energy Analysis

```python
# Energy budget analysis
energy_report = system.analyze_energy()
print(f"Total consumed: {energy_report['total_consumed']}")
print(f"Peak usage: {energy_report['peak_usage']}")
print(f"Efficiency: {energy_report['efficiency']}")
```

#### Oscillation Analysis

```python
# Frequency analysis
osc_analysis = system.analyze_oscillations()
print(f"Dominant frequency: {osc_analysis['dominant_freq']} Hz")
print(f"Power bands: {osc_analysis['power_bands']}")
```

### Visualization

#### Plotting Results

```python
import matplotlib.pyplot as plt

# Plot ignition events
plt.figure(figsize=(12, 6))
plt.plot(data["time"], data["ignition"], "r.", label="Ignitions")
plt.xlabel("Time (ms)")
plt.ylabel("Ignition Events")
plt.title("Ignition Timeline")
plt.legend()
plt.show()
```

#### 3D State Space

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data["free_energy"], data["precision"], data["allostatic_load"],
           c=data["time"], cmap='viridis')
ax.set_xlabel("Free Energy")
ax.set_ylabel("Precision")
ax.set_zlabel("Allostatic Load")
plt.title("3D State Space Trajectory")
plt.show()
```

---

## Troubleshooting

### Common Issues

#### GUI Won't Start

**Symptoms:**

- Error messages about missing modules
- Window doesn't appear

**Solutions:**

```bash
# Install missing dependencies
pip install matplotlib pyyaml numpy scipy tkinter

# Verify installation
python -c "import tkinter; print('Tkinter OK')"
python -c "import matplotlib; print('Matplotlib OK')"
```

#### Slow Performance

**Symptoms:**

- Low FPS counter
- Laggy interface
- High memory usage

**Solutions:**

- Reduce simulation speed slider

- Close other applications

- Decrease buffer size in code

- Use Ctrl+P to hide parameter panel

- Use Ctrl+L to hide log panel

#### Plots Not Updating

**Symptoms:**

- Static plots

- No real-time updates

- Frozen interface

**Solutions:**

- Verify simulation is running (not paused)

- Check FPS counter shows activity

- Try Reset (F8)

- Restart application

#### Memory Issues

**Symptoms:**

- System becomes slow over time

- Memory usage increases continuously

- Application crashes

**Solutions:**

- Export data periodically

- Reset simulation to clear buffers

- Reduce buffer_size in apgi_gui.py

- Monitor memory usage in System Metrics tab

#### API Connection Issues

**Symptoms:**

- Connection refused errors

- 404 errors

- Authentication failures

**Solutions:**

```bash
# Check if server is running
curl http://localhost:8000/health

# Verify port availability
netstat -an | grep 8000

# Check logs
python -m api.main --log-level DEBUG
```

### Debug Mode

#### GUI Debug Mode

```bash
# Run with debug output
python apgi_gui.py --debug

# Enable verbose logging
python apgi_gui.py --verbose
```

#### API Debug Mode

```bash
# Run with debug logging
python -m api.main --log-level DEBUG

# Enable auto-reload for development
uvicorn api.main:app --reload --log-level debug
```

### Getting Help

1. **Check Event Log**: Monitor GUI event log for error messages

2. **Export Diagnostics**: Tools → System Diagnostics

3. **Review Configuration**: Verify parameter values are reasonable

4. **Test Components**: Run individual component tests

5. **Check Documentation**: Review relevant documentation files

---

## Advanced Features

### Custom Configuration

#### Creating Custom Configs

```yaml
# custom_config.yaml
system:
  timestep_ms: 1.0
  random_seed: 42

hierarchy:
  levels: 4
  nodes_per_level: [256, 128, 64, 32]
  timescales_ms: [10, 50, 200, 500]

active_inference:
  learning_rate: 0.01
  prediction_threshold: 0.1

ignition:
  threshold_baseline: 2.0
  refractory_period_ms: 300

interoception:
  allostatic_ranges:
    heart_rate: [60, 100]  # bpm
    cortisol: [5, 25]       # μg/dL
```

#### Loading Custom Configs

```python
# Load custom configuration
system = APGISystem(config_path="custom_config.yaml")

# Or via GUI: File → Load Configuration
```

### Extension Development

#### Adding Custom Metrics

```python
class CustomMetric:
    def __init__(self, system):
        self.system = system
        self.history = []
    
    def calculate(self, state):
        # Custom metric calculation
        value = self.compute_custom_value(state)
        self.history.append(value)
        return value
    
    def compute_custom_value(self, state):
        # Implement your metric logic
        pass

# Register with system
custom_metric = CustomMetric(system)
system.register_metric("custom_metric", custom_metric)
```

#### Custom Visualization

```python
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CustomPlot:
    def __init__(self, parent, system):
        self.system = system
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack()
    
    def update(self):
        # Update plot with latest data
        self.ax.clear()
        data = self.system.get_metric_history("custom_metric")
        self.ax.plot(data)
        self.canvas.draw()
```

### Performance Optimization

#### Memory Management

```python
# Limit history size
system.set_max_history(10000)  # Keep last 10k steps

# Automatic cleanup
system.enable_auto_cleanup(interval_ms=60000)  # Clean every minute
```

#### Computational Optimization

```python
# Reduce precision for faster simulation
system.set_precision("single")  # Use float32 instead of float64

# Disable expensive features
system.disable_feature("oscillation_analysis")
system.disable_feature("detailed_logging")
```

### Integration Examples

#### Jupyter Notebook Integration

```python
# In Jupyter notebook
%matplotlib inline
from apgi_simulation.system import APGISystem
import matplotlib.pyplot as plt

# Run simulation and plot results
system = APGISystem()
results = system.run(duration_ms=5000)

# Interactive plotting
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(results["time"], results["ignition"], "r.")
plt.title("Ignition Events")

plt.subplot(2, 2, 2)
plt.plot(results["time"], results["free_energy"])
plt.title("Free Energy")

plt.tight_layout()
plt.show()
```

#### Web Integration

```python
# Flask web interface
from flask import Flask, render_template, jsonify
import threading

app = Flask(__name__)
system = APGISystem()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def get_status():
    return jsonify(system.get_state())

@app.route("/api/start")
def start_simulation():
    def run_sim():
        system.run(duration_ms=10000)
    
    thread = threading.Thread(target=run_sim)
    thread.start()
    return jsonify({"status": "started"})

if __name__ == "__main__":
    app.run(debug=True)
```

---

## Conclusion

This comprehensive guide covers all aspects of using the APGI system, from basic installation to advanced customization. The system provides a powerful platform for consciousness modeling with multiple interfaces suitable for different use cases:

- **GUI Application**: Best for interactive exploration and demonstrations
- **REST API**: Ideal for web integration and remote access
- **Programmatic Interface**: Perfect for research and custom applications

For additional support:

- Review the documentation in the `/docs` folder
- Check the `/examples` directory for code samples
- Run tests with `pytest tests/`
- Report issues on the project repository

Happy modeling with APGI! 🧠✨
