# APGI Framework - Test Enhancement

Comprehensive Test Enhancement for APGI Framework - Advanced Platform for General Intelligence Research.

## Overview

The APGI Framework provides tools for consciousness research, experimental paradigms, and falsification testing of consciousness theories. This repository contains enhanced testing infrastructure and comprehensive test coverage improvements.

## Features

Every feature below is marked **✅ verified** (code exists in this repository and is exercised by tests or an
entry point) or **🧭 planned** (described, but not implemented or not measurable today). See the
[status legend](#how-to-use-apgi).

- Advanced consciousness modeling and simulation — ✅ verified (`apgi_framework/system.py`, `apgi_framework/ignition/`)
- Falsification testing protocols — ✅ verified (`apgi_framework/falsification/`, `apgi_framework/protocols/`)
- Real-time monitoring and visualization — ✅ verified (`apgi_framework/gui/monitoring_dashboard.py`, `apgi_framework/monitoring/`)
- Cross-species validation paradigms — ✅ verified (`apgi_framework/research/cross_species_validation.py`)
- Device acquisition with offline-first upload — ✅ verified (`apgi_framework/acquisition/`, `tests/acquisition/`)
- Comprehensive test suite with high coverage — 🧭 planned (CI enforces `--cov-fail-under=90` on `tests/unit/`; no measured coverage figure is published here)

## How to Use APGI

This section is the complete usage guide (also kept as [docs/HOW-TO-USE-APGI.md](docs/HOW-TO-USE-APGI.md)).

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

> **Status legend.** Entry points, modules, API routes and methods marked **✅ verified** were checked against
> this repository. Items marked **🧭 planned** appear in the original how-to guide but have no implementation
> in the current code; they describe intended behavior and will not run today. Where the original guide named
> a path that does not exist, the working equivalent is given.

### System Overview

The **APGI (Allostatic Precision-Gated Ignition) System** is a computational framework for modeling consciousness based on:

- **Active Inference**: Variational free energy minimization
- **Predictive Processing**: Hierarchical prediction error minimization
- **Allostatic Regulation**: Interoceptive predictive processing
- **Ignition Dynamics**: Threshold-based global workspace broadcasting
- **Self-Modeling**: Minimal and narrative self-representations

#### Key Components

- **Active Inference Engine**: Bayesian inference and learning
- **Hierarchical Predictor**: 4-level predictive processing
- **Precision Weighting**: Dynamic precision modulation
- **Body Model**: Physiological state simulation
- **Ignition Threshold**: Consciousness event gating
- **Global Workspace**: Information sharing system
- **Self-Model**: Coherence tracking
- **Metabolic Budget**: Energy constraint management

### Installation & Setup

#### Prerequisites

- Python 3.11 or higher
- Git (for cloning the repository)
- Tk (`tkinter`) for the desktop GUI. On Debian/Ubuntu install `python3-tk`; on headless Linux also install `xvfb` and `xauth`
- Optional: PostgreSQL 14+ (for full API functionality)
- Optional: Redis 7+ (for caching, token blacklist and login-attempt tracking)

#### Step 1: Clone Repository

```bash
git clone https://github.com/apgiframework/apgi-system.git
cd apgi-system
```

#### Step 2: Install Dependencies

```bash
# Install the package in development mode
pip install -e .

# Or install specific requirements
pip install -r requirements.txt
```

#### Step 3: Verify Installation

```bash
# Test core system (✅ verified module path)
python -c "from apgi_framework.system import APGISystem; print('✅ Core system OK')"

# Test GUI
python -c "import tkinter; print('✅ Tkinter available')"

# Test API dependencies
python -c "import fastapi; print('✅ FastAPI available')"

# Validate all framework components
python -m apgi_framework.cli validate-system
```

#### Step 4: Environment Configuration (Optional)

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
# DATABASE_URL=postgresql://user:password@localhost:5432/apgi_api
# REDIS_URL=redis://localhost:6379/0
# JWT_SECRET_KEY=your-secret-key-here
```

A bare `postgresql://` URL is run through the `psycopg2` driver; `asyncpg` is used for the async engine.
Set a strong, random `JWT_SECRET_KEY` (64+ characters) before exposing the API.

### Getting Started

#### Quick Start Options

##### Option 1: GUI Application (Recommended for Beginners)

```bash
python -m apgi_gui.main      # ✅ verified
apgi-gui                     # console script, available after `pip install -e .`
```

The original guide used `python apgi_gui.py`, `python run_gui.py` and `python -m apgi_gui`; those files do not
exist in this repository.

##### Option 2: REST API (for Web Integration)

```bash
python -m api.main           # ✅ verified; defaults to port 8000 and picks another port if busy
```

##### Option 3: Programmatic Usage (for Developers)

```python
from apgi_framework.system import APGISystem   # ✅ verified (guide said apgi_simulation.system)

system = APGISystem(config_path="config/default.yaml")
results = system.run(duration_ms=10000.0)
```

#### First Run Checklist

- [ ] System installed without errors
- [ ] GUI launches successfully
- [ ] All visualization tabs load
- [ ] Simulation starts and runs smoothly
- [ ] Data export functionality works

### GUI Application Usage

#### Launching the GUI

```bash
# Method 1: As module
python -m apgi_gui.main

# Method 2: Installed console script
apgi-gui

# Method 3: Legacy launcher (deprecated, see Entry Points)
python GUI-Launcher.py
```

On a headless Linux machine run the GUI under a virtual display: `xvfb-run -a python -m apgi_gui.main`.

#### Interface Overview

The GUI consists of four main sections:

1. **Control Panel** (Left): Simulation controls and parameters
2. **Visualization Tabs** (Right): real-time monitoring tabs
3. **Menu Bar** (Top): File operations and advanced features
4. **Status Bar** (Bottom): System status and event log

#### Basic Workflow

##### 1. Start a Simulation

```text
Click "▶ Start" button OR press F5
```

The system will:

- Initialize all subsystems
- Begin real-time simulation at ~1000 Hz
- Update GUI at 10 Hz (100 ms intervals)
- Display live metrics in all tabs

##### 2. Monitor System State

Watch these key indicators:

- **Simulation Time**: Elapsed simulation time
- **Ignition Events**: Count of consciousness events
- **Workspace State**: Idle/Broadcasting status
- **Metabolic Reserves**: Energy level (0-100%)
- **Allostatic Load**: Cumulative stress (0-100%)
- **FPS**: GUI update rate

##### 3. Adjust Parameters

Use the Quick Parameters sliders:

- **Ignition Threshold** (1.0-5.0): Consciousness gating
- **Exteroceptive Precision** (0.1-10.0): Sensory confidence
- **Interoceptive Precision** (0.1-10.0): Body confidence
- **Arousal Level** (0.0-1.0): System activation
- **Stress Level** (0.0-1.0): Allostatic stress
- **Activity Level** (0.0-1.0): Behavioral activation
- **Learning Rate** (0.001-0.1): Adaptation speed
- **Attention Gain** (0.5-3.0): Attentional amplification

##### 4. Navigate Visualization Tabs

###### Tab 1: Neural Activity

- Ignition events (red dots)
- Global workspace activity (blue line)
- Precision dynamics (green/red lines)
- Free energy trajectory (purple line)

###### Tab 2: Interoception

- Heart rate monitoring
- Cortisol levels
- Allostatic load
- Metabolic reserves

###### Tab 3: System Metrics

- Somatic marker count
- Gamma band power (30-80 Hz)
- Beta band power (12-30 Hz)
- Performance metrics

###### Tab 4: Self-Model

- Minimal self coherence
- Depersonalization thresholds
- Narrative self integrity

###### Tab 5: Oscillations

- Real-time oscillation signals
- Multi-band power spectrum
- Delta, theta, alpha, beta, gamma bands

###### Tab 6: 3D State Space

- 3D trajectory visualization
- Free Energy × Precision × Allostatic Load
- Color-coded temporal evolution

> Tabs 1-3 (Neural Activity, Interoception, System Metrics) are ✅ verified in `apgi_gui/components/visualization_panel.py`.
> Tabs 4-6 are described by the original guide and have not been confirmed in the modular GUI.

#### Advanced GUI Features

##### Menu System

###### File Menu (Ctrl+N/O/S/E/Q)

- New Session: Start fresh simulation
- Load/Save Configuration: YAML parameter files
- Export Data: CSV/JSON formats
- Export Plot: Save visualizations
- Auto-save: Toggle periodic saving
- Exit: Close application

###### Simulation Menu (F5-F8)

- Start/Pause/Stop/Reset controls
- Run Preset Task: Experimental paradigms

###### Tools Menu

- Trigger Ignition Event: Force consciousness
- Induce Stressor: Add allostatic load
- Modulate Precision: Adjust precision weights
- System Diagnostics: Health check

###### Analysis Menu

- Ignition Statistics: Detailed metrics
- Energy Budget Report: Metabolic analysis
- Generate Report: Comprehensive summary

##### Keyboard Shortcuts

| Shortcut | Action | Status |
| ---------- | -------- | ------ |
| **F5** | Start Simulation | ✅ verified |
| **F6** | Pause/Resume | ✅ verified |
| **F7** | Stop Simulation | ✅ verified |
| **F8** | Reset System | ✅ verified |
| **Ctrl+N** | New Session | ✅ verified |
| **Ctrl+O** | Load Configuration | ✅ verified |
| **Ctrl+S** | Save Configuration | ✅ verified |
| **Ctrl+E** | Export Data | ✅ verified |
| **Ctrl+Q** | Exit | ✅ verified |
| **Ctrl++/-/0** | Zoom In/Out/Reset | ✅ verified (menu accelerators) |
| **Ctrl+Tab** | Cycle Tabs | per original guide |
| **F1** | Show Help | per original guide |

##### Running Experimental Tasks

1. **Simulation → Run Preset Task**
2. Select from:
   - Attentional Blink Task
   - Change Blindness Task
   - Binocular Rivalry Task
   - Masking Paradigm Task
   - Iowa Gambling Task
3. Click "Run Task"
4. Monitor progress and results

> The Iowa Gambling Task implementation is currently a stub (see [Experimental Tasks](#experimental-tasks)).

##### Manual Interventions

- **Trigger Ignition**: Tools → Trigger Ignition Event forces high arousal/stress for ignition
- **Induce Stressor**: Tools → Induce Stressor adds an allostatic load spike
- **Modulate Precision**: Tools → Modulate Precision applies a multiplicative factor to precision weights

#### Data Export

##### CSV Format

```csv
time,ignition,free_energy,extero_precision,intero_precision,...
0.00,0,1.234,1.0,0.8,...
0.01,0,1.245,1.0,0.8,...
0.02,1,2.456,1.5,0.9,...
```

##### JSON Format

```json
[
  {
    "time": 0.00,
    "ignition": 0,
    "free_energy": 1.234,
    "extero_precision": 1.0
  }
]
```

### REST API Usage

#### Starting the API Server

```bash
# Method 1: Direct launch
python -m api.main

# Method 2: Uvicorn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Method 3: Docker Compose
docker-compose up -d
```

#### API Documentation

Once running (any non-production `ENVIRONMENT`), access:

- **Swagger UI**: <http://localhost:8000/docs>
- **ReDoc**: <http://localhost:8000/redoc>
- **OpenAPI schema**: <http://localhost:8000/openapi.json>
- **Health Check**: <http://localhost:8000/v1/health>

The interactive docs, ReDoc and OpenAPI schema are disabled when `ENVIRONMENT=production`.

#### Endpoint Map (✅ verified against `api/routes/`)

The API is versioned under `/v1` and organised around **sessions**. The original guide's `/simulation/start`,
`/system/status`, `/parameters`, `/export/csv`, `/analysis/ignition-stats` and `/auth/token` routes do not exist.

| Area | Method and path |
| ---- | --------------- |
| Auth | `POST /v1/auth/login`, `POST /v1/auth/refresh`, `POST /v1/auth/logout` |
| Sessions | `POST /v1/sessions` (create), `POST /v1/sessions/{id}/start`, `/pause`, `/stop`, `/reset`, `DELETE /v1/sessions/{id}` |
| State | `GET /v1/sessions/{id}/state`, `/ignition-history`, `/interoception`, `/prediction-errors`, `/somatic-markers`, `/summary-statistics` |
| Export | `GET /v1/sessions/{id}/export?format=json\|csv`, `/summary`, `/timeseries`, `/events` |
| Tasks | `GET/POST /v1/tasks`, `/v1/sessions/{id}/tasks`, `/v1/tasks/{task_id}` |
| Health | `GET /v1/health`, `/v1/health/live`, `/v1/health/ready`, `GET /v1/version` |
| Admin and compliance | `/v1/admin/...`, `/v1/compliance/export`, `/v1/metrics` (system-admin permission), `/v1/webhooks/...`, `/v1/users/...` |

#### Typical Workflow

```bash
# 1. Authenticate (production deployments)
curl -X POST "http://localhost:8000/v1/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "user", "password": "pass"}'

# 2. Create a session (config_path, custom_config and description are optional)
curl -X POST "http://localhost:8000/v1/sessions" \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{"description": "demo run"}'

# 3. Start it and read its state
curl -X POST "http://localhost:8000/v1/sessions/<session_id>/start" -H "Authorization: Bearer <token>"
curl "http://localhost:8000/v1/sessions/<session_id>/state" -H "Authorization: Bearer <token>"
curl "http://localhost:8000/v1/sessions/<session_id>/summary-statistics" -H "Authorization: Bearer <token>"

# 4. Export the results
curl "http://localhost:8000/v1/sessions/<session_id>/export?format=csv" \
     -H "Authorization: Bearer <token>" -o simulation_data.csv

# 5. Stop the session
curl -X POST "http://localhost:8000/v1/sessions/<session_id>/stop" -H "Authorization: Bearer <token>"
```

Exact request and response schemas are in Swagger UI. Repeated failed logins lock the account after five attempts
(tracked in Redis when available, otherwise in memory).

### Programmatic Usage

#### Basic System Initialization

```python
from apgi_framework.system import APGISystem

# Initialize with default configuration (config/default.yaml)
system = APGISystem()

# Initialize with custom configuration
system = APGISystem(config_path="config/default.yaml")
```

#### Running Simulations

```python
import numpy as np

# Run for a specific duration (✅ verified signature: run(duration_ms, extero_input_fn=None))
results = system.run(duration_ms=10000.0)

# Run with a custom sensory input generator: a function of time returning a vector
results = system.run(duration_ms=5000.0, extero_input_fn=lambda t: np.random.randn(256))

# Run step by step (✅ verified signature: step(extero_input, context=None))
for step in range(1000):
    state = system.step(np.random.randn(256))
    print(f"Step {step}: Ignition={state['ignition']}")

# Reset the system
system.reset()
```

> The exact keys returned by `step()` depend on the configuration; inspect the returned dictionary.
> The original guide's `run(..., input_data=...)` argument does not exist; use `extero_input_fn`.

#### Accessing System State

```python
# Get current state (✅ verified)
state = system.get_state()
summary = system.get_state_summary()
overview = system.get_system_summary()
performance = system.get_performance_statistics()
```

🧭 **Planned (not implemented):** `system.neural_system.get_state()`, `system.interoception.get_state()`,
`system.ignition_system.get_state()` and `system.get_metrics()` with `ignition_count` / `avg_free_energy`.

#### Parameter Modification

Edit parameters in a YAML configuration file and pass it via `config_path`, or use the GUI sliders or
`python -m apgi_framework.cli set-params`.

🧭 **Planned (not implemented):**

```python
system.set_parameter("ignition_threshold", 2.5)
system.set_parameters({"ignition_threshold": 2.0, "arousal_level": 0.7, "stress_level": 0.3})
```

#### Data Collection

```python
import numpy as np
import pandas as pd

data = []
for step in range(1000):
    state = system.step(np.random.randn(256))
    data.append({
        "time": step,
        "ignition": state["ignition"],
        "free_energy": state["free_energy"],
        "workspace": state["workspace_activity"],
    })

df = pd.DataFrame(data)
df.to_csv("simulation_results.csv", index=False)
```

The key names above (`ignition`, `free_energy`, `workspace_activity`) follow the original guide; confirm them against
`system.get_state()` for your configuration.

### Experimental Tasks

#### Available Tasks

1. **Attentional Blink Task**: temporal attention limitations; target detection under rapid presentation
2. **Change Blindness Task**: visual perception failures; scene change detection
3. **Binocular Rivalry Task**: competitive perception; alternating dominance
4. **Masking Paradigm Task**: subliminal processing; conscious/unconscious perception
5. **Iowa Gambling Task**: decision-making under uncertainty; risk/reward learning

Implementations live under `apgi_framework` (see [docs/Experimental-Tasks.md](docs/Experimental-Tasks.md)).
The Iowa Gambling implementation is currently a stub that lacks the balance, schedule, stimulus and trial APIs its
tests expect, so treat its results as non-scientific until completed.

#### Running Tasks via GUI

1. **Simulation → Run Preset Task**
2. Select desired task
3. Configure task parameters
4. Click "Run Task"
5. Monitor progress in real-time

#### Running Tasks Programmatically

🧭 **Planned (not implemented):** the `apgi_simulation.experiments` package shown in the original guide does not exist.
The intended interface was:

```python
from apgi_simulation.experiments.tasks import AttentionalBlinkTask   # not available

task = AttentionalBlinkTask(system=system)
results = task.run()
performance = results["performance"]
metrics = results["metrics"]
```

For falsification tests and batch experiments use the CLI (`python -m apgi_framework.cli run-test primary --trials 1000`).

#### Custom Task Creation

🧭 **Planned (not implemented)** pattern from the original guide:

```python
class CustomTask(BaseTask):                  # BaseTask: not available
    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)

    def run(self):
        ...

    def analyze_results(self, results):
        ...
```

For custom experiments today, subclass the base class in `core/experiment.py` (see "Currently Implemented" below).

### Data Analysis & Export

#### Export Formats

Session data can be exported through the REST API (`GET /v1/sessions/{id}/export?format=json|csv`) and the GUI
(**File → Export Data**, Ctrl+E). Library-level helpers are in `apgi_framework/data_export.py`.

🧭 **Planned (not implemented):** `system.export_csv(...)`, `system.export_json(...)` and `system.export_config(...)`.

#### Analysis Tools

Use the GUI **Analysis** menu (Ignition Statistics, Energy Budget Report, Generate Report), the
`/v1/sessions/{id}/summary-statistics` API route, or the analysis package (`apgi-analysis`, `apgi_framework.analysis`).

🧭 **Planned (not implemented):** `system.analyze_ignitions()`, `system.analyze_energy()` and
`system.analyze_oscillations()`, which were described as returning ignition count, mean interval and rate; total
consumed energy, peak usage and efficiency; and dominant frequency and band powers.

#### Visualization

##### Plotting Results

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df["time"], df["ignition"], "r.", label="Ignitions")
plt.xlabel("Time (ms)")
plt.ylabel("Ignition Events")
plt.title("Ignition Timeline")
plt.legend()
plt.show()
```

##### 3D State Space

```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(df["free_energy"], df["precision"], df["allostatic_load"], c=df["time"], cmap="viridis")
ax.set_xlabel("Free Energy")
ax.set_ylabel("Precision")
ax.set_zlabel("Allostatic Load")
plt.title("3D State Space Trajectory")
plt.show()
```

### Troubleshooting

See also [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

#### GUI Won't Start

**Symptoms:** error messages about missing modules; the window does not appear.

```bash
# Install missing dependencies (tkinter ships with Python or the OS package python3-tk, not pip)
pip install -r requirements.txt

# Verify installation
python -c "import tkinter; print('Tkinter OK')"
python -c "import matplotlib; print('Matplotlib OK')"
```

On a headless server run under Xvfb: `xvfb-run -a python -m apgi_gui.main`.

#### Slow Performance

**Symptoms:** low FPS counter, laggy interface, high memory usage.

- Reduce the simulation speed slider
- Close other applications
- Decrease the buffer size in code
- Use Ctrl+P to hide the parameter panel and Ctrl+L to hide the log panel (per original guide)

#### Plots Not Updating

**Symptoms:** static plots, no real-time updates, frozen interface.

- Verify the simulation is running (not paused)
- Check that the FPS counter shows activity
- Try Reset (F8)
- Restart the application

#### Memory Issues

**Symptoms:** system slows over time, memory use climbs, application crashes.

- Export data periodically
- Reset the simulation to clear buffers
- Reduce the buffer size in the GUI code
- Monitor memory usage in the System Metrics tab

#### API Connection Issues

**Symptoms:** connection refused, 404 errors, authentication failures.

```bash
# Check if server is running
curl http://localhost:8000/v1/health

# Verify port availability
netstat -an | grep 8000
```

A 404 on `/health`, `/simulation/start` or similar means the path is not under `/v1/` (see the endpoint map above).
`api.main` falls back to another port when 8000 is taken, so read the startup log for the port actually used.

#### Debug Mode

```bash
# API with debug logging and auto-reload
uvicorn api.main:app --reload --log-level debug

# CLI with debug logging
python -m apgi_framework.cli --log-level DEBUG status
```

The original guide's `--debug` / `--verbose` GUI flags and `python -m api.main --log-level` option are not implemented.

#### Getting Help

1. **Check Event Log**: Monitor the GUI event log for error messages
2. **Export Diagnostics**: Tools → System Diagnostics, or `python -m apgi_framework.cli validate-system`
3. **Review Configuration**: Verify parameter values are reasonable
4. **Test Components**: Run individual component tests with `pytest tests/unit/`
5. **Check Documentation**: Review the files in `docs/`

### Advanced Features

#### Custom Configuration

##### Creating Custom Configs

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
    cortisol: [5, 25]      # μg/dL
```

Start from `config/default.yaml` (✅ verified) and validate with `python -m apgi_framework.cli generate-config` /
`validate-system`; key names in your file must match those in the default file.

##### Loading Custom Configs

```python
system = APGISystem(config_path="custom_config.yaml")
# Or via the GUI: File → Load Configuration
```

#### Extension Development

##### Adding Custom Metrics

🧭 **Planned (not implemented):** `system.register_metric(...)`. Until it exists, compute custom metrics from the
state dictionaries returned by `system.step()`:

```python
class CustomMetric:
    def __init__(self):
        self.history = []

    def calculate(self, state):
        value = self.compute_custom_value(state)
        self.history.append(value)
        return value

    def compute_custom_value(self, state):
        ...  # your metric logic
```

##### Custom Visualization

```python
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CustomPlot:
    def __init__(self, parent, history_fn):
        self.history_fn = history_fn
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack()

    def update(self):
        self.ax.clear()
        self.ax.plot(self.history_fn())
        self.canvas.draw()
```

#### Performance Optimization

🧭 **Planned (not implemented):** `set_max_history`, `enable_auto_cleanup`, `set_precision("single")` and
`disable_feature(...)`. Available today: run `system.get_performance_statistics()` / `system.log_performance()`,
use `utils/performance_profiler.py`, and see [docs/Benchmarks.md](docs/Benchmarks.md).

#### Integration Examples

##### Jupyter Notebook Integration

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from apgi_framework.system import APGISystem

system = APGISystem()
history = [system.step(np.random.randn(256)) for _ in range(2000)]

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot([s["free_energy"] for s in history])
plt.title("Free Energy")
plt.tight_layout()
plt.show()
```

##### Web Integration

Prefer the bundled REST API. A minimal custom Flask wrapper looks like this:

```python
import threading
from flask import Flask, jsonify
from apgi_framework.system import APGISystem

app = Flask(__name__)
system = APGISystem()

@app.route("/api/status")
def get_status():
    return jsonify(system.get_state_summary())

@app.route("/api/start")
def start_simulation():
    threading.Thread(target=lambda: system.run(duration_ms=10000.0)).start()
    return jsonify({"status": "started"})

if __name__ == "__main__":
    app.run()          # do not use debug=True on an exposed host
```

`get_state_summary()` must return JSON-serializable values for `jsonify`; convert NumPy arrays with `.tolist()` if needed.

### Conclusion

This guide covers installation through advanced customization. The system provides three interfaces:

- **GUI Application**: best for interactive exploration and demonstrations
- **REST API**: ideal for web integration and remote access
- **Programmatic Interface**: perfect for research and custom applications

For additional support:

- Review the documentation in the [`docs/`](docs/) folder
- Check the [`examples/`](examples/) directory for code samples
- Run tests with `pytest tests/`
- Report issues on the [project repository](https://github.com/apgiframework/apgi-system/issues)

Happy modeling with APGI! 🧠✨


## Device acquisition

Record from an amplifier (simulated driver included) with clock-synchronised event markers, a crash-safe local
recorder, an offline-first session state machine backed by a SQLCipher-encrypted store, resumable uploads and BIDS-EEG
export. See [docs/ACQUISITION.md](docs/ACQUISITION.md) and `examples/acquisition_quickstart.py`.

## Installation

### Basic Installation

```bash
pip install -e .
```

### Development Installation

```bash
pip install -e .[dev]
```

### All Dependencies

```bash
pip install -e .[all]
```

## Usage

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=apgi_framework

# Run specific test categories
pytest -m "unit"
pytest -m "integration"
pytest -m "gui"
```

### GUI Applications

```bash
# Launch main GUI
python GUI-Launcher.py
```

## Project Structure

- `apgi_framework/` - Core framework modules
- `tests/` - Comprehensive test suite
- `examples/` - Usage examples
- `docs/` - Documentation
- `utils/` - Utility scripts and tools

## Entry Points (Canonical)

### Recommended Entry Points

| Entry Point | Type | Purpose | Status |
|-------------|------|---------|--------|
| `apgi_gui/main.py` | Desktop GUI | **Primary GUI** - Component-based GUI with clean architecture | Active |
| `apgi_framework/cli.py` | CLI | **Primary CLI** - Command-line interface for tests, experiments | Active |
| `python -m apgi_framework` | Module | Direct module execution | Active |

### Legacy Entry Points (Deprecated)

> **Note**: These monolithic files are deprecated in favor of `apgi_gui/main.py`:

| Entry Point | Size | Status | Migration |
|-------------|------|--------|-----------|
| `GUI-Launcher.py` | 54KB | Deprecated | Use `apgi_gui/main.py` |
| `APGI_GUI.py` | 237KB | Deprecated | Use `apgi_gui/main.py` |
| `APGI_Application_GUI.py` | 300KB | Deprecated | Use `apgi_gui/main.py` |
| `Psychological_States_GUI.py` | 365KB | Deprecated | Use `apgi_gui/main.py` |
| `Assistant_GUI.py` | 359KB | Deprecated | Use `apgi_gui/main.py` |
| `Tests_GUI.py` | - | Deprecated | Use `apgi_gui/main.py` |
| `Utils_GUI.py` | - | Deprecated | Use `apgi_gui/main.py` |

### Utility Scripts

| Script | Type | Purpose |
|--------|------|---------|
| `setup.sh` | Installation | Environment setup and dependencies |
| `deploy.sh` | Deployment | Production deployment |
| `delete_pycache.py` | Maintenance | Cache cleanup |

## GUI Component Architecture (`apgi_gui/`)

The modern GUI uses a component-based architecture:

```
apgi_gui/
├── main.py              # Application entry point
├── mediator.py          # Component coordination
├── cli.py               # CLI interface
├── components/
│   ├── core.py          # Core component base
│   ├── control_panel.py # Experiment controls
│   ├── visualization_panel.py  # Data visualization
│   ├── menu_bar.py      # Application menus
│   └── status_bar.py    # Status indicators
└── controllers/
    └── simulation_controller.py  # Business logic
```

**Key Benefits:**
- Clean separation of concerns
- Reusable components
- Easier testing and maintenance
- Consistent UI patterns

## Utils Scripts

Each script is marked **✅ verified** (the file exists in `utils/`) or **🧭 planned** (listed in earlier docs but absent).
`python scripts/check_readme_status.py` enforces this.

### 1. CI/CD Pipeline (`utils/cicd_pipeline.py`) — 🧭 planned (file does not exist)

**Type**: Automation script
**Purpose**: Comprehensive CI/CD pipeline for testing, building, and deployment

### 2. Dependency Checker (`utils/check_dependencies.py`) — ✅ verified

**Type**: Validation script
**Purpose**: Check and validate all required dependencies

### 3. Performance Profiler (`utils/performance_profiler.py`) — ✅ verified

**Type**: Analysis tool
**Purpose**: Profile and analyze application performance

### 4. Data Validation (`utils/data_validation.py`) — ✅ verified

**Type**: Data quality tool
**Purpose**: Validate and assess data quality and integrity

### 5. Backup Manager (`utils/backup_manager.py`) — ✅ verified

**Type**: Utility script
**Purpose**: Automated backup and recovery system

### 6. Batch Processor (`utils/batch_processor.py`) — ✅ verified

**Type**: Processing tool
**Purpose**: Batch processing of experiments and data

### 7. Error Handler (`utils/error_handler.py`) — ✅ verified

**Type**: Error management
**Purpose**: Centralized error handling and reporting

### 8. Cache Manager (`utils/cache_manager.py`) — ✅ verified

**Type**: Performance utility
**Purpose**: Manage application cache and temporary data

### 9. Report Generator (`utils/report_generator.py`) — ✅ verified

**Type**: Documentation tool
**Purpose**: Generate comprehensive reports from test results

### 10. Sample Data Generator (`utils/sample_data_generator.py`) — ✅ verified

**Type**: Data utility
**Purpose**: Generate sample data for testing and development

### 11. GUI Testing Framework (`utils/gui_testing_framework.py`) — ✅ verified

**Type**: Testing tool
**Purpose**: Framework for testing GUI components and interactions

### 12. Parameter Validator (`utils/parameter_validator.py`) — ✅ verified

**Type**: Validation tool
**Purpose**: Validate framework parameters and configurations

### 13. Release Manager (`utils/release.py`) — ✅ verified

**Type**: Deployment tool
**Purpose**: Automated release management and versioning

### 14. Performance Dashboard (`utils/performance_dashboard.py`) — 🧭 planned (file does not exist)

**Type**: Monitoring tool
**Purpose**: Real-time performance monitoring dashboard

### 15. Data Quality Assessment (`utils/data_quality_assessment.py`) — ✅ verified

**Type**: Analysis tool
**Purpose**: Comprehensive data quality assessment and reporting

## APGI Framework Details

### Framework Overview

The APGI Framework is a comprehensive research platform for consciousness studies, neural signal processing, and adaptive intelligence systems. It provides tools for:

- **Neural Signal Processing**: EEG, ECG, pupillometry data analysis — ✅ verified (`apgi_framework/neural/eeg_processor.py`, `apgi_framework/neural/cardiac_processor.py`, `apgi_framework/neural/pupillometry_processor.py`)
- **Consciousness Research**: Threshold detection paradigms, phase transitions — ✅ verified (`apgi_framework/ignition/`, `apgi_framework/thermodynamic/`)
- **Adaptive Testing**: Bayesian parameter estimation, falsification testing — ✅ verified (`apgi_framework/adaptive/quest_plus_staircase.py`, `apgi_framework/falsification/`)
- **Machine Learning**: Classification tools, biomarker discovery — 🧭 planned (`apgi_framework/ml/` holds one integration module and no package `__init__.py`; classification lives in `apgi_framework/clinical/disorder_classification.py`)
- **Data Visualization**: Interactive dashboards, real-time monitoring — ✅ verified (`apgi_framework/gui/interactive_dashboard.py`, `apgi_framework/visualization/`)

### Framework Features

#### Core Capabilities

- Multi-modal physiological data processing — ✅ verified (`apgi_framework/neural/`)
- Real-time signal analysis and monitoring — ✅ verified (`apgi_framework/monitoring/`, `apgi_framework/gui/monitoring_dashboard.py`)
- Adaptive experimental paradigms — ✅ verified (`apgi_framework/adaptive/`)
- Comprehensive test coverage and validation — 🧭 planned (see the coverage note under Features)
- Modern GUI interfaces for experiment control — ✅ verified (`apgi_gui/main.py`)

### Research Applications

- Consciousness threshold detection — ✅ verified (`apgi_framework/ignition/`)
- Cross-species validation studies — ✅ verified (`apgi_framework/research/cross_species_validation.py`)
- Clinical biomarker identification — 🧭 planned (research notes in `docs/research/clinical_biomarkers.md`; the code only extracts model parameters and classifies states, and is not validated for clinical use)
- Neural dynamics analysis — ✅ verified (`apgi_framework/neural/oscillations.py`, `apgi_framework/neural/microstate_analysis.py`)
- Active inference modeling — ✅ verified (`tests/unit/test_active_inference.py`)

### CLI Usage

```bash
# Run tests (✅ verified subcommand: run-tests; the earlier `test --all` form does not exist)
python -m apgi_framework.cli run-tests

# Run with coverage
pytest --cov=apgi_framework

# Run specific categories
pytest -m unit
pytest -m integration
pytest -m gui
```

### GUI Usage

```bash
python GUI-Launcher.py          # ✅ verified (deprecated legacy launcher; prefer python -m apgi_gui.main)
python GUI.py                   # 🧭 planned: no such file
python GUI-Experiment-Registry.py   # 🧭 planned: no such file
```

### Framework Development Installation

```bash
git clone https://github.com/apgiframework/apgi-system.git   # ✅ verified URL (the old apgi-research/apgi-framework URL is not this repository)
cd apgi-system
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# GUI components (✅ verified extra)
pip install -e ".[gui]"

# Machine learning tools (✅ verified extra)
pip install -e ".[ml]"

# Neural signal processing (✅ verified extra)
pip install -e ".[neural]"

# All optional dependencies (✅ verified extra)
pip install -e ".[all]"
```

## Framework Quick Start

### Framework Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # ✅ verified
pytest tests/integration/   # ✅ verified
pytest tests/acquisition/   # ✅ verified
pytest tests/performance/   # 🧭 planned: directory does not exist (see benchmarks/)

# Run with coverage
pytest --cov=apgi_framework --cov-report=html
```

### Using the Framework

```python
# 🧭 planned: `APGIAgent` and `DataProcessor.analyze_eeg` do not exist in the current code and this will not run.
# For a working example see "Option 3: Programmatic Usage" above (`apgi_framework.system.APGISystem`, ✅ verified).
from apgi_framework import APGIAgent
from apgi_framework.data import DataProcessor

agent = APGIAgent(config_file="config.yaml")
processor = DataProcessor()
results = processor.analyze_eeg(data_file="eeg_data.csv")
```

### Framework GUI Applications

```bash
# Launch the main GUI
python GUI-Launcher.py

# Interactive dashboard (✅ verified: module has a __main__ block; the `apgi-dashboard` console script is 🧭 planned, its target `main` is missing)
python -m apgi_framework.gui.interactive_dashboard
```

### Framework Project Structure

```text
apgi-experiments/
├── apgi_framework/          # Core framework code
│   ├── adaptive/            # Adaptive testing modules
│   ├── analysis/            # Analysis tools
│   ├── clinical/            # Clinical applications
│   ├── config/              # Configuration management
│   ├── gui/                 # GUI components
│   ├── neural/              # Neural signal processing
│   └── testing/            # Test framework
├── tests/                   # Test suite (✅ verified)
├── examples/                # Usage examples (✅ verified)
├── docs/                    # Documentation (✅ verified)
├── research/                # Research modules (🧭 planned: no such top-level directory; see apgi_framework/research/)
└── utils/                   # Utility scripts (✅ verified)
```

## Documentation

- [API Documentation](docs/REST-API.md) — ✅ verified
- [Examples](docs/examples/) — ✅ verified
- [Developer Guide](docs/developer/) — ✅ verified
- [Research Applications](docs/research/) — ✅ verified

## Testing

The framework includes comprehensive testing:

- **Unit Tests**: Fast isolated tests for individual components — ✅ verified (`tests/unit/`)
- **Integration Tests**: Tests for component interactions — ✅ verified (`tests/integration/`)
- **Performance Tests**: Benchmarks and profiling — 🧭 planned (benchmarks live in `benchmarks/`; no `tests/performance/`)
- **GUI Tests**: User interface validation — ✅ verified (`-m gui`)
- **Research Tests**: Domain-specific validation — ✅ verified (`tests/test_cross_species_validation.py`)

Run tests with:

```bash
# All tests
pytest

# With coverage
pytest --cov=apgi_framework --cov-report=html

# Specific categories
pytest -m unit
pytest -m integration
pytest -m performance   # 🧭 planned: marker is not registered, so --strict-markers rejects it
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

<!-- identifiers:begin (generated by scripts/sync_identifiers.py; do not edit by hand) -->
| Identifier | Value |
|------------|-------|
| DOI | [10.5281/zenodo.21632264](https://doi.org/10.5281/zenodo.21632264) |
| Author ORCID | [0009-0005-0328-2896](https://orcid.org/0009-0005-0328-2896) |
| Repository | <https://github.com/apgiframework/apgi-system> |
| Version | 1.1.0 (released 2026-10-17) |

```bibtex
@software{pesochin_apgi_system_2026,
  title={APGI System: Allostatic Precision-Gated Ignition — testing and simulation platform},
  author={Pesochin, Dimitry},
  year={2026},
  version={1.1.0},
  doi={10.5281/zenodo.21632264},
  url={https://github.com/apgiframework/apgi-system},
  orcid={https://orcid.org/0009-0005-0328-2896}
}
```
<!-- identifiers:end -->

The experimental protocols in [`apgi_framework/protocols/`](apgi_framework/protocols/) follow APGI protocol specification
1.1.0, byte-identical to [apgi-framework](https://github.com/apgiframework/apgi-framework)
1.1.0. Validate them with `make protocols-check`; see [`apgi_framework/protocols/CHANGELOG.md`](apgi_framework/protocols/CHANGELOG.md)
for what changed from 1.0. Under 1.1.0 a failure to confirm a prediction is not falsification
(verdicts are CONFIRMED, FALSIFIED or INDETERMINATE; conflicting evidence is INDETERMINATE), and κ (ATP/bit) is
uncalibrated. Loading the specification does not by itself show that the existing simulation parameters implement
the registered reduced model; that mapping is not yet established.

### 🖥️ GUI Applications

#### Main GUI Applications

- GUI.py - Main comprehensive GUI application (244KB)
- GUI-Launcher.py - Centralized GUI launcher for all applications

#### Apps Directory GUIs

- apps/apgi_falsification_gui.py - Falsification testing GUI (74KB)
- apps/apgi_falsification_gui_refactored.py - Refactored falsification GUI
- apps/gui_template.py - GUI template (241KB)

#### Framework GUI Components

- apgi_framework/gui/coverage_visualization.py - Test coverage visualization
- apgi_framework/gui/enhanced_monitoring_dashboard.py - Enhanced monitoring dashboard
- apgi_framework/gui/interactive_dashboard.py - Interactive dashboard
- apgi_framework/gui/monitoring_dashboard.py - Basic monitoring dashboard
- apgi_framework/gui/parameter_estimation_gui.py - Parameter estimation GUI
- apgi_framework/gui/progress_monitoring.py - Progress monitoring GUI
- apgi_framework/gui/reporting_visualization.py - Reporting visualization
- apgi_framework/gui/results_viewer.py - Results viewer GUI
- apgi_framework/gui/session_management.py - Session management GUI
- apgi_framework/gui/task_configuration.py - Task configuration GUI

### ⌨️ CLI Interfaces

#### Primary CLI Entry Points

- apgi_framework/cli.py - Main command-line interface
- apgi_framework/main.py - Module entry point (python -m apgi_framework)
- apgi_framework/deployment/cli.py - Deployment CLI
- apgi_framework/validation/diagnostics_cli.py - Diagnostics CLI

#### CLI Tools

- tools/run_tests.py - Test runner CLI

### 🔧 Standalone Tools & Scripts

#### Setup & Deployment

- apgi_setup.py - Installation and setup script
- setup.sh - Unix/Linux setup script
- deploy.bat - Windows deployment script
- deploy.sh - Unix/Linux deployment script
- quick_deploy.py - Quick deployment tool
- delete_pycache.py - Cache cleanup utility

#### Analysis & Processing

- examples/data_loader.py - Data loading utility
- examples/coverage_collector_demo.py - Coverage collection demo

### 🧪 Testing Suites

#### Main Test Runner

- run_tests.py - Comprehensive test runner with GUI integration

#### Test Categories

- Unit Tests - Fast, isolated component tests
- Integration Tests - Cross-component integration tests
- Framework Tests - Core framework functionality tests
- Falsification Tests - Falsification testing validation
- GUI Tests - GUI component testing
- Performance Tests - Benchmark and performance tests
- Research Tests - Domain-specific experiment tests

#### Test Configuration

- pytest.ini - Pytest configuration (61 lines)
- pyproject.toml - Modern testing configuration (225 lines)

#### Key Test Files

- tests/test_gui_components.py - GUI component tests
- tests/test_cli_module.py - CLI module tests
- tests/test_falsification_coverage.py - Falsification coverage tests
- tests/framework/ - Framework-specific tests (9 files)
- tests/falsification/ - Falsification tests (7 files)
- tests/integration/ - Integration tests (4 files)

### 🚀 Module Entry Points

#### Framework Module

- python -m apgi_framework - Main framework module execution

#### Package Entry Points

- `apgi_framework/__init__.py` - Package initialization

## Quick Start (5 minutes)

**New to the system?** See the [Quick Start Guide](docs/QUICK-START-GUIDE.md) for a 5-minute introduction.

**Using GUI?** Check out the [GUI Visual Guide](docs/GUI-VISUAL-GUIDE.md) for a visual walkthrough.

**Using Web Interface?** Visit [APGI-Experiments.html](../../apgi-web/APGI-Experiments.html) for interactive web-based experiments and visualizations.

**Need help?** See the [How to Use APGI](#how-to-use-apgi) section and [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

### Installation Guide

1. **Set up the environment**

   ```bash
   # Create and activate virtual environment
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Unix/macOS
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Validate installation**

   ```bash
   python -m apgi_framework.cli validate-system
   ```

3. **Run the APGI Agent example**

   ```bash
   python -m core.models.apgi_agent
   ```

4. **Run the Somatic Agent example**

   ```bash
   python -m core.models.phase_transition
   ```

5. **Run experiments via GUI**

   ```bash
   python GUI-Launcher.py
   ```

## Currently Implemented

### Core Models

1. **APGI Agent** (`core/models/apgi_agent.py`)
   - Implements the core APGI framework with interoceptive and exteroceptive processing
   - Dynamic threshold mechanism for conscious access
   - Precision-weighted prediction errors
   - Somatic marker integration

2. **Phase Transition Model** (`core/models/phase_transition.py`)
   - Somatic marker-based decision making
   - Conscious vs unconscious processing modes
   - Expected free energy calculations

3. **Base Experiment Framework** (`core/experiment.py`)
   - Abstract base class for all experiments
   - Standardized data collection and analysis

### Implemented Experiments

1. **Interoceptive Gating Paradigm** (`experiments/interoceptive_gating/`)
   - Tests how interoceptive precision gates conscious access
   - Cardiac discrimination task simulation
   - Three conditions: interoceptive focus, exteroceptive focus, control
   - Threshold tracking and detection rate analysis

2. **AI Benchmarking** (`experiments/ai_benchmarking/`)
   - Compares different agent architectures in survival environments
   - Includes Random, Reactive, DQN, and APGI agents
   - Grid world environment with food, obstacles, and predators
   - Performance metrics: survival time, energy efficiency, food consumption

3. **GUI Interface** (`GUI-Launcher.py`)
   - Tkinter-based interface for running experiments
   - Parameter configuration and real-time logging
   - Supports all implemented experiments

4. **Web Interface** (`../../apgi-web/APGI-Experiments.html`)
   - Interactive web-based experiments and visualizations
   - Modern responsive design with real-time particle animations
   - Neural network visualizations and interactive parameter controls
   - Cross-platform accessibility through web browser

## Key Features

- **Modular Architecture**: Clean separation between models, experiments, and utilities
- **Standardized Experiments**: All experiments inherit from `BaseExperiment` for consistency
- **Multiple Interfaces**: Command-line, GUI, web interface, and programmatic access
- **Data Management**: Automatic data saving and visualization
- **Extensible Design**: Easy to add new models and experiments
- **Web-Based Visualization**: Interactive experiments with real-time neural animations

## Example Usage

### Running Falsification Tests

**Using the GUI:**

```bash
python GUI-Launcher.py
```

**Using the CLI:**

```bash

# Run primary falsification test

python -m apgi_framework.cli run-test primary --trials 1000

# Run all tests

python -m apgi_framework.cli run-batch --all-tests
```

**Using Python API:**

```python
from apgi_framework.main_controller import MainApplicationController

# Initialize system

controller = MainApplicationController()
controller.initialize_system()

# Run test

tests = controller.get_falsification_tests()
result = tests['primary'].run_test(n_trials=1000)
print(f"Falsified: {result.is_falsified}")
print(f"Confidence: {result.confidence_level:.2f}")

# Cleanup

controller.shutdown_system()
```

## Running the APGI Agent

```python
from core.models.apgi_agent import APGIAgent

# Create and run agent with default parameters

agent = APGIAgent()
agent.run_example()  # Runs simulation and shows plots
```

### Running an Experiment

```python
from research.interoceptive_gating.experiments.experiment import run_interoceptive_gating_experiment

# Run experiment with custom parameters

experiment = run_interoceptive_gating_experiment(
    n_participants=20,
    n_trials_per_condition=100
)
```

## Complete Documentation

Complete documentation is available in the `docs/` directory and web interface:

- **[APGI-Experiments.html](../../apgi-web/APGI-Experiments.html)** - Interactive web-based experiments and visualizations
- **[Quick Start Guide](docs/QUICK-START-GUIDE.md)** - Get started in 5 minutes
- **[GUI Visual Guide](docs/GUI-VISUAL-GUIDE.md)** - Visual walkthrough of GUI
- **[User Guide](docs/USER-GUIDE.md)** - Complete user manual
- **[CLI Reference](docs/CLI-REFERENCE.md)** - Command-line interface documentation
- **[Results Interpretation Guide](docs/RESULTS-INTERPRETATION_GUIDE.md)** - Understanding test results
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[How-To Guide](docs/HOW-TO-USE-APGI.md)** - Complete usage guide (also included above)

Browse the [`docs/`](docs/) directory for the full set of guides.
