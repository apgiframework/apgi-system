# APGI Assistant GUI - Comprehensive Documentation

## Overview

The APGI Assistant GUI is a production-ready graphical interface for Allostatic Precision-Gated Ignition (APGI) Assistant system. This application provides a comprehensive cognitive AI interface that combines advanced neural network processing, physiological state monitoring, biofeedback integration, real-time visualization, and professional export capabilities.

### Core Features

- **Cognitive State Tracking**: Real-time monitoring of AI cognitive states including confusion, focus, alertness, and processing modes
- **Physiological Integration**: Heart rate, heart rate variability (HRV), respiration breathing rate, and electrodermal activity monitoring
- **Biofeedback System**: Automated recommendations based on physiological state analysis
- **Energy Management**: Computational efficiency monitoring with battery-like energy tracking
- **Oscillatory Analysis**: Brain-like frequency pattern analysis and visualization
- **Professional UI/UX**: Multi-tabbed interface with lazy loading, progress indicators, and comprehensive error handling
- **Session Management**: Save/load sessions, configuration persistence, and history tracking
- **Advanced Export System**: Professional-grade export in PNG, JPG, PDF, SVG, CSV, and JSON formats
- **Dark Mode Theme Support**: Complete theme system with normal and high contrast modes
- **Advanced Keyboard Customization**: Comprehensive shortcut system with categorized key bindings
- **Visualization Tools**: Real-time plots, state timelines, and energy usage graphs
- **Advanced Neural Architecture**: Liquid Time-Constant Networks with Neural ODE integration
- **Structured Error Handling**: Comprehensive ErrorContext system for debugging and user feedback
- **Coherent Language Processing**: Context-aware response generation with query relevance
- **Robust Initialization**: Fixed infinite loop issues with proper progress dialog management

### Recent Improvements (v2.0+)

#### Language Processing Fixes

- **Response Coherence**: Fixed nonsensical responses by enabling language model integration
- **Query Context**: User input now properly incorporated into response generation
- **Intelligent Fallbacks**: Context-aware responses for specific query types (weather, jokes, technical, philosophical)
- **Language Model Integration**: Proper transformers integration with GPT-2 support

#### Initialization Fixes

- **Progress Dialog Cleanup**: Fixed stuck dialogs with proper frame destruction
- **State Synchronization**: Enhanced initialization state management
- **Emergency Cleanup**: Added safety mechanisms for stuck UI elements
- **Error Handling**: Improved debugging information for initialization issues

#### Performance Enhancements

- **Threshold Adaptation**: Fixed zero variance issues with proper history tracking
- **Surprise Sensitivity**: Enhanced correlation measurements with biologically plausible variation
- **Memory Management**: Optimized history tracking with automatic pruning
- **Energy Efficiency**: Improved computational cost monitoring

**Features advanced components:**

- Neural ODE integration with torchdiffeq (with Euler fallback)
- LinOSS (Linear Oscillator State-Space) decomposition
- Dual processing pathways (conscious/unconscious)
- Energy budget management and battery monitoring
- Precision estimation and surprise accumulation
- Optional language model integration (transformers)
- Biofeedback integration with physiological signal processing
- Research-grade implementation with proper fallbacks

**Library/module version** - designed to be imported by other applications
Suitable for server-side or headless deployments

**GUI features:**

- Multi-tabbed interface with lazy loading and progress indicators
- Real-time cognitive state visualization and monitoring
- History management with automatic pruning and memory optimization
- Professional export system (PNG, JPG, PDF, SVG, CSV, JSON)
- PDF report generation (reportlab integration)
- Structured error handling with ErrorContext system
- Safe widget operations with decorators
- Performance monitoring and optimization
- Dark mode theme support with high contrast option
- Comprehensive keyboard shortcut system
- Session management and configuration persistence
- Biofeedback dashboard with physiological monitoring
- Energy usage tracking and oscillatory analysis
- Production-ready GUI with robust error handling and user experience features

## Architecture

The primary application class that orchestrates all GUI components and manages the application lifecycle.

**Key Responsibilities:**

- Initialize and manage the tkinter GUI
- Handle assistant initialization with timeout protection
- Coordinate between UI threads and processing threads
- Manage state persistence and configuration
- Provide error handling and user feedback

### Core Neural Architecture

- **Liquid Time-Constant Networks**: Continuous-time neural dynamics with adaptive time constants
- **Neural ODE Integration**: Proper differential equation solving with torchdiffeq (fallback to Euler)
- **APGI-LFM2**: Allostatic Precision-Gated Ignition with Liquid Fractional Memory
- **Multi-Level Entropy**: Thermodynamic, Shannon, and variational entropy calculations
- **Surprise-Driven Gating**: Dynamic threshold control based on prediction error
- **LinOSS Decomposition**: Linear Oscillator State-Space decomposition for frequency analysis

#### 3. Threading Architecture

The application uses a multi-threaded design to maintain UI responsiveness:

- **Main Thread**: GUI updates and user interactions
- **Initialization Thread**: Assistant setup with timeout handling
- **Processing Thread**: Query processing with cancellation support
- **Update Thread**: Periodic display updates with debouncing

#### 4. Advanced Utility Classes

**DebouncedUpdater**: Prevents excessive GUI updates by batching rapid requests with operation-specific delays

**CancellableProgress**: Provides aggressive cancellation support for long operations with callback propagation

**ActionHistory**: Implements undo/redo functionality using Command pattern

**InputValidator**: Validates user inputs and physiological parameters

**UsageTracker**: Tracks usage patterns locally for analytics with automatic memory management

**ErrorContext**: Structured error reporting with timing and context information

**safe_widget_method**: Decorator for safe widget operations with existence validation

**DependencyNotifier**: User-facing notifications for optional dependency limitations

**PermissionValidator**: Enhanced file access validation with user guidance

**HistoryManager**: Automatic memory management with configurable limits and pruning

## User Interface

### Tab Structure

The application uses a tabbed interface with lazy loading for optimal performance:

#### Main Interface Tab

**Purpose**: Primary query processing and response display

**Components:**

- **Query Input Panel**:

  - Scrolled text widget with character counting (max 10,000 characters)
  - Real-time character counter with color coding
  - Keyboard shortcut support (Ctrl+Enter to process)

- **Physiological State Controls**:
  - Heart Rate slider (40-200 bpm)
  - HRV slider (10-200 ms)
  - Respiration slider (8-40 bpm)
  - Skin Conductance slider (0.5-20 µS)
  - Quick preset buttons (Relaxed, Normal, Stressed, Anxious)

- **Response Display**:
  - Formatted text display with tags for headings and normal text
  - Structured response sections (Answer, Explanation, Biofeedback)
  - Real-time cognitive state indicators

- **Cognitive State Display**:
  - Primary state indicator
  - Processing mode
  - Confidence levels
  - Surprise levels
  - Dominant frequency
  - Coherence metrics
  - Energy cost tracking

#### Cognitive Monitoring Tab

**Purpose**: Real-time cognitive state tracking and history

**Components:**

- **State Visualization Canvas**: Visual representation of current cognitive state with intensity rings

- **State Metrics Display**: Detailed metrics including attention allocation and oscillatory profiles

- **State History Tree**: Tabular history of cognitive states with timestamps and confidence levels

#### Oscillatory Analysis Tab

**Purpose**: Brain-like frequency pattern analysis

**Components:**

- **Power Spectrum Display**: Real-time matplotlib visualization of frequency band power

- **Frequency Distribution**: Band-specific power analysis with color coding

- **Oscillatory Metrics**: Detailed numerical metrics for coherence, entropy, and dominant frequencies

#### Biofeedback Tab

**Purpose**: Physiological state monitoring and recommendations

**Components:**

- **Baseline Calibration**: Manual calibration of resting physiological states

- **Current Physiology Display**: Real-time bar charts of physiological parameters

- **Recommendations Panel**: AI-generated biofeedback recommendations

- **Physiology History**: Historical trends of physiological data

#### Energy Management Tab

**Purpose**: Computational efficiency monitoring

**Components:**

- **Battery Status Display**: Visual battery indicator with percentage

- **Energy Usage History**: Time-series plot of energy consumption

- **Energy Statistics**: Detailed metrics on energy efficiency and budget

#### Performance Tab

**Purpose**: System performance metrics and query history

**Components:**

- **Performance Metrics**: Response times, surprise statistics, ignition statistics

- **Query History Tree**: Historical record of all queries with timestamps and states

#### Visualizations Tab

**Purpose**: Generate and display analytical plots

**Components:**

- **Control Panel**: Buttons for generating various plots

- **Visualization Display**: Canvas for displaying matplotlib figures

- **Export Functions**: Save plots to files

#### Settings Tab

**Purpose**: Configure assistant parameters and system settings

**Components:**

- **Model Configuration**: Input dimension, hidden dimension settings

- **Processing Options**: Adaptive processing, energy-aware computation toggles

- **System Information**: Dependency status and resource usage

- **Action Buttons**: Apply settings, reset to defaults, reset assistant

## Core Functionality

### Query Processing Pipeline

1. **Input Validation**:

   ```python
   # Validates query text and physiological parameters
   is_valid, errors = InputValidator.validate_query(query)
   is_valid, errors = InputValidator.validate_physiological(**physio_data)
   ```

2. **Processing Thread**:
   - Creates cancellable progress dialog
   - Processes query in background thread
   - Handles cancellation requests
   - Updates progress messages

3. **Response Handling**:
   - Formats response with structured sections
   - Updates cognitive state displays
   - Tracks usage metrics
   - Adds to history

4. **Visualization Updates**:
   - Debounced updates to prevent excessive refreshes
   - Thread-safe flag system for update coordination
   - Lazy loading of visualization components

### State Management

#### Cognitive States

The system tracks multiple cognitive dimensions:

- **Primary States**: confused, focused, alert, idle, working
- **Processing Modes**: analytical, creative, reactive, predictive
- **Confidence Levels**: Numeric and categorical confidence measures
- **Surprise Levels**: Response to unexpected inputs
- **Oscillatory Profiles**: Frequency-based brain-like patterns

#### Physiological States

Real-time monitoring of:

- **Heart Rate (HR)**: 40-200 bpm range
- **Heart Rate Variability (HRV)**: 10-200 ms range
- **Respiration Rate**: 8-40 bpm range
- **Electrodermal Activity (EDA)**: 0.5-20 µS range

#### Energy Management

- **Battery-like tracking**: 0-100% energy levels
- **Consumption monitoring**: Per-query energy usage
- **Efficiency metrics**: Energy per response ratio
- **Budget management**: Configurable energy limits

### Advanced Export System

#### Export Formats and Capabilities

The application provides comprehensive export functionality:

**Image Exports:**

- **PNG**: High-quality raster images with configurable DPI (150/300/600)
- **JPG**: Compressed images for web use
- **PDF**: Vector format for publications with metadata
- **SVG**: Scalable vector graphics for web

**Data Exports:**

- **CSV**: Structured data for spreadsheet analysis
  - Session data (queries, responses, timestamps)
  - Performance metrics (response times, accuracy)
  - Physiological data (HR, HRV, respiration, EDA)
  - Energy usage history
- **JSON**: Complete session export with full metadata
  - Session metadata and configuration
  - Complete query and state histories
  - Performance analytics
  - System information

**PDF Reports:**

- Professional reports with structured sections
- Executive summaries and performance analytics
- Tables and charts with proper formatting
- Metadata and timestamps

#### Export Features

- **Quality Settings**: Low (150 DPI), Medium (300 DPI), High (600 DPI)
- **Batch Export**: Save all plots simultaneously
- **Permission Validation**: Enhanced file access checking with user guidance
- **Progress Indicators**: Real-time export progress
- **Error Handling**: Comprehensive error context and recovery

### Theme System

#### Dark Mode Support

Complete theme implementation with:

- **Normal Theme**: Light theme with professional color palette
- **High Contrast Theme**: Dark theme optimized for accessibility
- **Dynamic Switching**: Real-time theme updates without restart
- **Color Mapping**: Comprehensive color definitions for all UI elements
- **State Colors**: Theme-appropriate colors for cognitive states
- **Export Integration**: Theme-aware plot generation

#### Theme Components

- **Background/Foreground**: Primary text and background colors
- **Accent Colors**: Highlight and interaction colors
- **Status Colors**: Success, error, warning, info indicators
- **State Colors**: Cognitive state visualization colors
- **Battery Colors**: Energy level indicators
- **Canvas Colors**: Plot and visualization backgrounds

### Advanced Keyboard Customization

#### Comprehensive Shortcut System

Categorized keyboard shortcuts for all major functions:

**File Operations:**

- Ctrl+N: New session
- Ctrl+S: Save session
- Ctrl+O: Load session
- Ctrl+Q: Exit application

**Advanced Export:**

- Ctrl+Shift+C: Export session as CSV
- Ctrl+Shift+M: Export metrics as CSV
- Ctrl+Shift+J: Export complete session as JSON
- Ctrl+Shift+P: Export session report as PDF

**Edit Operations:**

- Ctrl+Z: Undo
- Ctrl+Shift+Z: Redo
- Ctrl+C: Copy
- Ctrl+V: Paste
- Ctrl+X: Cut
- Ctrl+A: Select all
- Ctrl+L: Clear query

**View Operations:**

- F1: Show help
- F5: Refresh displays
- F11: Toggle fullscreen
- Ctrl+Plus/Minus/0: Zoom controls

**Navigation:**

- Ctrl+Tab/Ctrl+Shift+Tab: Tab navigation
- Ctrl+1-9: Direct tab switching
- Alt+Left/Right: Back/Forward navigation

**Processing:**

- Ctrl+Enter: Process query
- Ctrl+R: Reset assistant
- Ctrl+Space: Auto-complete
- Ctrl+Shift+P: Command palette

**Theme:**

- Ctrl+Alt+H: Toggle high contrast mode

#### Shortcut Management

- **Interactive Dialog**: Searchable shortcut reference
- **Categorized Display**: Organized by functionality
- **Real-time Updates**: Immediate shortcut registration
- **Customizable**: Extensible for new features

### Biofeedback System

The biofeedback system provides personalized recommendations based on physiological state analysis:

#### Analysis Pipeline

1. **Baseline Comparison**: Compare current state to calibrated baseline
2. **Stress Assessment**: Evaluate stress levels from physiological patterns
3. **Recommendation Generation**: AI-generated suggestions for state optimization
4. **Real-time Updates**: Continuous monitoring and adjustment

#### Recommendation Types

- **Relaxation techniques**: Breathing exercises, meditation suggestions
- **Focus enhancement**: Environmental adjustments, timing recommendations
- **Energy management**: Break suggestions, activity modifications
- **Stress reduction**: Immediate interventions and long-term strategies

### Advanced Neural Architecture

#### Liquid Time-Constant Networks

The core neural architecture implements continuous-time dynamics:

```python
class LiquidTimeConstantLayers(nn.Module):
    """Proper LTC implementation with Neural ODE integration"""
    
    def forward(self, x):
        # ODE: dx/dt = (1/τ) * (-x + σ(W·input + b))
        return odeint(self.ode_func, x, t)
```

#### APGI-LFM2 Architecture

- **Dual-Pathway Processing**: Separate analytical and creative pathways
- **Surprise-Driven Gating**: Dynamic threshold control
- **Precision Estimation**: Adaptive confidence weighting
- **Global Workspace**: Consciousness simulation with metabolic costs

### Error Handling and Recovery

#### Enhanced Error Context System

The application uses a comprehensive `ErrorContext` system that provides:

- **Structured Error Reporting**: Each error logged with operation context and timing
- **User-Facing Messages**: Clear contextual error messages when `user_facing=True`
- **Performance Timing**: Elapsed time tracking for all operations
- **Consistent Logging**: Unified error format across the application

#### Multi-Level Error Handling

1. **Input Validation**: Prevent invalid data entry with `InputValidator`
2. **Processing Errors**: Graceful handling of assistant failures
3. **UI Errors**: Widget existence validation with `safe_widget_method` decorator
4. **System Errors**: Comprehensive logging and user feedback

#### Recovery Mechanisms

- **Automatic Retry**: For transient failures
- **Assistant Reset**: Clear corrupted state and reinitialize
- **Configuration Restore**: Fallback to default settings
- **Error Pattern Tracking**: Monitor error patterns and suggest solutions

### Session Management

#### Session Persistence

- **Automatic Saving**: Periodic configuration saves
- **Manual Save/Load**: User-controlled session management
- **Configuration Export/Import**: Share settings between instances
- **History Preservation**: Maintain query and response history

#### Data Storage

**Configuration File**: JSON file in user home directory
**Session Files**: Timestamped JSON files with full session state
**Log Files**: Rotating log files with debug information (`apgi_gui.log`)
**History**: In-memory storage with configurable limits

#### Thread-Safe Operations

All session operations use thread-safe mechanisms:

- **Lock-based coordination** for shared state access
- **Atomic operations** for configuration updates
- **Error context** for all file operations

### Performance Optimization

#### Lazy Loading

Tabs are created on-demand to reduce startup time and memory usage:

```python
def on_tab_changed(self, event):
    current_tab = self.notebook.select()
    tab_name = self.notebook.tab(current_tab, 'text')
    
    if tab_name not in self.tabs_created:
        # Create tab content only when first accessed
        if tab_name == 'Cognitive Monitoring':
            self.create_cognitive_monitoring_tab()
```

#### Debounced Updates

Rapid updates are batched to prevent UI freezing:

```python
class DebouncedUpdater:
    def schedule_update(self, root, update_id, callback):
        # Cancel existing update
        # Schedule new update with delay
```

#### Thread Safety

All GUI updates are coordinated through thread-safe mechanisms:

- **Lock-based coordination** for shared state
- **Event-driven updates** using tkinter events
- **Atomic flag operations** for update coordination

### Accessibility Features

#### Keyboard Navigation

- **Tab switching**: Ctrl+Tab/Ctrl+Shift+Tab
- **Menu access**: Alt+key combinations
- **Shortcut keys**: Comprehensive keyboard shortcuts
- **Focus management**: Visible focus indicators

## Usage Guide

### Getting Started

1. **Launch Application**: Run `python APGI_GUI.py`
2. **Wait for Initialization**: Assistant initializes automatically with timeout protection
3. **Enter Query**: Type your question in the main interface (max 10,000 characters)
4. **Set Physiology**: Adjust sliders or use presets (Relaxed, Normal, Stressed, Anxious)
5. **Process Query**: Click "Process Query" or press Ctrl+Enter
6. **Review Results**: Examine response and cognitive state indicators

### Advanced Features

#### Biofeedback Calibration

1. **Relax for 2-3 minutes** in a quiet environment
2. **Click "Calibrate Baseline"** in the Biofeedback tab
3. **Follow on-screen instructions** for accurate calibration
4. **Verify calibration status** shows "Calibrated ✓"

#### Visualization Generation

1. **Accumulate data** through several queries
2. **Navigate to Visualizations tab**
3. **Select desired plot type** (State Timeline, Energy Plot, Oscillatory Spectrum)
4. **Generate and save plots** as PNG/JPG/SVG files

#### Session Operations

1. **Save Session**: File → Save Session (Ctrl+S)
2. **Load Session**: File → Load Session (Ctrl+O)
3. **Export Configuration**: File → Export Configuration
4. **Import Configuration**: File → Import Configuration

#### Advanced Usage

For advanced users and researchers:

1. **Direct API Usage**: Import APGI-Assistant.py as a library
2. **Custom Configuration**: Modify model parameters in Settings tab
3. **Batch Processing**: Use session management for multiple queries
4. **Export Analysis**: Save cognitive state data for external analysis

### Troubleshooting

#### Common Issues

**Assistant Initialization Fails**:

- Check system resources (memory, CPU)
- Verify dependency installation: `pip install -r requirements.txt`
- Review log files: `apgi_gui.log` for detailed errors
- Try resetting assistant: Tools → Reset Assistant (Ctrl+R)
- Check torchdiffeq availability for ODE integration

**Visualizations Not Working**:

- Install matplotlib: `pip install matplotlib`
- Install PIL/Pillow: `pip install Pillow`
- Check system graphics drivers
- Verify matplotlib backend compatibility

**Performance Issues**:

- Close unnecessary tabs to reduce memory usage
- Reduce update frequency in Settings tab
- Monitor energy usage in Energy tab
- Check system resource usage with psutil
- Disable advanced features if needed

#### Error Recovery

1. **Check Logs**: Help → View Logs
2. **Reset Assistant**: Tools → Reset Assistant
3. **Clear History**: Edit → Clear History
4. **Restart Application**: Complete fresh start

## Development and Extension

### Code Structure

#### Main Files

- **APGI_GUI.py**: Primary GUI application (v2.0) - Main entry point
- **APGI-Assistant.py**: Core assistant logic with LTC networks
- **requirements.txt**: Dependency specifications
- **pyproject.toml**: Code formatting and linting configuration
- **README.md**: This documentation
- **delete_pycache.py**: Utility script for cleaning Python cache files

#### Key Classes

- **APGIGUI**: Main application controller
- **DebouncedUpdater**: Update batching utility
- **CancellableProgress**: Progress dialog with cancellation
- **ActionHistory**: Undo/redo functionality
- **InputValidator**: Input validation utilities

### Extension Points

#### Adding New Tabs

1. **Create tab frame** in `create_widgets()`
2. **Implement tab creation method** (e.g., `create_new_tab()`)
3. **Add tab to lazy loading** in `on_tab_changed()`
4. **Update navigation** and help documentation

#### Custom Visualizations

1. **Extend APGIVisualizer class**
2. **Add plot generation methods**
3. **Integrate with visualization tab**
4. **Add save/export functionality**

#### Additional Metrics

1. **Define new metric collection** in assistant
2. **Add display widgets** to appropriate tabs
3. **Implement update methods** with thread safety
4. **Add to performance tracking**

## Appendix C: Keyboard Shortcuts Reference

| Shortcut | Action |

| **Ctrl+N** | New session |
| **Ctrl+S** | Save session |
| **Ctrl+O** | Load session |
| **Ctrl+Q** | Exit application |
| **Ctrl+Shift+C** | Export session as CSV |
| **Ctrl+Shift+M** | Export metrics as CSV |
| **Ctrl+Shift+J** | Export session as JSON |
| **Ctrl+Shift+P** | Export report as PDF |
| **Ctrl+L** | Clear query input |
| **Ctrl+,** | Open settings |
| **Ctrl+Z** | Undo last action |
| **Ctrl+Enter** | Process query |
| **Ctrl+R** | Reset assistant |
| **Ctrl+Plus** | Increase font size |
| **Ctrl+Minus** | Decrease font size |
| **Ctrl+0** | Reset font size |
| **Ctrl+Tab** | Next tab |
| **Ctrl+Shift+Tab** | Previous tab |
| **F1** | Quick start guide |
| **F5** | Refresh displays |
| **F11** | Toggle fullscreen |
| **Ctrl+Alt+H** | Toggle high contrast theme |

## Future Features

- 🔄 Advanced Analytics Dashboard - Could be expanded for deeper insights
- 🔄 Voice Input/Output - Potential accessibility enhancement
- 🔄 Screen reader compatibility - Additional accessibility support
- 🔄 High DPI scaling optimization - Visual enhancement
- 🔮 AI-powered usage pattern analysis
- 🔮 Plugin system for custom extensions
- 🔮 Cloud synchronization for sessions
- 🔮 Advanced collaboration features
