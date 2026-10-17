# APGI Application GUI - Comprehensive Test Checklist

> **Note**: This checklist reflects the actual implementation in `@/Users/apgiframework/Sites/PYTHON/apgi-system/APGI_Application_GUI.py`.
> **Test Status**: ✅ VERIFIED (Based on static analysis and runtime testing)

## Overview

The `APGI_Application_GUI.py` is an **APGI Application GUI** - a modern graphical interface built with customtkinter featuring a sidebar with APGI parameters, falsification test controls, console output, and matplotlib visualizations. Includes comprehensive error handling, keyboard shortcuts, and theme management.

## Dependencies Check

- [x] **customtkinter**: Modern tkinter alternative (ctk.CTk main window)
- [x] **tkinter**: For menu bar (compatibility)
- [x] **matplotlib**: Required for plotting (Agg backend for threading)
- [x] **NumPy**: Required for numerical computations
- [x] **pandas**: Data manipulation and analysis
- [x] **PIL/Pillow**: Optional for image handling
- [x] **APGI Framework**: Core framework with constants
- [x] **GUI Components**: ThemeManager, ToolTip, KeyboardManager, UndoRedoManager
- [x] **Structured Logging**: `apgi_framework.logging.centralized_logging`

## GUI Launch Verification

- [x] GUI window opens without errors (`python APGI_Application_GUI.py`)
- [x] Window extends `ctk.CTk` (customtkinter main window)
- [x] Window title set via customtkinter
- [x] Window size: 70% of screen with min 1200x800
- [x] Window centered on screen
- [x] Resizable with min/max constraints
- [x] Appearance mode: Light (default) / Dark (configurable)
- [x] Color theme: Blue (default) / configurable
- [x] Grid layout: 2 columns (sidebar + main), 2 rows (content + status)

## Data Loading Verification

- [x] **GUIConstants** imported from `apgi_framework.config.constants`
- [x] **GUIConfig** initialized with all constants
- [x] **Theme Manager** loaded via `get_theme_manager()`
- [x] **Structured Logger** initialized
- [x] **Test Runners** initialized (8 falsification tests)
- [x] **Keyboard Manager** for shortcuts
- [x] **Undo/Redo Manager** for parameter changes
- [x] **Thread Pool** configured (4 threads)
- [x] **Active Figures Set** for memory management (max 50)
- [x] **Input Validator** singleton instance

## Core Classes & Functions

- [x] `GUIConfig` - Central configuration with paths, sizes, colors, defaults
- [x] `APGIFrameworkGUI` - Main GUI class extending `ctk.CTk`
- [x] `InputValidator` - Singleton for input validation with range checking
- [x] `GUIErrorHandler` - Centralized error handling with logging
- [x] `FigureManager` - Memory-managed matplotlib figure creation
- [x] `create_sidebar()` - Left sidebar with parameters and tests
- [x] `create_main_content_area()` - Right panel with visualizations
- [x] `create_menu_bar()` - Traditional tkinter menu bar
- [x] `create_status_bar()` - Bottom status bar
- [x] `create_apgi_parameters_section()` - Parameter inputs with validation
- [x] `create_falsification_tests_section()` - Test selection and controls
- [x] `create_console_output()` - Scrollable console with timestamps
- [x] `add_parameter_tooltips()` - Context-sensitive help
- [x] `setup_keyboard_shortcuts()` - Keyboard navigation
- [x] `run_test()` - Execute selected falsification test
- [x] `display_system_status()` - Show comprehensive status dialog

## Layout Structure (customtkinter)

### Grid Configuration

- [x] Column 0: Sidebar (width=350, fixed)
- [x] Column 1: Main content (weight=1, expandable)
- [x] Row 0: Main content area (weight=1, expandable)
- [x] Row 1: Status bar (height=30, fixed)

### Sidebar (Left Panel)

- [x] Width: 350px, corner_radius=0
- [x] fg_color: "#f0f0f0" (light gray)
- [x] CTkScrollableFrame for content
- [x] Grid layout: 1 column, expand vertically

### Main Content (Right Panel)

- [x] Matplotlib canvas area for plots
- [x] Console output at bottom (scrollable)
- [x] Grid weight=1 for expansion

### Status Bar (Bottom)

- [x] Height: 30px, fg_color: "#e0e0e0"
- [x] Status label: "Ready"
- [x] System status label: "System: Initializing..."
- [x] Spans both columns

## Sidebar Sections

### Controls Frame (Top)

- [x] **Clear Console** button - Clears console output
- [x] **Quit** button - Gracefully exits application
- [x] Grid: 2 columns for side-by-side buttons

### APGI Parameters Section

- [x] Title: " APGI Parameters"
- [x] 6 Parameter input fields:
  - [x] Exteroceptive Precision (0.1-10.0, default: 1.0)
  - [x] Interoceptive Precision (0.1-10.0, default: 1.0)
  - [x] Somatic Gain (0.0-2.0, default: 0.5)
  - [x] Threshold (0.0-2.0, default: 0.5)
  - [x] Steepness (1.0-20.0, default: 5.0)
  - [x] Number of Trials (1-10000, default: 100)
- [x] CTkEntry widgets with validation
- [x] Tooltips for each parameter
- [x] Dynamic defaults from ConfigManager (when available)

### Falsification Tests Section

- [x] Title: " Falsification Tests"
- [x] 8 Test checkboxes:
  - [x] Test 1: Synthetic EEG ML Validation
  - [x] Test 2: Behavioral Bayesian Analysis
  - [x] Test 3: Active Inference Agent
  - [x] Test 4: Phase Transition Dynamics
  - [x] Test 5: Evolutionary Emergence
  - [x] Test 6: Liquid Network Plasticity
  - [x] Test 7: TMS Causal Manipulation
  - [x] Test 8: Psychophysical Threshold
- [x] **Run Selected Tests** button (green, prominent)
- [x] All tests selected by default
- [x] Individual checkbox toggles

## Main Content Area

### Matplotlib Visualization Panel

- [x] CTkFrame with expand=True
- [x] Matplotlib Figure with Agg backend (thread-safe)
- [x] FigureCanvasTkAgg integration
- [x] Subplot configurations (1x1, 1x2, 2x2)
- [x] DPI: 100 (plot), 300 (export)
- [x] Auto-clear before new plots
- [x] Memory limit: 50 figures max

### Console Output Panel

- [x] CTkScrollableFrame
- [x] CTkLabel with timestamps
- [x] Monospace font (Courier)
- [x] Auto-scroll to bottom
- [x] Max lines limit (1000)
- [x] Log levels: INFO, WARNING, ERROR, SUCCESS
- [x] Color-coded messages
- [x] Export to file capability

## Menu Bar (tkinter.Menu)

### File Menu

- [x] Load Configuration
- [x] Save Configuration
- [x] Export Results
- [x] Exit

### Edit Menu

- [x] Undo (Ctrl+Z)
- [x] Redo (Ctrl+Y)
- [x] Cut
- [x] Copy
- [x] Paste
- [x] Clear Console

### View Menu

- [x] Light/Dark Mode toggle
- [x] Show/Hide Sidebar
- [x] Increase Font Size
- [x] Decrease Font Size
- [x] Reset Zoom

### Tools Menu

- [x] Run All Tests
- [x] Stop Running Tests
- [x] System Diagnostics
- [x] Clear Cache

### Window Menu

- [x] Minimize
- [x] Maximize
- [x] Full Screen (F11)

### Help Menu

- [x] Documentation
- [x] Keyboard Shortcuts (Ctrl+?)
- [x] About APGI Application GUI

## Falsification Test System

### Test Runners (8 Tests)

- [x] `VP_01_Synthetic_EEG_ML_Validation`
- [x] `VP_02_Behavioral_Bayesian_Analysis`
- [x] `VP_03_Active_Inference_Agent`
- [x] `VP_04_Phase_Transition_Dynamics`
- [x] `VP_05_Evolutionary_Emergence`
- [x] `VP_06_Liquid_Network_Plasticity`
- [x] `VP_07_TMS_Causal_Manipulation`
- [x] `VP_08_Psychophysical_Threshold`

### Test Execution

- [x] `run_test(test_name)` - Individual test runner
- [x] Thread-based execution (non-blocking)
- [x] Progress updates to console
- [x] Parameter passing to tests
- [x] Result collection and display
- [x] Error handling with stack traces

### Thread Safety

- [x] `_test_lock` threading.Lock
- [x] `_test_running` flag
- [x] `sidebar_buttons` tracking for disable during tests
- [x] `run_in_thread` decorator from framework

## Configuration System

### GUIConstants (from apgi_framework)

- [x] Window ratios: 0.8 width, 0.8 height
- [x] Min size: 1200x800
- [x] Max size: 1920x1080
- [x] Sidebar width: 350
- [x] Status bar height: 30
- [x] Thread pool: 4
- [x] Plot DPI: 100
- [x] Export DPI: 300
- [x] Console max lines: 1000
- [x] Validation timeout: 5 seconds

### Colors

- [x] Sidebar BG: "#f0f0f0"
- [x] Main BG: "#ffffff"
- [x] Status bar BG: "#e0e0e0"
- [x] Success: "#4CAF50"
- [x] Warning: "#FFC107"
- [x] Error: "#F44336"
- [x] Info: "#2196F3"

### Default Parameters

- [x] Exteroceptive precision: 1.0
- [x] Interoceptive precision: 1.0
- [x] Somatic gain: 0.5
- [x] Threshold: 0.5
- [x] Steepness: 5.0
- [x] Number of trials: 100
- [x] Number of participants: 1
- [x] Session duration: 60

## Input Validation System

### InputValidator Class (Singleton)

- [x] `validate_float(value, min, max, name)` - Float validation
- [x] `validate_int(value, min, max, name)` - Integer validation
- [x] `validate_string(value, max_length, name)` - String validation
- [x] Range checking with inclusive bounds
- [x] Error messages with parameter names
- [x] Returns (is_valid, error_message) tuple

### Widget-Level Validation

- [x] CTkEntry validation callbacks
- [x] Real-time error display
- [x] Visual feedback (red border on error)
- [x] Clamp values to valid ranges
- [x] Tooltip hints for valid ranges

## Error Handling System

### GUIErrorHandler Class

- [x] `handle_error(error, context, show_dialog)` - Main handler
- [x] Automatic error logging
- [x] User-friendly error dialogs
- [x] Stack trace capture
- [x] Recovery suggestions

### Error Types Handled

- [x] ValidationError - Input validation failures
- [x] TestError - Test execution failures
- [x] ConfigError - Configuration problems
- [x] ExportError - Export operation failures
- [x] UnexpectedError - Catch-all for unknown errors

### Error Display

- [x] Console logging with colors
- [x] Dialog boxes for critical errors
- [x] Status bar updates
- [x] Non-blocking error handling

## Keyboard Shortcuts

### Navigation

- [x] **Tab** - Next widget
- [x] **Shift+Tab** - Previous widget
- [x] **Ctrl+Tab** - Next test checkbox
- [x] **Ctrl+Shift+Tab** - Previous test checkbox

### Actions

- [x] **Ctrl+R** - Run selected tests
- [x] **Ctrl+S** - Stop running tests
- [x] **Ctrl+T** - Toggle theme (Light/Dark)
- [x] **Ctrl+E** - Export results
- [x] **Ctrl+C** - Clear console
- [x] **Ctrl+L** - Load configuration
- [x] **Ctrl+Shift+S** - Save configuration

### Edit

- [x] **Ctrl+Z** - Undo parameter change
- [x] **Ctrl+Y** - Redo parameter change

### View

- [x] **F11** - Toggle fullscreen
- [x] **Ctrl+Plus** - Increase font size
- [x] **Ctrl+Minus** - Decrease font size
- [x] **Ctrl+0** - Reset zoom

### Help

- [x] **F1** - Documentation
- [x] **Ctrl+?** - Keyboard shortcuts dialog

## Theme Management

### customtkinter Themes

- [x] Appearance modes: Light, Dark, System
- [x] Color themes: blue, green, dark-blue
- [x] `ctk.set_appearance_mode(mode)` - Dynamic switching
- [x] `ctk.set_default_color_theme(theme)` - Color scheme

### Custom Colors (GUIConfig.COLORS)

- [x] Sidebar background: #f0f0f0
- [x] Main background: #ffffff
- [x] Status bar: #e0e0e0
- [x] Success: #4CAF50 (green)
- [x] Warning: #FFC107 (amber)
- [x] Error: #F44336 (red)
- [x] Info: #2196F3 (blue)

### Theme Manager Integration

- [x] Load theme from config file
- [x] Save theme preference
- [x] Auto-apply on startup
- [x] Per-user theme storage

## Console System

### Console Features

- [x] `log_to_console(message, level)` - Main logging method
- [x] Timestamp prefix: [HH:MM:SS]
- [x] Color-coded levels:
  - [x] INFO - Blue
  - [x] WARNING - Yellow
  - [x] ERROR - Red
  - [x] SUCCESS - Green

- [x] Max lines: 1000 (auto-pruning)

### Test Output

- [x] Real-time test progress
- [x] Test start/end messages
- [x] Parameter values displayed
- [x] Results summary
- [x] Error details with tracebacks

### Figure Management

- [x] `create_figure(figsize, dpi)` - Memory-managed creation
- [x] `active_figures` set tracking
- [x] Max 50 figures limit
- [x] Automatic cleanup
- [x] `close_all_figures()` - Bulk cleanup

## System Status Dialog

### Status Display (`display_system_status()`)

- [x] Modal dialog with CTkToplevel
- [x] System Information:
  - [x] Python version
  - [x] Platform (OS)
  - [x] Working directory

- [x] Configuration Status:
  - [x] Config file path
  - [x] Data directory
  - [x] Results directory

- [x] Dependencies Status:
  - [x] NumPy version
  - [x] Matplotlib version
  - [x] Pandas version
  - [x] APGI Framework availability

- [x] Test Status:
  - [x] Last test run time
  - [x] Tests passed/failed
  - [x] Total execution time

- [x] Close button
- [x] Copy to clipboard button
- [x] Refresh button

## Runtime Verification Summary

| Component | Status |
| --------- | ------ |
| GUI Launch | ✅ Functional |
| customtkinter | ✅ CTk main window |
| Sidebar | ✅ 350px with scroll |
| APGI Parameters | ✅ 6 inputs with validation |
| Falsification Tests | ✅ 8 test checkboxes |
| Console Output | ✅ Scrollable with colors |
| Matplotlib | ✅ Agg backend, thread-safe |
| Menu Bar | ✅ tkinter.Menu |
| Theme System | ✅ Light/Dark modes |
| Keyboard Shortcuts | ✅ Full bindings |
| Input Validation | ✅ Range checking |
| Error Handling | ✅ Centralized |
| Threading | ✅ Non-blocking tests |
| Figure Management | ✅ Memory limit 50 |
