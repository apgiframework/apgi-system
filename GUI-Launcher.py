#!/usr/bin/env python3
"""
Comprehensive GUI Launcher for APGI Framework

This script provides a centralized launcher to access all GUI applications
and tools
in the APGI Framework, organized by category for easy navigation.

Usage:
    python GUI-Launcher.py              # Launch GUI
    python GUI-Launcher.py --help       # Show help
    python GUI-Launcher.py --version    # Show version
"""

# Deprecated legacy entry point (README: "Legacy Entry Points (Deprecated)").
try:
    from apgi_framework.utils.deprecation import warn_deprecated_module

    warn_deprecated_module("GUI-Launcher", "apgi_gui.main", since="1.1.0", removal="2.0.0")
except ImportError:  # framework not importable: the README and docstring still state the deprecation
    pass

import argparse
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


# Configuration constants
class UIConfig:
    """UI configuration constants."""

    # Window scaling factors for different screen sizes
    SCALE_4K = 0.6
    SCALE_QHD = 0.7
    SCALE_FHD = 0.8
    SCALE_SMALL = 0.85

    # Layout spacing (pixels)
    MAIN_CONTAINER_PADDING = 30
    HEADER_SPACING = 25
    BOTTOM_BUTTON_SPACING = 25
    CATEGORY_SPACING = (20, 15)
    CARD_CONTENT_PADDING = (20, 15)
    CARD_BUTTON_PADDING = (20, 0)

    # Button dimensions
    BUTTON_PADDING_LARGE = (20, 10)
    BUTTON_PADDING_LAUNCH = (25, 12)
    BUTTON_PADDING_CLOSE = (20, 8)

    # Font sizes
    FONT_ICON_SIZE = 20
    FONT_TITLE_SIZE = 36
    FONT_SUBTITLE_SIZE = 13

    # Text wrapping
    DESCRIPTION_WRAP_LENGTH = 1200  # Increased for better use of width


class ComprehensiveGUILauncher:
    """Comprehensive launcher for all APGI Framework GUI applications."""

    def __init__(self):
        """Initialize the comprehensive launcher."""
        self.root = tk.Tk()
        self.root.title("APGI Framework - Comprehensive GUI Launcher")

        # Adaptive window sizing based on screen resolution and DPI
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Use DPI-aware scaling without hard caps
        # Scale factor for different screen sizes
        if screen_width >= 3840:  # 4K displays
            scale_factor = UIConfig.SCALE_4K
        elif screen_width >= 2560:  # QHD/2K displays
            scale_factor = UIConfig.SCALE_QHD
        elif screen_width >= 1920:  # Full HD
            scale_factor = UIConfig.SCALE_FHD
        else:  # Smaller displays
            scale_factor = UIConfig.SCALE_SMALL

        window_width = int(screen_width * scale_factor)
        window_height = int(screen_height * scale_factor)

        # Center window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.resizable(True, True)

        # Set window icon and styling
        self.root.configure(bg="#ecf0f1")

        # Configure styles
        self.setup_styles()

        # Define GUI applications
        self.gui_apps = self.define_gui_applications()

        # Check application availability
        self.check_app_availability()

        # Create widgets
        self.create_widgets()

    def setup_styles(self):
        """Setup custom styles for better appearance."""
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Configure custom styles
        self.style.configure(
            "Title.TLabel",
            font=("Helvetica", 34, "bold"),
            background="#ecf0f1",
            foreground="#1f2d3d",
        )

        self.style.configure(
            "Subtitle.TLabel",
            font=("Helvetica", 12),
            background="#ecf0f1",
            foreground="#5c6b77",
        )

        self.style.configure(
            "Category.TLabel",
            font=("Helvetica", 18, "bold"),
            background="#ecf0f1",
            foreground="#22313f",
        )

        self.style.configure(
            "App.TLabel",
            font=("Helvetica", 13, "bold"),
            background="#ffffff",
            foreground="#1f2d3d",
        )

        self.style.configure(
            "Desc.TLabel",
            font=("Helvetica", 10),
            background="#ffffff",
            foreground="#4d5d6c",
        )

        self.style.configure(
            "Path.TLabel",
            font=("Courier", 9, "italic"),
            background="#ffffff",
            foreground="#6c7a89",
        )

        self.style.configure(
            "Launch.TButton",
            font=("Helvetica", 11, "bold"),
            padding=UIConfig.BUTTON_PADDING_LARGE,
        )

        # High-contrast ttk button styles
        self.style.configure(
            "Primary.TButton",
            font=("Helvetica", 11, "bold"),
            background="#3498db",
            foreground="#ffffff",
            padding=UIConfig.BUTTON_PADDING_LARGE,
        )
        self.style.map(
            "Primary.TButton",
            background=[("active", "#2d86c5")],
            foreground=[("active", "#ffffff")],
        )
        self.style.configure(
            "Secondary.TButton",
            font=("Helvetica", 11, "bold"),
            background="#7f8c8d",
            foreground="#ffffff",
            padding=UIConfig.BUTTON_PADDING_LARGE,
        )
        self.style.map(
            "Secondary.TButton",
            background=[("active", "#6c7778")],
            foreground=[("active", "#ffffff")],
        )
        self.style.configure(
            "Danger.TButton",
            font=("Helvetica", 11, "bold"),
            background="#e74c3c",
            foreground="#ffffff",
            padding=UIConfig.BUTTON_PADDING_LARGE,
        )
        self.style.map(
            "Danger.TButton",
            background=[("active", "#c0392b")],
            foreground=[("active", "#ffffff")],
        )

        self.style.configure(
            "Status.TLabel",
            font=("Helvetica", 9, "bold"),
            background="#ffffff",
        )

    def check_app_availability(self):
        """Check which applications are available."""
        self.app_status = {}
        current_dir = Path(__file__).parent

        for category, apps in self.gui_apps.items():
            for app in apps:
                script_path = current_dir / app["file"]
                self.app_status[app["file"]] = script_path.exists()

    def define_gui_applications(self):
        """Define all GUI applications and entry points organized by category."""
        return {
            "Core Applications": [
                {
                    "name": "APGI GUI",
                    "file": "APGI_GUI.py",
                    "description": "Main APGI Framework GUI application",
                    "icon": "[Main]",
                    "command": self.launch_apgi_gui,
                },
                {
                    "name": "APGI Application GUI",
                    "file": "APGI_Application_GUI.py",
                    "description": "Application-level GUI for APGI Framework",
                    "icon": "[App]",
                    "command": self.launch_apgi_application_gui,
                },
                {
                    "name": "Assistant GUI",
                    "file": "Assistant_GUI.py",
                    "description": "AI Assistant interface with GUI",
                    "icon": "[AI]",
                    "command": self.launch_assistant_gui,
                },
                {
                    "name": "AI Assistant",
                    "file": "AI_Assistant.py",
                    "description": "Standalone AI Assistant application",
                    "icon": "[Bot]",
                    "command": self.launch_ai_assistant,
                },
                {
                    "name": "APGI Simulation GUI",
                    "file": "APGI_Simulation_GUI.py",
                    "description": "Simulation visualization and control GUI",
                    "icon": "[Sim]",
                    "command": self.launch_apgi_simulation_gui,
                },
                {
                    "name": "Psychological States GUI",
                    "file": "Psychological_States_GUI.py",
                    "description": "Psychological state visualization and analysis",
                    "icon": "[Brain]",
                    "command": self.launch_psychological_states_gui,
                },
            ],
            "Analysis & Visualization": [
                {
                    "name": "Parameter Estimation GUI",
                    "file": "apgi_framework/gui/parameter_estimation_gui.py",
                    "description": "Parameter estimation and analysis tools",
                    "icon": "[Data]",
                    "command": self.launch_parameter_estimation,
                },
                {
                    "name": "Interactive Dashboard",
                    "file": "apgi_framework/gui/interactive_dashboard.py",
                    "description": "Web-based interactive dashboard (requires Flask)",
                    "icon": "[Web]",
                    "command": self.launch_interactive_dashboard,
                },
                {
                    "name": "Monitoring Dashboard",
                    "file": "apgi_framework/gui/monitoring_dashboard.py",
                    "description": "Real-time monitoring dashboard",
                    "icon": "[Chart]",
                    "command": self.launch_monitoring_dashboard,
                },
                {
                    "name": "Web Monitoring Dashboard",
                    "file": "apgi_framework/gui/web_monitoring_dashboard.py",
                    "description": "Web-based real-time monitoring",
                    "icon": "[Browser]",
                    "command": self.launch_web_monitoring_dashboard,
                },
                {
                    "name": "Reporting & Visualization",
                    "file": "apgi_framework/gui/reporting_visualization.py",
                    "description": "Generate reports and visualizations",
                    "icon": "[Report]",
                    "command": self.launch_reporting_visualization,
                },
                {
                    "name": "Enhanced Monitoring Dashboard",
                    "file": "apgi_framework/gui/enhanced_monitoring_dashboard.py",
                    "description": "Enhanced monitoring with advanced features",
                    "icon": "[Monitor]",
                    "command": self.launch_enhanced_monitoring_dashboard,
                },
                {
                    "name": "Results Viewer",
                    "file": "apgi_framework/gui/results_viewer.py",
                    "description": "View and analyze experiment results",
                    "icon": "[Results]",
                    "command": self.launch_results_viewer,
                },
                {
                    "name": "Coverage Visualization",
                    "file": "apgi_framework/gui/coverage_visualization.py",
                    "description": "Test coverage visualization tool",
                    "icon": "[Coverage]",
                    "command": self.launch_coverage_visualization,
                },
                {
                    "name": "Real-time Data Stream",
                    "file": "apgi_framework/gui/realtime_data_stream.py",
                    "description": "Real-time data streaming visualization",
                    "icon": "[Stream]",
                    "command": self.launch_realtime_data_stream,
                },
            ],
            "Configuration & Management": [
                {
                    "name": "Task Configuration",
                    "file": "apgi_framework/gui/task_configuration.py",
                    "description": "Configure experimental tasks",
                    "icon": "[Config]",
                    "command": self.launch_task_configuration,
                },
                {
                    "name": "Session Management",
                    "file": "apgi_framework/gui/session_management.py",
                    "description": "Manage experimental sessions",
                    "icon": "[Session]",
                    "command": self.launch_session_management,
                },
                {
                    "name": "Progress Monitoring",
                    "file": "apgi_framework/gui/progress_monitoring.py",
                    "description": "Monitor experiment progress",
                    "icon": "[Data]",
                    "command": self.launch_progress_monitoring,
                },
                {
                    "name": "Error Handling",
                    "file": "apgi_framework/gui/error_handling.py",
                    "description": "Error handling and logging interface",
                    "icon": "[Warning]",
                    "command": self.launch_error_handling,
                },
                {
                    "name": "Error Logging Utils",
                    "file": "apgi_framework/gui/error_logging_utils.py",
                    "description": "Error logging utility functions",
                    "icon": "[Log]",
                    "command": self.launch_error_logging_utils,
                },
                {
                    "name": "apgi GUI Main",
                    "file": "apgi_gui/main.py",
                    "description": "apgi_gui main application entry point",
                    "icon": "[GUI]",
                    "command": self.launch_apgi_gui_main,
                },
            ],
            "Development & Testing": [
                {
                    "name": "Tests GUI",
                    "file": "Tests_GUI.py",
                    "description": "GUI to run all tests folder scripts with summary panel",
                    "icon": "[Tests]",
                    "command": self.launch_tests_gui,
                },
                {
                    "name": "Utils GUI",
                    "file": "Utils_GUI.py",
                    "description": "GUI to run all utility scripts from utils folder",
                    "icon": "[Utils]",
                    "command": self.launch_utils_gui,
                },
                {
                    "name": "GUI Template",
                    "file": "apps/gui_template.py",
                    "description": "GUI template for development",
                    "icon": "[Template]",
                    "command": self.launch_gui_template_main,
                },
                {
                    "name": "APGI Design",
                    "file": "apps/apgi-design.py",
                    "description": "APGI design template and visualization system",
                    "icon": "[Design]",
                    "command": self.launch_apgi_design,
                },
                {
                    "name": "Script Runner GUI",
                    "file": "utils/script_runner_gui.py",
                    "description": "GUI for running utility scripts",
                    "icon": "[Runner]",
                    "command": self.launch_script_runner_gui,
                },
                {
                    "name": "Framework Testing Main",
                    "file": "apgi_framework/testing/main.py",
                    "description": "Testing framework main entry point",
                    "icon": "[Test]",
                    "command": self.launch_framework_testing_main,
                },
                {
                    "name": "GUI Test Runner",
                    "file": "apgi_framework/testing/gui_test_runner.py",
                    "description": "GUI-based test runner",
                    "icon": "[Test]",
                    "command": self.launch_gui_test_runner,
                },
            ],
            "CLI Tools & Framework": [
                {
                    "name": "Framework CLI",
                    "file": "apgi_framework/cli.py",
                    "description": "Main command-line interface for APGI Framework",
                    "icon": "[Terminal]",
                    "command": self.launch_framework_cli,
                },
                {
                    "name": "apgi GUI CLI",
                    "file": "apgi_gui/cli.py",
                    "description": "apgi_gui command-line interface",
                    "icon": "[CLI]",
                    "command": self.launch_apgi_gui_cli,
                },
                {
                    "name": "Diagnostics CLI",
                    "file": "apgi_framework/validation/diagnostics_cli.py",
                    "description": "System diagnostics and validation tools",
                    "icon": "[Diag]",
                    "command": self.launch_diagnostics_cli,
                },
                {
                    "name": "Deployment CLI",
                    "file": "apgi_framework/deployment/cli.py",
                    "description": "Deployment automation and management",
                    "icon": "[Deploy]",
                    "command": self.launch_deployment_cli,
                },
                {
                    "name": "Deployment Automation",
                    "file": "apgi_framework/deployment/automation.py",
                    "description": "Deployment automation scripts",
                    "icon": "[Auto]",
                    "command": self.launch_deployment_automation,
                },
                {
                    "name": "Main Controller",
                    "file": "apgi_framework/main_controller.py",
                    "description": "Framework main controller entry point",
                    "icon": "[Ctrl]",
                    "command": self.launch_main_controller,
                },
                {
                    "name": "Installation Validator",
                    "file": "apgi_framework/installation_validator.py",
                    "description": "Validate installation and dependencies",
                    "icon": "[Check]",
                    "command": self.launch_installation_validator,
                },
                {
                    "name": "Module Mode",
                    "file": "apgi_framework/__main__.py",
                    "description": "Run framework as module (python -m apgi_framework)",
                    "icon": "[Mod]",
                    "command": self.launch_module_mode,
                },
            ],
            "API & Backend": [
                {
                    "name": "API Server",
                    "file": "api/main.py",
                    "description": "API server main entry point",
                    "icon": "[API]",
                    "command": self.launch_api_server,
                },
                {
                    "name": "Celery App",
                    "file": "api/celery_app.py",
                    "description": "Celery task queue application",
                    "icon": "[Queue]",
                    "command": self.launch_celery_app,
                },
            ],
            "Testing & Benchmarks": [
                {
                    "name": "Comprehensive Test Runner",
                    "file": "benchmarks/run_tests.py",
                    "description": "Run comprehensive test suite",
                    "icon": "[Test]",
                    "command": self.launch_test_runner,
                },
                {
                    "name": "Coverage Runner",
                    "file": "benchmarks/run_coverage.py",
                    "description": "Run test coverage analysis",
                    "icon": "[Coverage]",
                    "command": self.launch_coverage_runner,
                },
                {
                    "name": "Performance Benchmarks",
                    "file": "benchmarks/test_performance.py",
                    "description": "Run performance benchmarks",
                    "icon": "[Bench]",
                    "command": self.launch_performance_benchmarks,
                },
                {
                    "name": "Critical Path Profiling",
                    "file": "benchmarks/critical_path_profiling.py",
                    "description": "Profile critical code paths",
                    "icon": "[Profile]",
                    "command": self.launch_critical_path_profiling,
                },
            ],
            "Utilities & Tools": [
                {
                    "name": "Delete Cache",
                    "file": "delete_pycache.py",
                    "description": "Clean Python cache files",
                    "icon": "[Clean]",
                    "command": self.launch_delete_cache,
                },
                {
                    "name": "Backup Manager",
                    "file": "utils/backup_manager.py",
                    "description": "Manage system backups",
                    "icon": "[Backup]",
                    "command": self.launch_backup_manager,
                },
                {
                    "name": "Diagnostics",
                    "file": "utils/diagnostics.py",
                    "description": "System diagnostics utilities",
                    "icon": "[Diag]",
                    "command": self.launch_diagnostics,
                },
                {
                    "name": "Performance Dashboard",
                    "file": "utils/static_dashboard_generator.py",
                    "description": "Generate static performance monitoring dashboards",
                    "icon": "[Chart]",
                    "command": self.launch_performance_dashboard,
                },
                {
                    "name": "Pipeline Visualization",
                    "file": "utils/pipeline_visualization.py",
                    "description": "Visualize data pipelines",
                    "icon": "[Pipe]",
                    "command": self.launch_pipeline_visualization,
                },
                {
                    "name": "Report Generator",
                    "file": "utils/report_generator.py",
                    "description": "Generate system reports",
                    "icon": "[Report]",
                    "command": self.launch_report_generator,
                },
                {
                    "name": "Tutorial",
                    "file": "utils/tutorial.py",
                    "description": "Interactive system tutorial",
                    "icon": "[Learn]",
                    "command": self.launch_tutorial,
                },
                {
                    "name": "Validate App",
                    "file": "utils/validate_app.py",
                    "description": "Validate application configuration",
                    "icon": "[Validate]",
                    "command": self.launch_validate_app,
                },
                {
                    "name": "Dependency Checker",
                    "file": "utils/dependency_checker.py",
                    "description": "Check system dependencies",
                    "icon": "[Check]",
                    "command": self.launch_dependency_checker,
                },
                {
                    "name": "Config Manager",
                    "file": "utils/config_manager.py",
                    "description": "Manage system configuration",
                    "icon": "[Config]",
                    "command": self.launch_config_manager,
                },
                {
                    "name": "Cache Manager",
                    "file": "utils/cache_manager.py",
                    "description": "Manage system cache",
                    "icon": "[Cache]",
                    "command": self.launch_cache_manager,
                },
                {
                    "name": "Data Processor",
                    "file": "utils/data_processor.py",
                    "description": "Process and transform data",
                    "icon": "[Data]",
                    "command": self.launch_data_processor,
                },
                {
                    "name": "Batch Processor",
                    "file": "utils/batch_processor.py",
                    "description": "Batch data processing",
                    "icon": "[Batch]",
                    "command": self.launch_batch_processor,
                },
                {
                    "name": "Sample Data Generator",
                    "file": "utils/sample_data_generator.py",
                    "description": "Generate sample test data",
                    "icon": "[Data]",
                    "command": self.launch_sample_data_generator,
                },
                {
                    "name": "Demo Analysis",
                    "file": "utils/demo_analysis.py",
                    "description": "Run demo analysis",
                    "icon": "[Demo]",
                    "command": self.launch_demo_analysis,
                },
                {
                    "name": "Basic Simulation",
                    "file": "utils/basic_simulation.py",
                    "description": "Run basic simulations",
                    "icon": "[Sim]",
                    "command": self.launch_basic_simulation,
                },
                {
                    "name": "Gap Analyzer",
                    "file": "utils/analyze_gaps.py",
                    "description": "Analyze system gaps",
                    "icon": "[Analyze]",
                    "command": self.launch_gap_analyzer,
                },
                {
                    "name": "Data Validation",
                    "file": "utils/data_validation.py",
                    "description": "Validate data integrity",
                    "icon": "[Validate]",
                    "command": self.launch_data_validation,
                },
                {
                    "name": "Error Handler",
                    "file": "utils/error_handler.py",
                    "description": "Error handling utilities",
                    "icon": "[Error]",
                    "command": self.launch_error_handler,
                },
                {
                    "name": "Parameter Validator",
                    "file": "utils/parameter_validator.py",
                    "description": "Validate system parameters",
                    "icon": "[Check]",
                    "command": self.launch_parameter_validator,
                },
                {
                    "name": "GUI Testing Framework",
                    "file": "utils/gui_testing_framework.py",
                    "description": "Framework for GUI testing",
                    "icon": "[Test]",
                    "command": self.launch_gui_testing_framework,
                },
                {
                    "name": "Deployment Validator",
                    "file": "apgi_framework/deployment/deployment_validator.py",
                    "description": "Validate deployment configuration",
                    "icon": "[Deploy]",
                    "command": self.launch_deployment_validator,
                },
            ],
            "Examples & Tutorials": [
                {
                    "name": "Primary Falsification Test",
                    "file": "examples/01_run_primary_falsification_test.py",
                    "description": "Run primary falsification test example",
                    "icon": "[Test]",
                    "command": self.launch_primary_falsification_test,
                },
                {
                    "name": "Batch Processing Config",
                    "file": "examples/02_batch_processing_configurations.py",
                    "description": "Batch processing configuration example",
                    "icon": "[Batch]",
                    "command": self.launch_batch_processing_config,
                },
                {
                    "name": "Custom Analysis Results",
                    "file": "examples/03_custom_analysis_saved_results.py",
                    "description": "Custom analysis of saved results",
                    "icon": "[Analysis]",
                    "command": self.launch_custom_analysis_results,
                },
                {
                    "name": "Extending Falsification Criteria",
                    "file": "examples/04_extending_falsification_criteria.py",
                    "description": "Extend falsification criteria example",
                    "icon": "[Extend]",
                    "command": self.launch_extending_falsification_criteria,
                },
                {
                    "name": "Data Loader Example",
                    "file": "examples/08_data_loader.py",
                    "description": "Data loading utility example",
                    "icon": "[Data]",
                    "command": self.launch_data_loader_example,
                },
                {
                    "name": "Coverage Collector Demo",
                    "file": "examples/05_coverage_collector.py",
                    "description": "Coverage collection demonstration",
                    "icon": "[Coverage]",
                    "command": self.launch_coverage_collector_demo,
                },
            ],
        }

    def create_widgets(self):
        """Create launcher widgets with organized layout."""
        # Main container
        main_container = tk.Frame(self.root, bg="#ecf0f1")
        main_container.pack(
            fill=tk.BOTH,
            expand=True,
            padx=UIConfig.MAIN_CONTAINER_PADDING,
            pady=UIConfig.MAIN_CONTAINER_PADDING,
        )

        # Header section
        header_frame = tk.Frame(main_container, bg="#ecf0f1")
        header_frame.pack(fill=tk.X, pady=(0, UIConfig.HEADER_SPACING))

        # Title and exit button container
        title_container = tk.Frame(header_frame, bg="#ecf0f1")
        title_container.pack(fill=tk.X)

        # Title
        title_label = tk.Label(
            title_container,
            text="APGI Framework Launcher",
            font=("Helvetica", UIConfig.FONT_TITLE_SIZE, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50",
        )
        title_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Exit button (top right)
        exit_button = ttk.Button(
            title_container,
            text="Exit",
            command=self.root.quit,
            style="Danger.TButton",
        )
        exit_button.pack(side=tk.RIGHT, padx=(20, 0))

        # Scrollable area for applications
        canvas = tk.Canvas(main_container, bg="#ecf0f1", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#ecf0f1")

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window(
            (0, 0), window=scrollable_frame, anchor="nw", width=canvas.winfo_width()
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        # Bind canvas resize to update window width
        def _on_canvas_configure(event):
            canvas.itemconfig(canvas.find_all()[0], width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)

        # Create application sections
        self.create_application_sections(scrollable_frame)

        # Pack scrollable area
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel for scrolling (cross-platform)
        def _on_mousewheel(event):
            if sys.platform == "darwin":
                canvas.yview_scroll(int(-1 * event.delta), "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_linux(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel_linux)
        canvas.bind_all("<Button-5>", _on_mousewheel_linux)

    def create_application_sections(self, parent):
        """Create application sections for each category."""
        for category, apps in self.gui_apps.items():
            # Category section
            category_frame = tk.Frame(parent, bg="#ecf0f1")
            category_frame.pack(fill=tk.BOTH, expand=True, pady=UIConfig.CATEGORY_SPACING)

            # Category header with count
            available_count = sum(1 for app in apps if self.app_status.get(app["file"], False))
            total_count = len(apps)

            category_label = tk.Label(
                category_frame,
                text=f"{category} ({available_count}/{total_count})",
                font=("Helvetica", 18, "bold"),
                bg="#ecf0f1",
                fg="#2c3e50",
                anchor="w",
            )
            category_label.pack(fill=tk.X, pady=(0, 12))

            # Applications grid
            apps_frame = tk.Frame(category_frame, bg="#ecf0f1")
            apps_frame.pack(fill=tk.BOTH, expand=True)

            # Create application cards
            for i, app in enumerate(apps):
                self.create_application_card(apps_frame, app, i)

    def create_application_card(self, parent, app, index):
        """Create an individual application card with enhanced styling."""
        # Card frame
        card_frame = tk.Frame(
            parent,
            bg="#ffffff",
            relief=tk.RIDGE,
            bd=2,
            highlightthickness=1,
            highlightbackground="#d0d3d4",
        )
        card_frame.pack(fill=tk.X, pady=8, padx=0)

        # Check availability
        is_available = self.app_status.get(app["file"], False)

        # Card content
        content_frame = tk.Frame(card_frame, bg="#ffffff")
        content_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=UIConfig.CARD_CONTENT_PADDING[0],
            pady=UIConfig.CARD_CONTENT_PADDING[1],
        )

        # Left side - Icon and info
        info_frame = tk.Frame(content_frame, bg="#ffffff")
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Icon and name
        title_frame = tk.Frame(info_frame, bg="#ffffff")
        title_frame.pack(fill=tk.X, pady=(0, 8))

        # Status indicator
        status_color = "#27ae60" if is_available else "#e74c3c"
        status_text = "✓" if is_available else "✗"
        status_label = tk.Label(
            title_frame,
            text=status_text,
            font=("Helvetica", 16, "bold"),
            bg="#ffffff",
            fg=status_color,
        )
        status_label.pack(side=tk.LEFT, padx=(0, 12))

        # Icon
        icon_label = tk.Label(
            title_frame,
            text=app["icon"],
            font=("Helvetica", UIConfig.FONT_ICON_SIZE),
            bg="#ffffff",
            fg="#2c3e50",
        )
        icon_label.pack(side=tk.LEFT, padx=(0, 12))

        # Name
        name_label = tk.Label(
            title_frame,
            text=app["name"],
            font=("Helvetica", 14, "bold"),
            bg="#ffffff",
            fg="#2c3e50" if is_available else "#95a5a6",
            anchor="w",
        )
        name_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Description
        desc_label = tk.Label(
            info_frame,
            text=app["description"],
            font=("Helvetica", 10),
            bg="#ffffff",
            fg="#7f8c8d" if is_available else "#bdc3c7",
            anchor="w",
            justify=tk.LEFT,
            wraplength=1000,  # Increased for better width utilization
        )
        desc_label.pack(fill=tk.X)

        # File path
        file_label = tk.Label(
            info_frame,
            text=f"File: {app['file']}",
            font=("Courier", 9, "italic"),
            bg="#ffffff",
            fg="#95a5a6" if is_available else "#bdc3c7",
            anchor="w",
        )
        file_label.pack(fill=tk.X, pady=(8, 0))

        # Right side - Launch button
        button_frame = tk.Frame(content_frame, bg="#ffffff")
        button_frame.pack(side=tk.RIGHT, padx=UIConfig.CARD_BUTTON_PADDING)

        if is_available:
            launch_button = ttk.Button(
                button_frame,
                text="Launch",
                command=app["command"],
                style="Primary.TButton",
            )
            launch_button.pack()
        else:
            disabled_button = ttk.Button(
                button_frame,
                text="Unavailable",
                state=tk.DISABLED,
                style="Secondary.TButton",
            )
            disabled_button.pack()

    # Launch methods for each application
    def launch_parameter_estimation(self):
        """Launch Parameter Estimation GUI."""
        self.launch_python_script(
            "apgi_framework/gui/parameter_estimation_gui.py", "Parameter Estimation GUI"
        )

    def launch_interactive_dashboard(self):
        """Launch Interactive Dashboard."""
        self.launch_python_script(
            "apgi_framework/gui/interactive_dashboard.py", "Interactive Dashboard"
        )

    def launch_monitoring_dashboard(self):
        """Launch Monitoring Dashboard."""
        self.launch_python_script(
            "apgi_framework/gui/monitoring_dashboard.py", "Monitoring Dashboard"
        )

    def launch_web_monitoring_dashboard(self):
        """Launch Web Monitoring Dashboard."""
        self.launch_python_script(
            "apgi_framework/gui/web_monitoring_dashboard.py", "Web Monitoring Dashboard"
        )

    def launch_reporting_visualization(self):
        """Launch Reporting & Visualization."""
        self.launch_python_script(
            "apgi_framework/gui/reporting_visualization.py", "Reporting & Visualization"
        )

    def launch_task_configuration(self):
        """Launch Task Configuration."""
        self.launch_python_script("apgi_framework/gui/task_configuration.py", "Task Configuration")

    def launch_session_management(self):
        """Launch Session Management."""
        self.launch_python_script("apgi_framework/gui/session_management.py", "Session Management")

    def launch_progress_monitoring(self):
        """Launch Progress Monitoring."""
        self.launch_python_script(
            "apgi_framework/gui/progress_monitoring.py", "Progress Monitoring"
        )

    def launch_error_handling(self):
        """Launch Error Handling Demo."""
        self.launch_python_script("apgi_framework/gui/error_handling.py", "Error Handling Demo")

    # Additional GUI launch methods
    def launch_coverage_visualization(self):
        """Launch Coverage Visualization."""
        self.launch_python_script(
            "apgi_framework/gui/coverage_visualization.py", "Coverage Visualization"
        )

    def launch_enhanced_monitoring_dashboard(self):
        """Launch Enhanced Monitoring Dashboard."""
        self.launch_python_script(
            "apgi_framework/gui/enhanced_monitoring_dashboard.py",
            "Enhanced Monitoring Dashboard",
        )

    def launch_results_viewer(self):
        """Launch Results Viewer."""
        self.launch_python_script("apgi_framework/gui/results_viewer.py", "Results Viewer")

    def launch_gui_template_main(self):
        """Launch GUI Template (Main)."""
        self.launch_python_script("apps/gui_template.py", "GUI Template (Main)")

    # CLI Tools launch methods
    def launch_test_runner(self):
        """Launch Comprehensive Test Runner."""
        self.launch_python_script("benchmarks/run_tests.py", "Comprehensive Test Runner")

    def launch_framework_cli(self):
        """Launch Framework CLI."""
        self.launch_python_script("apgi_framework/cli.py", "Framework CLI")

    def launch_diagnostics_cli(self):
        """Launch Diagnostics CLI."""
        self.launch_python_script("apgi_framework/validation/diagnostics_cli.py", "Diagnostics CLI")

    def launch_deployment_cli(self):
        """Launch Deployment CLI."""
        self.launch_python_script("apgi_framework/deployment/cli.py", "Deployment CLI")

    def launch_setup_script(self):
        """Launch Setup Script."""
        self.launch_python_script("utils/install_dependencies.py", "Setup Script")

    def launch_quick_deploy(self):
        """Launch Quick Deploy."""
        self.launch_python_script(
            "apgi_framework/deployment/deployment_validator.py", "Quick Deploy"
        )

    def launch_delete_cache(self):
        """Launch Delete Cache."""
        self.launch_python_script("delete_pycache.py", "Delete Cache")

    # Examples & Demos launch methods
    def launch_data_loader_example(self):
        """Launch Data Loader Example."""
        self.launch_python_script("examples/08_data_loader.py", "Data Loader Example")

    def launch_coverage_collector_demo(self):
        """Launch Coverage Collector Demo."""
        self.launch_python_script("examples/05_coverage_collector.py", "Coverage Collector Demo")

    def launch_primary_falsification_test(self):
        """Launch Primary Falsification Test."""
        self.launch_python_script(
            "examples/01_run_primary_falsification_test.py",
            "Primary Falsification Test",
        )

    def launch_batch_processing_config(self):
        """Launch Batch Processing Config."""
        self.launch_python_script(
            "examples/02_batch_processing_configurations.py", "Batch Processing Config"
        )

    def launch_custom_analysis_results(self):
        """Launch Custom Analysis Results."""
        self.launch_python_script(
            "examples/03_custom_analysis_saved_results.py", "Custom Analysis Results"
        )

    def launch_extending_falsification_criteria(self):
        """Launch Extending Falsification Criteria."""
        self.launch_python_script(
            "examples/04_extending_falsification_criteria.py",
            "Extending Falsification Criteria",
        )

    # Core Applications launch methods
    def launch_apgi_gui(self):
        """Launch APGI GUI."""
        self.launch_python_script("APGI_GUI.py", "APGI GUI")

    def launch_apgi_application_gui(self):
        """Launch APGI Application GUI."""
        self.launch_python_script("APGI_Application_GUI.py", "APGI Application GUI")

    def launch_assistant_gui(self):
        """Launch Assistant GUI."""
        self.launch_python_script("Assistant_GUI.py", "Assistant GUI")

    def launch_ai_assistant(self):
        """Launch AI Assistant."""
        self.launch_python_script("AI_Assistant.py", "AI Assistant")

    def launch_apgi_simulation_gui(self):
        """Launch APGI Simulation GUI."""
        self.launch_python_script("APGI_Simulation_GUI.py", "APGI Simulation GUI")

    def launch_psychological_states_gui(self):
        """Launch Psychological States GUI."""
        self.launch_python_script("Psychological_States_GUI.py", "Psychological States GUI")

    # Additional Visualization launch methods
    def launch_realtime_data_stream(self):
        """Launch Real-time Data Stream."""
        self.launch_python_script(
            "apgi_framework/gui/realtime_data_stream.py", "Real-time Data Stream"
        )

    # Configuration & Management launch methods
    def launch_error_logging_utils(self):
        """Launch Error Logging Utils."""
        self.launch_python_script(
            "apgi_framework/gui/error_logging_utils.py", "Error Logging Utils"
        )

    def launch_apgi_gui_main(self):
        """Launch apgi GUI Main."""
        self.launch_python_script("apgi_gui/main.py", "apgi GUI Main")

    # Development & Testing launch methods
    def launch_tests_gui(self):
        """Launch Tests GUI."""
        self.launch_python_script("Tests_GUI.py", "Tests GUI")

    def launch_utils_gui(self):
        """Launch Utils GUI."""
        self.launch_python_script("Utils_GUI.py", "Utils GUI")

    def launch_apgi_design(self):
        """Launch APGI Design."""
        self.launch_python_script("apps/apgi-design.py", "APGI Design")

    def launch_script_runner_gui(self):
        """Launch Script Runner GUI."""
        self.launch_python_script("utils/script_runner_gui.py", "Script Runner GUI")

    def launch_framework_testing_main(self):
        """Launch Framework Testing Main."""
        self.launch_python_script("apgi_framework/testing/main.py", "Framework Testing Main")

    def launch_gui_test_runner(self):
        """Launch GUI Test Runner."""
        self.launch_python_script("apgi_framework/testing/gui_test_runner.py", "GUI Test Runner")

    # CLI Tools & Framework launch methods
    def launch_apgi_gui_cli(self):
        """Launch apgi GUI CLI."""
        self.launch_python_script("apgi_gui/cli.py", "apgi GUI CLI")

    def launch_deployment_automation(self):
        """Launch Deployment Automation."""
        self.launch_python_script(
            "apgi_framework/deployment/automation.py", "Deployment Automation"
        )

    def launch_main_controller(self):
        """Launch Main Controller."""
        self.launch_python_script("apgi_framework/main_controller.py", "Main Controller")

    def launch_installation_validator(self):
        """Launch Installation Validator."""
        self.launch_python_script(
            "apgi_framework/installation_validator.py", "Installation Validator"
        )

    def launch_module_mode(self):
        """Launch Module Mode."""
        self.launch_python_script("apgi_framework/__main__.py", "Module Mode")

    # API & Backend launch methods
    def launch_api_server(self):
        """Launch API Server."""
        self.launch_python_script("api/main.py", "API Server")

    def launch_celery_app(self):
        """Launch Celery App."""
        self.launch_python_script("api/celery_app.py", "Celery App")

    # Testing & Benchmarks launch methods
    def launch_coverage_runner(self):
        """Launch Coverage Runner."""
        self.launch_python_script("benchmarks/run_coverage.py", "Coverage Runner")

    def launch_performance_benchmarks(self):
        """Launch Performance Benchmarks."""
        self.launch_python_script("benchmarks/test_performance.py", "Performance Benchmarks")

    def launch_critical_path_profiling(self):
        """Launch Critical Path Profiling."""
        self.launch_python_script(
            "benchmarks/critical_path_profiling.py", "Critical Path Profiling"
        )

    # Utilities & Tools launch methods
    def launch_backup_manager(self):
        """Launch Backup Manager."""
        self.launch_python_script("utils/backup_manager.py", "Backup Manager")

    def launch_diagnostics(self):
        """Launch Diagnostics."""
        self.launch_python_script("utils/diagnostics.py", "Diagnostics")

    def launch_performance_dashboard(self):
        """Launch Performance Dashboard."""
        self.launch_python_script(
            "utils/static_dashboard_generator.py", "Performance Dashboard Generator"
        )

    def launch_pipeline_visualization(self):
        """Launch Pipeline Visualization."""
        self.launch_python_script("utils/pipeline_visualization.py", "Pipeline Visualization")

    def launch_report_generator(self):
        """Launch Report Generator."""
        self.launch_python_script("utils/report_generator.py", "Report Generator")

    def launch_tutorial(self):
        """Launch Tutorial."""
        self.launch_python_script("utils/tutorial.py", "Tutorial")

    def launch_validate_app(self):
        """Launch Validate App."""
        self.launch_python_script("utils/validate_app.py", "Validate App")

    def launch_dependency_checker(self):
        """Launch Dependency Checker."""
        self.launch_python_script("utils/dependency_checker.py", "Dependency Checker")

    def launch_config_manager(self):
        """Launch Config Manager."""
        self.launch_python_script("utils/config_manager.py", "Config Manager")

    def launch_cache_manager(self):
        """Launch Cache Manager."""
        self.launch_python_script("utils/cache_manager.py", "Cache Manager")

    def launch_data_processor(self):
        """Launch Data Processor."""
        self.launch_python_script("utils/data_processor.py", "Data Processor")

    def launch_batch_processor(self):
        """Launch Batch Processor."""
        self.launch_python_script("utils/batch_processor.py", "Batch Processor")

    def launch_sample_data_generator(self):
        """Launch Sample Data Generator."""
        self.launch_python_script("utils/sample_data_generator.py", "Sample Data Generator")

    def launch_demo_analysis(self):
        """Launch Demo Analysis."""
        self.launch_python_script("utils/demo_analysis.py", "Demo Analysis")

    def launch_basic_simulation(self):
        """Launch Basic Simulation."""
        self.launch_python_script("utils/basic_simulation.py", "Basic Simulation")

    def launch_gap_analyzer(self):
        """Launch Gap Analyzer."""
        self.launch_python_script("utils/analyze_gaps.py", "Gap Analyzer")

    def launch_data_validation(self):
        """Launch Data Validation."""
        self.launch_python_script("utils/data_validation.py", "Data Validation")

    def launch_error_handler(self):
        """Launch Error Handler."""
        self.launch_python_script("utils/error_handler.py", "Error Handler")

    def launch_parameter_validator(self):
        """Launch Parameter Validator."""
        self.launch_python_script("utils/parameter_validator.py", "Parameter Validator")

    def launch_gui_testing_framework(self):
        """Launch GUI Testing Framework."""
        self.launch_python_script("utils/gui_testing_framework.py", "GUI Testing Framework")

    def launch_deployment_validator(self):
        """Launch Deployment Validator."""
        self.launch_python_script(
            "apgi_framework/deployment/deployment_validator.py", "Deployment Validator"
        )

    def launch_python_script(self, script_path, app_name):
        """Launch a Python script in a separate process."""
        try:
            print(f"Launching {app_name} ({script_path})...")

            # Get the absolute path to the script
            current_dir = Path(__file__).parent.absolute()
            script_full_path = current_dir / script_path

            if not script_full_path.exists():
                messagebox.showerror(
                    "File Not Found",
                    f"The script {script_path} was not found.\n"
                    f"Expected path: {script_full_path}",
                )
                return

            # Launch in a separate thread to avoid blocking the GUI
            def run_script():
                try:
                    process = subprocess.Popen(
                        [sys.executable, str(script_full_path)],
                        cwd=current_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    # Verify process actually started before showing success message
                    # Give it a brief moment to start and check if it's still running
                    import time

                    time.sleep(0.1)
                    if process.poll() is None:
                        # Process is still running - successful launch for long-running apps
                        print(f"Successfully launched {app_name}")
                    else:
                        # Process exited quickly - check exit code to determine success
                        return_code = process.poll()
                        if return_code == 0:
                            # Process completed successfully (e.g., example scripts)
                            print(f"Successfully launched {app_name}")
                        else:
                            # Process failed with non-zero exit code
                            stdout, stderr = process.communicate()
                            error_msg = stderr if stderr else stdout
                            print(f"Failed to launch {app_name}: {error_msg}")
                except Exception as e:
                    messagebox.showerror("Launch Error", f"Failed to launch {app_name}: {str(e)}")

            thread = threading.Thread(target=run_script, daemon=True)
            thread.start()

        except Exception as e:
            messagebox.showerror(
                "System Error", f"System error while launching {app_name}: {str(e)}"
            )

    def run(self):
        """Run the launcher."""
        self.root.mainloop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="APGI Framework GUI Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python GUI-Launcher.py              # Launch GUI window
    python GUI-Launcher.py --list       # List available applications
    python GUI-Launcher.py --version    # Show version
    """,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available GUI applications without launching GUI",
    )
    parser.add_argument("--version", action="store_true", help="Show version information")

    args = parser.parse_args()

    if args.version:
        print("APGI Framework GUI Launcher v1.0")
        print(f"Python version: {sys.version.split()[0]}")
        return

    # Check if we can display GUI (check for DISPLAY on Unix-like systems)
    if sys.platform.startswith("linux") and not sys.platform.startswith("darwin"):
        display = os.environ.get("DISPLAY")
        if not display:
            print("Error: No DISPLAY environment variable set.")
            print("GUI applications require a display server.")
            print("Use --list flag to see available applications without GUI.")
            sys.exit(1)

    # List mode
    if args.list:
        current_dir = Path(__file__).parent
        launcher = ComprehensiveGUILauncher()
        launcher.root.withdraw()  # Hide the GUI window in list mode
        print("\nAvailable GUI Applications:")
        print("=" * 60)
        for category, apps in launcher.gui_apps.items():
            print(f"\n{category}:")
            for app in apps:
                script_path = current_dir / app["file"]
                status = "✓" if script_path.exists() else "✗"
                print(f"  [{status}] {app['name']}")
                print(f"      {app['file']}")
        launcher.root.destroy()  # Clean up Tk resources
        return

    # Normal GUI mode
    try:
        print("Starting APGI Framework GUI Launcher...")
        launcher = ComprehensiveGUILauncher()
        launcher.run()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
