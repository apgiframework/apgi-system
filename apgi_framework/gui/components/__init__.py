"""Components package for APGI Framework GUI.

This package contains modular GUI components extracted from the monolithic
apgi_falsification_gui.py file to improve maintainability and testability.
"""

from .loading_indicator import LoadingIndicator, ProgressDialog
from .logging_panel import LoggingPanel, create_logging_panel
from .main_gui_controller import MainGUIController, create_main_gui_controller
from .parameter_config_panel import ParameterConfigPanel, create_parameter_config_panel
from .results_visualization_panel import (
    ResultsVisualizationPanel,
    create_results_visualization_panel,
)
from .test_execution_panel import ExecutionPanel, create_test_execution_panel

__all__ = [
    # Main components
    "ParameterConfigPanel",
    "ExecutionPanel",
    "ResultsVisualizationPanel",
    "LoggingPanel",
    "MainGUIController",
    "LoadingIndicator",
    "ProgressDialog",
    # Factory functions
    "create_parameter_config_panel",
    "create_test_execution_panel",
    "create_results_visualization_panel",
    "create_logging_panel",
    "create_main_gui_controller",
]

# Version information
__version__ = "1.0.0"
__author__ = "APGI Framework Team"
__description__ = "Modular GUI components for APGI Framework"
