"""
APGI Framework Testing System

A comprehensive platform for implementing and validating the Allostatic Precision-Gated Ignition
(APGI) Framework through systematic testing.
"""

__version__ = "1.1.0"  # keep in step with __version__.py

__author__ = "APGI"

# Import all submodules for testing
from . import (
    analysis,
    clinical,
    data,
    falsification,
    gui,
    monitoring,
    processing,
    reporting,
    research,
    security,
    validation,
    visualization,
)
from .collaboration import CollaborationManager
from .computation import IntensiveComputation
from .core.active_inference import ActiveInferenceEngine
from .core.free_energy import FreeEnergyCalculator
from .core.predictive_processing import HierarchicalPredictor
from .engines import (
    APGIEquation,
    PrecisionCalculator,
    PrecisionWeighting,
    PredictionErrorProcessor,
    SomaticMarkerEngine,
    ThresholdManager,
)
from .exceptions import APGIFrameworkError, MathematicalError, SimulationError
from .fusion import DataFusion
from .notification import NotificationManager

__all__ = [
    "APGIEquation",
    "PrecisionCalculator",
    "PredictionErrorProcessor",
    "SomaticMarkerEngine",
    "ThresholdManager",
    "APGIFrameworkError",
    "MathematicalError",
    "SimulationError",
    "ActiveInferenceEngine",
    "FreeEnergyCalculator",
    "HierarchicalPredictor",
    "PrecisionWeighting",
]
