"""Adaptive staircase algorithms and stimulus control for parameter estimation tasks.

This module provides adaptive algorithms for optimal stimulus selection and
stimulus control systems for the three core parameter estimation tasks.
"""

from ..research.core_mechanisms.phase_transition import (
    ConsciousnessDetector,
    PhaseTransitionAnalyzer,
    SomaticAgent,
)
from .quest_plus_staircase import QuestPlusStaircase
from .stimulus_generators import (
    CO2PuffController,
    GaborPatchGenerator,
    HeartbeatSynchronizer,
    ToneGenerator,
)
from .task_control import (
    PrecisionTimer,
    ResponseCollector,
    SessionManager,
    TaskStateMachine,
)

__all__ = [
    "QuestPlusStaircase",
    "GaborPatchGenerator",
    "ToneGenerator",
    "CO2PuffController",
    "HeartbeatSynchronizer",
    "PrecisionTimer",
    "TaskStateMachine",
    "ResponseCollector",
    "SessionManager",
]
