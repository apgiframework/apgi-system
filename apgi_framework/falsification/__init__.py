"""
APGI Framework Testing Module

This module implements the primary falsification testing framework for the APGI
(Allostatic Precision-Gated Ignition) Framework, including consciousness assessment,
neural signature validation, and experimental control mechanisms.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

# Other supporting modules
from .ai_acc_validation import AIACCValidator
from .consciousness_assessment import (
    ConsciousnessAssessment,
    ConsciousnessAssessmentSimulator,
    ConsciousnessValidator,
)
from .consciousness_without_ignition import ConsciousnessWithoutIgnitionTest
from .edge_case_interpreter import EdgeCaseInterpreter, EdgeCaseType, FrameworkBoundary
from .experimental_control import ExperimentalControlValidator
from .mock_engine import FalsificationEngine

# Core test implementations
from .primary_falsification import FalsificationResult, PrimaryFalsificationTest
from .result_interpretation import FalsificationInterpreter, ResultLogger
from .soma_bias import SomaBiasTest
from .threshold_insensitivity import ThresholdInsensitivityTest

__all__ = [
    "ConsciousnessAssessment",
    "ConsciousnessAssessmentSimulator",
    "ConsciousnessValidator",
    "AIACCValidator",
    "ExperimentalControlValidator",
    "FalsificationInterpreter",
    "ResultLogger",
    "EdgeCaseInterpreter",
    "EdgeCaseType",
    "FrameworkBoundary",
    "ConsciousnessWithoutIgnitionTest",
    "ThresholdInsensitivityTest",
    "SomaBiasTest",
    "PrimaryFalsificationTest",
    "FalsificationResult",
    "FalsificationEngine",
]
