"""Documentation and training system for APGI Framework.

Provides comprehensive documentation, troubleshooting guides, and
in-system help for the parameter estimation system.
"""

from .help_system import InSystemHelpSystem
from .troubleshooting import TroubleshootingGuide
from .user_manual import ParameterEstimationUserManual

__all__ = [
    "ParameterEstimationUserManual",
    "TroubleshootingGuide",
    "InSystemHelpSystem",
]
