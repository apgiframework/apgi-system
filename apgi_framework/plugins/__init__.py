"""Plugin architecture for APGI Framework.

Provides extensible plugin system for third-party engine registration.
"""

from apgi_framework.plugins.manager import PluginManager
from apgi_framework.plugins.registry import PluginRegistry, PluginSpec

__all__ = ["PluginRegistry", "PluginSpec", "PluginManager"]
