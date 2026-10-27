"""Error logging utilities for APGI Framework.

Provides integration between error handling and configuration system.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_error_log_dir(gui_config_path: Optional[Path] = None) -> Path:
    """Get the configured error log directory.

    Args:
        gui_config_path: Optional path to GUI config file (unused)

    Returns:
        Path to the error log directory
    """
    # Use default log directory as GUI config is no longer available
    log_dir = Path.home() / ".apgi" / "error_logs"
    logger.debug(f"Using default error log directory: {log_dir}")
    return log_dir
