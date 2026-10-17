"""
Cross-platform utilities for resource loading and platform detection.

This module provides platform-agnostic functions for:
- Detecting the current operating system
- Determining if running as a bundled executable
- Resolving resource paths in both development and bundled environments
- Getting platform-appropriate directories for user-writable data

Error Handling:
- Graceful degradation for missing resources
- Fallback behavior for platform-specific features
- Logging for debugging bundled executables
"""

import logging
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Optional

# Configure logging for error handling
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_platform() -> str:
    """
    Get current platform identifier.

    Returns
    -------
    str
        'windows', 'macos', or 'linux'
    """
    system = platform.system().lower()

    if system == "windows":
        return "windows"
    elif system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    else:
        return "unknown"


def is_bundled() -> bool:
    """
    Check if running as bundled executable.

    PyInstaller sets sys.frozen and sys._MEIPASS when running as executable.
    py2app sets sys.frozen when running as .app bundle.

    Returns:
        True if bundled, False if running from source
    """
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def get_resource_path(relative_path: str) -> Path:
    """
    Get absolute path to resource, works for dev and bundled environments.

    In development: returns path relative to project root
    In bundled executable: returns path relative to temporary extraction directory

    Args:
        relative_path: Path relative to application root (e.g., 'config/default.yaml')

    Returns:
        Absolute path to resource

    Raises:
        ValueError: If relative_path is None or invalid

    Example:
        >>> config_path = get_resource_path('config/default.yaml')
        >>> if config_path.exists():
        ...     with open(config_path) as f:
        ...         config = yaml.safe_load(f)
    """
    try:
        if not relative_path:
            logger.warning("Empty resource path provided, returning base path")
            relative_path = ""

        if is_bundled():
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = Path(getattr(sys, "_MEIPASS", ""))
            logger.debug(f"Running as bundled executable, base path: {base_path}")
        else:
            # Running from source - use project root
            # This file is in apgi_simulation/, so go up one level
            base_path = Path(__file__).parent.parent
            logger.debug(f"Running from source, base path: {base_path}")

        resource_path = base_path / relative_path

        # Log warning if resource doesn't exist
        if not resource_path.exists():
            logger.warning(f"Resource path does not exist: {resource_path}")

        return resource_path

    except Exception as e:
        logger.error(f"Error resolving resource path '{relative_path}': {e}")
        # Return a safe fallback path
        return Path.cwd() / relative_path


def get_config_dir() -> Path:
    """
    Get platform-appropriate configuration directory.

    Returns user-writable directory for storing configuration files.
    Creates the directory if it doesn't exist.

    Returns:
        Path to config directory (user-writable)

    Platform-specific locations:
        - Windows: %APPDATA%/APGI System/
        - macOS: ~/Library/Application Support/APGI System/
        - Linux: ~/.config/apgi-system/

    Raises:
        PermissionError: If directory cannot be created due to permissions
        OSError: If directory creation fails for other reasons
    """
    app_name = "APGI System"
    current_platform = get_platform()

    try:
        if current_platform == "windows":
            # Use APPDATA on Windows
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
            config_dir = base / app_name
        elif current_platform == "macos":
            # Use Application Support on macOS
            config_dir = Path.home() / "Library" / "Application Support" / app_name
        else:
            # Use XDG config directory on Linux
            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            config_dir = base / "apgi-system"

        # Create directory if it doesn't exist
        config_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Config directory: {config_dir}")

        return config_dir

    except PermissionError as e:
        logger.error(f"Permission denied creating config directory: {e}")
        # Fallback to temp directory
        fallback_dir = Path.home() / ".apgi_config"
        logger.warning(f"Using fallback config directory: {fallback_dir}")
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            return fallback_dir
        except Exception as fallback_error:
            logger.critical(f"Cannot create fallback config directory: {fallback_error}")
            raise
    except Exception as e:
        logger.error(f"Error creating config directory: {e}")
        raise


def get_data_dir() -> Path:
    """
    Get platform-appropriate data directory.

    Returns user-writable directory for storing application data files.
    Creates the directory if it doesn't exist.

    Returns:
        Path to data directory (user-writable)

    Platform-specific locations:
        - Windows: %LOCALAPPDATA%/APGI System/
        - macOS: ~/Library/Application Support/APGI System/Data/
        - Linux: ~/.local/share/apgi-system/

    Raises:
        PermissionError: If directory cannot be created due to permissions
        OSError: If directory creation fails for other reasons
    """
    app_name = "APGI System"
    current_platform = get_platform()

    try:
        if current_platform == "windows":
            # Use LOCALAPPDATA on Windows
            base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            data_dir = base / app_name
        elif current_platform == "macos":
            # Use Application Support/Data on macOS
            data_dir = Path.home() / "Library" / "Application Support" / app_name / "Data"
        else:
            # Use XDG data directory on Linux
            base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
            data_dir = base / "apgi-system"

        # Create directory if it doesn't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data directory: {data_dir}")

        return data_dir

    except PermissionError as e:
        logger.error(f"Permission denied creating data directory: {e}")
        # Fallback to temp directory
        fallback_dir = Path.home() / ".apgi_data"
        logger.warning(f"Using fallback data directory: {fallback_dir}")
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            return fallback_dir
        except Exception as fallback_error:
            logger.critical(f"Cannot create fallback data directory: {fallback_error}")
            raise
    except Exception as e:
        logger.error(f"Error creating data directory: {e}")
        raise


def get_base_path() -> Path:
    """
    Get the base path of the application.

    Returns:
        Path to application base directory
    """
    try:
        if is_bundled():
            return Path(getattr(sys, "_MEIPASS", ""))
        else:
            return Path(__file__).parent.parent
    except Exception as e:
        logger.error(f"Error getting base path: {e}")
        return Path.cwd()


def load_resource_with_fallback(
    relative_path: str, fallback_content: Optional[str] = None
) -> Optional[str]:
    """
    Load resource file with fallback behavior.

    Attempts to load a resource file and returns its content.
    If the file doesn't exist or cannot be read, returns fallback content.

    Args:
        relative_path: Path relative to application root
        fallback_content: Content to return if resource cannot be loaded

    Returns:
        File content as string, or fallback_content if loading fails

    Example:
        >>> config_text = load_resource_with_fallback('config/default.yaml', 'threshold: 2.0')
    """
    try:
        resource_path = get_resource_path(relative_path)

        if not resource_path.exists():
            logger.warning(f"Resource not found: {relative_path}, using fallback")
            return fallback_content

        with open(resource_path, "r", encoding="utf-8") as f:
            content = f.read()
            logger.debug(f"Successfully loaded resource: {relative_path}")
            return content

    except PermissionError as e:
        logger.error(f"Permission denied reading resource '{relative_path}': {e}")
        return fallback_content
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading resource '{relative_path}': {e}")
        return fallback_content
    except Exception as e:
        logger.error(f"Error loading resource '{relative_path}': {e}")
        return fallback_content


def safe_write_file(file_path: Path, content: str, create_backup: bool = True) -> bool:
    """
    Safely write content to a file with error handling.

    Args:
        file_path: Path to file to write
        content: Content to write
        create_backup: Whether to create backup of existing file

    Returns:
        True if write succeeded, False otherwise

    Example:
        >>> success = safe_write_file(Path('config.yaml'), 'threshold: 2.0')
    """
    try:
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create backup if file exists
        if create_backup and file_path.exists():
            backup_path = file_path.with_suffix(file_path.suffix + ".backup")
            try:
                import shutil

                shutil.copy2(file_path, backup_path)
                logger.debug(f"Created backup: {backup_path}")
            except Exception as backup_error:
                logger.warning(f"Could not create backup: {backup_error}")

        # Write content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Successfully wrote file: {file_path}")
        return True

    except PermissionError as e:
        logger.error(f"Permission denied writing to '{file_path}': {e}")
        return False
    except OSError as e:
        logger.error(f"OS error writing to '{file_path}': {e}")
        return False
    except Exception as e:
        logger.error(f"Error writing to '{file_path}': {e}")
        return False


def check_required_libraries() -> Dict[str, bool]:
    """
    Check if required libraries are available.

    Returns:
        Dictionary with library names as keys and availability as boolean values

    Example:
        >>> libs = check_required_libraries()
        >>> if not libs['numpy']:
        ...     print("NumPy is not available")
    """
    required_libs = {
        "numpy": False,
        "yaml": False,
        "tkinter": False,
        "matplotlib": False,
        "scipy": False,
    }

    for lib_name in required_libs.keys():
        try:
            if lib_name == "yaml":
                __import__("yaml")
            else:
                __import__(lib_name)
            required_libs[lib_name] = True
            logger.debug(f"Library '{lib_name}' is available")
        except ImportError:
            logger.warning(f"Library '{lib_name}' is not available")
            required_libs[lib_name] = False

    return required_libs
