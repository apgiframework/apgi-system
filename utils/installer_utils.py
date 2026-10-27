"""
Installation utilities for the APGI system.

This module provides utilities for installing and managing the APGI system.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class InstallerError(Exception):
    """Custom exception for installer errors."""


def check_python_version(min_version: tuple = (3, 8)) -> bool:
    """Check if Python version meets requirements.

    Parameters
    ----------
    min_version : tuple, optional
        Minimum Python version, by default (3, 8)

    Returns
    -------
    bool
        True if version is sufficient
    """
    return sys.version_info >= min_version


def install_package(package: str, upgrade: bool = False) -> bool:
    """Install a Python package using pip.

    Parameters
    ----------
    package : str
        Package name to install
    upgrade : bool, optional
        Whether to upgrade if already installed, by default False

    Returns
    -------
    bool
        True if installation successful
    """
    try:
        cmd = ["pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package)

        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False


def install_requirements(requirements_file: str = "requirements.txt") -> bool:
    """Install packages from requirements file.

    Parameters
    ----------
    requirements_file : str, optional
        Path to requirements file, by default "requirements.txt"

    Returns
    -------
    bool
        True if installation successful
    """
    if not os.path.exists(requirements_file):
        print(f"Requirements file {requirements_file} not found")
        return False

    try:
        subprocess.run(
            ["pip", "install", "-r", requirements_file],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        return False


def check_package_installed(package: str) -> bool:
    """Check if a package is installed.

    Parameters
    ----------
    package : str
        Package name to check

    Returns
    -------
    bool
        True if package is installed
    """
    try:
        __import__(package)
        return True
    except ImportError:
        return False


def create_virtual_environment(venv_path: str = "venv") -> bool:
    """Create a Python virtual environment.

    Parameters
    ----------
    venv_path : str, optional
        Path for virtual environment, by default "venv"

    Returns
    -------
    bool
        True if creation successful
    """
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to create virtual environment: {e}")
        return False


def get_system_info() -> Dict[str, Any]:
    """Get system information for installation.

    Returns
    -------
    Dict[str, Any]
        System information
    """
    import platform

    import psutil

    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
    }


def validate_installation() -> Dict[str, bool]:
    """Validate APGI system installation.

    Returns
    -------
    Dict[str, bool]
        Validation results
    """
    results = {}

    # Check core packages
    core_packages = [
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "sklearn",
        "networkx",
        "torch",
        "jax",
    ]

    for package in core_packages:
        results[f"core_{package}"] = check_package_installed(package)

    # Check API packages
    api_packages = ["fastapi", "uvicorn", "pydantic", "sqlalchemy"]

    for package in api_packages:
        results[f"api_{package}"] = check_package_installed(package)

    # Check system packages
    system_packages = ["redis", "psycopg2"]

    for package in system_packages:
        results[f"system_{package}"] = check_package_installed(package)

    # Check GUI
    results["gui_tkinter"] = check_package_installed("tkinter")

    return results


def extract_version_from_pyproject(pyproject_path: str = "pyproject.toml") -> str:
    """Extract version from pyproject.toml file.

    Parameters
    ----------
    pyproject_path : str, optional
        Path to pyproject.toml file, by default "pyproject.toml"

    Returns
    -------
    str
        Version string
    """
    pyproject_file = Path(pyproject_path)

    if not pyproject_file.exists():
        raise InstallerError(f"pyproject.toml file not found: {pyproject_path}")

    try:
        import tomllib

        with open(pyproject_file, "rb") as fh:
            config = tomllib.load(fh)
        version = config.get("project", {}).get("version")

        if not version:
            raise InstallerError("Version not found in pyproject.toml")

        return str(version)
    except InstallerError:
        raise
    except Exception as e:
        raise InstallerError(f"Failed to parse pyproject.toml: {e}")


def normalize_path_for_inno(path: str) -> str:
    """Normalize path for Inno Setup script.

    Parameters
    ----------
    path : str
        Path to normalize

    Returns
    -------
    str
        Normalized path
    """
    # Convert forward slashes to backslashes for Windows
    normalized = path.replace("/", "\\")

    # Ensure absolute paths start with drive letter
    if normalized.startswith("\\"):
        normalized = "C:" + normalized

    return normalized


def generate_registry_entries(
    app_name: str, version: str, install_path: str
) -> List[Dict[str, str]]:
    """Generate Windows registry entries for application.

    Parameters
    ----------
    app_name : str
        Application name
    version : str
        Application version
    install_path : str
        Installation path

    Returns
    -------
    List[Dict[str, str]]
        Registry entries
    """
    entries = [
        {
            "key": f"SOFTWARE\\{app_name}",
            "value_name": "DisplayName",
            "value_data": app_name,
            "value_type": "REG_SZ",
        },
        {
            "key": f"SOFTWARE\\{app_name}",
            "value_name": "DisplayVersion",
            "value_data": version,
            "value_type": "REG_SZ",
        },
        {
            "key": f"SOFTWARE\\{app_name}",
            "value_name": "InstallLocation",
            "value_data": install_path,
            "value_type": "REG_SZ",
        },
        {
            "key": f"SOFTWARE\\{app_name}",
            "value_name": "Publisher",
            "value_data": "APGI System",
            "value_type": "REG_SZ",
        },
    ]

    return entries


def generate_inno_setup_script(
    app_name: str,
    version: str,
    source_dir: str,
    output_dir: str,
    executable_name: Optional[str] = None,
) -> str:
    """Generate Inno Setup script for Windows installer.

    Parameters
    ----------
    app_name : str
        Application name
    version : str
        Application version
    source_dir : str
        Source directory
    output_dir : str
        Output directory
    executable_name : str, optional
        Executable name, by default None

    Returns
    -------
    str
        Inno Setup script content
    """
    if executable_name is None:
        executable_name = app_name.lower().replace(" ", "_") + ".exe"

    script = f"""
[Setup]
AppName={app_name}
AppVersion={version}
DefaultDirName={{pf}}\\{app_name}
DefaultGroupName={app_name}
OutputDir={output_dir}
OutputBaseFilename={app_name}_setup_v{version}
SetupIconFile=resources\\icons\\apgi.ico
Compression=lzma
SolidCompression=yes

[Files]
Source: "{source_dir}\\*"; DestDir: "{{app}}"; Flags: recursesubdirs createallsubdirs
Source: "{source_dir}\\{executable_name}"; DestDir: "{{app}}"; Flags: ignoreversion

[Icons]
Name: "{{group}}\\{app_name}"; Filename: "{{app}}\\{executable_name}"
Name: "{{commondesktop}}\\{app_name}"; Filename: "{{app}}\\{executable_name}"

[Registry]
"""

    # Add registry entries
    registry_entries = generate_registry_entries(app_name, version, "{{app}}")
    for entry in registry_entries:
        script += f'Root: HKCU; Subkey: "{entry["key"]}"; ValueType: {entry["value_type"]}; ValueName: "{entry["value_name"]}"; ValueData: "{entry["value_data"]}"\n'

    script += f"""
[Run]
Filename: "{{app}}\\{executable_name}"; Description: "Launch application"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{{app}}"
"""

    return script.strip()
