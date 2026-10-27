"""Installation Manager for automated dependency installation.

Handles installation of LSL, Stan, eye tracker SDKs, and other dependencies
required for the APGI Framework parameter estimation system.
"""

import logging
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional


class DependencyStatus(Enum):
    """Status of a dependency installation."""

    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DependencyInfo:
    """Information about a system dependency."""

    name: str
    description: str
    required: bool = True
    install_command: Optional[str] = None
    check_command: Optional[str] = None
    status: DependencyStatus = DependencyStatus.NOT_INSTALLED
    version: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class InstallationReport:
    """Report of installation process."""

    dependencies: Dict[str, DependencyInfo] = field(default_factory=dict)
    success: bool = False
    total_dependencies: int = 0
    installed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0

    def add_dependency(self, dep: DependencyInfo) -> None:
        """Add dependency to report."""
        self.dependencies[dep.name] = dep
        self.total_dependencies += 1

        if dep.status == DependencyStatus.INSTALLED:
            self.installed_count += 1
        elif dep.status == DependencyStatus.FAILED:
            self.failed_count += 1
        elif dep.status == DependencyStatus.SKIPPED:
            self.skipped_count += 1


class InstallationManager:
    """Manages automated installation of APGI Framework dependencies.

    Handles installation of:
    - Python packages (numpy, scipy, pystan, etc.)
    - Lab Streaming Layer (LSL)
    - Stan/PyMC3 for Bayesian modeling
    - Eye tracker SDKs (Tobii, EyeLink, etc.)
    - EEG system interfaces
    """

    def __init__(self, install_dir: Optional[Path] = None):
        """Initialize installation manager.

        Args:
            install_dir: Directory for installing dependencies. If None, uses default.
        """
        self.logger = logging.getLogger(__name__)
        self.install_dir = install_dir or Path.home() / ".apgi_framework"
        self.install_dir.mkdir(parents=True, exist_ok=True)

        self.system = platform.system()
        self.python_version = sys.version_info

        # Define dependencies
        self.dependencies = self._define_dependencies()

    def _define_dependencies(self) -> Dict[str, DependencyInfo]:
        """Define all system dependencies."""
        deps = {}

        # Core Python packages
        deps["numpy"] = DependencyInfo(
            name="numpy",
            description="Numerical computing library",
            required=True,
            install_command="python -m pip install numpy>=1.20.0",
            check_command='python -c "import numpy; print(numpy.__version__)"',
        )

        deps["scipy"] = DependencyInfo(
            name="scipy",
            description="Scientific computing library",
            required=True,
            install_command="python -m pip install scipy>=1.7.0",
            check_command='python -c "import scipy; print(scipy.__version__)"',
        )

        deps["pandas"] = DependencyInfo(
            name="pandas",
            description="Data manipulation library",
            required=True,
            install_command="python -m pip install pandas>=1.3.0",
            check_command='python -c "import pandas; print(pandas.__version__)"',
        )

        # Bayesian modeling
        deps["pystan"] = DependencyInfo(
            name="pystan",
            description="Python interface to Stan for Bayesian modeling",
            required=True,
            install_command="python -m pip install pystan>=3.0.0",
            check_command='python -c "import stan; print(stan.__version__)"',
        )

        deps["arviz"] = DependencyInfo(
            name="arviz",
            description="Bayesian model diagnostics and visualization",
            required=True,
            install_command="python -m pip install arviz>=0.11.0",
            check_command='python -c "import arviz; print(arviz.__version__)"',
        )

        # Lab Streaming Layer
        deps["pylsl"] = DependencyInfo(
            name="pylsl",
            description="Lab Streaming Layer for real-time data streaming",
            required=True,
            install_command="python -m pip install pylsl>=1.16.0",
            check_command='python -c "import pylsl; print(pylsl.__version__)"',
        )

        # Signal processing
        deps["mne"] = DependencyInfo(
            name="mne",
            description="EEG/MEG analysis library",
            required=True,
            install_command="python -m pip install mne>=1.0.0",
            check_command='python -c "import mne; print(mne.__version__)"',
        )

        # Eye tracking (optional)
        deps["tobii_research"] = DependencyInfo(
            name="tobii_research",
            description="Tobii eye tracker SDK",
            required=False,
            install_command="python -m pip install tobii-research",
            check_command='python -c "import tobii_research; print(tobii_research.__version__)"',
        )

        # Visualization
        deps["matplotlib"] = DependencyInfo(
            name="matplotlib",
            description="Plotting library",
            required=True,
            install_command="python -m pip install matplotlib>=3.4.0",
            check_command='python -c "import matplotlib; print(matplotlib.__version__)"',
        )

        deps["seaborn"] = DependencyInfo(
            name="seaborn",
            description="Statistical visualization",
            required=True,
            install_command="python -m pip install seaborn>=0.11.0",
            check_command='python -c "import seaborn; print(seaborn.__version__)"',
        )

        # GUI
        deps["pyqt5"] = DependencyInfo(
            name="pyqt5",
            description="GUI framework",
            required=True,
            install_command="python -m pip install PyQt5>=5.15.0",
            check_command='python -c "import PyQt5.QtCore; print(PyQt5.QtCore.PYQT_VERSION_STR)"',
        )

        return deps

    def check_dependency(self, dep: DependencyInfo) -> DependencyInfo:
        """Check if a dependency is installed.

        Args:
            dep: Dependency information.

        Returns:
            Updated dependency information with status.
        """
        if not dep.check_command:
            dep.status = DependencyStatus.SKIPPED
            return dep

        try:
            # Use shlex.split() to safely parse command arguments
            command_args = shlex.split(dep.check_command)
            result = subprocess.run(command_args, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                dep.status = DependencyStatus.INSTALLED
                dep.version = result.stdout.strip()
                self.logger.info(f"{dep.name} is installed (version: {dep.version})")
            else:
                dep.status = DependencyStatus.NOT_INSTALLED
                self.logger.warning(f"{dep.name} is not installed")

        except Exception as e:
            dep.status = DependencyStatus.NOT_INSTALLED
            dep.error_message = str(e)
            self.logger.warning(f"Error checking {dep.name}: {e}")

        return dep

    def install_dependency(self, dep: DependencyInfo) -> DependencyInfo:
        """Install a dependency.

        Args:
            dep: Dependency information.

        Returns:
            Updated dependency information with installation status.
        """
        if not dep.install_command:
            dep.status = DependencyStatus.SKIPPED
            dep.error_message = "No install command available"
            return dep

        self.logger.info(f"Installing {dep.name}...")

        try:
            # Use shlex.split() to safely parse command arguments
            command_args = shlex.split(dep.install_command)
            result = subprocess.run(
                command_args,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
            )

            if result.returncode == 0:
                # Verify installation
                dep = self.check_dependency(dep)
                if dep.status == DependencyStatus.INSTALLED:
                    self.logger.info(f"Successfully installed {dep.name}")
                else:
                    dep.status = DependencyStatus.FAILED
                    dep.error_message = "Installation completed but verification failed"
            else:
                dep.status = DependencyStatus.FAILED
                dep.error_message = result.stderr
                self.logger.error(f"Failed to install {dep.name}: {result.stderr}")

        except subprocess.TimeoutExpired:
            dep.status = DependencyStatus.FAILED
            dep.error_message = "Installation timeout"
            self.logger.error(f"Installation of {dep.name} timed out")
        except Exception as e:
            dep.status = DependencyStatus.FAILED
            dep.error_message = str(e)
            self.logger.error(f"Error installing {dep.name}: {e}")

        return dep

    def check_all_dependencies(self) -> InstallationReport:
        """Check status of all dependencies.

        Returns:
            Installation report with dependency status.
        """
        self.logger.info("Checking all dependencies...")
        report = InstallationReport()

        for name, dep in self.dependencies.items():
            updated_dep = self.check_dependency(dep)
            self.dependencies[name] = updated_dep
            report.add_dependency(updated_dep)

        return report

    def install_all_dependencies(self, skip_optional: bool = False) -> InstallationReport:
        """Install all missing dependencies.

        Args:
            skip_optional: If True, skip optional dependencies.

        Returns:
            Installation report.
        """
        self.logger.info("Starting dependency installation...")
        report = InstallationReport()

        for name, dep in self.dependencies.items():
            # Skip optional dependencies if requested
            if skip_optional and not dep.required:
                dep.status = DependencyStatus.SKIPPED
                report.add_dependency(dep)
                continue

            # Check if already installed
            dep = self.check_dependency(dep)

            # Install if not installed
            if dep.status == DependencyStatus.NOT_INSTALLED:
                dep = self.install_dependency(dep)

            self.dependencies[name] = dep
            report.add_dependency(dep)

        # Determine overall success
        report.success = report.failed_count == 0 and all(
            dep.status != DependencyStatus.NOT_INSTALLED
            for dep in report.dependencies.values()
            if dep.required
        )

        self.logger.info(
            f"Installation complete: {report.installed_count} installed, "
            f"{report.failed_count} failed, {report.skipped_count} skipped"
        )

        return report

    def install_lsl(self) -> bool:
        """Install Lab Streaming Layer.

        Returns:
            True if successful, False otherwise.
        """
        self.logger.info("Installing Lab Streaming Layer...")

        dep = self.dependencies.get("pylsl")
        if not dep:
            return False

        dep = self.install_dependency(dep)
        self.dependencies["pylsl"] = dep

        return dep.status == DependencyStatus.INSTALLED

    def install_stan(self) -> bool:
        """Install Stan for Bayesian modeling.

        Returns:
            True if successful, False otherwise.
        """
        self.logger.info("Installing Stan...")

        dep = self.dependencies.get("pystan")
        if not dep:
            return False

        dep = self.install_dependency(dep)
        self.dependencies["pystan"] = dep

        return dep.status == DependencyStatus.INSTALLED

    def install_eye_tracker_sdk(self, tracker_type: str = "tobii") -> bool:
        """Install eye tracker SDK.

        Args:
            tracker_type: Type of eye tracker ('tobii', 'eyelink', etc.).

        Returns:
            True if successful, False otherwise.
        """
        self.logger.info(f"Installing {tracker_type} eye tracker SDK...")

        if tracker_type.lower() == "tobii":
            dep = self.dependencies.get("tobii_research")
            if dep:
                dep = self.install_dependency(dep)
                self.dependencies["tobii_research"] = dep
                return dep.status == DependencyStatus.INSTALLED

        self.logger.warning(f"Eye tracker SDK for {tracker_type} not configured")
        return False

    def generate_requirements_file(self, output_path: Path) -> None:
        """Generate requirements.txt file from dependencies.

        Args:
            output_path: Path to save requirements.txt.
        """
        self.logger.info(f"Generating requirements file: {output_path}")

        with open(output_path, "w") as f:
            f.write("# APGI Framework Parameter Estimation System Requirements\n")
            f.write("# Generated automatically by InstallationManager\n\n")

            # Required dependencies
            f.write("# Core dependencies\n")
            for name, dep in self.dependencies.items():
                if dep.required and dep.install_command:
                    # Extract package specification from install command
                    if "pip install" in dep.install_command:
                        package = dep.install_command.replace("pip install", "").strip()
                        f.write(f"{package}\n")

            f.write("\n# Optional dependencies\n")
            for name, dep in self.dependencies.items():
                if not dep.required and dep.install_command:
                    if "pip install" in dep.install_command:
                        package = dep.install_command.replace("pip install", "").strip()
                        f.write(f"# {package}\n")

        self.logger.info("Requirements file generated successfully")

    def get_installation_status(self) -> Dict[str, str]:
        """Get current installation status of all dependencies.

        Returns:
            Dictionary mapping dependency names to status strings.
        """
        status = {}
        for name, dep in self.dependencies.items():
            status[name] = f"{dep.status.value} ({dep.version or 'N/A'})"
        return status
