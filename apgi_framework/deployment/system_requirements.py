"""System Requirements Validator for compatibility checking and performance assessment.

Validates system requirements including hardware, software, and performance
capabilities for running the APGI Framework parameter estimation system.
"""

import logging
import platform
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import psutil


class RequirementLevel(Enum):
    """Level of requirement severity."""

    MINIMUM = "minimum"
    RECOMMENDED = "recommended"
    OPTIMAL = "optimal"


class RequirementStatus(Enum):
    """Status of requirement check."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class SystemRequirement:
    """Definition of a system requirement."""

    name: str
    description: str
    level: RequirementLevel
    status: RequirementStatus = RequirementStatus.FAILED
    actual_value: Optional[str] = None
    required_value: Optional[str] = None
    message: Optional[str] = None


@dataclass
class SystemRequirementsReport:
    """Report of system requirements validation."""

    requirements: List[SystemRequirement] = field(default_factory=list)
    overall_status: RequirementStatus = RequirementStatus.FAILED

    system_info: Dict[str, str] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def passed_count(self) -> int:
        """Number of passed requirements."""
        return sum(1 for req in self.requirements if req.status == RequirementStatus.PASSED)

    @property
    def warning_count(self) -> int:
        """Number of warnings."""
        return sum(1 for req in self.requirements if req.status == RequirementStatus.WARNING)

    @property
    def failed_count(self) -> int:
        """Number of failed requirements."""
        return sum(1 for req in self.requirements if req.status == RequirementStatus.FAILED)

    @property
    def can_run(self) -> bool:
        """Whether system can run the framework."""
        # Can run if no minimum requirements failed
        minimum_failed = any(
            req.status == RequirementStatus.FAILED and req.level == RequirementLevel.MINIMUM
            for req in self.requirements
        )
        return not minimum_failed


class SystemRequirementsValidator:
    """Validates system requirements for APGI Framework.

    Checks hardware capabilities, software dependencies, and performance
    characteristics to ensure the system can run parameter estimation tasks.
    """

    def __init__(self) -> None:
        """Initialize system requirements validator."""
        self.logger = logging.getLogger(__name__)

        # Define minimum requirements
        self.min_ram_gb = 8
        self.rec_ram_gb = 16
        self.opt_ram_gb = 32

        self.min_cpu_cores = 4
        self.rec_cpu_cores = 8
        self.opt_cpu_cores = 16

        self.min_disk_gb = 10
        self.rec_disk_gb = 50
        self.opt_disk_gb = 100

        self.min_python_version = (3, 8)
        self.rec_python_version = (3, 9)
        self.opt_python_version = (3, 10)

    def validate_system(self) -> SystemRequirementsReport:
        """Validate all system requirements.

        Returns:
            Complete system requirements report.
        """
        self.logger.info("Validating system requirements...")

        report = SystemRequirementsReport()

        # Collect system information
        report.system_info = self._collect_system_info()

        # Check hardware requirements
        report.requirements.extend(self._check_hardware_requirements())

        # Check software requirements
        report.requirements.extend(self._check_software_requirements())

        # Check performance capabilities
        report.performance_metrics = self._assess_performance()
        report.requirements.extend(self._check_performance_requirements(report.performance_metrics))

        # Determine overall status
        if report.failed_count > 0:
            report.overall_status = RequirementStatus.FAILED
        elif report.warning_count > 0:
            report.overall_status = RequirementStatus.WARNING
        else:
            report.overall_status = RequirementStatus.PASSED

        self.logger.info(f"System validation complete: {report.overall_status.value}")
        self.logger.info(
            f"Passed: {report.passed_count}, Warnings: {report.warning_count}, Failed: {report.failed_count}"
        )

        return report

    def _collect_system_info(self) -> Dict[str, str]:
        """Collect basic system information."""
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
        }

    def _check_hardware_requirements(self) -> List[SystemRequirement]:
        """Check hardware requirements."""
        requirements = []

        # RAM check
        ram_gb = psutil.virtual_memory().total / (1024**3)
        ram_req = SystemRequirement(
            name="RAM",
            description="System memory (RAM)",
            level=RequirementLevel.MINIMUM,
            actual_value=f"{ram_gb:.1f} GB",
            required_value=f"{self.min_ram_gb} GB minimum, {self.rec_ram_gb} GB recommended",
        )

        if ram_gb >= self.opt_ram_gb:
            ram_req.status = RequirementStatus.PASSED
            ram_req.message = "Optimal RAM available"
        elif ram_gb >= self.rec_ram_gb:
            ram_req.status = RequirementStatus.PASSED
            ram_req.message = "Recommended RAM available"
        elif ram_gb >= self.min_ram_gb:
            ram_req.status = RequirementStatus.WARNING
            ram_req.message = "Minimum RAM met, but more recommended for optimal performance"
        else:
            ram_req.status = RequirementStatus.FAILED
            ram_req.message = f"Insufficient RAM: {ram_gb:.1f} GB < {self.min_ram_gb} GB minimum"

        requirements.append(ram_req)

        # CPU cores check
        cpu_cores: Optional[int] = psutil.cpu_count(logical=False) or psutil.cpu_count()
        cpu_req = SystemRequirement(
            name="CPU Cores",
            description="Number of CPU cores",
            level=RequirementLevel.MINIMUM,
            actual_value=str(cpu_cores),
            required_value=f"{self.min_cpu_cores} minimum, {self.rec_cpu_cores} recommended",
        )

        if cpu_cores is not None and cpu_cores >= self.opt_cpu_cores:
            cpu_req.status = RequirementStatus.PASSED
            cpu_req.message = "Optimal CPU cores available"
        elif cpu_cores is not None and cpu_cores >= self.rec_cpu_cores:
            cpu_req.status = RequirementStatus.PASSED
            cpu_req.message = "Recommended CPU cores available"
        elif cpu_cores is not None and cpu_cores >= self.min_cpu_cores:
            cpu_req.status = RequirementStatus.WARNING
            cpu_req.message = "Minimum CPU cores met, but more recommended"
        else:
            cpu_req.status = RequirementStatus.FAILED
            cpu_req.message = f"Insufficient CPU cores: {cpu_cores} < {self.min_cpu_cores} minimum"

        requirements.append(cpu_req)

        # Disk space check
        disk_gb = psutil.disk_usage("/").free / (1024**3)
        disk_req = SystemRequirement(
            name="Disk Space",
            description="Available disk space",
            level=RequirementLevel.MINIMUM,
            actual_value=f"{disk_gb:.1f} GB",
            required_value=f"{self.min_disk_gb} GB minimum, {self.rec_disk_gb} GB recommended",
        )

        if disk_gb >= self.opt_disk_gb:
            disk_req.status = RequirementStatus.PASSED
            disk_req.message = "Optimal disk space available"
        elif disk_gb >= self.rec_disk_gb:
            disk_req.status = RequirementStatus.PASSED
            disk_req.message = "Recommended disk space available"
        elif disk_gb >= self.min_disk_gb:
            disk_req.status = RequirementStatus.WARNING
            disk_req.message = "Minimum disk space met, but more recommended"
        else:
            disk_req.status = RequirementStatus.FAILED
            disk_req.message = (
                f"Insufficient disk space: {disk_gb:.1f} GB < {self.min_disk_gb} GB minimum"
            )

        requirements.append(disk_req)

        return requirements

    def _check_software_requirements(self) -> List[SystemRequirement]:
        """Check software requirements."""
        requirements = []

        # Python version check
        python_version = sys.version_info[:2]
        python_req = SystemRequirement(
            name="Python Version",
            description="Python interpreter version",
            level=RequirementLevel.MINIMUM,
            actual_value=f"{python_version[0]}.{python_version[1]}",
            required_value=f"{self.min_python_version[0]}.{self.min_python_version[1]}+ minimum",
        )

        if python_version >= self.opt_python_version:
            python_req.status = RequirementStatus.PASSED
            python_req.message = "Optimal Python version"
        elif python_version >= self.rec_python_version:
            python_req.status = RequirementStatus.PASSED
            python_req.message = "Recommended Python version"
        elif python_version >= self.min_python_version:
            python_req.status = RequirementStatus.WARNING
            python_req.message = "Minimum Python version met, but newer version recommended"
        else:
            python_req.status = RequirementStatus.FAILED
            python_req.message = f"Python version too old: {python_version[0]}.{python_version[1]} < {self.min_python_version[0]}.{self.min_python_version[1]}"

        requirements.append(python_req)

        # Operating system check
        os_name = platform.system()
        os_req = SystemRequirement(
            name="Operating System",
            description="Operating system compatibility",
            level=RequirementLevel.MINIMUM,
            actual_value=os_name,
            required_value="Windows, Linux, or macOS",
        )

        if os_name in ["Windows", "Linux", "Darwin"]:
            os_req.status = RequirementStatus.PASSED
            os_req.message = f"Supported operating system: {os_name}"
        else:
            os_req.status = RequirementStatus.WARNING
            os_req.message = f"Untested operating system: {os_name}"

        requirements.append(os_req)

        return requirements

    def _assess_performance(self) -> Dict[str, float]:
        """Assess system performance capabilities."""
        metrics = {}

        # CPU frequency
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            metrics["cpu_freq_mhz"] = cpu_freq.current

        # CPU usage
        metrics["cpu_usage_percent"] = psutil.cpu_percent(interval=1)

        # Memory usage
        mem = psutil.virtual_memory()
        metrics["memory_usage_percent"] = mem.percent
        metrics["memory_available_gb"] = mem.available / (1024**3)

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            metrics["disk_read_mb_s"] = disk_io.read_bytes / (1024**2)
            metrics["disk_write_mb_s"] = disk_io.write_bytes / (1024**2)

        return metrics

    def _check_performance_requirements(self, metrics: Dict[str, float]) -> List[SystemRequirement]:
        """Check performance requirements."""
        requirements = []

        # CPU frequency check
        if "cpu_freq_mhz" in metrics:
            cpu_freq = metrics["cpu_freq_mhz"]
            freq_req = SystemRequirement(
                name="CPU Frequency",
                description="CPU clock speed",
                level=RequirementLevel.RECOMMENDED,
                actual_value=f"{cpu_freq:.0f} MHz",
                required_value="2000 MHz minimum, 3000 MHz recommended",
            )

            if cpu_freq >= 3000:
                freq_req.status = RequirementStatus.PASSED
                freq_req.message = "Good CPU frequency"
            elif cpu_freq >= 2000:
                freq_req.status = RequirementStatus.WARNING
                freq_req.message = "CPU frequency adequate but higher recommended"
            else:
                freq_req.status = RequirementStatus.WARNING
                freq_req.message = "Low CPU frequency may impact performance"

            requirements.append(freq_req)

        # Memory availability check
        if "memory_available_gb" in metrics:
            mem_avail = metrics["memory_available_gb"]
            mem_req = SystemRequirement(
                name="Available Memory",
                description="Currently available RAM",
                level=RequirementLevel.RECOMMENDED,
                actual_value=f"{mem_avail:.1f} GB",
                required_value="4 GB minimum available",
            )

            if mem_avail >= 8:
                mem_req.status = RequirementStatus.PASSED
                mem_req.message = "Sufficient memory available"
            elif mem_avail >= 4:
                mem_req.status = RequirementStatus.WARNING
                mem_req.message = "Limited memory available, close other applications"
            else:
                mem_req.status = RequirementStatus.FAILED
                mem_req.message = "Insufficient memory available, close applications"

            requirements.append(mem_req)

        return requirements

    def check_real_time_capability(self) -> Tuple[bool, str]:
        """Check if system can handle real-time processing.

        Returns:
            Tuple of (capable, message).
        """
        # Check CPU and memory
        cpu_cores: Optional[int] = psutil.cpu_count(logical=False) or psutil.cpu_count()
        ram_gb = psutil.virtual_memory().total / (1024**3)

        if cpu_cores is not None and cpu_cores >= self.rec_cpu_cores and ram_gb >= self.rec_ram_gb:
            return True, "System capable of real-time processing"
        elif (
            cpu_cores is not None and cpu_cores >= self.min_cpu_cores and ram_gb >= self.min_ram_gb
        ):
            return (
                True,
                "System may handle real-time processing with reduced performance",
            )
        else:
            return False, "System may not handle real-time processing reliably"

    def estimate_max_sampling_rate(self) -> int:
        """Estimate maximum sampling rate system can handle.

        Returns:
            Estimated maximum sampling rate in Hz.
        """
        cpu_cores: int = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
        ram_gb: float = psutil.virtual_memory().total / (1024**3)

        # Simple heuristic based on hardware
        if cpu_cores >= 16 and ram_gb >= 32:
            return 10000  # High-end system
        elif cpu_cores >= 8 and ram_gb >= 16:
            return 5000  # Mid-range system
        elif cpu_cores >= 4 and ram_gb >= 8:
            return 2000  # Minimum system
        else:
            return 1000  # Limited system

    def get_optimization_recommendations(self, report: SystemRequirementsReport) -> List[str]:
        """Get optimization recommendations based on system validation.

        Args:
            report: System requirements report.

        Returns:
            List of optimization recommendations.
        """
        recommendations = []

        # Check for warnings and failures
        for req in report.requirements:
            if req.status == RequirementStatus.FAILED:
                if req.name == "RAM":
                    recommendations.append("Upgrade system RAM to at least 8 GB")
                elif req.name == "CPU Cores":
                    recommendations.append("Consider upgrading to a CPU with more cores")
                elif req.name == "Disk Space":
                    recommendations.append("Free up disk space or add additional storage")
                elif req.name == "Python Version":
                    recommendations.append("Upgrade Python to version 3.8 or higher")
                elif req.name == "Available Memory":
                    recommendations.append("Close other applications to free up memory")

            elif req.status == RequirementStatus.WARNING:
                if req.name == "RAM":
                    recommendations.append(
                        "Consider upgrading RAM to 16 GB for optimal performance"
                    )
                elif req.name == "CPU Cores":
                    recommendations.append("More CPU cores would improve processing speed")
                elif req.name == "CPU Frequency":
                    recommendations.append(
                        "Higher CPU frequency would improve real-time performance"
                    )

        # General recommendations
        if report.performance_metrics.get("cpu_usage_percent", 0) > 80:
            recommendations.append("High CPU usage detected, close background applications")

        if report.performance_metrics.get("memory_usage_percent", 0) > 80:
            recommendations.append("High memory usage detected, close unnecessary applications")

        return recommendations
