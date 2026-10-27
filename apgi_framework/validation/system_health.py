"""System Health Checker.

Provides comprehensive system health checks and diagnostics for the APGI Framework.
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import numpy as np


@dataclass
class HealthCheckResult:
    """Result of system health check."""

    timestamp: datetime
    overall_status: str  # "healthy", "warning", "critical"
    checks_passed: int
    checks_failed: int
    checks_warning: int

    component_status: Dict[str, str] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.overall_status == "healthy"

    def get_report(self) -> str:
        """Get formatted health check report."""
        lines = []
        lines.append("=" * 60)
        lines.append("APGI FRAMEWORK SYSTEM HEALTH CHECK")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Overall Status: {self.overall_status.upper()}")
        lines.append(
            f"Checks: {self.checks_passed} passed, {self.checks_warning} warnings, {self.checks_failed} failed"
        )
        lines.append("")

        # Component status
        if self.component_status:
            lines.append("Component Status:")
            for component, status in self.component_status.items():
                status_icon = "✓" if status == "healthy" else "⚠️" if status == "warning" else "❌"
                lines.append(f"  {status_icon} {component}: {status}")
            lines.append("")

        # Issues
        if self.issues:
            lines.append("CRITICAL ISSUES:")
            for issue in self.issues:
                lines.append(f"  ❌ {issue}")
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("RECOMMENDATIONS:")
            for rec in self.recommendations:
                lines.append(f"  💡 {rec}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


class SystemHealthChecker:
    """Comprehensive system health checker for APGI Framework.

    Performs checks on:
    - Python environment
    - Required dependencies
    - Configuration validity
    - Data storage
    - Computational resources
    - Component availability
    """

    def __init__(self) -> None:
        self.check_history: List[HealthCheckResult] = []

    def run_full_health_check(self) -> HealthCheckResult:
        """Run complete system health check.

        Returns:
            HealthCheckResult with detailed status
        """
        timestamp = datetime.now()
        component_status = {}
        issues = []
        warnings: List[str] = []
        recommendations = []

        checks_passed = 0
        checks_failed = 0
        checks_warning = 0

        # Check Python environment
        status, check_issues, check_warnings = self._check_python_environment()
        component_status["Python Environment"] = status
        issues.extend(check_issues)
        warnings.extend(check_warnings)
        if status == "healthy":
            checks_passed += 1
        elif status == "warning":
            checks_warning += 1
        else:
            checks_failed += 1

        # Check dependencies
        status, check_issues, check_warnings = self._check_dependencies()
        component_status["Dependencies"] = status
        issues.extend(check_issues)
        warnings.extend(check_warnings)
        if status == "healthy":
            checks_passed += 1
        elif status == "warning":
            checks_warning += 1
        else:
            checks_failed += 1

        # Check configuration
        status, check_issues, check_warnings = self._check_configuration()
        component_status["Configuration"] = status
        issues.extend(check_issues)
        warnings.extend(check_warnings)
        if status == "healthy":
            checks_passed += 1
        elif status == "warning":
            checks_warning += 1
        else:
            checks_failed += 1

        # Check data storage
        status, check_issues, check_warnings = self._check_data_storage()
        component_status["Data Storage"] = status
        issues.extend(check_issues)
        warnings.extend(check_warnings)
        if status == "healthy":
            checks_passed += 1
        elif status == "warning":
            checks_warning += 1
        else:
            checks_failed += 1

        # Check computational resources
        status, check_issues, check_warnings = self._check_computational_resources()
        component_status["Computational Resources"] = status
        issues.extend(check_issues)
        warnings.extend(check_warnings)
        if status == "healthy":
            checks_passed += 1
        elif status == "warning":
            checks_warning += 1
        else:
            checks_failed += 1

        # Check core components
        status, check_issues, check_warnings = self._check_core_components()
        component_status["Core Components"] = status
        issues.extend(check_issues)
        warnings.extend(check_warnings)
        if status == "healthy":
            checks_passed += 1
        elif status == "warning":
            checks_warning += 1
        else:
            checks_failed += 1

        # Generate recommendations
        recommendations = self._generate_recommendations(issues, warnings)

        # Determine overall status
        if checks_failed > 0:
            overall_status = "critical"
        elif checks_warning > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        result = HealthCheckResult(
            timestamp=timestamp,
            overall_status=overall_status,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            checks_warning=checks_warning,
            component_status=component_status,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
        )

        self.check_history.append(result)
        return result

    def _check_python_environment(self) -> tuple:
        """Check Python environment."""
        issues = []
        warnings: List[str] = []

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            issues.append(
                f"Python version {python_version.major}.{python_version.minor} is too old (requires 3.8+)"
            )
        elif python_version < (3, 9):
            warnings.append(
                f"Python {python_version.major}.{python_version.minor} is supported but 3.9+ recommended"
            )

        # Check platform
        platform = sys.platform
        if platform not in ["win32", "linux", "darwin"]:
            warnings.append(f"Platform {platform} may have limited support")

        status = "critical" if issues else "warning" if warnings else "healthy"
        return status, issues, warnings

    def _check_dependencies(self) -> tuple:
        """Check required dependencies."""
        issues = []
        warnings: List[str] = []

        required_packages = {
            "numpy": "1.20.0",
            "scipy": "1.7.0",
        }

        optional_packages = {
            "matplotlib": "3.3.0",
            "pandas": "1.3.0",
        }

        # Check required packages
        for package, min_version in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                issues.append(f"Required package '{package}' not found")

        # Check optional packages
        for package, min_version in optional_packages.items():
            try:
                __import__(package)
            except ImportError:
                warnings.append(
                    f"Optional package '{package}' not found (some features may be unavailable)"
                )

        status = "critical" if issues else "warning" if warnings else "healthy"
        return status, issues, warnings

    def _check_configuration(self) -> tuple:
        """Check configuration validity."""
        issues = []
        warnings: List[str] = []

        try:
            from ..config import get_config_manager

            config_manager = get_config_manager()

            # Check APGI parameters
            apgi_params = config_manager.get_apgi_parameters()
            if apgi_params.extero_precision <= 0:
                issues.append("Invalid exteroceptive precision in configuration")
            if apgi_params.intero_precision <= 0:
                issues.append("Invalid interoceptive precision in configuration")

            # Check experimental config
            exp_config = config_manager.get_experimental_config()
            if exp_config.n_trials <= 0:
                issues.append("Invalid trial count in configuration")
            if exp_config.n_participants <= 0:
                issues.append("Invalid participant count in configuration")

        except Exception as e:
            issues.append(f"Configuration error: {str(e)}")

        status = "critical" if issues else "warning" if warnings else "healthy"
        return status, issues, warnings

    def _check_data_storage(self) -> tuple:
        """Check data storage availability."""
        issues = []
        warnings: List[str] = []

        # Check results directory
        results_dir = "results"
        if not os.path.exists(results_dir):
            try:
                os.makedirs(results_dir)
                warnings.append(f"Created missing results directory: {results_dir}")
            except Exception as e:
                issues.append(f"Cannot create results directory: {str(e)}")

        # Check write permissions
        if os.path.exists(results_dir):
            test_file = os.path.join(results_dir, ".health_check_test")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                issues.append(f"No write permission in results directory: {str(e)}")

        # Check disk space (if possible)
        try:
            import shutil

            total, used, free = shutil.disk_usage(
                results_dir if os.path.exists(results_dir) else "."
            )
            free_gb = free / (1024**3)
            if free_gb < 1.0:
                issues.append(f"Low disk space: {free_gb:.2f} GB available")
            elif free_gb < 5.0:
                warnings.append(f"Limited disk space: {free_gb:.2f} GB available")
        except Exception:
            pass  # Disk space check not critical

        status = "critical" if issues else "warning" if warnings else "healthy"
        return status, issues, warnings

    def _check_computational_resources(self) -> tuple:
        """Check computational resources."""
        issues = []
        warnings: List[str] = []

        # Check NumPy functionality
        try:
            test_array = np.random.rand(1000, 1000)
            result = np.dot(test_array, test_array.T)
            if not np.isfinite(result).all():
                issues.append("NumPy computation produced invalid results")
        except Exception as e:
            issues.append(f"NumPy computation failed: {str(e)}")

        # Check random number generation
        try:
            random_values = np.random.rand(1000)
            if len(np.unique(random_values)) < 900:
                warnings.append("Random number generator may have low entropy")
        except Exception as e:
            issues.append(f"Random number generation failed: {str(e)}")

        status = "critical" if issues else "warning" if warnings else "healthy"
        return status, issues, warnings

    def _check_core_components(self) -> tuple:
        """Check core APGI Framework components."""
        issues = []
        warnings: List[str] = []

        # Check core modules
        core_modules = [
            "apgi_framework.core.equation",
            "apgi_framework.simulators.p3b_simulator",
            "apgi_framework.simulators.gamma_simulator",
            "apgi_framework.falsification.primary_falsification_test",
        ]

        for module_name in core_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                issues.append(f"Core module '{module_name}' cannot be imported: {str(e)}")
            except Exception as e:
                warnings.append(f"Core module '{module_name}' import warning: {str(e)}")

        # Test basic functionality
        try:
            from apgi_framework.core.equation import APGIEquation  # type: ignore[import-not-found]

            equation = APGIEquation()
            test_result = equation.calculate_surprise(1.0, 1.0, 2.0, 1.5, 1.0)
            if not np.isfinite(test_result):
                issues.append("APGI equation produces invalid results")
        except Exception as e:
            issues.append(f"APGI equation test failed: {str(e)}")

        status = "critical" if issues else "warning" if warnings else "healthy"
        return status, issues, warnings

    def _generate_recommendations(self, issues: List[str], warnings: List[str]) -> List[str]:
        """Generate recommendations based on issues and warnings."""
        recommendations = []

        if any("Python version" in issue for issue in issues):
            recommendations.append("Upgrade Python to version 3.9 or higher")

        if any("package" in issue.lower() for issue in issues):
            recommendations.append("Install missing packages: pip install -r requirements.txt")

        if any("disk space" in issue.lower() for issue in issues):
            recommendations.append("Free up disk space or change output directory")

        if any("permission" in issue.lower() for issue in issues):
            recommendations.append("Check file system permissions for results directory")

        if any("configuration" in issue.lower() for issue in issues):
            recommendations.append("Review and fix configuration parameters")

        if any("module" in issue.lower() for issue in issues):
            recommendations.append("Reinstall APGI Framework or check installation integrity")

        if not recommendations and warnings:
            recommendations.append("Address warnings to ensure optimal performance")

        if not issues and not warnings:
            recommendations.append("System is healthy - no action required")

        return recommendations

    def check_component(self, component_name: str) -> HealthCheckResult:
        """Check specific component health.

        Args:
            component_name: Name of component to check

        Returns:
            HealthCheckResult for the component
        """
        timestamp = datetime.now()
        component_status = {}
        issues = []
        warnings: List[str] = []

        if component_name.lower() == "python":
            status, check_issues, check_warnings = self._check_python_environment()
            component_status["Python Environment"] = status
            issues.extend(check_issues)
            warnings.extend(check_warnings)

        elif component_name.lower() == "dependencies":
            status, check_issues, check_warnings = self._check_dependencies()
            component_status["Dependencies"] = status
            issues.extend(check_issues)
            warnings.extend(check_warnings)

        elif component_name.lower() == "configuration":
            status, check_issues, check_warnings = self._check_configuration()
            component_status["Configuration"] = status
            issues.extend(check_issues)
            warnings.extend(check_warnings)

        elif component_name.lower() == "storage":
            status, check_issues, check_warnings = self._check_data_storage()
            component_status["Data Storage"] = status
            issues.extend(check_issues)
            warnings.extend(check_warnings)

        elif component_name.lower() == "resources":
            status, check_issues, check_warnings = self._check_computational_resources()
            component_status["Computational Resources"] = status
            issues.extend(check_issues)
            warnings.extend(check_warnings)

        elif component_name.lower() == "core":
            status, check_issues, check_warnings = self._check_core_components()
            component_status["Core Components"] = status
            issues.extend(check_issues)
            warnings.extend(check_warnings)

        else:
            issues.append(f"Unknown component: {component_name}")
            status = "critical"

        checks_passed = 1 if status == "healthy" else 0
        checks_failed = 1 if status == "critical" else 0
        checks_warning = 1 if status == "warning" else 0

        overall_status = status
        recommendations = self._generate_recommendations(issues, warnings)

        return HealthCheckResult(
            timestamp=timestamp,
            overall_status=overall_status,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            checks_warning=checks_warning,
            component_status=component_status,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
        )

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get detailed diagnostic information.

        Returns:
            Dictionary with diagnostic information
        """
        info = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "numpy_version": None,
            "scipy_version": None,
            "working_directory": os.getcwd(),
            "results_directory_exists": os.path.exists("results"),
        }

        try:
            import numpy

            info["numpy_version"] = numpy.__version__
        except ImportError:
            info["numpy_version"] = "Not installed"

        try:
            import scipy

            info["scipy_version"] = scipy.__version__
        except ImportError:
            info["scipy_version"] = "Not installed"

        return info

    def get_health_history(self, n_recent: int = 10) -> List[HealthCheckResult]:
        """Get recent health check history."""
        return self.check_history[-n_recent:]


# Global health checker instance
_health_checker = None


def get_health_checker() -> SystemHealthChecker:
    """Get global system health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = SystemHealthChecker()
    return _health_checker
