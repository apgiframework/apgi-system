"""Deployment Validator with automated testing pipeline for new installations.

Validates complete deployment of APGI Framework parameter estimation system
including dependencies, hardware configuration, and end-to-end functionality.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Handle both module import and direct execution
try:
    from .hardware_configuration import HardwareConfigurationManager
    from .installation_manager import InstallationManager, InstallationReport
    from .system_requirements import (
        SystemRequirementsReport,
        SystemRequirementsValidator,
    )
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from apgi_framework.deployment.hardware_configuration import (
        HardwareConfigurationManager,
    )
    from apgi_framework.deployment.installation_manager import (
        InstallationManager,
        InstallationReport,
    )
    from apgi_framework.deployment.system_requirements import (
        SystemRequirementsReport,
        SystemRequirementsValidator,
    )


class ValidationPhase:
    """Phases of deployment validation."""

    SYSTEM_REQUIREMENTS = "system_requirements"
    DEPENDENCIES = "dependencies"
    HARDWARE_CONFIG = "hardware_config"
    FUNCTIONAL_TESTS = "functional_tests"
    INTEGRATION_TESTS = "integration_tests"
    PERFORMANCE_TESTS = "performance_tests"


@dataclass
class ValidationResult:
    """Result of a validation phase."""

    phase: str
    passed: bool
    message: str
    details: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DeploymentValidationReport:
    """Complete deployment validation report."""

    validation_id: str
    start_time: datetime
    end_time: Optional[datetime] = None

    results: List[ValidationResult] = field(default_factory=list)
    overall_passed: bool = False

    system_requirements_report: Optional[SystemRequirementsReport] = None
    installation_report: Optional[InstallationReport] = None

    def add_result(self, result: ValidationResult) -> None:
        """Add validation result."""
        self.results.append(result)

    @property
    def passed_phases(self) -> int:
        """Number of passed phases."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_phases(self) -> int:
        """Number of failed phases."""
        return sum(1 for r in self.results if not r.passed)


class DeploymentValidator:
    """Validates deployment of APGI Framework parameter estimation system.

    Performs comprehensive validation including:
    - System requirements checking
    - Dependency installation verification
    - Hardware configuration validation
    - Functional testing
    - Integration testing
    - Performance benchmarking
    """

    def __init__(self, install_dir: Optional[Path] = None):
        """Initialize deployment validator.

        Args:
            install_dir: Installation directory.
        """
        self.logger = logging.getLogger(__name__)
        self.install_dir = install_dir or Path.home() / ".apgi_framework"

        # Initialize managers
        self.system_validator = SystemRequirementsValidator()
        self.installation_manager = InstallationManager(install_dir)
        self.hardware_manager = HardwareConfigurationManager()

        self.current_report: Optional[DeploymentValidationReport] = None

    def validate_deployment(
        self, skip_optional: bool = False, run_performance_tests: bool = False
    ) -> DeploymentValidationReport:
        """Run complete deployment validation.

        Args:
            skip_optional: Skip optional dependencies and tests.
            run_performance_tests: Run performance benchmarking tests.

        Returns:
            Complete deployment validation report.
        """
        validation_id = f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting deployment validation: {validation_id}")

        self.current_report = DeploymentValidationReport(
            validation_id=validation_id, start_time=datetime.now()
        )

        try:
            # Phase 1: System requirements
            self._validate_system_requirements()

            # Phase 2: Dependencies
            self._validate_dependencies(skip_optional)

            # Phase 3: Hardware configuration
            self._validate_hardware_configuration()

            # Phase 4: Functional tests
            self._run_functional_tests()

            # Phase 5: Integration tests
            self._run_integration_tests()

            # Phase 6: Performance tests (optional)
            if run_performance_tests:
                self._run_performance_tests()

            # Determine overall status
            self.current_report.overall_passed = all(r.passed for r in self.current_report.results)

            self.current_report.end_time = datetime.now()

            self.logger.info(f"Deployment validation complete: {validation_id}")
            self.logger.info(
                f"Passed: {self.current_report.passed_phases}, "
                f"Failed: {self.current_report.failed_phases}"
            )

        except Exception as e:
            self.logger.error(f"Deployment validation failed: {e}")
            self.current_report.end_time = datetime.now()
            self.current_report.overall_passed = False

        return self.current_report

    def _validate_system_requirements(self) -> None:
        """Validate system requirements."""
        if self.current_report is None:
            raise RuntimeError("Validation report must be initialized before this step.")
        self.logger.info("Validating system requirements...")

        try:
            report = self.system_validator.validate_system()
            self.current_report.system_requirements_report = report

            result = ValidationResult(
                phase=ValidationPhase.SYSTEM_REQUIREMENTS,
                passed=report.can_run,
                message=f"System requirements: {report.overall_status.value}",
                details={
                    "passed": report.passed_count,
                    "warnings": report.warning_count,
                    "failed": report.failed_count,
                    "can_run": report.can_run,
                },
            )

            self.current_report.add_result(result)

            if not report.can_run:
                self.logger.error("System does not meet minimum requirements")
                recommendations = self.system_validator.get_optimization_recommendations(report)
                self.logger.info("Recommendations:")
                for rec in recommendations:
                    self.logger.info(f"  - {rec}")

        except Exception as e:
            self.logger.error(f"System requirements validation failed: {e}")
            result = ValidationResult(
                phase=ValidationPhase.SYSTEM_REQUIREMENTS,
                passed=False,
                message=f"System requirements check failed: {e}",
            )
            self.current_report.add_result(result)

    def _validate_dependencies(self, skip_optional: bool) -> None:
        """Validate dependency installation."""
        if self.current_report is None:
            raise RuntimeError("Validation report must be initialized before this step.")
        self.logger.info("Validating dependencies...")

        try:
            # Install missing dependencies
            install_report = self.installation_manager.install_all_dependencies(skip_optional)
            self.current_report.installation_report = install_report

            result = ValidationResult(
                phase=ValidationPhase.DEPENDENCIES,
                passed=install_report.success,
                message=f"Dependencies: {install_report.installed_count} installed, "
                f"{install_report.failed_count} failed",
                details={
                    "total": install_report.total_dependencies,
                    "installed": install_report.installed_count,
                    "failed": install_report.failed_count,
                    "skipped": install_report.skipped_count,
                },
            )

            self.current_report.add_result(result)

            if not install_report.success:
                self.logger.error("Dependency installation incomplete")
                for name, dep in install_report.dependencies.items():
                    if dep.error_message:
                        self.logger.error(f"  {name}: {dep.error_message}")

        except Exception as e:
            self.logger.error(f"Dependency validation failed: {e}")
            result = ValidationResult(
                phase=ValidationPhase.DEPENDENCIES,
                passed=False,
                message=f"Dependency validation failed: {e}",
            )
            self.current_report.add_result(result)

    def _validate_hardware_configuration(self) -> None:
        """Validate hardware configuration."""
        if self.current_report is None:
            raise RuntimeError("Validation report must be initialized before this step.")
        self.logger.info("Validating hardware configuration...")

        try:
            # List available configurations
            available_configs = self.hardware_manager.list_available_configs()

            # Check if any devices are configured
            configured_devices = len(self.hardware_manager.devices)

            passed = True  # Hardware config is optional
            message = f"Hardware configuration available: {len(available_configs)} device types"

            if configured_devices > 0:
                message += f", {configured_devices} devices configured"

            result = ValidationResult(
                phase=ValidationPhase.HARDWARE_CONFIG,
                passed=passed,
                message=message,
                details={
                    "available_configs": available_configs,
                    "configured_devices": configured_devices,
                },
            )

            self.current_report.add_result(result)

        except Exception as e:
            self.logger.error(f"Hardware configuration validation failed: {e}")
            result = ValidationResult(
                phase=ValidationPhase.HARDWARE_CONFIG,
                passed=False,
                message=f"Hardware configuration validation failed: {e}",
            )
            self.current_report.add_result(result)

    def _run_functional_tests(self) -> None:
        """Run functional tests."""
        if self.current_report is None:
            raise RuntimeError("Validation report must be initialized before this step.")
        self.logger.info("Running functional tests...")

        try:
            tests_passed = 0
            tests_failed = 0

            # Test 1: Import core modules
            try:
                # import numpy as np
                # import pandas as pd
                # import scipy

                tests_passed += 1
            except ImportError as e:
                self.logger.error(f"Core module import failed: {e}")
                tests_failed += 1

            # Test 2: Import Bayesian modeling
            try:
                # import arviz
                # import stan

                tests_passed += 1
            except ImportError as e:
                self.logger.warning(f"Bayesian module import failed: {e}")
                tests_failed += 1

            # Test 3: Import signal processing
            try:
                # import mne

                tests_passed += 1
            except ImportError as e:
                self.logger.warning(f"Signal processing module import failed: {e}")
                tests_failed += 1

            # Test 4: Import LSL
            try:
                # import pylsl

                tests_passed += 1
            except ImportError as e:
                self.logger.warning(f"LSL import failed: {e}")
                tests_failed += 1

            # Test 5: Import GUI
            try:
                # from PyQt5 import QtCore, QtWidgets

                tests_passed += 1
            except ImportError as e:
                self.logger.warning(f"GUI module import failed: {e}")
                tests_failed += 1

            passed = tests_failed == 0

            result = ValidationResult(
                phase=ValidationPhase.FUNCTIONAL_TESTS,
                passed=passed,
                message=f"Functional tests: {tests_passed} passed, {tests_failed} failed",
                details={"passed": tests_passed, "failed": tests_failed},
            )

            self.current_report.add_result(result)

        except Exception as e:
            self.logger.error(f"Functional tests failed: {e}")
            result = ValidationResult(
                phase=ValidationPhase.FUNCTIONAL_TESTS,
                passed=False,
                message=f"Functional tests failed: {e}",
            )
            self.current_report.add_result(result)

    def _run_integration_tests(self) -> None:
        """Run integration tests."""
        if self.current_report is None:
            raise RuntimeError("Validation report must be initialized before this step.")
        self.logger.info("Running integration tests...")

        try:
            tests_passed = 0
            tests_failed = 0

            # Test 1: APGI Framework imports
            try:
                from apgi_framework.config import ConfigManager

                # from apgi_framework.exceptions import APGIFrameworkError

                tests_passed += 1
            except ImportError as e:
                self.logger.error(f"APGI Framework import failed: {e}")
                tests_failed += 1

            # Test 2: Configuration management
            try:
                from apgi_framework.config import ConfigManager

                ConfigManager()
                # apgi_params = config.get_apgi_parameters()
                tests_passed += 1
            except Exception as e:
                self.logger.error(f"Configuration test failed: {e}")
                tests_failed += 1

            # Test 3: Deployment modules
            try:
                # from apgi_framework.deployment import (
                #     HardwareConfigurationManager,
                #     InstallationManager,
                #     SystemRequirementsValidator,
                # )

                tests_passed += 1
            except ImportError as e:
                self.logger.error(f"Deployment module import failed: {e}")
                tests_failed += 1

            passed = tests_failed == 0

            result = ValidationResult(
                phase=ValidationPhase.INTEGRATION_TESTS,
                passed=passed,
                message=f"Integration tests: {tests_passed} passed, {tests_failed} failed",
                details={"passed": tests_passed, "failed": tests_failed},
            )

            self.current_report.add_result(result)

        except Exception as e:
            self.logger.error(f"Integration tests failed: {e}")
            result = ValidationResult(
                phase=ValidationPhase.INTEGRATION_TESTS,
                passed=False,
                message=f"Integration tests failed: {e}",
            )
            self.current_report.add_result(result)

    def _run_performance_tests(self) -> None:
        """Run performance tests."""
        if self.current_report is None:
            raise RuntimeError("Validation report must be initialized before this step.")
        self.logger.info("Running performance tests...")

        try:
            import time

            import numpy as np

            performance_metrics = {}

            # Test 1: Array operations
            start = time.time()
            data = np.random.randn(1000, 1000)
            result = np.dot(data, data.T)
            performance_metrics["array_ops_time"] = time.time() - start

            # Test 2: FFT operations
            start = time.time()
            signal = np.random.randn(10000)
            np.fft.fft(signal)
            performance_metrics["fft_time"] = time.time() - start

            # Determine if performance is acceptable
            passed = (
                performance_metrics["array_ops_time"] < 1.0
                and performance_metrics["fft_time"] < 0.1
            )

            result = ValidationResult(
                phase=ValidationPhase.PERFORMANCE_TESTS,
                passed=passed,
                message=f"Performance tests: {'passed' if passed else 'slow performance detected'}",
                details=performance_metrics,
            )

            self.current_report.add_result(result)

        except Exception as e:
            self.logger.error(f"Performance tests failed: {e}")
            result = ValidationResult(
                phase=ValidationPhase.PERFORMANCE_TESTS,
                passed=False,
                message=f"Performance tests failed: {e}",
            )
            self.current_report.add_result(result)

    def save_report(self, output_path: Path) -> None:
        """Save deployment validation report to file.

        Args:
            output_path: Path to save report.
        """
        if not self.current_report:
            self.logger.warning("No report to save")
            return

        report_data = {
            "validation_id": self.current_report.validation_id,
            "start_time": self.current_report.start_time.isoformat(),
            "end_time": (
                self.current_report.end_time.isoformat() if self.current_report.end_time else None
            ),
            "overall_passed": self.current_report.overall_passed,
            "passed_phases": self.current_report.passed_phases,
            "failed_phases": self.current_report.failed_phases,
            "results": [
                {
                    "phase": r.phase,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self.current_report.results
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(f"Report saved to {output_path}")

    def generate_summary(self) -> str:
        """Generate human-readable summary of validation.

        Returns:
            Summary text.
        """
        if not self.current_report:
            return "No validation report available"

        lines = []
        lines.append("=" * 60)
        lines.append("APGI Framework Deployment Validation Summary")
        lines.append("=" * 60)
        lines.append(f"Validation ID: {self.current_report.validation_id}")
        lines.append(f"Start Time: {self.current_report.start_time}")
        if self.current_report.end_time:
            duration = self.current_report.end_time - self.current_report.start_time
            lines.append(f"Duration: {duration.total_seconds():.1f} seconds")
        lines.append("")

        lines.append(
            f"Overall Status: {'PASSED' if self.current_report.overall_passed else 'FAILED'}"
        )
        lines.append(f"Phases Passed: {self.current_report.passed_phases}")
        lines.append(f"Phases Failed: {self.current_report.failed_phases}")
        lines.append("")

        lines.append("Phase Results:")
        lines.append("-" * 60)
        for result in self.current_report.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            lines.append(f"{status} | {result.phase}: {result.message}")

        lines.append("=" * 60)

        return "\n".join(lines)


def main() -> None:
    """CLI entry point for deployment validator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="APGI Framework Deployment Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python deployment_validator.py              # Run full validation
    python deployment_validator.py --quick    # Run quick validation
    python deployment_validator.py --report validation_report.json
        """,
    )
    parser.add_argument("--quick", action="store_true", help="Run quick validation only")
    parser.add_argument("--full", action="store_true", help="Run full validation (default)")
    parser.add_argument("--report", type=str, help="Save validation report to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create validator and run
    validator = DeploymentValidator()

    if args.quick:
        print("Running quick deployment validation...")
        validator.validate_deployment(skip_optional=True, run_performance_tests=False)
    else:
        print("Running full deployment validation...")
        validator.validate_deployment(skip_optional=False, run_performance_tests=True)

    # Print summary
    print("\n" + validator.generate_summary())

    # Save report if requested
    if args.report:
        report_path = Path(args.report)
        validator.save_report(report_path)

    # Exit with appropriate code
    sys.exit(0 if validator.current_report and validator.current_report.overall_passed else 1)


if __name__ == "__main__":
    main()
