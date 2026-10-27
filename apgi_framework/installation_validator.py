"""Installation Validation for APGI Framework Test Enhancement.

This module provides comprehensive validation of the installation,
system requirements checking, and configuration validation.

Requirements: System deployment
"""

import importlib
import json
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from apgi_framework.logging.standardized_logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    severity: str = "error"  # error, warning, info


class InstallationValidator:
    """Comprehensive installation and system validation."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize validator with project root."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.logger = logging.getLogger(__name__)
        self.results: List[ValidationResult] = []

    def validate_all(self) -> Tuple[bool, List[ValidationResult]]:
        """Run all validation checks."""
        self.results.clear()

        logger.info("Starting APGI Framework installation validation...")
        logger.info("=" * 60)

        # System requirements
        self._validate_python_version()
        self._validate_platform()
        self._validate_disk_space()
        self._validate_memory()

        # Core dependencies
        self._validate_core_dependencies()
        self._validate_optional_dependencies()

        # Framework modules
        self._validate_framework_modules()
        self._validate_test_enhancement_modules()

        # File system structure
        self._validate_directory_structure()
        self._validate_configuration_files()

        # Functional tests
        self._validate_basic_functionality()
        self._validate_test_execution()

        # Generate summary
        self._print_validation_summary()

        # Return overall result
        has_errors = any(r.passed is False and r.severity == "error" for r in self.results)
        return not has_errors, self.results

    def _validate_python_version(self) -> None:
        """Validate Python version requirements."""
        min_version = (3, 8)
        current_version = sys.version_info[:2]

        if current_version >= min_version:
            self.results.append(
                ValidationResult(
                    name="Python Version",
                    passed=True,
                    message=f"Python {current_version[0]}.{current_version[1]} is supported",
                    details={
                        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                    },
                )
            )
        else:
            self.results.append(
                ValidationResult(
                    name="Python Version",
                    passed=False,
                    message=f"Python {min_version[0]}.{min_version[1]}+ required, found {current_version[0]}.{current_version[1]}",
                    severity="error",
                )
            )

    def _validate_platform(self) -> None:
        """Validate platform compatibility."""
        current_platform = platform.system()
        supported_platforms = ["Windows", "Linux", "Darwin"]  # Darwin = macOS

        if current_platform in supported_platforms:
            self.results.append(
                ValidationResult(
                    name="Platform Compatibility",
                    passed=True,
                    message=f"Platform {current_platform} is supported",
                    details={
                        "platform": current_platform,
                        "version": platform.version(),
                    },
                )
            )
        else:
            self.results.append(
                ValidationResult(
                    name="Platform Compatibility",
                    passed=False,
                    message=f"Platform {current_platform} may not be fully supported",
                    severity="warning",
                )
            )

    def _validate_disk_space(self) -> None:
        """Validate available disk space."""
        try:
            import shutil

            free_space_gb = shutil.disk_usage(self.project_root).free / (1024**3)

            if free_space_gb >= 1.0:
                self.results.append(
                    ValidationResult(
                        name="Disk Space",
                        passed=True,
                        message=f"Sufficient disk space available: {free_space_gb:.1f} GB",
                        details={"free_space_gb": free_space_gb},
                    )
                )
            else:
                self.results.append(
                    ValidationResult(
                        name="Disk Space",
                        passed=False,
                        message=f"Low disk space: {free_space_gb:.1f} GB available",
                        severity="warning",
                    )
                )
        except Exception as e:
            self.results.append(
                ValidationResult(
                    name="Disk Space",
                    passed=False,
                    message=f"Could not check disk space: {e}",
                    severity="warning",
                )
            )

    def _validate_memory(self) -> None:
        """Validate available memory."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            if available_gb >= 2.0:
                self.results.append(
                    ValidationResult(
                        name="Available Memory",
                        passed=True,
                        message=f"Sufficient memory available: {available_gb:.1f} GB",
                        details={
                            "available_memory_gb": available_gb,
                            "total_memory_gb": memory.total / (1024**3),
                        },
                    )
                )
            else:
                self.results.append(
                    ValidationResult(
                        name="Available Memory",
                        passed=False,
                        message=f"Low memory: {available_gb:.1f} GB available",
                        severity="warning",
                    )
                )
        except ImportError:
            self.results.append(
                ValidationResult(
                    name="Available Memory",
                    passed=True,
                    message="Memory check skipped (psutil not available)",
                    severity="info",
                )
            )
        except Exception as e:
            self.results.append(
                ValidationResult(
                    name="Available Memory",
                    passed=False,
                    message=f"Could not check memory: {e}",
                    severity="warning",
                )
            )

    def _validate_core_dependencies(self) -> None:
        """Validate core Python dependencies."""
        core_dependencies = [
            ("numpy", "1.20.0"),
            ("scipy", "1.7.0"),
            ("matplotlib", "3.5.0"),
            ("pandas", "1.3.0"),
            ("coverage", "6.0"),
            ("pytest", "7.0"),
            ("hypothesis", "6.0"),
        ]

        for package, min_version in core_dependencies:
            self._check_package(package, min_version, required=True)

    def _validate_optional_dependencies(self) -> None:
        """Validate optional dependencies."""
        optional_dependencies = [
            ("PySide6", "6.0", "GUI functionality"),
            ("psutil", "5.8", "System monitoring"),
            ("memory_profiler", "0.60", "Memory profiling"),
            ("black", "22.0", "Code formatting"),
            ("flake8", "5.0", "Code linting"),
            ("mypy", "1.0", "Type checking"),
        ]

        for package, min_version, description in optional_dependencies:
            result = self._check_package(package, min_version, required=False)
            if result and not result.passed:
                result.message += f" (needed for {description})"

    def _check_package(
        self, package_name: str, min_version: str, required: bool = True
    ) -> Optional[ValidationResult]:
        """Check if a package is installed and meets version requirements."""
        try:
            module = importlib.import_module(package_name)

            # Try to get version
            version = None
            for attr in ["__version__", "version", "VERSION"]:
                if hasattr(module, attr):
                    version = getattr(module, attr)
                    break

            if version:
                # Simple version comparison (works for most cases)
                try:
                    from packaging import version as pkg_version

                    if pkg_version.parse(str(version)) >= pkg_version.parse(min_version):
                        result = ValidationResult(
                            name=f"Package: {package_name}",
                            passed=True,
                            message=f"{package_name} {version} installed (>= {min_version})",
                            details={
                                "version": str(version),
                                "min_version": min_version,
                            },
                        )
                    else:
                        result = ValidationResult(
                            name=f"Package: {package_name}",
                            passed=False,
                            message=f"{package_name} {version} is below minimum {min_version}",
                            severity="error" if required else "warning",
                        )
                except ImportError:
                    # Fallback to string comparison if packaging not available
                    result = ValidationResult(
                        name=f"Package: {package_name}",
                        passed=True,
                        message=f"{package_name} {version} installed (version check skipped)",
                        severity="info",
                    )
            else:
                result = ValidationResult(
                    name=f"Package: {package_name}",
                    passed=True,
                    message=f"{package_name} installed (version unknown)",
                    severity="info",
                )

        except ImportError:
            result = ValidationResult(
                name=f"Package: {package_name}",
                passed=False,
                message=f"{package_name} not installed",
                severity="error" if required else "warning",
            )

        self.results.append(result)
        return result

    def _validate_framework_modules(self) -> None:
        """Validate core framework modules can be imported."""
        core_modules = [
            "apgi_framework",
            "apgi_framework.core",
            "apgi_framework.analysis",
            "apgi_framework.data",
            "apgi_framework.utils",
        ]

        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
                self.results.append(
                    ValidationResult(
                        name=f"Module: {module_name}",
                        passed=True,
                        message=f"Successfully imported {module_name}",
                    )
                )
            except ImportError as e:
                self.results.append(
                    ValidationResult(
                        name=f"Module: {module_name}",
                        passed=False,
                        message=f"Failed to import {module_name}: {e}",
                        severity="error",
                    )
                )

    def _validate_test_enhancement_modules(self) -> None:
        """Validate test enhancement modules can be imported."""
        test_modules = [
            "apgi_framework.testing",
            "apgi_framework.testing.main",
            "apgi_framework.testing.batch_runner",
            "apgi_framework.testing.activity_logger",
        ]

        for module_name in test_modules:
            try:
                importlib.import_module(module_name)
                self.results.append(
                    ValidationResult(
                        name=f"Test Module: {module_name}",
                        passed=True,
                        message=f"Successfully imported {module_name}",
                    )
                )
            except ImportError as e:
                self.results.append(
                    ValidationResult(
                        name=f"Test Module: {module_name}",
                        passed=False,
                        message=f"Failed to import {module_name}: {e}",
                        severity="error",
                    )
                )

    def _validate_directory_structure(self) -> None:
        """Validate required directory structure."""
        required_dirs = ["apgi_framework", "tests", "logs", "config"]

        optional_dirs = [
            "test_reports",
            "coverage_reports",
            "session_data",
            "apgi_outputs",
            "data/examples",
        ]

        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                self.results.append(
                    ValidationResult(
                        name=f"Directory: {dir_name}",
                        passed=True,
                        message=f"Required directory exists: {dir_path}",
                    )
                )
            else:
                self.results.append(
                    ValidationResult(
                        name=f"Directory: {dir_name}",
                        passed=False,
                        message=f"Required directory missing: {dir_path}",
                        severity="error",
                    )
                )

        for dir_name in optional_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.results.append(
                        ValidationResult(
                            name=f"Directory: {dir_name}",
                            passed=True,
                            message=f"Created optional directory: {dir_path}",
                            severity="info",
                        )
                    )
                except Exception as e:
                    self.results.append(
                        ValidationResult(
                            name=f"Directory: {dir_name}",
                            passed=False,
                            message=f"Could not create directory {dir_path}: {e}",
                            severity="warning",
                        )
                    )

    def _validate_configuration_files(self) -> None:
        """Validate configuration files."""
        config_template = self.project_root / "config" / "test_config_template.json"

        if config_template.exists():
            try:
                with open(config_template, "r") as f:
                    json.load(f)

                self.results.append(
                    ValidationResult(
                        name="Configuration Template",
                        passed=True,
                        message="Configuration template is valid JSON",
                        details={"file": str(config_template)},
                    )
                )
            except json.JSONDecodeError as e:
                self.results.append(
                    ValidationResult(
                        name="Configuration Template",
                        passed=False,
                        message=f"Configuration template has invalid JSON: {e}",
                        severity="error",
                    )
                )
        else:
            self.results.append(
                ValidationResult(
                    name="Configuration Template",
                    passed=False,
                    message="Configuration template not found",
                    severity="warning",
                )
            )

    def _validate_basic_functionality(self) -> None:
        """Validate basic framework functionality."""
        try:
            # Test basic imports and instantiation
            from apgi_framework.testing.main import ApplicationConfig

            ApplicationConfig(mode="cli", project_root=str(self.project_root))

            self.results.append(
                ValidationResult(
                    name="Basic Functionality",
                    passed=True,
                    message="Basic framework functionality works correctly",
                )
            )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    name="Basic Functionality",
                    passed=False,
                    message=f"Basic functionality test failed: {e}",
                    severity="error",
                )
            )

    def _validate_test_execution(self) -> None:
        """Validate test execution capability."""
        try:
            # Try to run a simple test discovery
            # Initialize with default config
            from apgi_framework.config import ConfigManager
            from apgi_framework.testing.batch_runner import BatchTestRunner

            config = ConfigManager()
            BatchTestRunner(config_manager=config)

            # Just test initialization, not actual test execution
            self.results.append(
                ValidationResult(
                    name="Test Execution",
                    passed=True,
                    message="Test execution components initialized successfully",
                )
            )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    name="Test Execution",
                    passed=False,
                    message=f"Test execution validation failed: {e}",
                    severity="error",
                )
            )

    def _print_validation_summary(self) -> None:
        """Print validation summary."""
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)

        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        errors = [r for r in self.results if not r.passed and r.severity == "error"]
        warnings = [r for r in self.results if not r.passed and r.severity == "warning"]

        logger.info(f"Total Checks: {total_count}")
        logger.info(f"Passed: {passed_count}")
        logger.info(f"Errors: {len(errors)}")
        logger.info(f"Warnings: {len(warnings)}")

        if errors:
            logger.info(f"\nERRORS ({len(errors)}):")
            for error in errors:
                logger.info(f"  ✗ {error.name}: {error.message}")

        if warnings:
            logger.info(f"\nWARNINGS ({len(warnings)}):")
            for warning in warnings:
                logger.info(f"  ⚠ {warning.name}: {warning.message}")

        # Show successful checks
        successes = [r for r in self.results if r.passed]
        if successes:
            logger.info(f"\nSUCCESSFUL CHECKS ({len(successes)}):")
            for success in successes:
                logger.info(f"  ✓ {success.name}: {success.message}")

        logger.info("=" * 60)

        if not errors:
            logger.info("🎉 Installation validation completed successfully!")
        else:
            logger.info("❌ Installation validation found errors that need to be addressed.")

    def save_validation_report(self, output_file: str) -> None:
        """Save validation report to file."""
        report_data = {
            "validation_timestamp": str(sys.version_info),
            "project_root": str(self.project_root),
            "platform": {
                "system": platform.system(),
                "version": platform.version(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                    "details": r.details,
                }
                for r in self.results
            ],
            "summary": {
                "total_checks": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "errors": sum(1 for r in self.results if not r.passed and r.severity == "error"),
                "warnings": sum(
                    1 for r in self.results if not r.passed and r.severity == "warning"
                ),
            },
        }

        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Validation report saved to: {output_file}")


def main() -> None:
    """Main entry point for installation validation."""
    import argparse

    parser = argparse.ArgumentParser(description="APGI Framework Installation Validator")
    parser.add_argument("--project-root", type=str, help="Project root directory")
    parser.add_argument("--output", type=str, help="Output file for validation report")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Run validation
    validator = InstallationValidator(args.project_root)
    success, results = validator.validate_all()

    # Save report if requested
    if args.output:
        validator.save_validation_report(args.output)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
