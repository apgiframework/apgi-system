"""Error Handler for Comprehensive Test Enhancement System.

Provides comprehensive diagnostic capture, error categorization, and resolution
guidance for test execution errors, framework issues, and environmental problems.
"""

import json
import os
import platform
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, cast

# Declare psutil for conditional import
psutil: Optional[Any] = None

try:
    import psutil  # type: ignore[no-redef]

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from ..logging.standardized_logging import get_logger


class ErrorCategory(Enum):
    """Categories for test execution errors."""

    TEST_FAILURE = "test_failure"
    FRAMEWORK_ISSUE = "framework_issue"
    ENVIRONMENTAL = "environmental"
    DEPENDENCY_MISSING = "dependency_missing"
    CONFIGURATION_ERROR = "configuration_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT_ERROR = "timeout_error"
    IMPORT_ERROR = "import_error"
    SYNTAX_ERROR = "syntax_error"
    ASSERTION_ERROR = "assertion_error"


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemState:
    """System state information at time of error."""

    timestamp: datetime
    python_version: str
    platform_info: str
    memory_usage: Dict[str, float]
    disk_usage: Dict[str, float]
    cpu_usage: float
    environment_variables: Dict[str, str]
    working_directory: str
    process_id: int

    @classmethod
    def capture_current(cls) -> "SystemState":
        """Capture current system state."""
        if HAS_PSUTIL:
            # Memory usage
            memory_info = psutil.virtual_memory()  # type: ignore
            memory_usage = {
                "total_gb": memory_info.total / (1024**3),
                "available_gb": memory_info.available / (1024**3),
                "percent_used": memory_info.percent,
                "free_gb": memory_info.free / (1024**3),
            }

            # Disk usage
            disk = psutil.disk_usage("/")  # type: ignore
            disk_usage = {
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percent_used": (disk.used / disk.total) * 100,
            }

            cpu_usage = psutil.cpu_percent(interval=1)  # type: ignore
        else:
            # Dummy values when psutil not available
            memory_usage = {
                "total_gb": 8.0,
                "available_gb": 4.0,
                "percent_used": 50.0,
                "free_gb": 4.0,
            }
            disk_usage = {
                "total_gb": 256.0,
                "free_gb": 128.0,
                "percent_used": 50.0,
            }
            cpu_usage = 50.0

        # Environment variables (filtered for security)
        safe_env_vars = {}
        for key, value in os.environ.items():
            if not any(
                sensitive in key.lower() for sensitive in ["password", "token", "key", "secret"]
            ):
                safe_env_vars[key] = value[:100]  # Limit length

        return cls(
            timestamp=datetime.now(),
            python_version=sys.version,
            platform_info=platform.platform(),
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            cpu_usage=cpu_usage,
            environment_variables=safe_env_vars,
            working_directory=os.getcwd(),
            process_id=os.getpid(),
        )


@dataclass
class Context:
    """Context information for test execution."""

    test_name: Optional[str] = None
    test_file: Optional[str] = None
    test_class: Optional[str] = None
    test_method: Optional[str] = None
    test_parameters: Dict[str, Any] = field(default_factory=dict)
    test_fixtures: List[str] = field(default_factory=list)
    test_marks: List[str] = field(default_factory=list)
    execution_time: Optional[float] = None


@dataclass
class ResolutionGuidance:
    """Resolution guidance for specific error types."""

    title: str
    description: str
    steps: List[str]
    code_example: Optional[str] = None
    documentation_links: List[str] = field(default_factory=list)
    automated_fix_available: bool = False
    estimated_fix_time: Optional[str] = None
    success_probability: float = 0.5


@dataclass
class DiagnosticInfo:
    """Comprehensive diagnostic information for an error."""

    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    original_exception: Exception
    message: str
    user_friendly_message: str
    stack_trace: List[str]
    system_state: SystemState
    test_context: Context
    resolution_guidance: List[ResolutionGuidance]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic info to dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "exception_type": type(self.original_exception).__name__,
            "message": self.message,
            "user_friendly_message": self.user_friendly_message,
            "timestamp": self.system_state.timestamp.isoformat(),
            "test_name": self.test_context.test_name,
            "test_file": self.test_context.test_file,
            "python_version": self.system_state.python_version,
            "platform": self.system_state.platform_info,
            "memory_usage_percent": self.system_state.memory_usage.get("percent_used", 0),
            "cpu_usage_percent": self.system_state.cpu_usage,
            "resolution_count": len(self.resolution_guidance),
            "metadata": self.metadata,
        }


class ErrorHandler:
    """Comprehensive error handler for test enhancement system.

    Provides error categorization, diagnostic capture, and resolution guidance
    for test failures, framework issues, and environmental problems.
    """

    def __init__(self) -> None:
        self.logger = get_logger("test_error_handler")
        self.error_patterns = self._load_error_patterns()
        self.resolution_database = self._build_resolution_database()

    def _load_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load known error patterns and their classifications."""
        return {
            # Test failure patterns
            "AssertionError": {
                "category": ErrorCategory.TEST_FAILURE,
                "severity": ErrorSeverity.MEDIUM,
                "keywords": ["assert", "expected", "actual"],
            },
            "ValueError": {
                "category": ErrorCategory.TEST_FAILURE,
                "severity": ErrorSeverity.MEDIUM,
                "keywords": ["invalid", "value", "range"],
            },
            # Framework issue patterns
            "ImportError": {
                "category": ErrorCategory.FRAMEWORK_ISSUE,
                "severity": ErrorSeverity.HIGH,
                "keywords": ["import", "module", "package"],
            },
            "ModuleNotFoundError": {
                "category": ErrorCategory.DEPENDENCY_MISSING,
                "severity": ErrorSeverity.HIGH,
                "keywords": ["module", "found", "install"],
            },
            # Environmental patterns
            "FileNotFoundError": {
                "category": ErrorCategory.ENVIRONMENTAL,
                "severity": ErrorSeverity.MEDIUM,
                "keywords": ["file", "directory", "path"],
            },
            "PermissionError": {
                "category": ErrorCategory.ENVIRONMENTAL,
                "severity": ErrorSeverity.MEDIUM,
                "keywords": ["permission", "access", "denied"],
            },
            # Resource patterns
            "MemoryError": {
                "category": ErrorCategory.RESOURCE_EXHAUSTION,
                "severity": ErrorSeverity.CRITICAL,
                "keywords": ["memory", "allocation", "out of memory"],
            },
            "TimeoutError": {
                "category": ErrorCategory.TIMEOUT_ERROR,
                "severity": ErrorSeverity.MEDIUM,
                "keywords": ["timeout", "time", "exceeded"],
            },
            # Configuration patterns
            "ConfigurationError": {
                "category": ErrorCategory.CONFIGURATION_ERROR,
                "severity": ErrorSeverity.MEDIUM,
                "keywords": ["config", "configuration", "setting"],
            },
            "SyntaxError": {
                "category": ErrorCategory.SYNTAX_ERROR,
                "severity": ErrorSeverity.HIGH,
                "keywords": ["syntax", "invalid", "parsing"],
            },
        }

    def _build_resolution_database(
        self,
    ) -> Dict[ErrorCategory, List[ResolutionGuidance]]:
        """Build database of resolution guidance for different error categories."""
        return {
            ErrorCategory.TEST_FAILURE: [
                ResolutionGuidance(
                    title="Review Test Logic",
                    description="Check test assertions and expected values",
                    steps=[
                        "Review the failing assertion in the test",
                        "Verify expected values are correct",
                        "Check if test data has changed",
                        "Ensure test logic matches requirements",
                    ],
                    estimated_fix_time="5-15 minutes",
                    success_probability=0.8,
                ),
                ResolutionGuidance(
                    title="Update Test Data",
                    description="Refresh or correct test data and fixtures",
                    steps=[
                        "Check if test data files exist and are accessible",
                        "Verify test data format matches expectations",
                        "Update fixture data if requirements changed",
                        "Regenerate synthetic test data if needed",
                    ],
                    estimated_fix_time="10-30 minutes",
                    success_probability=0.7,
                ),
            ],
            ErrorCategory.FRAMEWORK_ISSUE: [
                ResolutionGuidance(
                    title="Check Framework Installation",
                    description="Verify test framework is properly installed",
                    steps=[
                        "Check pytest installation: pip show pytest",
                        "Verify all test dependencies are installed",
                        "Update test framework to latest version",
                        "Check for conflicting package versions",
                    ],
                    code_example="pip install --upgrade pytest pytest-cov hypothesis",
                    estimated_fix_time="5-10 minutes",
                    success_probability=0.9,
                ),
                ResolutionGuidance(
                    title="Framework Configuration",
                    description="Check test framework configuration files",
                    steps=[
                        "Verify pytest.ini or pyproject.toml configuration",
                        "Check test discovery patterns",
                        "Validate plugin configurations",
                        "Ensure test paths are correct",
                    ],
                    estimated_fix_time="10-20 minutes",
                    success_probability=0.8,
                ),
            ],
            ErrorCategory.ENVIRONMENTAL: [
                ResolutionGuidance(
                    title="Check File System",
                    description="Verify file and directory permissions",
                    steps=[
                        "Check if required files and directories exist",
                        "Verify read/write permissions",
                        "Ensure sufficient disk space",
                        "Check file paths are correct for current OS",
                    ],
                    estimated_fix_time="5-15 minutes",
                    success_probability=0.9,
                ),
                ResolutionGuidance(
                    title="Environment Setup",
                    description="Configure test environment properly",
                    steps=[
                        "Set required environment variables",
                        "Configure test database connections",
                        "Setup test data directories",
                        "Initialize test configuration files",
                    ],
                    estimated_fix_time="15-30 minutes",
                    success_probability=0.8,
                ),
            ],
            ErrorCategory.DEPENDENCY_MISSING: [
                ResolutionGuidance(
                    title="Install Missing Dependencies",
                    description="Install required packages and modules",
                    steps=[
                        "Identify missing package from error message",
                        "Install using pip: pip install <package>",
                        "Check requirements.txt for version constraints",
                        "Restart test execution after installation",
                    ],
                    code_example="pip install -r requirements.txt",
                    automated_fix_available=True,
                    estimated_fix_time="2-5 minutes",
                    success_probability=0.95,
                )
            ],
            ErrorCategory.RESOURCE_EXHAUSTION: [
                ResolutionGuidance(
                    title="Optimize Memory Usage",
                    description="Reduce memory consumption during tests",
                    steps=[
                        "Run tests in smaller batches",
                        "Clear test data between tests",
                        "Use memory-efficient data structures",
                        "Consider running tests sequentially",
                    ],
                    code_example="pytest --maxfail=1 -x tests/",
                    estimated_fix_time="10-30 minutes",
                    success_probability=0.7,
                ),
                ResolutionGuidance(
                    title="System Resources",
                    description="Check and free system resources",
                    steps=[
                        "Close unnecessary applications",
                        "Check available memory and disk space",
                        "Restart system if needed",
                        "Consider running on machine with more resources",
                    ],
                    estimated_fix_time="5-15 minutes",
                    success_probability=0.8,
                ),
            ],
            ErrorCategory.TIMEOUT_ERROR: [
                ResolutionGuidance(
                    title="Adjust Timeout Settings",
                    description="Increase timeout values for slow operations",
                    steps=[
                        "Identify which operation is timing out",
                        "Increase timeout in test configuration",
                        "Optimize slow test operations",
                        "Consider parallel execution limits",
                    ],
                    code_example="pytest --timeout=300 tests/",
                    estimated_fix_time="5-10 minutes",
                    success_probability=0.9,
                )
            ],
            ErrorCategory.CONFIGURATION_ERROR: [
                ResolutionGuidance(
                    title="Fix Configuration",
                    description="Correct configuration file errors",
                    steps=[
                        "Validate configuration file syntax",
                        "Check configuration values are correct",
                        "Restore from backup if available",
                        "Use default configuration as template",
                    ],
                    estimated_fix_time="10-20 minutes",
                    success_probability=0.8,
                )
            ],
        }

    def handle_error(
        self,
        exception: Exception,
        test_context: Optional[Context] = None,
        **metadata: Any,
    ) -> DiagnosticInfo:
        """Handle an error with comprehensive diagnostic capture.

        Args:
            exception: The exception that occurred
            test_context: Context information about the test
            **metadata: Additional metadata about the error

        Returns:
            DiagnosticInfo: Comprehensive diagnostic information
        """
        # Generate unique error ID
        error_id = f"TEST_ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(exception)}"

        # Capture system state
        system_state = SystemState.capture_current()

        # Categorize error
        category, severity = self._categorize_error(exception)

        # Extract stack trace
        stack_trace = traceback.format_exception(
            type(exception), exception, exception.__traceback__
        )

        # Generate user-friendly message
        user_friendly_message = self._generate_user_message(exception, category)

        # Get resolution guidance
        resolution_guidance = self._get_resolution_guidance(category, exception)

        # Create diagnostic info
        diagnostic_info = DiagnosticInfo(
            error_id=error_id,
            category=category,
            severity=severity,
            original_exception=exception,
            message=str(exception),
            user_friendly_message=user_friendly_message,
            stack_trace=stack_trace,
            system_state=system_state,
            test_context=test_context or Context(),
            resolution_guidance=resolution_guidance,
            metadata=metadata,
        )

        # Log the error
        self._log_diagnostic_info(diagnostic_info)

        return diagnostic_info

    def _categorize_error(self, exception: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Categorize error and determine severity."""
        exception_name = type(exception).__name__
        exception_message = str(exception).lower()

        # Check known patterns
        if exception_name in self.error_patterns:
            pattern = self.error_patterns[exception_name]
            return pattern["category"], pattern["severity"]

        # Analyze message for keywords
        for pattern_name, pattern_info in self.error_patterns.items():
            keywords = pattern_info.get("keywords", [])
            if any(keyword in exception_message for keyword in keywords):
                return pattern_info["category"], pattern_info["severity"]

        # Default categorization
        if "test" in exception_message or "assert" in exception_message:
            return ErrorCategory.TEST_FAILURE, ErrorSeverity.MEDIUM
        elif "import" in exception_message or "module" in exception_message:
            return ErrorCategory.FRAMEWORK_ISSUE, ErrorSeverity.HIGH
        elif "file" in exception_message or "directory" in exception_message:
            return ErrorCategory.ENVIRONMENTAL, ErrorSeverity.MEDIUM
        else:
            return ErrorCategory.FRAMEWORK_ISSUE, ErrorSeverity.MEDIUM

    def _generate_user_message(self, exception: Exception, category: ErrorCategory) -> str:
        """Generate user-friendly error message."""
        category_messages = {
            ErrorCategory.TEST_FAILURE: "A test failed due to an assertion or logic error.",
            ErrorCategory.FRAMEWORK_ISSUE: "There's an issue with the test framework or its configuration.",
            ErrorCategory.ENVIRONMENTAL: "There's a problem with the test environment or file system.",
            ErrorCategory.DEPENDENCY_MISSING: "A required dependency or module is missing.",
            ErrorCategory.CONFIGURATION_ERROR: "There's an error in the test configuration.",
            ErrorCategory.RESOURCE_EXHAUSTION: "The system has run out of available resources.",
            ErrorCategory.TIMEOUT_ERROR: "A test operation has exceeded its timeout limit.",
            ErrorCategory.IMPORT_ERROR: "There's a problem importing a required module.",
            ErrorCategory.SYNTAX_ERROR: "There's a syntax error in the test code.",
            ErrorCategory.ASSERTION_ERROR: "A test assertion has failed.",
        }

        base_message = category_messages.get(category, "An error occurred during test execution.")
        return f"{base_message} Error details: {str(exception)}"

    def _get_resolution_guidance(
        self, category: ErrorCategory, exception: Exception
    ) -> List[ResolutionGuidance]:
        """Get resolution guidance for the error category."""
        guidance_list = self.resolution_database.get(category, [])

        # Add specific guidance based on exception details
        exception_message = str(exception).lower()

        if category == ErrorCategory.DEPENDENCY_MISSING:
            # Try to extract package name
            if "no module named" in exception_message:
                module_name = (
                    exception_message.split("'")[1] if "'" in exception_message else "unknown"
                )
                specific_guidance = ResolutionGuidance(
                    title=f"Install {module_name}",
                    description=f"Install the missing {module_name} module",
                    steps=[f"Run: pip install {module_name}"],
                    code_example=f"pip install {module_name}",
                    automated_fix_available=True,
                    estimated_fix_time="1-2 minutes",
                    success_probability=0.95,
                )
                return cast(List[ResolutionGuidance], [specific_guidance] + guidance_list)

        return cast(List[ResolutionGuidance], guidance_list)

    def _log_diagnostic_info(self, diagnostic_info: DiagnosticInfo) -> None:
        """Log diagnostic information with appropriate level."""
        # Log summary with appropriate method
        if diagnostic_info.severity == ErrorSeverity.LOW:
            self.logger.info(
                f"Test Error {diagnostic_info.error_id}: {diagnostic_info.category.value} - {diagnostic_info.message}"
            )
        elif diagnostic_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(
                f"Test Error {diagnostic_info.error_id}: {diagnostic_info.category.value} - {diagnostic_info.message}"
            )
        elif diagnostic_info.severity == ErrorSeverity.HIGH:
            self.logger.error(
                f"Test Error {diagnostic_info.error_id}: {diagnostic_info.category.value} - {diagnostic_info.message}"
            )
        elif diagnostic_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(
                f"Test Error {diagnostic_info.error_id}: {diagnostic_info.category.value} - {diagnostic_info.message}"
            )

        # Log detailed information at debug level
        self.logger.debug(f"Diagnostic details: {json.dumps(diagnostic_info.to_dict(), indent=2)}")

    def get_common_error_patterns(self) -> Dict[str, int]:
        """Get statistics on common error patterns (would be implemented with error tracking)."""
        # This would be implemented with actual error tracking in a real system
        return {
            "ImportError": 15,
            "AssertionError": 25,
            "FileNotFoundError": 8,
            "ValueError": 12,
            "TimeoutError": 5,
        }

    def suggest_preventive_measures(self, error_history: List[DiagnosticInfo]) -> List[str]:
        """Suggest preventive measures based on error history."""
        suggestions = []

        # Analyze error patterns
        category_counts: Dict[ErrorCategory, int] = {}
        for error in error_history:
            category_counts[error.category] = category_counts.get(error.category, 0) + 1

        # Generate suggestions based on frequent error types
        if category_counts.get(ErrorCategory.DEPENDENCY_MISSING, 0) > 2:
            suggestions.append("Consider using a requirements.txt file and virtual environment")

        if category_counts.get(ErrorCategory.RESOURCE_EXHAUSTION, 0) > 1:
            suggestions.append(
                "Consider running tests in smaller batches or on more powerful hardware"
            )

        if category_counts.get(ErrorCategory.TIMEOUT_ERROR, 0) > 2:
            suggestions.append("Review and optimize slow test operations")

        if category_counts.get(ErrorCategory.ENVIRONMENTAL, 0) > 3:
            suggestions.append("Standardize test environment setup with configuration management")

        return suggestions
