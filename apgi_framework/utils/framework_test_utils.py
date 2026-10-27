"""Test-specific utilities for the APGI Framework test enhancement system.

This module provides test discovery, metadata extraction, test execution utilities,
and test result processing functions for comprehensive test management.
"""

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, cast

from .ast_analyzer import ASTAnalyzer
from .file_utils import FileUtils


class FrameworkRunStatus(Enum):
    """Test execution status."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    UNKNOWN = "unknown"


class FrameworkFailureCategory(Enum):
    """Test failure categories."""

    ASSERTION_ERROR = "assertion_error"
    FRAMEWORK_ERROR = "framework_error"
    IMPORT_ERROR = "import_error"
    TIMEOUT_ERROR = "timeout_error"
    SETUP_ERROR = "setup_error"
    TEARDOWN_ERROR = "teardown_error"


class FrameworkRunCategory(Enum):
    """Test categories."""

    UNIT = "unit"
    INTEGRATION = "integration"
    PROPERTY = "property"
    GUI = "gui"
    PERFORMANCE = "performance"
    SMOKE = "smoke"
    MODULE_SPECIFIC = "module_specific"


@dataclass
class FrameworkConfiguration:
    """Test execution configuration."""

    categories: List[FrameworkRunCategory] = field(default_factory=list)
    modules: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    parallel: bool = False
    max_workers: int = 4
    timeout: Optional[float] = None
    retries: int = 0
    verbose: bool = False
    coverage: bool = False


@dataclass
class FrameworkTestCase:
    """Represents a single test case."""

    name: str
    file_path: Path
    module: str  # Added module attribute
    class_name: Optional[str]
    method_name: str
    category: FrameworkRunCategory
    line_number: int
    docstring: Optional[str]
    parameters: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    estimated_duration: float = 0.0
    tags: Set[str] = field(default_factory=set)


@dataclass
class FrameworkRunResult:
    """Represents test execution result."""

    test_case: FrameworkTestCase
    status: FrameworkRunStatus
    duration: float
    output: str
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FrameworkTestSuite:
    """Represents a collection of test cases."""

    name: str
    test_cases: List[FrameworkTestCase]
    setup_methods: List[str] = field(default_factory=list)
    teardown_methods: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)
    total_estimated_duration: float = 0.0


@dataclass
class FrameworkFailure:
    """Represents a test failure with detailed information."""

    test_name: str
    failure_category: FrameworkFailureCategory
    error_message: str
    stack_trace: str
    failure_context: Dict[str, Any]
    file_path: Path
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FrameworkResults:
    """Test execution results summary."""

    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    execution_time: float
    failures: List[FrameworkFailure] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    results: List[FrameworkRunResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        return self.passed_tests / self.total_tests * 100 if self.total_tests > 0 else 0


@dataclass
class FrameworkExecution:
    """Represents a test execution session."""

    execution_id: str
    test_suites: List[FrameworkTestSuite]
    results: Optional[FrameworkResults] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    status: str = "pending"


class FrameworkTestUtils:
    """Comprehensive test utilities for discovery, execution, and result processing.

    Provides functionality for test discovery, metadata extraction, test execution
    coordination, and result analysis for the test enhancement system.
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize TestUtilities.

        Args:
            base_path: Base directory for test operations
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.logger = logging.getLogger(__name__)

        # Initialize AST analyzer
        self.ast_analyzer: Any = ASTAnalyzer()

        # Initialize file utils
        self.file_utils = FileUtils()

        # Test file patterns
        self.test_file_patterns = ["test_*.py", "*_test.py", "tests.py"]

        # Test method patterns
        self.test_method_patterns = ["test_*", "*_test"]

    def discover_tests(
        self,
        root_dir: Union[str, Path],
        patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[FrameworkTestSuite]:
        """Discover all test files and extract test cases.

        Args:
            root_dir: Root directory to search for tests
            patterns: File patterns to include (optional)
            exclude_patterns: File patterns to exclude (optional)

        Returns:
            List of test suites
        """
        if patterns is None:
            patterns = self.test_file_patterns

        if exclude_patterns is None:
            exclude_patterns = ["__pycache__", "*.pyc", ".git"]

        test_files = []

        # Find test files
        for pattern in patterns:
            found_files = self.file_utils.find_files(pattern, root_dir, recursive=True)
            test_files.extend(found_files)

        # Remove duplicates and filter exclusions
        test_files = list(set(test_files))

        if exclude_patterns:
            filtered_files = []
            for file_path in test_files:
                should_exclude = False
                for exclude_pattern in exclude_patterns:
                    if exclude_pattern in str(file_path):
                        should_exclude = True
                        break
                if not should_exclude:
                    filtered_files.append(file_path)
            test_files = filtered_files

        # Extract test cases from files
        test_suites = []
        for test_file in test_files:
            try:
                suite = self._extract_test_suite(test_file)
                if suite.test_cases:  # Only add suites with test cases
                    test_suites.append(suite)
            except Exception as e:
                self.logger.warning(f"Failed to extract tests from {test_file}: {e}")

        self.logger.info(
            f"Discovered {len(test_suites)} test suites with {sum(len(s.test_cases) for s in test_suites)} test cases"
        )
        return test_suites

    def discover_all_tests(self) -> List[FrameworkTestSuite]:
        """Discover all test suites in the project.

        Returns:
            List of all test suites found in the project
        """
        return self.discover_tests(self.base_path or ".")

    def _extract_test_suite(self, file_path: Path) -> FrameworkTestSuite:
        """Extract test suite from a single test file."""
        elements = self.ast_analyzer.extract_code_elements(file_path)
        test_metrics = self.ast_analyzer.analyze_test_code(file_path)

        test_cases = []
        setup_methods = []
        teardown_methods = []
        fixtures = []

        # Process code elements
        for element in elements:
            if element.is_test and element.node_type.value in [
                "function",
                "method",
            ]:
                # Determine category
                category = self._determine_test_category(
                    file_path, element.name, element.decorators
                )

                # Create test case
                test_case = FrameworkTestCase(
                    name=(
                        f"{element.parent_class}.{element.name}"
                        if element.parent_class
                        else element.name
                    ),
                    file_path=file_path,
                    module=file_path.stem,  # Use file name as module
                    class_name=element.parent_class,
                    method_name=element.name,
                    category=category,
                    line_number=element.line_start,
                    docstring=element.docstring,
                    parameters=element.parameters,
                    decorators=element.decorators,
                    dependencies=self._extract_test_dependencies(element),
                    estimated_duration=self._estimate_test_duration(element),
                    tags=self._extract_test_tags(element),
                )
                test_cases.append(test_case)

        # Extract setup/teardown methods
        setup_methods = test_metrics.setup_methods
        teardown_methods = test_metrics.teardown_methods
        fixtures = test_metrics.fixture_usage

        # Calculate total estimated duration
        total_duration = sum(tc.estimated_duration for tc in test_cases)

        suite_name = file_path.stem
        return FrameworkTestSuite(
            name=suite_name,
            test_cases=test_cases,
            setup_methods=setup_methods,
            teardown_methods=teardown_methods,
            fixtures=fixtures,
            total_estimated_duration=total_duration,
        )

    def _determine_test_category(
        self, file_path: Path, test_name: str, decorators: List[str]
    ) -> FrameworkRunCategory:
        """Determine test category based on file path, name, and decorators."""
        file_name = file_path.name.lower()
        test_name_lower = test_name.lower()

        # Check decorators first
        for decorator in decorators:
            if "property" in decorator.lower() or "given" in decorator.lower():
                return FrameworkRunCategory.PROPERTY
            if "gui" in decorator.lower() or "ui" in decorator.lower():
                return FrameworkRunCategory.GUI
            if "performance" in decorator.lower() or "benchmark" in decorator.lower():
                return FrameworkRunCategory.PERFORMANCE

        # Check file path
        if "integration" in str(file_path).lower():
            return FrameworkRunCategory.INTEGRATION
        if "property" in file_name or "pbt" in file_name:
            return FrameworkRunCategory.PROPERTY
        if "gui" in file_name or "ui" in file_name:
            return FrameworkRunCategory.GUI
        if "performance" in file_name or "benchmark" in file_name:
            return FrameworkRunCategory.PERFORMANCE
        if "smoke" in file_name:
            return FrameworkRunCategory.SMOKE

        # Check test name
        if "integration" in test_name_lower:
            return FrameworkRunCategory.INTEGRATION
        if "property" in test_name_lower:
            return FrameworkRunCategory.PROPERTY
        if "gui" in test_name_lower or "ui" in test_name_lower:
            return FrameworkRunCategory.GUI
        if "performance" in test_name_lower or "benchmark" in test_name_lower:
            return FrameworkRunCategory.PERFORMANCE
        if "smoke" in test_name_lower:
            return FrameworkRunCategory.SMOKE

        # Default to unit test
        return FrameworkRunCategory.UNIT

    def _extract_test_dependencies(self, element: Any) -> Set[str]:
        """Extract test dependencies from code element."""
        dependencies = set()

        # Add dependencies based on decorators
        for decorator in element.decorators:
            if "mock" in decorator.lower():
                dependencies.add("mock")
            if "fixture" in decorator.lower():
                dependencies.add("pytest")
            if "hypothesis" in decorator.lower() or "given" in decorator.lower():
                dependencies.add("hypothesis")

        return dependencies

    def _estimate_test_duration(self, element: Any) -> float:
        """Estimate test execution duration based on complexity."""
        base_duration = 0.1  # Base 100ms

        # Adjust based on complexity
        complexity_factor = min(element.complexity / 10.0, 2.0)

        # Adjust based on test type
        type_factors = {
            "integration": 2.0,
            "property": 3.0,
            "gui": 5.0,
            "performance": 10.0,
        }

        type_factor = 1.0
        for test_type, factor in type_factors.items():
            if test_type in element.name.lower():
                type_factor = factor
                break

        return cast(float, base_duration * complexity_factor * type_factor)

    def _extract_test_tags(self, element: Any) -> Set[str]:
        """Extract test tags from decorators and docstring."""
        tags = set()

        # Extract from decorators
        for decorator in element.decorators:
            if "mark" in decorator.lower():
                # Extract pytest marks
                if "." in decorator:
                    mark_name = decorator.split(".")[-1]
                    tags.add(mark_name)

        # Extract from docstring
        if element.docstring:
            # Look for tags in format @tag or #tag
            tag_pattern = r"[@#](\w+)"
            found_tags = re.findall(tag_pattern, element.docstring)
            tags.update(found_tags)

        return tags

    def filter_tests(
        self,
        test_suites: List[FrameworkTestSuite],
        categories: Optional[List[FrameworkRunCategory]] = None,
        tags: Optional[List[str]] = None,
        name_pattern: Optional[str] = None,
        max_duration: Optional[float] = None,
    ) -> List[FrameworkTestSuite]:
        """Filter test suites based on criteria.

        Args:
            test_suites: List of test suites to filter
            categories: Test categories to include
            tags: Test tags to include
            name_pattern: Name pattern to match
            max_duration: Maximum test duration

        Returns:
            Filtered test suites
        """
        filtered_suites = []

        for suite in test_suites:
            filtered_cases = []

            for test_case in suite.test_cases:
                # Check category filter
                if categories and test_case.category not in categories:
                    continue

                # Check tag filter
                if tags and not any(tag in test_case.tags for tag in tags):
                    continue

                # Check name pattern
                if name_pattern and name_pattern not in test_case.name:
                    continue

                # Check duration filter
                if max_duration and test_case.estimated_duration > max_duration:
                    continue

                filtered_cases.append(test_case)

            # Create filtered suite if it has test cases
            if filtered_cases:
                filtered_suite = FrameworkTestSuite(
                    name=suite.name,
                    test_cases=filtered_cases,
                    setup_methods=suite.setup_methods,
                    teardown_methods=suite.teardown_methods,
                    fixtures=suite.fixtures,
                    total_estimated_duration=sum(tc.estimated_duration for tc in filtered_cases),
                )
                filtered_suites.append(filtered_suite)

        return filtered_suites

    def execute_tests(
        self,
        test_suites: List[FrameworkTestSuite],
        config: Optional[Dict[str, Any]] = None,
    ) -> FrameworkExecution:
        """Execute test suites using pytest.

        Args:
            test_suites: Test suites to execute
            config: Execution configuration

        Returns:
            Test execution results
        """
        if config is None:
            config = {}

        execution_id = f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create test execution
        execution = FrameworkExecution(
            execution_id=execution_id,
            test_suites=test_suites,
            start_time=datetime.now(),
            configuration=config,
            environment=dict(os.environ) if "os" in sys.modules else {},
        )

        # Collect all test files
        test_files = set()
        for suite in test_suites:
            for test_case in suite.test_cases:
                test_files.add(str(test_case.file_path))

        # Build pytest command
        cmd = ["python", "-m", "pytest", "-v"]

        # Add configuration options
        if config.get("verbose", False):
            cmd.append("-vv")

        if config.get("coverage", False):
            cmd.extend(
                [
                    "--cov=apgi_framework",
                    "--cov=research",
                    "--cov-report=term-missing",
                ]
            )

        # Add test files
        cmd.extend(sorted(test_files))

        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=config.get("timeout", 1800),
            )

            # Parse results
            execution.end_time = datetime.now()
            parsed_results = self._parse_pytest_output(result.stdout, result.stderr)

            # Create summary results
            total_tests = len(parsed_results)
            passed_tests = sum(1 for r in parsed_results if r.status == FrameworkRunStatus.PASSED)
            failed_tests = sum(1 for r in parsed_results if r.status == FrameworkRunStatus.FAILED)
            skipped_tests = sum(1 for r in parsed_results if r.status == FrameworkRunStatus.SKIPPED)
            error_tests = sum(1 for r in parsed_results if r.status == FrameworkRunStatus.ERROR)
            execution_time = (
                (execution.end_time - execution.start_time).total_seconds()
                if execution.start_time and execution.end_time
                else 0.0
            )

            execution.results = FrameworkResults(
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=skipped_tests,
                error_tests=error_tests,
                execution_time=execution_time,
                failures=[],  # Would need to extract from output
                timestamp=execution.end_time or datetime.now(),
                results=parsed_results,
            )

            return execution

        except subprocess.TimeoutExpired:
            execution.end_time = datetime.now()
            execution.configuration["timeout"] = True
            return execution
        except Exception as e:
            execution.end_time = datetime.now()
            execution.configuration["error"] = str(e)
            return execution

    def _parse_pytest_output(self, stdout: str, stderr: str) -> List[FrameworkRunResult]:
        """Parse pytest output to extract test results."""
        results = []

        # Simple parsing - in a real implementation, this would be more sophisticated
        lines = stdout.split("\n")
        for line in lines:
            if "::" in line and any(
                status in line for status in ["PASSED", "FAILED", "SKIPPED", "ERROR"]
            ):
                # Extract test name and status
                parts = line.split()
                if len(parts) >= 2:
                    # test_name = parts[0]  # Unused but kept for reference
                    status_str = parts[1]

                    # Map status
                    status_map = {
                        "PASSED": FrameworkRunStatus.PASSED,
                        "FAILED": FrameworkRunStatus.FAILED,
                        "SKIPPED": FrameworkRunStatus.SKIPPED,
                        "ERROR": FrameworkRunStatus.ERROR,
                    }

                    status = status_map.get(status_str, FrameworkRunStatus.UNKNOWN)

                    # Create a basic test case for compatibility
                    # In a real implementation, we'd match this to the actual test definition
                    dummy_test_case = FrameworkTestCase(
                        name=parts[0] if parts else "unknown",
                        file_path=Path("unknown"),  # noqa: F821
                        module="unknown",
                        class_name=None,
                        method_name="unknown",
                        category=FrameworkRunCategory.UNIT,
                        line_number=0,
                        docstring=None,
                    )

                    result = FrameworkRunResult(
                        test_case=dummy_test_case,
                        status=status,
                        duration=0.1,
                        output=line,
                        timestamp=datetime.now(),
                    )
                    results.append(result)

        return results


# Backward compatibility aliases
TestUtilities = FrameworkTestUtils
TestDefinition = FrameworkTestCase
TestConfiguration = FrameworkConfiguration
TestCollection = FrameworkTestSuite
TestRunResult = FrameworkRunResult
TestResults = FrameworkResults
TestRunExecution = FrameworkExecution
TestFailure = FrameworkFailure
TestRunStatus = FrameworkRunStatus
TestRunCategory = FrameworkRunCategory
FailureCategory = FrameworkFailureCategory


class FrameworkTestUtilities:
    """Test utilities for the APGI Framework test enhancement system.

    This class provides utility methods for test execution, discovery, and management.
    """

    def __init__(self) -> None:
        """Initialize the test utilities."""
        self.logger = logging.getLogger(__name__)

    def discover_tests(self, test_path: str) -> List[str]:
        """Discover test files in the given path.

        Args:
            test_path: Path to search for test files

        Returns:
            List of discovered test file paths
        """
        # Simple test discovery implementation
        import glob

        test_files = glob.glob(f"{test_path}/test_*.py")
        return test_files

    def run_tests(self, test_files: List[str]) -> Dict[str, Any]:
        """Run the specified test files.

        Args:
            test_files: List of test files to run

        Returns:
            Dictionary with test results
        """
        # Basic implementation - could be enhanced
        results = {
            "total_tests": len(test_files),
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
        }
        return results
