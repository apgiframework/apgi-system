"""Anti-Pattern Detector for Test Code Analysis.

Analyzes test code using AST parsing to detect common anti-patterns,
provides improvement suggestions, and gives resource requirement feedback.
"""

import ast
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logging.standardized_logging import get_logger


class AntiPatternType(Enum):
    """Types of anti-patterns in test code."""

    NO_ASSERTIONS = "no_assertions"
    TOO_MANY_ASSERTIONS = "too_many_assertions"
    HARDCODED_VALUES = "hardcoded_values"
    SLEEP_IN_TESTS = "sleep_in_tests"
    PRINT_DEBUGGING = "print_debugging"
    OVERLY_COMPLEX = "overly_complex"
    MISSING_DOCSTRING = "missing_docstring"
    POOR_TEST_NAMING = "poor_test_naming"
    SHARED_STATE = "shared_state"
    EXTERNAL_DEPENDENCIES = "external_dependencies"
    MAGIC_NUMBERS = "magic_numbers"
    DUPLICATE_CODE = "duplicate_code"
    BROAD_EXCEPTION_HANDLING = "broad_exception_handling"
    RESOURCE_LEAKS = "resource_leaks"
    SLOW_OPERATIONS = "slow_operations"


class Severity(Enum):
    """Severity levels for anti-patterns."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ImprovementSuggestion:
    """Suggestion for improving test code."""

    title: str
    description: str
    code_example: Optional[str] = None
    rationale: str = ""
    effort_level: str = "medium"  # low, medium, high
    impact: str = "medium"  # low, medium, high


@dataclass
class ResourceRequirement:
    """Resource requirements for test execution."""

    memory_mb: Optional[int] = None
    execution_time_seconds: Optional[float] = None
    network_required: bool = False
    file_system_access: bool = False
    external_services: List[str] = field(default_factory=list)
    cpu_intensive: bool = False


@dataclass
class AntiPattern:
    """Detected anti-pattern in test code."""

    pattern_type: AntiPatternType
    severity: Severity
    location: str  # file:line:column
    message: str
    code_snippet: str
    suggestions: List[ImprovementSuggestion]
    resource_impact: Optional[ResourceRequirement] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_type": self.pattern_type.value,
            "severity": self.severity.value,
            "location": self.location,
            "message": self.message,
            "code_snippet": self.code_snippet,
            "suggestions_count": len(self.suggestions),
            "has_resource_impact": self.resource_impact is not None,
        }


class TestCodeAnalyzer(ast.NodeVisitor):
    """AST visitor for analyzing test code patterns."""

    def __init__(self, source_code: str, filename: str):
        self.source_code = source_code
        self.filename = filename
        self.source_lines = source_code.splitlines()
        self.anti_patterns: List[AntiPattern] = []

        # Analysis state
        self.current_function: Optional[str] = None
        self.current_class: Optional[str] = None
        self.assertion_count = 0
        self.function_complexity = 0
        self.imports: set[str] = set()
        self.global_variables: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statements."""
        if node.module:
            for alias in node.names:
                self.imports.add(f"{node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        old_function = self.current_function
        old_assertion_count = self.assertion_count
        old_complexity = self.function_complexity

        self.current_function = node.name
        self.assertion_count = 0
        self.function_complexity = 0

        # Check if this is a test function
        is_test_function = node.name.startswith("test_") or any(
            decorator.id == "pytest.mark" if hasattr(decorator, "id") else False
            for decorator in node.decorator_list
            if hasattr(decorator, "id")
        )

        if is_test_function:
            self._analyze_test_function(node)

        self.generic_visit(node)

        # Restore previous state
        self.current_function = old_function
        self.assertion_count = old_assertion_count
        self.function_complexity = old_complexity

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        old_class = self.current_class
        self.current_class = node.name

        # Check for test class patterns
        if node.name.startswith("Test") or node.name.endswith("Test"):
            self._analyze_test_class(node)

        self.generic_visit(node)
        self.current_class = old_class

    def visit_Assert(self, node: ast.Assert) -> None:
        """Visit assert statements."""
        self.assertion_count += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls."""
        # Check for problematic function calls
        if hasattr(node.func, "attr"):
            if node.func.attr in ["sleep", "time"]:
                self._add_anti_pattern(
                    AntiPatternType.SLEEP_IN_TESTS,
                    Severity.HIGH,
                    node,
                    "Sleep calls in tests make tests slow and unreliable",
                )

        if hasattr(node.func, "id"):
            if node.func.id == "print":
                self._add_anti_pattern(
                    AntiPatternType.PRINT_DEBUGGING,
                    Severity.MEDIUM,
                    node,
                    "Print statements should be replaced with proper logging",
                )

        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        """Visit try-except blocks."""
        # Check for overly broad exception handling
        for handler in node.handlers:
            if handler.type is None or (
                hasattr(handler.type, "id") and handler.type.id == "Exception"
            ):
                self._add_anti_pattern(
                    AntiPatternType.BROAD_EXCEPTION_HANDLING,
                    Severity.MEDIUM,
                    node,
                    "Catching broad exceptions can hide real issues",
                )

        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        """Visit constant literals (including magic numbers)."""
        # Check for magic numbers (excluding common test values)
        if isinstance(node.value, (int, float)) and node.value not in [
            0,
            1,
            -1,
            2,
            10,
            100,
        ]:
            self._add_anti_pattern(
                AntiPatternType.MAGIC_NUMBERS,
                Severity.LOW,
                node,
                f"Magic number {node.value} should be replaced with a named constant",
            )

        self.generic_visit(node)

    def _analyze_test_function(self, node: ast.FunctionDef) -> None:
        """Analyze a test function for anti-patterns."""
        # Check for missing docstring
        if not ast.get_docstring(node):
            self._add_anti_pattern(
                AntiPatternType.MISSING_DOCSTRING,
                Severity.LOW,
                node,
                "Test functions should have docstrings explaining what they test",
            )

        # Check function name quality
        if not self._is_good_test_name(node.name):
            self._add_anti_pattern(
                AntiPatternType.POOR_TEST_NAMING,
                Severity.MEDIUM,
                node,
                "Test names should be descriptive and follow naming conventions",
            )

        # Calculate complexity
        complexity = self._calculate_complexity(node)
        if complexity > 10:
            self._add_anti_pattern(
                AntiPatternType.OVERLY_COMPLEX,
                Severity.HIGH,
                node,
                f"Test function is too complex (complexity: {complexity})",
            )

        # Check for resource-intensive operations
        resource_req = self._analyze_resource_requirements(node)
        if resource_req.memory_mb and resource_req.memory_mb > 100:
            self._add_anti_pattern(
                AntiPatternType.SLOW_OPERATIONS,
                Severity.MEDIUM,
                node,
                "Test may consume excessive memory",
                resource_req,
            )

    def _analyze_test_class(self, node: ast.ClassDef) -> None:
        """Analyze a test class for anti-patterns."""
        # Check for shared state issues
        class_variables = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_variables.append(target.id)

        if class_variables:
            self._add_anti_pattern(
                AntiPatternType.SHARED_STATE,
                Severity.HIGH,
                node,
                f"Class variables may cause shared state issues: {', '.join(class_variables)}",
            )

    def _is_good_test_name(self, name: str) -> bool:
        """Check if test name follows good practices."""
        # Should start with test_
        if not name.startswith("test_"):
            return False

        # Should be descriptive (more than just test_something)
        parts = name.split("_")
        if len(parts) < 3:
            return False

        # Should not contain numbers (usually indicates poor naming)
        if any(char.isdigit() for char in name):
            return False

        return True

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _analyze_resource_requirements(self, node: ast.FunctionDef) -> ResourceRequirement:
        """Analyze resource requirements of a test function."""
        req = ResourceRequirement()

        # Check for network operations
        network_indicators = ["requests", "urllib", "http", "socket", "aiohttp"]
        if any(indicator in self.imports for indicator in network_indicators):
            req.network_required = True

        # Check for file operations
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and hasattr(child.func, "id"):
                if child.func.id in ["open"]:
                    req.file_system_access = True

        # Check for CPU-intensive operations
        cpu_indicators = ["numpy", "scipy", "pandas", "sklearn"]
        if any(indicator in self.imports for indicator in cpu_indicators):
            req.cpu_intensive = True
            req.memory_mb = 50  # Estimate

        # Estimate execution time based on operations
        operation_count = len([n for n in ast.walk(node) if isinstance(n, ast.Call)])
        if operation_count > 20:
            req.execution_time_seconds = operation_count * 0.1

        return req

    def _add_anti_pattern(
        self,
        pattern_type: AntiPatternType,
        severity: Severity,
        node: ast.AST,
        message: str,
        resource_impact: Optional[ResourceRequirement] = None,
    ) -> None:
        """Add an anti-pattern to the results."""
        if hasattr(node, "lineno") and hasattr(node, "col_offset"):
            location = f"{self.filename}:{node.lineno}:{node.col_offset}"
        else:
            location = f"{self.filename}:unknown"

        # Extract code snippet
        if hasattr(node, "lineno") and node.lineno <= len(self.source_lines):
            code_snippet = self.source_lines[node.lineno - 1].strip()
        else:
            code_snippet = "Code not available"

        # Generate suggestions
        suggestions = self._generate_suggestions(pattern_type, node)

        anti_pattern = AntiPattern(
            pattern_type=pattern_type,
            severity=severity,
            location=location,
            message=message,
            code_snippet=code_snippet,
            suggestions=suggestions,
            resource_impact=resource_impact,
        )

        self.anti_patterns.append(anti_pattern)

    def _generate_suggestions(
        self, pattern_type: AntiPatternType, node: ast.AST
    ) -> List[ImprovementSuggestion]:
        """Generate improvement suggestions for an anti-pattern."""
        suggestions_map = {
            AntiPatternType.NO_ASSERTIONS: [
                ImprovementSuggestion(
                    title="Add Assertions",
                    description="Add assert statements to verify expected behavior",
                    code_example="assert result == expected_value",
                    rationale="Tests without assertions don't verify anything",
                    effort_level="low",
                    impact="high",
                )
            ],
            AntiPatternType.TOO_MANY_ASSERTIONS: [
                ImprovementSuggestion(
                    title="Split Test Function",
                    description="Break down into multiple focused test functions",
                    code_example="def test_specific_behavior():\n    # Test one thing",
                    rationale="Each test should verify one specific behavior",
                    effort_level="medium",
                    impact="high",
                )
            ],
            AntiPatternType.SLEEP_IN_TESTS: [
                ImprovementSuggestion(
                    title="Use Proper Synchronization",
                    description="Replace sleep with proper waiting mechanisms",
                    code_example="wait_for_condition(lambda: condition_met(), timeout=5)",
                    rationale="Sleep makes tests slow and unreliable",
                    effort_level="medium",
                    impact="high",
                )
            ],
            AntiPatternType.PRINT_DEBUGGING: [
                ImprovementSuggestion(
                    title="Use Logging",
                    description="Replace print statements with proper logging",
                    code_example="logger.debug('Debug information: %s', value)",
                    rationale="Print statements clutter test output",
                    effort_level="low",
                    impact="medium",
                )
            ],
            AntiPatternType.OVERLY_COMPLEX: [
                ImprovementSuggestion(
                    title="Simplify Test Logic",
                    description="Break complex test into simpler, focused tests",
                    code_example="# Split into multiple test methods",
                    rationale="Complex tests are hard to understand and maintain",
                    effort_level="high",
                    impact="high",
                )
            ],
            AntiPatternType.MISSING_DOCSTRING: [
                ImprovementSuggestion(
                    title="Add Docstring",
                    description="Add docstring explaining what the test verifies",
                    code_example='"""Test that user login works with valid credentials."""',
                    rationale="Docstrings help understand test purpose",
                    effort_level="low",
                    impact="medium",
                )
            ],
            AntiPatternType.POOR_TEST_NAMING: [
                ImprovementSuggestion(
                    title="Improve Test Name",
                    description="Use descriptive names that explain what is being tested",
                    code_example="test_user_login_with_valid_credentials_succeeds",
                    rationale="Good names make tests self-documenting",
                    effort_level="low",
                    impact="medium",
                )
            ],
            AntiPatternType.SHARED_STATE: [
                ImprovementSuggestion(
                    title="Use Test Fixtures",
                    description="Replace shared state with proper test fixtures",
                    code_example="@pytest.fixture\ndef user_data():\n    return {'name': 'test'}",
                    rationale="Shared state causes test interdependencies",
                    effort_level="medium",
                    impact="high",
                )
            ],
            AntiPatternType.MAGIC_NUMBERS: [
                ImprovementSuggestion(
                    title="Use Named Constants",
                    description="Replace magic numbers with named constants",
                    code_example="EXPECTED_COUNT = 42\nassert len(results) == EXPECTED_COUNT",
                    rationale="Named constants make tests more readable",
                    effort_level="low",
                    impact="medium",
                )
            ],
            AntiPatternType.BROAD_EXCEPTION_HANDLING: [
                ImprovementSuggestion(
                    title="Catch Specific Exceptions",
                    description="Catch only the specific exceptions you expect",
                    code_example="except ValueError as e:\n    # Handle specific error",
                    rationale="Broad exception handling can hide real issues",
                    effort_level="low",
                    impact="medium",
                )
            ],
        }

        return suggestions_map.get(pattern_type, [])


class AntiPatternDetector:
    """Detects anti-patterns in test code using AST analysis.

    Provides comprehensive analysis of test code quality, identifies
    common anti-patterns, and suggests improvements.
    """

    def __init__(self) -> None:
        self.logger = get_logger("anti_pattern_detector")

    def analyze_file(self, file_path: Path) -> List[AntiPattern]:
        """Analyze a single test file for anti-patterns.

        Args:
            file_path: Path to the test file

        Returns:
            List of detected anti-patterns
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            return self.analyze_code(source_code, str(file_path))

        except Exception as e:
            self.logger.error(f"Failed to analyze file {file_path}", exception=e)
            return []

    def analyze_code(self, source_code: str, filename: str = "<string>") -> List[AntiPattern]:
        """Analyze source code for anti-patterns.

        Args:
            source_code: Python source code to analyze
            filename: Name of the file (for error reporting)

        Returns:
            List of detected anti-patterns
        """
        try:
            tree = ast.parse(source_code, filename=filename)
            analyzer = TestCodeAnalyzer(source_code, filename)
            analyzer.visit(tree)

            # Post-processing checks
            self._post_process_analysis(analyzer)

            return analyzer.anti_patterns

        except SyntaxError as e:
            self.logger.error(f"Syntax error in {filename}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to analyze code in {filename}", exception=e)
            return []

    def analyze_directory(
        self, directory_path: Path, pattern: str = "test_*.py"
    ) -> Dict[str, List[AntiPattern]]:
        """Analyze all test files in a directory.

        Args:
            directory_path: Path to directory containing test files
            pattern: File pattern to match (default: test_*.py)

        Returns:
            Dictionary mapping file paths to lists of anti-patterns
        """
        results = {}

        for file_path in directory_path.rglob(pattern):
            if file_path.is_file():
                anti_patterns = self.analyze_file(file_path)
                if anti_patterns:
                    results[str(file_path)] = anti_patterns

        return results

    def _post_process_analysis(self, analyzer: TestCodeAnalyzer) -> None:
        """Post-process analysis results to add additional checks."""
        # Check for functions with no assertions
        for node in ast.walk(ast.parse(analyzer.source_code)):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                # Count assertions in this function
                assertion_count = 0
                for child in ast.walk(node):
                    if isinstance(child, ast.Assert):
                        assertion_count += 1

                if assertion_count == 0:
                    analyzer._add_anti_pattern(
                        AntiPatternType.NO_ASSERTIONS,
                        Severity.HIGH,
                        node,
                        "Test function has no assertions",
                    )
                elif assertion_count > 10:
                    analyzer._add_anti_pattern(
                        AntiPatternType.TOO_MANY_ASSERTIONS,
                        Severity.MEDIUM,
                        node,
                        f"Test function has too many assertions ({assertion_count})",
                    )

    def generate_report(self, anti_patterns: List[AntiPattern]) -> Dict[str, Any]:
        """Generate a comprehensive report of anti-patterns.

        Args:
            anti_patterns: List of detected anti-patterns

        Returns:
            Report dictionary with statistics and recommendations
        """
        if not anti_patterns:
            return {
                "total_issues": 0,
                "by_severity": {},
                "by_type": {},
                "recommendations": [],
            }

        # Count by severity
        by_severity: Dict[str, int] = {}
        for pattern in anti_patterns:
            severity = pattern.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1

        # Count by type
        by_type: Dict[str, int] = {}
        for pattern in anti_patterns:
            pattern_type = pattern.pattern_type.value
            by_type[pattern_type] = by_type.get(pattern_type, 0) + 1

        # Generate recommendations
        recommendations = self._generate_recommendations(anti_patterns)

        return {
            "total_issues": len(anti_patterns),
            "by_severity": by_severity,
            "by_type": by_type,
            "recommendations": recommendations,
            "issues": [pattern.to_dict() for pattern in anti_patterns],
        }

    def _generate_recommendations(self, anti_patterns: List[AntiPattern]) -> List[str]:
        """Generate high-level recommendations based on detected patterns."""
        recommendations = []

        # Count pattern types
        type_counts: Dict[str, int] = {}
        for pattern in anti_patterns:
            type_counts[pattern.pattern_type.value] = (
                type_counts.get(pattern.pattern_type.value, 0) + 1
            )

        # Generate recommendations based on common issues
        if type_counts.get(AntiPatternType.NO_ASSERTIONS.value, 0) > 3:
            recommendations.append(
                "Consider implementing a test coverage tool to ensure all tests have assertions"
            )

        if type_counts.get(AntiPatternType.SLEEP_IN_TESTS.value, 0) > 1:
            recommendations.append("Replace sleep calls with proper synchronization mechanisms")

        if type_counts.get(AntiPatternType.OVERLY_COMPLEX.value, 0) > 2:
            recommendations.append("Break down complex tests into smaller, focused test functions")

        if type_counts.get(AntiPatternType.SHARED_STATE.value, 0) > 1:
            recommendations.append(
                "Implement proper test isolation using fixtures and setup/teardown methods"
            )

        # Resource-related recommendations
        resource_intensive_count = sum(1 for p in anti_patterns if p.resource_impact)
        if resource_intensive_count > 5:
            recommendations.append(
                "Consider optimizing resource-intensive tests or running them separately"
            )

        return recommendations
