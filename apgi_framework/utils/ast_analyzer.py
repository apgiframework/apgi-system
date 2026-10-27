"""AST (Abstract Syntax Tree) analysis utilities for Python code analysis.

This module provides comprehensive Python AST parsing and analysis capabilities
for function/class/method extraction, code complexity analysis, and test code
analysis for the test enhancement system.
"""

import ast
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, cast

from .file_utils import FileUtils


class NodeType(Enum):
    """Types of AST nodes we analyze."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    ASYNC_FUNCTION = "async_function"
    PROPERTY = "property"
    STATICMETHOD = "staticmethod"
    CLASSMETHOD = "classmethod"


@dataclass
class CodeElement:
    """Represents a code element (function, class, method, etc.)."""

    name: str
    node_type: NodeType
    line_start: int
    line_end: int
    docstring: Optional[str]
    parameters: List[str]
    return_annotation: Optional[str]
    decorators: List[str]
    complexity: int
    is_test: bool
    is_private: bool
    parent_class: Optional[str] = None


@dataclass
class ComplexityMetrics:
    """Code complexity metrics."""

    cyclomatic_complexity: int
    cognitive_complexity: int
    lines_of_code: int
    logical_lines: int
    comment_lines: int
    blank_lines: int
    maintainability_index: float


@dataclass
class TestMetrics:
    """Test-specific code metrics."""

    assertion_count: int
    test_methods: List[str]
    setup_methods: List[str]
    teardown_methods: List[str]
    fixture_usage: List[str]
    mock_usage: List[str]
    has_property_tests: bool
    test_categories: List[str]


class ASTAnalyzer:
    """Comprehensive AST analysis for Python code.

    Provides parsing, analysis, and extraction capabilities for code elements,
    complexity metrics, and test-specific analysis.
    """

    def __init__(self) -> None:
        """Initialize AST analyzer."""
        self.file_utils = FileUtils()
        self.logger = logging.getLogger(__name__)

    def parse_file(self, file_path: Union[str, Path]) -> ast.AST:
        """Parse Python file into AST.

        Args:
            file_path: Path to Python file

        Returns:
            AST root node

        Raises:
            SyntaxError: If file has syntax errors
            FileNotFoundError: If file doesn't exist
        """
        try:
            content = self.file_utils.safe_read_text(file_path)
            tree = ast.parse(content, filename=str(file_path))
            self.logger.debug(f"Successfully parsed {file_path}")
            return cast(ast.AST, tree)
        except SyntaxError as e:
            self.logger.error(f"Syntax error in {file_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to parse {file_path}: {e}")
            raise

    def extract_code_elements(self, file_path: Union[str, Path]) -> List[CodeElement]:
        """Extract all code elements (functions, classes, methods) from file.

        Args:
            file_path: Path to Python file

        Returns:
            List of code elements
        """
        tree = self.parse_file(file_path)
        elements = []

        class ElementVisitor(ast.NodeVisitor):
            def __init__(self, analyzer: "ASTAnalyzer") -> None:
                self.analyzer = analyzer
                self.current_class: Optional[str] = None

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                # Extract class information
                element = self._create_element_from_class(node)
                elements.append(element)

                # Visit class methods
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                element = self._create_element_from_function(node)
                elements.append(element)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                element = self._create_element_from_function(node, is_async=True)
                elements.append(element)

            def _create_element_from_class(self, node: ast.ClassDef) -> CodeElement:
                return CodeElement(
                    name=node.name,
                    node_type=NodeType.CLASS,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    docstring=ast.get_docstring(node),
                    parameters=[],
                    return_annotation=None,
                    decorators=[self.analyzer._get_decorator_name(d) for d in node.decorator_list],
                    complexity=self.analyzer._calculate_complexity(node),
                    is_test=self.analyzer._is_test_class(node),
                    is_private=node.name.startswith("_"),
                    parent_class=None,
                )

            def _create_element_from_function(
                self,
                node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
                is_async: bool = False,
            ) -> CodeElement:
                node_type = NodeType.ASYNC_FUNCTION if is_async else NodeType.FUNCTION

                # Determine if it's a method and its type
                if self.current_class:
                    if any(
                        self.analyzer._get_decorator_name(d) == "property"
                        for d in node.decorator_list
                    ):
                        node_type = NodeType.PROPERTY
                    elif any(
                        self.analyzer._get_decorator_name(d) == "staticmethod"
                        for d in node.decorator_list
                    ):
                        node_type = NodeType.STATICMETHOD
                    elif any(
                        self.analyzer._get_decorator_name(d) == "classmethod"
                        for d in node.decorator_list
                    ):
                        node_type = NodeType.CLASSMETHOD
                    else:
                        node_type = NodeType.METHOD

                return CodeElement(
                    name=node.name,
                    node_type=node_type,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    docstring=ast.get_docstring(node),
                    parameters=self.analyzer._extract_parameters(node),
                    return_annotation=self.analyzer._get_return_annotation(node),
                    decorators=[self.analyzer._get_decorator_name(d) for d in node.decorator_list],
                    complexity=self.analyzer._calculate_complexity(node),
                    is_test=self.analyzer._is_test_function(node),
                    is_private=node.name.startswith("_"),
                    parent_class=self.current_class,
                )

        visitor = ElementVisitor(self)
        visitor.visit(tree)

        self.logger.debug(f"Extracted {len(elements)} code elements from {file_path}")
        return elements

    def calculate_complexity_metrics(self, file_path: Union[str, Path]) -> ComplexityMetrics:
        """Calculate comprehensive complexity metrics for file.

        Args:
            file_path: Path to Python file

        Returns:
            Complexity metrics
        """
        tree = self.parse_file(file_path)
        content = self.file_utils.safe_read_text(file_path)
        lines = content.splitlines()

        # Calculate various metrics
        cyclomatic = self._calculate_cyclomatic_complexity(tree)
        cognitive = self._calculate_cognitive_complexity(tree)

        # Line metrics
        total_lines = len(lines)
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
        blank_lines = sum(1 for line in lines if not line.strip())
        logical_lines = total_lines - comment_lines - blank_lines

        # Maintainability index (simplified version)
        maintainability = self._calculate_maintainability_index(
            logical_lines, cyclomatic, total_lines
        )

        return ComplexityMetrics(
            cyclomatic_complexity=cyclomatic,
            cognitive_complexity=cognitive,
            lines_of_code=total_lines,
            logical_lines=logical_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            maintainability_index=maintainability,
        )

    def analyze_test_code(self, file_path: Union[str, Path]) -> TestMetrics:
        """Analyze test-specific metrics and patterns.

        Args:
            file_path: Path to test file

        Returns:
            Test metrics
        """
        tree = self.parse_file(file_path)

        test_methods = []
        setup_methods = []
        teardown_methods = []
        fixture_usage = []
        mock_usage = []
        assertion_count = 0
        has_property_tests = False
        test_categories = []

        class TestVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                if self._is_test_method(node):
                    test_methods.append(node.name)
                elif self._is_setup_method(node):
                    setup_methods.append(node.name)
                elif self._is_teardown_method(node):
                    teardown_methods.append(node.name)

                # Count assertions
                nonlocal assertion_count
                assertion_count += self._count_assertions(node)

                # Check for property tests
                nonlocal has_property_tests
                if self._has_hypothesis_decorators(node):
                    has_property_tests = True

                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                # Check for fixture usage
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["fixture", "pytest.fixture"]:
                        fixture_usage.append(node.func.id)
                    elif "mock" in node.func.id.lower() or "Mock" in node.func.id:
                        mock_usage.append(node.func.id)

                self.generic_visit(node)

            def _is_test_method(self, node: ast.FunctionDef) -> bool:
                return node.name.startswith("test_") or any(
                    "test" in self._get_decorator_name(d).lower() for d in node.decorator_list
                )

            def _is_setup_method(self, node: ast.FunctionDef) -> bool:
                setup_names = ["setUp", "setup", "setup_method", "setup_class"]
                return node.name in setup_names or any(
                    "setup" in self._get_decorator_name(d).lower() for d in node.decorator_list
                )

            def _is_teardown_method(self, node: ast.FunctionDef) -> bool:
                teardown_names = [
                    "tearDown",
                    "teardown",
                    "teardown_method",
                    "teardown_class",
                ]
                return node.name in teardown_names or any(
                    "teardown" in self._get_decorator_name(d).lower() for d in node.decorator_list
                )

            def _count_assertions(self, node: ast.FunctionDef) -> int:
                count = 0
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute):
                            if child.func.attr.startswith("assert"):
                                count += 1
                        elif isinstance(child.func, ast.Name):
                            if child.func.id.startswith("assert"):
                                count += 1
                return count

            def _has_hypothesis_decorators(self, node: ast.FunctionDef) -> bool:
                for decorator in node.decorator_list:
                    decorator_name = self._get_decorator_name(decorator)
                    if "given" in decorator_name or "hypothesis" in decorator_name.lower():
                        return True
                return False

            def _get_decorator_name(self, decorator: ast.expr) -> str:
                if isinstance(decorator, ast.Name):
                    return decorator.id
                elif isinstance(decorator, ast.Attribute):
                    return (
                        f"{decorator.value.id}.{decorator.attr}"
                        if isinstance(decorator.value, ast.Name)
                        else decorator.attr
                    )
                elif isinstance(decorator, ast.Call):
                    return self._get_decorator_name(decorator.func)
                return str(decorator)

        visitor = TestVisitor()
        visitor.visit(tree)

        # Determine test categories based on file name and content
        file_name = Path(file_path).name.lower()
        if "unit" in file_name:
            test_categories.append("unit")
        if "integration" in file_name:
            test_categories.append("integration")
        if "property" in file_name or has_property_tests:
            test_categories.append("property")
        if "gui" in file_name or "ui" in file_name:
            test_categories.append("gui")

        return TestMetrics(
            assertion_count=assertion_count,
            test_methods=test_methods,
            setup_methods=setup_methods,
            teardown_methods=teardown_methods,
            fixture_usage=fixture_usage,
            mock_usage=mock_usage,
            has_property_tests=has_property_tests,
            test_categories=test_categories,
        )

    def find_functions_by_name(
        self, file_path: Union[str, Path], name_pattern: str
    ) -> List[CodeElement]:
        """Find functions matching name pattern.

        Args:
            file_path: Path to Python file
            name_pattern: Pattern to match (supports wildcards)

        Returns:
            List of matching functions
        """
        elements = self.extract_code_elements(file_path)

        import fnmatch

        matching = []
        for element in elements:
            if element.node_type in [
                NodeType.FUNCTION,
                NodeType.METHOD,
                NodeType.ASYNC_FUNCTION,
            ]:
                if fnmatch.fnmatch(element.name, name_pattern):
                    matching.append(element)

        return matching

    def find_classes_by_name(
        self, file_path: Union[str, Path], name_pattern: str
    ) -> List[CodeElement]:
        """Find classes matching name pattern.

        Args:
            file_path: Path to Python file
            name_pattern: Pattern to match (supports wildcards)

        Returns:
            List of matching classes
        """
        elements = self.extract_code_elements(file_path)

        import fnmatch

        matching = []
        for element in elements:
            if element.node_type == NodeType.CLASS:
                if fnmatch.fnmatch(element.name, name_pattern):
                    matching.append(element)

        return matching

    def get_imports(self, file_path: Union[str, Path]) -> Dict[str, List[str]]:
        """Extract all imports from file.

        Args:
            file_path: Path to Python file

        Returns:
            Dictionary with 'imports' and 'from_imports' lists
        """
        tree = self.parse_file(file_path)

        imports = []
        from_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    from_imports.append(f"{module}.{alias.name}")

        return {"imports": imports, "from_imports": from_imports}

    def get_dependencies(self, file_path: Union[str, Path]) -> Set[str]:
        """Get all module dependencies from imports.

        Args:
            file_path: Path to Python file

        Returns:
            Set of module names
        """
        imports_data = self.get_imports(file_path)
        dependencies = set()

        # Extract top-level module names
        for imp in imports_data["imports"]:
            dependencies.add(imp.split(".")[0])

        for imp in imports_data["from_imports"]:
            if imp and "." in imp:
                dependencies.add(imp.split(".")[0])

        return dependencies

    def _extract_parameters(
        self, func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> List[str]:
        """Extract parameter names from function definition."""
        params = []

        # Regular arguments
        for arg in func_node.args.args:
            params.append(arg.arg)

        # *args
        if func_node.args.vararg:
            params.append(f"*{func_node.args.vararg.arg}")

        # **kwargs
        if func_node.args.kwarg:
            params.append(f"**{func_node.args.kwarg.arg}")

        return params

    def _get_return_annotation(
        self, func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> Optional[str]:
        """Get return type annotation as string."""
        if func_node.returns:
            return ast.unparse(func_node.returns)
        return None

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Get decorator name as string."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            if isinstance(decorator.value, ast.Name):
                return f"{decorator.value.id}.{decorator.attr}"
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        return str(decorator)

    def _is_test_function(self, func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if function is a test function."""
        # Check name
        if func_node.name.startswith("test_"):
            return True

        # Check decorators
        for decorator in func_node.decorator_list:
            decorator_name = self._get_decorator_name(decorator)
            if "test" in decorator_name.lower():
                return True

        return False

    def _is_test_class(self, class_node: ast.ClassDef) -> bool:
        """Check if class is a test class."""
        # Check name
        if class_node.name.startswith("Test") or class_node.name.endswith("Test"):
            return True

        # Check base classes
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                if "Test" in base.id:
                    return True
            elif isinstance(base, ast.Attribute):
                if "Test" in base.attr:
                    return True

        return False

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a node."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1

        return complexity

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate total cyclomatic complexity for entire file."""
        total_complexity = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_complexity += self._calculate_complexity(node)

        return total_complexity

    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity (simplified version)."""
        # This is a simplified version - full cognitive complexity is more complex

        class CognitiveVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.complexity = 0
                self.nesting = 0

            def visit_If(self, node: ast.If) -> None:
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1

            def visit_While(self, node: ast.While) -> None:
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1

            def visit_For(self, node: ast.For) -> None:
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1

        visitor = CognitiveVisitor()
        visitor.visit(tree)
        return cast(int, visitor.complexity)

    def _calculate_maintainability_index(
        self, logical_lines: int, cyclomatic: int, total_lines: int
    ) -> float:
        """Calculate maintainability index (simplified version)."""
        import math

        # Simplified maintainability index calculation
        # Real formula is more complex and includes Halstead metrics
        if logical_lines == 0:
            return 100.0

        volume = logical_lines * math.log2(max(1, logical_lines))
        mi = max(
            0,
            (171 - 5.2 * math.log(volume) - 0.23 * cyclomatic - 16.2 * math.log(total_lines))
            * 100
            / 171,
        )

        return round(mi, 2)
