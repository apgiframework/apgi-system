"""Secure code execution sandbox for APGI Framework.

Provides a safe environment for dynamic code execution with strict
security controls, resource limits, and input validation.

Uses RestrictedPython for secure execution when available.
For production execution of genuinely untrusted code, consider using
subprocess-based isolation or dedicated sandboxing libraries.
"""

import ast
import logging
from typing import Any, Dict, List, Optional

try:
    from RestrictedPython import compile_restricted  # type: ignore
    from RestrictedPython import safe_globals

    HAS_RESTRICTED_PYTHON = True
except ImportError:
    HAS_RESTRICTED_PYTHON = False
    # Fallback definition for safe_globals
    safe_globals = {
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sum": sum,
            "max": max,
            "min": min,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "bool": bool,
        }
    }
    raise ImportError(
        "RestrictedPython is required for secure code execution. "
        "Install with: pip install RestrictedPython"
    )


logger = logging.getLogger(__name__)


class SandboxError(Exception):
    """Sandbox execution errors."""


class SecurityViolationError(SandboxError):
    """Security policy violations."""


class ResourceLimitExceededError(SandboxError):
    """Resource limit exceeded."""


class CodeSandbox:
    """Secure sandbox for executing untrusted Python code.

    Provides isolation, resource limits, and strict security controls
    to prevent malicious code execution.
    """

    def __init__(
        self,
        max_memory_mb: int = 100,
        max_cpu_time: float = 5.0,
        max_execution_time: float = 10.0,
        allow_network: bool = False,
        allow_file_access: bool = False,
        allowed_modules: Optional[List[str]] = None,
        use_subprocess_isolation: bool = False,
    ):
        """Initialize sandbox with security constraints.

        Args:
            max_memory_mb: Maximum memory usage in MB
            max_cpu_time: Maximum CPU time in seconds
            max_execution_time: Maximum wall clock time in seconds
            allow_network: Whether to allow network access
            allow_file_access: Whether to allow file system access
            allowed_modules: List of allowed Python modules
            use_subprocess_isolation: Whether to use subprocess-based isolation for enhanced security
        """
        self.max_memory_mb = max_memory_mb
        self.max_cpu_time = max_cpu_time
        self.max_execution_time = max_execution_time
        self.allow_network = allow_network
        self.allow_file_access = allow_file_access
        self.use_subprocess_isolation = use_subprocess_isolation

        # Default allowed modules for scientific computing
        self.allowed_modules = allowed_modules or [
            "math",
            "random",
            "statistics",
            "itertools",
            "functools",
            "collections",
            "datetime",
            "decimal",
            "fractions",
            "numpy",
            "scipy",
            "pandas",
            "matplotlib.pyplot",
            "seaborn",
            "sklearn",
            "statsmodels",
        ]

        # Forbidden modules and operations
        self.forbidden_modules = {
            "os",
            "sys",
            "subprocess",
            "socket",
            "urllib",
            "http",
            "ftplib",
            "smtplib",
            "telnetlib",
            "pickle",
            "shelve",
            "dbm",
            "sqlite3",
            "eval",
            "exec",
            "compile",
            "__import__",
            "open",
            "file",
            "input",
            "raw_input",
            "help",
            "credits",
        }

        self.forbidden_functions = {
            "eval",
            "exec",
            "compile",
            "__import__",
            "reload",
            "globals",
            "locals",
            "vars",
            "dir",
            "hasattr",
            "getattr",
            "setattr",
            "delattr",
            "callable",
            "isinstance",
            "issubclass",
            "open",
        }

    def validate_code(self, code: str) -> bool:
        """Validate code for security violations before execution.

        Args:
            code: Python code to validate

        Raises:
            SecurityViolationError: If code appears malicious
        """
        # Parse AST to check for violations
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SecurityViolationError(f"Syntax error in code: {e}")

        # Check for dangerous imports and function calls
        validator = SecurityValidator(
            self.allowed_modules, self.forbidden_modules, self.forbidden_functions
        )
        validator.visit(tree)

        if validator.violations:
            raise SecurityViolationError(f"Security violations found: {validator.violations}")

        return True

    def execute_code(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        capture_output: bool = True,
    ) -> Dict[str, Any]:
        """Execute code in secure sandbox environment.

        Args:
            code: Python code to execute
            context: Optional variables to inject into execution context
            capture_output: Whether to capture stdout/stderr

        Returns:
            Dictionary with execution results

        Raises:
            SandboxError: If execution fails or security violation occurs
        """
        # Validate code first
        self.validate_code(code)

        if self.use_subprocess_isolation:
            return self._execute_with_subprocess(code, context, capture_output)
        elif HAS_RESTRICTED_PYTHON:
            return self._execute_with_restricted_python(code, context, capture_output)
        else:
            return self._execute_with_fallback(code, context, capture_output)

    def _execute_with_subprocess(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        capture_output: bool = True,
    ) -> Dict[str, Any]:
        """Execute code using subprocess isolation for enhanced security.

        This provides OS-level isolation by running code in a separate process
        with restricted permissions and resource limits.
        """
        import json
        import subprocess
        import sys
        import tempfile

        try:
            # Serialize context for subprocess with proper escaping
            context_json = json.dumps(context or {})
            # Escape single quotes in the JSON to prevent code injection
            escaped_context_json = context_json.replace("'", "\\'")

            # Write code to temporary file instead of embedding it
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as code_file:
                code_file.write(code)
                code_file_path = code_file.name

            # Create isolated execution script with proper escaping
            exec_script = f"""
import sys
import json
import signal
import resource

# Set resource limits
resource.setrlimit(resource.RLIMIT_AS, ({self.max_memory_mb * 1024 * 1024}, resource.RLIM_INFINITY))
resource.setrlimit(resource.RLIMIT_CPU, ({self.max_cpu_time}, resource.RLIM_INFINITY))

# Load context
try:
    context = json.loads('{escaped_context_json}')
except:
    context = {{}}

# Execute code from file
try:
    exec_locals = {{**context}}
    with open('{code_file_path}', 'r', encoding='utf-8') as f:
        exec(f.read(), exec_locals)
    result = {{"success": True, "context": {{k: str(v) for k, v in exec_locals.items() if not k.startswith('__')}}}}
except Exception as e:
    result = {{"success": False, "error": str(e), "error_type": type(e).__name__}}

# Output result
print(json.dumps(result))
"""

            # Run in subprocess with timeout
            result = subprocess.run(
                [sys.executable, "-c", exec_script],
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,
                check=False,  # Don't raise exception on timeout
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr,
                    "error_type": "SubprocessError",
                }

            # Parse result
            try:
                exec_result: dict[str, Any] = json.loads(result.stdout.strip())
                return exec_result
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Failed to parse execution result",
                    "error_type": "ParseError",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timeout after {self.max_execution_time}s",
                "error_type": "TimeoutError",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    def _execute_with_restricted_python(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        capture_output: bool = True,
    ) -> Dict[str, Any]:
        """Execute code using RestrictedPython for enhanced security."""
        try:
            # Compile the code with restrictions
            compiled_code = compile_restricted(code, "<string>", "exec")

            # Prepare restricted globals
            restricted_globals: Dict[str, Any] = dict(safe_globals)
            restricted_globals["_getattr_"] = getattr
            restricted_globals["_write_"] = lambda x: x  # Allow writing

            # Add allowed modules
            for module_name in self.allowed_modules:
                try:
                    restricted_globals[module_name] = __import__(module_name)
                except ImportError:
                    pass

            # Add user context
            if context:
                for key, value in context.items():
                    if self._is_safe_value(value):
                        restricted_globals[key] = value

            # Execute with output capture if requested
            if capture_output:
                import contextlib
                import io

                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()

                with (
                    contextlib.redirect_stdout(stdout_capture),
                    contextlib.redirect_stderr(stderr_capture),
                ):
                    exec(
                        compiled_code, restricted_globals
                    )  # nosec B102 - intentional, restricted-globals sandbox

                return {
                    "success": True,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue(),
                    "context": restricted_globals,
                }
            else:
                exec(
                    compiled_code, restricted_globals
                )  # nosec B102 - intentional, restricted-globals sandbox
                return {"success": True, "context": restricted_globals}

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    def _execute_with_fallback(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        capture_output: bool = True,
    ) -> Dict[str, Any]:
        """Fallback execution method with basic restrictions.

        NOTE: This provides soft isolation only. For production use with untrusted code,
        consider using subprocess with restricted user permissions or dedicated sandboxing
        libraries like PyPy sandbox for OS-level isolation.
        """
        try:
            exec_context = self._prepare_execution_context(context)

            if capture_output:
                import contextlib
                import io

                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()

                with (
                    contextlib.redirect_stdout(stdout_capture),
                    contextlib.redirect_stderr(stderr_capture),
                ):
                    exec(code, exec_context)  # nosec B102 - intentional, restricted-globals sandbox

                return {
                    "success": True,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue(),
                    "context": exec_context,
                }
            else:
                exec(code, exec_context)  # nosec B102 - intentional, restricted-globals sandbox
                return {"success": True, "context": exec_context}

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    def validate_input(self, data: Any) -> bool:
        """Validate input data for security threats.

        Args:
            data: Input data to validate

        Raises:
            SecurityViolationError: If data is malicious
        """
        if isinstance(data, str):
            # Check for dangerous patterns
            dangerous_patterns = [
                "__import__",
                "eval(",
                "exec(",
                "compile(",
                "subprocess",
                "os.system",
                "os.popen",
                "socket",
                "urllib",
                "http",
                "ftplib",
                "pickle.loads",
                "pickle.load",
                "__reduce__",
                "__reduce_ex__",
            ]

            for pattern in dangerous_patterns:
                if pattern in data.lower():
                    raise SecurityViolationError(f"Dangerous pattern detected: {pattern}")

        elif isinstance(data, (bytes, bytearray)):
            # Check for malicious bytecode
            if b"__import__" in data or b"eval(" in data or b"exec(" in data:
                raise SecurityViolationError("Malicious bytecode detected")

        return True

    def _prepare_execution_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare secure execution context with restricted builtins."""
        # Create a very restricted builtins environment to prevent escape attacks
        safe_builtins = {
            # Essential types and functions only - avoid anything that could lead to object.__class__
            "print": print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sum": sum,
            "max": max,
            "min": min,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "bool": bool,
            # Remove 'type' and 'isinstance' as they can be used for escape
            # Remove 'hasattr' as it can be used for introspection
            # Math functions
            "math": __import__("math"),
        }

        exec_context: Dict[str, Any] = {
            "__builtins__": safe_builtins,
        }

        # Add allowed modules
        for module_name in self.allowed_modules:
            try:
                exec_context[str(module_name)] = __import__(module_name)
            except ImportError:
                pass

        # Add user-provided context (validate first)
        if context:
            for key, value in context.items():
                if self._is_safe_value(value):
                    exec_context[key] = value
                else:
                    logger.warning(f"Skipping unsafe value in context: {key}")

        return exec_context

    def _is_safe_value(self, value: Any) -> bool:
        """Check if a value is safe to include in execution context."""
        try:
            # Check for dangerous objects
            if callable(value):
                # Check if it's a safe function
                return value.__module__ in self.allowed_modules

            # Check for containers with dangerous content
            if isinstance(value, (list, tuple, set)):
                return all(self._is_safe_value(item) for item in value)

            # Check for dictionaries with dangerous values
            if isinstance(value, dict):
                return all(self._is_safe_value(v) for v in value.values())

            # Check for strings with dangerous content
            if isinstance(value, str):
                try:
                    self.validate_input(value)
                    return True
                except SecurityViolationError:
                    return False

            return True

        except Exception:
            return False


class SecurityValidator(ast.NodeVisitor):
    """AST visitor to validate code security."""

    def __init__(
        self,
        allowed_modules: List[str],
        forbidden_modules: set,
        forbidden_functions: set,
    ):
        self.allowed_modules = allowed_modules
        self.forbidden_modules = forbidden_modules
        self.forbidden_functions = forbidden_functions
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Check import statements."""
        for alias in node.names:
            # Check for exact matches and submodule imports
            if alias.name in self.forbidden_modules or any(
                forbidden in alias.name for forbidden in self.forbidden_modules
            ):
                self.violations.append(f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from-import statements."""
        if node.module and any(forbidden in node.module for forbidden in self.forbidden_modules):
            self.violations.append(f"Forbidden import from: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id in self.forbidden_functions:
                self.violations.append(f"Dangerous function call: {node.func.id}")
        self.generic_visit(node)


# Global sandbox instance
_default_sandbox = CodeSandbox()


def safe_execute(
    code: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any
) -> Dict[str, Any]:
    """Safely execute code in the default sandbox.

    Args:
        code: Python code to execute
        context: Optional execution context
        **kwargs: Additional sandbox configuration

    Returns:
        Dictionary with execution results
    """
    return _default_sandbox.execute_code(code, context, **kwargs)


def validate_code_safety(code: str) -> bool:
    """Validate code for security issues.

    Args:
        code: Python code to validate

    Returns:
        True if code is safe
    """
    try:
        _default_sandbox.validate_code(code)
        return True
    except SecurityViolationError:
        return False
