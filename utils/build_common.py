"""
Build utilities for the APGI system.

This module provides common build and development utilities.
"""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_command(
    cmd: List[str], cwd: Optional[str] = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and return the result.

    Parameters
    ----------
    cmd : List[str]
        Command to run
    cwd : str, optional
        Working directory
    check : bool, optional
        Whether to check return code, by default True

    Returns
    -------
    subprocess.CompletedProcess
        Command result
    """
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def get_project_root() -> Path:
    """Get the project root directory.

    Returns
    -------
    Path
        Project root path
    """
    return Path(__file__).parent.parent


def get_build_config() -> Dict[str, Any]:
    """Get build configuration.

    Returns
    -------
    Dict[str, Any]
        Build configuration
    """
    return {
        "project_name": "apgi-system",
        "version": "0.1.0",
        "python_requires": ">=3.8",
        "dependencies": [
            "numpy",
            "scipy",
            "matplotlib",
            "pandas",
            "sklearn",
            "networkx",
            "torch",
            "jax",
            "fastapi",
            "uvicorn",
            "pydantic",
            "sqlalchemy",
            "redis",
            "psycopg2-binary",
        ],
    }


def check_build_environment() -> Dict[str, bool]:
    """Check if build environment is ready.

    Returns
    -------
    Dict[str, bool]
        Environment check results
    """
    checks = {}

    # Check Python version
    import sys

    checks["python_version"] = sys.version_info >= (3, 8)

    # Check essential tools
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        checks["git"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        checks["git"] = False

    # Check pip
    try:
        subprocess.run(["pip", "--version"], capture_output=True, check=True)
        checks["pip"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        checks["pip"] = False

    return checks


_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist", ".tox"}
_CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".ini"}
_DATA_SUFFIXES = {".csv", ".txt", ".md"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".icns"}
_ICON_SUFFIXES = {".ico", ".icns", ".png", ".jpg"}


def _normalize_package(name: str) -> str:
    """Lower-case a distribution/import name and unify ``-``/``.`` with ``_``."""
    return re.sub(r"[-.]+", "_", name.strip().lower())


def _requirement_name(requirement: str) -> str:
    """Distribution name of a requirement line (``numpy[extra]>=2 ; python_version...``)."""
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return _normalize_package(match.group(1)) if match else ""


def _find_project_files(start: Path) -> Path:
    """Directory holding requirements.txt/pyproject.toml, searching up from ``start``."""
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return get_project_root()


def _imported_top_level_modules(source_files: List[Path]) -> set:
    """Top-level module names imported by ``source_files``; files that fail to parse count as empty."""
    modules: set = set()
    for source_file in source_files:
        try:
            tree = ast.parse(source_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def analyze_dependencies(project_path: Optional[str] = None) -> Dict[str, Any]:
    """Analyze the third-party dependencies imported by a Python file or a project.

    Parameters
    ----------
    project_path : str, optional
        A ``.py`` file or a directory (all ``*.py`` files below it are scanned), by default
        the project root.

    Returns
    -------
    Dict[str, Any]
        ``requirements_txt``: imported packages that are declared in the nearest
        ``requirements.txt``; ``pyproject_toml``: those declared in ``pyproject.toml``
        dependencies; ``total_dependencies``: number of distinct packages in either set.
        Standard-library and development-tool imports are ignored. A missing path or a file
        that does not parse yields empty sets.
    """
    target = get_project_root() if project_path is None else Path(project_path)
    empty: Dict[str, Any] = {
        "requirements_txt": set(),
        "pyproject_toml": set(),
        "total_dependencies": 0,
    }
    if not target.exists():
        return empty

    if target.is_dir():
        sources = [
            f
            for f in sorted(target.rglob("*.py"))
            if not _SKIP_DIRS.intersection(f.relative_to(target).parts)
        ]
        root = _find_project_files(target)
    else:
        sources = [target]
        root = _find_project_files(target.resolve().parent)

    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    imported = {
        name
        for name in _imported_top_level_modules(sources)
        if name not in stdlib and not should_exclude_module(name)
    }

    requirements: set = set()
    requirements_file = root / "requirements.txt"
    if requirements_file.exists():
        for line in requirements_file.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line and not line.startswith(("-", "git+", "http")):
                requirements.add(_requirement_name(line))

    pyproject_requirements: set = set()
    pyproject_file = root / "pyproject.toml"
    if pyproject_file.exists():
        try:
            with open(pyproject_file, "rb") as f:
                config = tomllib.load(f)
            pyproject_requirements = {
                _requirement_name(dep) for dep in config.get("project", {}).get("dependencies", [])
            }
        except (OSError, tomllib.TOMLDecodeError):
            pass

    # Import names differ from distribution names for a few packages
    aliases = {"sklearn": "scikit_learn", "yaml": "pyyaml", "cv2": "opencv_python", "PIL": "pillow"}

    def declared(name: str, names: set) -> bool:
        return _normalize_package(name) in names or aliases.get(name, "") in names

    from_requirements = {name for name in imported if declared(name, requirements)}
    from_pyproject = {name for name in imported if declared(name, pyproject_requirements)}
    return {
        "requirements_txt": from_requirements,
        "pyproject_toml": from_pyproject,
        "total_dependencies": len(from_requirements | from_pyproject),
    }


def collect_resources(
    project_path: Optional[str] = None, resource_dirs: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """Collect project resource files, grouped by kind.

    Parameters
    ----------
    project_path : str, optional
        Project directory, by default the project root
    resource_dirs : List[str], optional
        Restrict the search to these sub-directories, by default the whole project

    Returns
    -------
    Dict[str, List[str]]
        ``config_files`` (yaml/yml/json/toml/ini), ``data_files`` (csv/txt/md),
        ``resource_files`` (images plus everything under ``resources/``) and ``icon_files``
        (ico/icns/png/jpg), as paths relative to ``project_path``. Directories are never
        listed; a missing project directory yields empty lists.
    """
    project_root = get_project_root() if project_path is None else Path(project_path)
    resources: Dict[str, List[str]] = {
        "config_files": [],
        "data_files": [],
        "resource_files": [],
        "icon_files": [],
    }
    if not project_root.is_dir():
        return resources

    if resource_dirs is None:
        search_roots = [project_root]
    else:
        search_roots = [project_root / d for d in resource_dirs if (project_root / d).is_dir()]

    resources_dir = project_root / "resources"
    seen: set = set()
    for search_root in search_roots:
        for path in sorted(search_root.rglob("*")):
            if not path.is_file() or path in seen:
                continue
            relative = path.relative_to(project_root)
            if _SKIP_DIRS.intersection(relative.parts):
                continue
            seen.add(path)
            suffix = path.suffix.lower()
            entry = str(relative)
            if suffix in _CONFIG_SUFFIXES:
                resources["config_files"].append(entry)
            if suffix in _DATA_SUFFIXES:
                resources["data_files"].append(entry)
            if suffix in _IMAGE_SUFFIXES or resources_dir in path.parents:
                resources["resource_files"].append(entry)
            if suffix in _ICON_SUFFIXES:
                resources["icon_files"].append(entry)

    return resources


def get_version(project_path: Optional[str] = None) -> str:
    """Get project version.

    Parameters
    ----------
    project_path : str, optional
        Path to project, by default None (current directory)

    Returns
    -------
    str
        Project version
    """
    if project_path is None:
        project_root = get_project_root()
    else:
        project_root = Path(project_path)

    # Try VERSION file
    version_file = project_root / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()

    # Try pyproject.toml
    pyproject_file = project_root / "pyproject.toml"
    if pyproject_file.exists():
        try:
            with open(pyproject_file, "rb") as f:
                version: str = tomllib.load(f).get("project", {}).get("version", "0.1.0")
            return version
        except (OSError, tomllib.TOMLDecodeError):
            pass

    return "0.1.0"


# Modules PyInstaller cannot discover statically, by top-level package
HIDDEN_IMPORTS: Dict[str, List[str]] = {
    "numpy": [
        "numpy.random._pickle",
        "numpy.random._common",
        "numpy.random._bounded_integers",
        "numpy.random.bit_generator",
    ],
    "scipy": [
        "scipy._lib.messagestream",
        "scipy.special._ufuncs_cxx",
        "scipy.linalg.cython_blas",
        "scipy.linalg.cython_lapack",
        "scipy.sparse.csgraph._validation",
    ],
    "matplotlib": [
        "matplotlib.backends.backend_tkagg",
        "matplotlib.backends.backend_agg",
        "matplotlib.backends.backend_svg",
        "matplotlib.backends.backend_pdf",
    ],
    "tkinter": [
        "tkinter.ttk",
        "tkinter.filedialog",
        "tkinter.messagebox",
        "tkinter.simpledialog",
        "tkinter.scrolledtext",
        "tkinter.colorchooser",
    ],
    "sklearn": ["sklearn.utils._cython_blas", "sklearn.utils._openmp_helpers"],
    "torch": ["torch._C", "torch._C._dynamo"],
    "jax": ["jaxlib.xla_extension"],
}


def detect_hidden_imports(project_path: Optional[str] = None) -> List[str]:
    """Hidden imports a packaged build needs.

    Parameters
    ----------
    project_path : str, optional
        Either a package name (``"scipy"``, ``"tkinter"``, ...) to look up, or a project
        directory whose ``*.py`` files are scanned for the packages they import; by default
        the project root.

    Returns
    -------
    List[str]
        Hidden imports (empty for an unknown package name)
    """
    if project_path is not None and not Path(project_path).exists():
        return list(HIDDEN_IMPORTS.get(project_path, []))

    root = get_project_root() if project_path is None else Path(project_path)
    sources = [
        f
        for f in sorted(root.rglob("*.py"))
        if not _SKIP_DIRS.intersection(f.relative_to(root).parts)
    ]
    hidden: List[str] = []
    for package in sorted(_imported_top_level_modules(sources)):
        for module in HIDDEN_IMPORTS.get(package, []):
            if module not in hidden:
                hidden.append(module)
    return hidden


# Alias for backward compatibility
get_hidden_imports = detect_hidden_imports


def should_exclude_module(module_name: str) -> bool:
    """Whether a module (or any submodule of an excluded package) is left out of builds.

    Parameters
    ----------
    module_name : str
        Dotted module name, e.g. ``"pytest.fixtures"``

    Returns
    -------
    bool
        True if the module's top-level package is in :func:`get_excluded_modules`
    """
    return module_name.split(".")[0].lower() in set(get_excluded_modules())


def get_excluded_modules() -> List[str]:
    """Get list of commonly excluded modules.

    Returns
    -------
    List[str]
        List of excluded modules
    """
    return [
        "test",
        "tests",
        "pytest",
        "_pytest",
        "hypothesis",
        "setuptools",
        "pip",
        "wheel",
        "debug",
        "pdb",
        "ipdb",
        "jupyter",
        "ipython",
        "sphinx",
        "docs",
        "examples",
        "samples",
        "mypy",
        "flake8",
        "black",
        "coverage",
        "coveralls",
        "tkinter.test",
        "unittest",
        "doctest",
        "argparse",
        "email",
        "html",
        "http",
        "urllib",
        "xml",
        "xmlrpc",
    ]
