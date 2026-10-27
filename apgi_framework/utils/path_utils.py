"""Path handling utilities for APGI Framework
Standardizes path operations using pathlib for cross-platform compatibility.
"""

import os
import platform
import shutil
import tempfile
import threading
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from apgi_framework.logging.standardized_logging import get_logger

logger = get_logger(__name__)


class PathManager:
    """Centralized path management using pathlib for cross-platform compatibility.

    This class provides a unified interface for all path operations in the APGI Framework,
    ensuring consistent behavior across Windows, macOS, and Linux.
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """Initialize path manager.

        Args:
            base_path: Base path for relative operations. Defaults to current working directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.platform = platform.system().lower()

        # Platform-specific settings
        self.case_sensitive = self.platform not in ["windows"]
        self.path_separator = os.sep

        # Common directories
        self._setup_common_directories()

    def _setup_common_directories(self) -> None:
        """Setup common framework directories."""
        self.dirs: Dict[str, Path] = {
            "root": self.base_path,
            "data": self.base_path / "data",
            "results": self.base_path / "results",
            "logs": self.base_path / "logs",
            "config": self.base_path / "config",
            "temp": self.base_path / "temp",
            "exports": self.base_path / "exports",
            "figures": self.base_path / "figures",
            "reports": self.base_path / "reports",
            "cache": self.base_path / "cache",
            "session_state": self.base_path / "session_state",
            "examples": self.base_path / "examples",
            "tests": self.base_path / "tests",
            "docs": self.base_path / "docs",
        }

    def resolve_path(self, path: Union[str, Path], relative_to: Optional[str] = None) -> Path:
        """Resolve path to absolute Path object.

        Args:
            path: Input path (string or Path)
            relative_to: Base directory for relative paths. If None, uses base_path.

        Returns:
            Resolved absolute Path object
        """
        if isinstance(path, str):
            # Handle environment variables
            path = os.path.expandvars(path)
            # Handle user home directory
            path = os.path.expanduser(path)

        input_path = Path(path)

        # Determine base for relative paths
        if relative_to and relative_to in self.dirs:
            base = self.dirs[relative_to]
        elif relative_to:
            base = Path(relative_to)
        else:
            base = self.base_path

        # Resolve to absolute path
        if input_path.is_absolute():
            resolved = input_path.resolve()
        else:
            resolved = (base / input_path).resolve()

        return resolved

    def ensure_dir(self, path: Union[str, Path], relative_to: Optional[str] = None) -> Path:
        """Ensure directory exists, create if necessary.

        Args:
            path: Directory path
            relative_to: Base directory for relative paths

        Returns:
            Path object for the directory
        """
        resolved_path = self.resolve_path(path, relative_to)
        resolved_path.mkdir(parents=True, exist_ok=True)
        return resolved_path

    def get_dir(self, dir_name: str) -> Path:
        """Get predefined directory path.

        Args:
            dir_name: Name of predefined directory

        Returns:
            Path object for the directory

        Raises:
            KeyError: If directory name is not predefined
        """
        if dir_name not in self.dirs:
            raise KeyError(f"Unknown directory: {dir_name}. Available: {list(self.dirs.keys())}")

        return self.dirs[dir_name]

    def ensure_common_dirs(self) -> Dict[str, Path]:
        """Ensure all common directories exist.

        Returns:
            Dictionary of directory paths
        """
        created_dirs = {}
        for name, path in self.dirs.items():
            if name != "root":  # Don't create root if it doesn't exist
                created_dirs[name] = self.ensure_dir(path)
            else:
                created_dirs[name] = path

        return created_dirs

    def join_paths(self, *paths: Union[str, Path]) -> Path:
        """Join multiple path components.

        Args:
            *paths: Path components to join

        Returns:
            Joined Path object
        """
        result = Path(paths[0])
        for path in paths[1:]:
            result = result / path
        return result

    def get_relative_path(
        self, path: Union[str, Path], base: Optional[Union[str, Path]] = None
    ) -> Path:
        """Get relative path from base.

        Args:
            path: Target path
            base: Base path. If None, uses base_path.

        Returns:
            Relative path
        """
        target_path = self.resolve_path(path)
        base_path = self.resolve_path(base) if base else self.base_path

        try:
            return target_path.relative_to(base_path)
        except ValueError:
            # Paths are on different drives (Windows) or otherwise not relative
            warnings.warn(f"Cannot make path relative: {target_path} to {base_path}")
            return target_path

    def normalize_path(self, path: Union[str, Path]) -> Path:
        """Normalize path for cross-platform compatibility.

        Args:
            path: Input path

        Returns:
            Normalized Path object
        """
        if isinstance(path, str):
            # Convert forward slashes to OS-specific separators
            path = path.replace("/", os.sep).replace("\\", os.sep)

        path_obj = Path(path)
        return path_obj.resolve()

    def is_subdirectory(self, child: Union[str, Path], parent: Union[str, Path]) -> bool:
        """Check if child path is within parent directory.

        Args:
            child: Child path
            parent: Parent path

        Returns:
            True if child is within parent
        """
        try:
            child_path = self.resolve_path(child)
            parent_path = self.resolve_path(parent)
            child_path.relative_to(parent_path)
            return True
        except ValueError:
            return False

    def find_files(
        self,
        pattern: str,
        root_dir: Optional[Union[str, Path]] = None,
        recursive: bool = True,
    ) -> List[Path]:
        """Find files matching pattern.

        Args:
            pattern: Glob pattern to match
            root_dir: Directory to search in. If None, uses base_path.
            recursive: Whether to search recursively

        Returns:
            List of matching file paths
        """
        search_dir = self.resolve_path(root_dir) if root_dir else self.base_path

        if recursive:
            matches = list(search_dir.rglob(pattern))
        else:
            matches = list(search_dir.glob(pattern))

        return sorted(matches)

    def get_file_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive file information.

        Args:
            path: File path

        Returns:
            Dictionary with file information
        """
        resolved_path = self.resolve_path(path)

        if not resolved_path.exists():
            return {"exists": False}

        stat = resolved_path.stat()

        info = {
            "exists": True,
            "path": resolved_path,
            "name": resolved_path.name,
            "stem": resolved_path.stem,
            "suffix": resolved_path.suffix,
            "is_file": resolved_path.is_file(),
            "is_dir": resolved_path.is_dir(),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
        }

        if resolved_path.is_file():
            info["extension"] = resolved_path.suffix.lower()
            info["size_mb"] = stat.st_size / (1024 * 1024)

        return info

    def safe_filename(self, filename: str, max_length: int = 255) -> str:
        """Create safe filename for current platform.

        Args:
            filename: Original filename
            max_length: Maximum filename length

        Returns:
            Safe filename
        """
        # Platform-specific invalid characters
        if self.platform == "windows":
            invalid_chars = '<>:"/\\|?*'
            reserved_names = {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                "COM1",
                "COM2",
                "COM3",
                "COM4",
                "COM5",
                "COM6",
                "COM7",
                "COM8",
                "COM9",
                "LPT1",
                "LPT2",
                "LPT3",
                "LPT4",
                "LPT5",
                "LPT6",
                "LPT7",
                "LPT8",
                "LPT9",
            }
        else:
            invalid_chars = "\0/"
            reserved_names = set()

        # Remove invalid characters
        safe_name = "".join(c for c in filename if c not in invalid_chars)

        # Remove trailing spaces and dots (Windows)
        safe_name = safe_name.rstrip(" .")

        # Check for reserved names
        name_without_ext = Path(safe_name).stem
        if name_without_ext.upper() in reserved_names:
            safe_name = f"_{safe_name}"

        # Ensure reasonable length
        if len(safe_name) > max_length:
            name_part = Path(safe_name).stem
            ext_part = Path(safe_name).suffix
            max_name_length = max_length - len(ext_part)
            safe_name = name_part[:max_name_length] + ext_part

        # Ensure not empty
        if not safe_name or safe_name in [".", ".."]:
            safe_name = "unnamed_file"

        return safe_name

    def get_temp_dir(self) -> Path:
        """Get framework-specific temporary directory.

        Returns:
            Path to temporary directory
        """
        temp_dir = self.ensure_dir("temp")
        return temp_dir

    @contextmanager
    def temp_file(self, suffix: str = "", prefix: str = "apgi_") -> Any:
        """Context manager for temporary file.

        Args:
            suffix: File suffix
            prefix: File prefix

        Yields:
            Path to temporary file
        """
        temp_dir = self.get_temp_dir()

        # Create temporary file
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=temp_dir)
        os.close(fd)  # Close file descriptor

        temp_file_path = Path(temp_path)

        try:
            yield temp_file_path
        finally:
            # Clean up
            if temp_file_path.exists():
                temp_file_path.unlink()

    @contextmanager
    def temp_dir_context(self, prefix: str = "apgi_temp_") -> Any:
        """Context manager for temporary directory.

        Args:
            prefix: Directory prefix

        Yields:
            Path to temporary directory
        """
        temp_base = self.get_temp_dir()
        temp_dir_path = Path(tempfile.mkdtemp(prefix=prefix, dir=temp_base))

        try:
            yield temp_dir_path
        finally:
            # Clean up
            if temp_dir_path.exists():
                shutil.rmtree(temp_dir_path)

    def backup_file(
        self, file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """Create backup of file.

        Args:
            file_path: File to backup
            backup_dir: Directory for backup. If None, uses 'backups' subdirectory.

        Returns:
            Path to backup file
        """
        source_path = self.resolve_path(file_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if backup_dir:
            backup_directory = self.ensure_dir(backup_dir)
        else:
            backup_directory = self.ensure_dir("backups")

        # Create backup filename with timestamp
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source_path.stem}_{timestamp}{source_path.suffix}"
        backup_path = backup_directory / backup_name

        # Copy file
        shutil.copy2(source_path, backup_path)

        return backup_path

    def copy_file(
        self, src: Union[str, Path], dst: Union[str, Path], create_dirs: bool = True
    ) -> Path:
        """Copy file with proper path handling.

        Args:
            src: Source file
            dst: Destination path
            create_dirs: Whether to create destination directories

        Returns:
            Path to copied file
        """
        source_path = self.resolve_path(src)
        dest_path = self.resolve_path(dst)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if create_dirs:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        return Path(shutil.copy2(source_path, dest_path))

    def move_file(
        self, src: Union[str, Path], dst: Union[str, Path], create_dirs: bool = True
    ) -> Path:
        """Move file with proper path handling.

        Args:
            src: Source file
            dst: Destination path
            create_dirs: Whether to create destination directories

        Returns:
            Path to moved file
        """
        source_path = self.resolve_path(src)
        dest_path = self.resolve_path(dst)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if create_dirs:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        return Path(shutil.move(str(source_path), str(dest_path)))

    def get_platform_info(self) -> Dict[str, Any]:
        """Get platform-specific path information.

        Returns:
            Dictionary with platform information
        """
        return {
            "platform": self.platform,
            "case_sensitive": self.case_sensitive,
            "path_separator": self.path_separator,
            "base_path": str(self.base_path),
            "home_directory": str(Path.home()),
            "temp_directory": str(Path(tempfile.gettempdir())),
            "current_directory": str(Path.cwd()),
        }


# Global path manager instance
_default_path_manager = None
_path_lock = threading.Lock()


def get_path_manager(base_path: Optional[Union[str, Path]] = None) -> PathManager:
    """Get global path manager instance.

    Args:
        base_path: Base path for path manager

    Returns:
        PathManager instance
    """
    global _default_path_manager
    with _path_lock:
        if _default_path_manager is None:
            _default_path_manager = PathManager(base_path)
    return _default_path_manager


def resolve_path(path: Union[str, Path], relative_to: Optional[str] = None) -> Path:
    """Convenience function to resolve path."""
    return get_path_manager().resolve_path(path, relative_to)


def ensure_dir(path: Union[str, Path], relative_to: Optional[str] = None) -> Path:
    """Convenience function to ensure directory exists."""
    return get_path_manager().ensure_dir(path, relative_to)


def get_framework_dir(dir_name: str) -> Path:
    """Convenience function to get framework directory."""
    return get_path_manager().get_dir(dir_name)


def safe_filename(filename: str, max_length: int = 255) -> str:
    """Convenience function to create safe filename."""
    return get_path_manager().safe_filename(filename, max_length)


# Migration utilities for existing code
def migrate_from_os_path() -> Dict[str, str]:
    """Provide guidance for migrating from os.path to pathlib.

    Returns:
        Dictionary with migration examples
    """
    return {
        "os.path.join": 'PathManager.join_paths() or Path() / "subdir"',
        "os.path.exists": "Path.exists()",
        "os.path.isdir": "Path.is_dir()",
        "os.path.isfile": "Path.is_file()",
        "os.path.abspath": "Path.resolve()",
        "os.path.basename": "Path.name",
        "os.path.dirname": "Path.parent",
        "os.path.splitext": "Path.stem and Path.suffix",
        "os.makedirs": "Path.mkdir(parents=True, exist_ok=True)",
        "os.path.expanduser": "Path.expanduser()",
        "os.path.expandvars": "Path.expandvars() (via PathManager)",
    }


if __name__ == "__main__":
    # Example usage
    pm = PathManager()

    logger.info(f"Platform info: {pm.get_platform_info()}")
    logger.info(f"Common directories: {pm.ensure_common_dirs()}")

    # Test file operations
    with pm.temp_file(suffix=".txt") as temp_file:
        temp_file.write_text("Hello, APGI Framework!")
        logger.info(f"Created temp file: {temp_file}")
        logger.info(f"File info: {pm.get_file_info(temp_file)}")

    # Test safe filename
    unsafe_name = '<>:"/\\|?*.txt'
    safe_name = pm.safe_filename(unsafe_name)
    logger.info(f"Safe filename: {unsafe_name} -> {safe_name}")

    logger.info("Path utilities test completed.")
