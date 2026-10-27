"""File operation utilities for the APGI Framework test enhancement system.

This module provides safe file operations with comprehensive error handling,
path manipulation and validation utilities, and temporary file management.
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, TextIO, Union, cast

from .path_utils import get_path_manager


class FileOperationError(Exception):
    """Exception raised for file operation errors."""


class FileUtils:
    """Comprehensive file operation utilities with error handling and validation.

    Provides safe file operations, path manipulation, temporary file management,
    and file content operations for the test enhancement system.
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """Initialize FileUtils with optional base path.

        Args:
            base_path: Base directory for relative operations
        """
        self.path_manager = get_path_manager(base_path)
        self.logger = logging.getLogger(__name__)

    def safe_read_text(
        self,
        file_path: Union[str, Path],
        encoding: str = "utf-8",
        fallback_encodings: Optional[List[str]] = None,
    ) -> str:
        """Safely read text file with encoding fallback.

        Args:
            file_path: Path to the file
            encoding: Primary encoding to try
            fallback_encodings: List of fallback encodings

        Returns:
            File content as string

        Raises:
            FileOperationError: If file cannot be read
        """
        if fallback_encodings is None:
            fallback_encodings = ["utf-8", "latin-1", "cp1252"]

        resolved_path = self.path_manager.resolve_path(file_path)

        if not resolved_path.exists():
            raise FileOperationError(f"File not found: {resolved_path}")

        if not resolved_path.is_file():
            raise FileOperationError(f"Path is not a file: {resolved_path}")

        # Try primary encoding first
        encodings_to_try = [encoding] + [enc for enc in fallback_encodings if enc != encoding]

        for enc in encodings_to_try:
            try:
                with open(resolved_path, "r", encoding=enc) as f:
                    content = f.read()
                    self.logger.debug(f"Successfully read {resolved_path} with encoding {enc}")
                    return content
            except UnicodeDecodeError:
                self.logger.debug(f"Failed to read {resolved_path} with encoding {enc}")
                continue

        raise FileOperationError(
            f"Could not read file {resolved_path} with any encoding: {encodings_to_try}"
        )

    def safe_write_text(
        self,
        file_path: Union[str, Path],
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True,
        backup: bool = False,
    ) -> Path:
        """Safely write text to file with optional backup.

        Args:
            file_path: Path to write to
            content: Text content to write
            encoding: Text encoding
            create_dirs: Whether to create parent directories
            backup: Whether to backup existing file

        Returns:
            Path to written file

        Raises:
            FileOperationError: If file cannot be written
        """
        resolved_path = self.path_manager.resolve_path(file_path)

        try:
            # Create parent directories if needed
            if create_dirs:
                resolved_path.parent.mkdir(parents=True, exist_ok=True)

            # Backup existing file if requested
            if backup and resolved_path.exists():
                backup_path = self.create_backup(resolved_path)
                self.logger.info(f"Created backup: {backup_path}")

            # Write content
            with open(resolved_path, "w", encoding=encoding) as f:
                f.write(content)

            self.logger.debug(f"Successfully wrote {len(content)} characters to {resolved_path}")
            return resolved_path

        except Exception as e:
            raise FileOperationError(f"Failed to write file {resolved_path}: {e}")

    def safe_read_binary(self, file_path: Union[str, Path]) -> bytes:
        """Safely read binary file.

        Args:
            file_path: Path to the file

        Returns:
            File content as bytes

        Raises:
            FileOperationError: If file cannot be read
        """
        resolved_path = self.path_manager.resolve_path(file_path)

        if not resolved_path.exists():
            raise FileOperationError(f"File not found: {resolved_path}")

        if not resolved_path.is_file():
            raise FileOperationError(f"Path is not a file: {resolved_path}")

        try:
            with open(resolved_path, "rb") as f:
                content = f.read()
                self.logger.debug(f"Successfully read {len(content)} bytes from {resolved_path}")
                return content
        except Exception as e:
            raise FileOperationError(f"Failed to read binary file {resolved_path}: {e}")

    def safe_write_binary(
        self,
        file_path: Union[str, Path],
        content: bytes,
        create_dirs: bool = True,
        backup: bool = False,
    ) -> Path:
        """Safely write binary content to file.

        Args:
            file_path: Path to write to
            content: Binary content to write
            create_dirs: Whether to create parent directories
            backup: Whether to backup existing file

        Returns:
            Path to written file

        Raises:
            FileOperationError: If file cannot be written
        """
        resolved_path = self.path_manager.resolve_path(file_path)

        try:
            # Create parent directories if needed
            if create_dirs:
                resolved_path.parent.mkdir(parents=True, exist_ok=True)

            # Backup existing file if requested
            if backup and resolved_path.exists():
                backup_path = self.create_backup(resolved_path)
                self.logger.info(f"Created backup: {backup_path}")

            # Write content
            with open(resolved_path, "wb") as f:
                f.write(content)

            self.logger.debug(f"Successfully wrote {len(content)} bytes to {resolved_path}")
            return resolved_path

        except Exception as e:
            raise FileOperationError(f"Failed to write binary file {resolved_path}: {e}")

    def safe_copy(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        create_dirs: bool = True,
        preserve_metadata: bool = True,
    ) -> Path:
        """Safely copy file with error handling.

        Args:
            src: Source file path
            dst: Destination file path
            create_dirs: Whether to create destination directories
            preserve_metadata: Whether to preserve file metadata

        Returns:
            Path to copied file

        Raises:
            FileOperationError: If copy operation fails
        """
        src_path = self.path_manager.resolve_path(src)
        dst_path = self.path_manager.resolve_path(dst)

        if not src_path.exists():
            raise FileOperationError(f"Source file not found: {src_path}")

        if not src_path.is_file():
            raise FileOperationError(f"Source is not a file: {src_path}")

        try:
            # Create destination directories if needed
            if create_dirs:
                dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            if preserve_metadata:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)

            self.logger.debug(f"Successfully copied {src_path} to {dst_path}")
            return dst_path

        except Exception as e:
            raise FileOperationError(f"Failed to copy {src_path} to {dst_path}: {e}")

    def safe_move(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        create_dirs: bool = True,
    ) -> Path:
        """Safely move file with error handling.

        Args:
            src: Source file path
            dst: Destination file path
            create_dirs: Whether to create destination directories

        Returns:
            Path to moved file

        Raises:
            FileOperationError: If move operation fails
        """
        src_path = self.path_manager.resolve_path(src)
        dst_path = self.path_manager.resolve_path(dst)

        if not src_path.exists():
            raise FileOperationError(f"Source file not found: {src_path}")

        try:
            # Create destination directories if needed
            if create_dirs:
                dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Move file
            moved_path = shutil.move(str(src_path), str(dst_path))
            result_path = Path(moved_path)

            self.logger.debug(f"Successfully moved {src_path} to {result_path}")
            return result_path

        except Exception as e:
            raise FileOperationError(f"Failed to move {src_path} to {dst_path}: {e}")

    def safe_delete(self, file_path: Union[str, Path], backup: bool = False) -> bool:
        """Safely delete file with optional backup.

        Args:
            file_path: Path to file to delete
            backup: Whether to create backup before deletion

        Returns:
            True if file was deleted, False if it didn't exist

        Raises:
            FileOperationError: If deletion fails
        """
        resolved_path = self.path_manager.resolve_path(file_path)

        if not resolved_path.exists():
            return False

        if not resolved_path.is_file():
            raise FileOperationError(f"Path is not a file: {resolved_path}")

        try:
            # Create backup if requested
            if backup:
                backup_path = self.create_backup(resolved_path)
                self.logger.info(f"Created backup before deletion: {backup_path}")

            # Delete file
            resolved_path.unlink()
            self.logger.debug(f"Successfully deleted {resolved_path}")
            return True

        except Exception as e:
            raise FileOperationError(f"Failed to delete {resolved_path}: {e}")

    def create_backup(
        self,
        file_path: Union[str, Path],
        backup_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Create backup of file with timestamp.

        Args:
            file_path: Path to file to backup
            backup_dir: Directory for backup (optional)

        Returns:
            Path to backup file

        Raises:
            FileOperationError: If backup creation fails
        """
        src_path = self.path_manager.resolve_path(file_path)

        if not src_path.exists():
            raise FileOperationError(f"Source file not found: {src_path}")

        # Determine backup directory
        if backup_dir:
            backup_directory = self.path_manager.resolve_path(backup_dir)
            backup_directory.mkdir(parents=True, exist_ok=True)
        else:
            backup_directory = src_path.parent / "backups"
            backup_directory.mkdir(exist_ok=True)

        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{src_path.stem}_{timestamp}{src_path.suffix}"
        backup_path = backup_directory / backup_name

        try:
            shutil.copy2(src_path, backup_path)
            self.logger.debug(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            raise FileOperationError(f"Failed to create backup of {src_path}: {e}")

    def validate_path(
        self,
        path: Union[str, Path],
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        readable: bool = False,
        writable: bool = False,
    ) -> bool:
        """Validate path with various criteria.

        Args:
            path: Path to validate
            must_exist: Path must exist
            must_be_file: Path must be a file
            must_be_dir: Path must be a directory
            readable: Path must be readable
            writable: Path must be writable

        Returns:
            True if all criteria are met

        Raises:
            FileOperationError: If validation fails
        """
        resolved_path = self.path_manager.resolve_path(path)

        if must_exist and not resolved_path.exists():
            raise FileOperationError(f"Path does not exist: {resolved_path}")

        if resolved_path.exists():
            if must_be_file and not resolved_path.is_file():
                raise FileOperationError(f"Path is not a file: {resolved_path}")

            if must_be_dir and not resolved_path.is_dir():
                raise FileOperationError(f"Path is not a directory: {resolved_path}")

            if readable and not os.access(resolved_path, os.R_OK):
                raise FileOperationError(f"Path is not readable: {resolved_path}")

            if writable and not os.access(resolved_path, os.W_OK):
                raise FileOperationError(f"Path is not writable: {resolved_path}")

        return True

    def get_file_hash(self, file_path: Union[str, Path], algorithm: str = "sha256") -> str:
        """Calculate file hash.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256, etc.)

        Returns:
            Hexadecimal hash string

        Raises:
            FileOperationError: If hash calculation fails
        """
        resolved_path = self.path_manager.resolve_path(file_path)

        if not resolved_path.exists():
            raise FileOperationError(f"File not found: {resolved_path}")

        if not resolved_path.is_file():
            raise FileOperationError(f"Path is not a file: {resolved_path}")

        try:
            hash_obj = hashlib.new(algorithm)
            with open(resolved_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            raise FileOperationError(f"Failed to calculate hash for {resolved_path}: {e}")

    def find_files(
        self,
        pattern: str,
        root_dir: Optional[Union[str, Path]] = None,
        recursive: bool = True,
        include_dirs: bool = False,
    ) -> List[Path]:
        """Find files matching pattern.

        Args:
            pattern: Glob pattern to match
            root_dir: Directory to search in
            recursive: Whether to search recursively
            include_dirs: Whether to include directories in results

        Returns:
            List of matching file paths
        """
        search_dir = (
            self.path_manager.resolve_path(root_dir) if root_dir else self.path_manager.base_path
        )

        if recursive:
            matches = list(search_dir.rglob(pattern))
        else:
            matches = list(search_dir.glob(pattern))

        # Filter by type if requested
        if not include_dirs:
            matches = [p for p in matches if p.is_file()]

        return sorted(matches)

    def read_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Read JSON file safely.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON data

        Raises:
            FileOperationError: If JSON cannot be read or parsed
        """
        try:
            content = self.safe_read_text(file_path)
            return cast(Dict[str, Any], json.loads(content))
        except json.JSONDecodeError as e:
            raise FileOperationError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            raise FileOperationError(f"Failed to read JSON from {file_path}: {e}")

    def write_json(
        self,
        file_path: Union[str, Path],
        data: Dict[str, Any],
        indent: int = 2,
        create_dirs: bool = True,
        backup: bool = False,
    ) -> Path:
        """Write data to JSON file safely.

        Args:
            file_path: Path to write JSON to
            data: Data to serialize
            indent: JSON indentation
            create_dirs: Whether to create parent directories
            backup: Whether to backup existing file

        Returns:
            Path to written file

        Raises:
            FileOperationError: If JSON cannot be written
        """
        try:
            json_content = json.dumps(data, indent=indent, ensure_ascii=False)
            return self.safe_write_text(
                file_path, json_content, create_dirs=create_dirs, backup=backup
            )
        except Exception as e:
            raise FileOperationError(f"Failed to write JSON to {file_path}: {e}")

    @contextmanager
    def temporary_file(
        self,
        suffix: str = "",
        prefix: str = "apgi_test_",
        text_mode: bool = True,
    ) -> Iterator[Union[TextIO, BinaryIO]]:
        """Context manager for temporary file.

        Args:
            suffix: File suffix
            prefix: File prefix
            text_mode: Whether to open in text mode

        Yields:
            File object
        """
        temp_dir = self.path_manager.get_temp_dir()

        mode = "w+" if text_mode else "w+b"

        with tempfile.NamedTemporaryFile(
            mode=mode,
            suffix=suffix,
            prefix=prefix,
            dir=temp_dir,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            try:
                # Cast to expected type for compatibility
                if text_mode:
                    import typing

                    yield typing.cast(TextIO, temp_file)
                else:
                    yield typing.cast(BinaryIO, temp_file)
            finally:
                # Ensure file is closed before cleanup
                try:
                    temp_file.close()
                except Exception:
                    pass
                # Clean up
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temporary file {temp_path}: {e}")

    @contextmanager
    def temporary_directory(self, prefix: str = "apgi_test_") -> Iterator[Path]:
        """Context manager for temporary directory.

        Args:
            prefix: Directory prefix

        Yields:
            Path to temporary directory
        """
        temp_base = self.path_manager.get_temp_dir()
        temp_dir_path = Path(tempfile.mkdtemp(prefix=prefix, dir=temp_base))

        try:
            yield temp_dir_path
        finally:
            # Clean up
            if temp_dir_path.exists():
                shutil.rmtree(temp_dir_path)

    def get_file_stats(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive file statistics.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file statistics

        Raises:
            FileOperationError: If file stats cannot be retrieved
        """
        resolved_path = self.path_manager.resolve_path(file_path)

        if not resolved_path.exists():
            raise FileOperationError(f"File not found: {resolved_path}")

        try:
            stat = resolved_path.stat()

            stats = {
                "path": str(resolved_path),
                "name": resolved_path.name,
                "stem": resolved_path.stem,
                "suffix": resolved_path.suffix,
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 * 1024),
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "accessed": datetime.fromtimestamp(stat.st_atime),
                "is_file": resolved_path.is_file(),
                "is_dir": resolved_path.is_dir(),
                "is_symlink": resolved_path.is_symlink(),
                "readable": os.access(resolved_path, os.R_OK),
                "writable": os.access(resolved_path, os.W_OK),
                "executable": os.access(resolved_path, os.X_OK),
            }

            # Add line count for text files
            if resolved_path.is_file() and resolved_path.suffix in [
                ".py",
                ".txt",
                ".md",
                ".json",
                ".yaml",
                ".yml",
            ]:
                try:
                    content = self.safe_read_text(resolved_path)
                    stats["line_count"] = len(content.splitlines())
                    stats["char_count"] = len(content)
                except Exception:
                    stats["line_count"] = None
                    stats["char_count"] = None

            return stats

        except Exception as e:
            raise FileOperationError(f"Failed to get stats for {resolved_path}: {e}")

    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """Clean up old temporary files.

        Args:
            max_age_hours: Maximum age of files to keep

        Returns:
            Number of files cleaned up
        """
        temp_dir = self.path_manager.get_temp_dir()
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_hours * 3600

        cleaned_count = 0

        try:
            for file_path in temp_dir.rglob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                            self.logger.debug(f"Cleaned up old temp file: {file_path}")
                        except Exception as e:
                            self.logger.warning(f"Failed to clean up {file_path}: {e}")

        except Exception as e:
            self.logger.error(f"Error during temp file cleanup: {e}")

        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} temporary files")

        return cleaned_count
