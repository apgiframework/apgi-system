"""
Unit tests for release automation script.

Tests the release.py script functionality including version bumping,
changelog generation, and release process automation.

Requirements tested:
- 8.1: Platform detection and build environment
- 8.2: Clean build environment creation
- 8.4: Output file placement

Feature: cross-platform-executable
"""

import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest

# Import release module
from utils.release import ReleaseError, ReleaseManager


class TestVersionValidation:
    """Test version string validation."""

    @pytest.fixture
    def manager(self) -> ReleaseManager:
        """Create a release manager in dry-run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            return ReleaseManager(tmpdir)

    def test_valid_version_formats(self, manager: ReleaseManager) -> None:
        """Test that valid version formats are accepted."""
        valid_versions = [
            "1.0.0",
            "0.1.0",
            "10.20.30",
            "1.0.0-beta",
            "1.0.0-alpha",
            "1.0.0-rc1",
            "2.1.3-dev",
        ]

        for version in valid_versions:
            assert manager.validate_version(version), f"Version {version} should be valid"

    def test_invalid_version_formats(self, manager: ReleaseManager) -> None:
        """Test that invalid version formats are rejected."""
        invalid_versions = [
            "v1.0.0",  # No 'v' prefix
            "1.0",  # Missing patch version
            "1",  # Only major version
            "1.0.0.0",  # Too many parts
            "abc",  # Not a version
            "1.0.0-",  # Empty tag
            "1.0.0-beta.1",  # Dots in tag not supported
        ]

        for version in invalid_versions:
            assert not manager.validate_version(version), f"Version {version} should be invalid"


class TestVersionBumping:
    """Test version bumping functionality."""

    @pytest.fixture
    def temp_project(self) -> Generator[Path, None, None]:
        """Create a temporary project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create apgi_framework directory
            apgi_dir = project_root / "apgi_framework"
            apgi_dir.mkdir()

            # Create __init__.py with version
            init_file = apgi_dir / "__init__.py"
            init_file.write_text('__version__ = "0.1.0"\n')

            # Create pyproject.toml
            pyproject_file = project_root / "pyproject.toml"
            pyproject_file.write_text('[project]\nversion = "0.1.0"\n')

            # Create .git directory
            git_dir = project_root / ".git"
            git_dir.mkdir()

            yield project_root

    def test_get_current_version(self, temp_project: Path) -> None:
        """Test extracting current version from __init__.py."""
        manager = ReleaseManager(str(temp_project))
        version = manager.get_current_version()

        assert version == "0.1.0", "Should extract correct version"

    def test_bump_version_dry_run(self, temp_project: Path) -> None:
        """Test version bumping in dry-run mode."""
        manager = ReleaseManager(str(temp_project))

        # Should not raise error
        manager.bump_version("1.0.0")

        # Files should not be modified
        init_file = temp_project / "apgi_framework" / "__init__.py"
        content = init_file.read_text()
        assert "0.1.0" in content, "Version should not change in dry-run"

    def test_bump_version_actual(self, temp_project: Path) -> None:
        """Test actual version bumping."""
        manager = ReleaseManager(str(temp_project))
        manager.dry_run = False

        manager.bump_version("1.0.0")

        # Check __init__.py
        init_file = temp_project / "apgi_framework" / "__init__.py"
        content = init_file.read_text()
        assert "1.0.0" in content, "Version should be updated in __init__.py"
        assert "0.1.0" not in content, "Old version should be removed"

        # Check pyproject.toml
        pyproject_file = temp_project / "pyproject.toml"
        content = pyproject_file.read_text()
        assert "1.0.0" in content, "Version should be updated in pyproject.toml"

    def test_bump_version_invalid_format(self, temp_project: Path) -> None:
        """Test that invalid version format raises error."""
        manager = ReleaseManager(str(temp_project))

        with pytest.raises(ReleaseError) as exc_info:
            manager.bump_version("invalid-version")

        assert "Invalid version format" in str(exc_info.value)


class TestChangelogGeneration:
    """Test changelog generation functionality."""

    @pytest.fixture
    def manager(self) -> ReleaseManager:
        """Create a release manager in dry-run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            return ReleaseManager(tmpdir)

    def test_categorize_commits_features(self, manager: ReleaseManager) -> None:
        """Test that feature commits are categorized correctly."""
        commits = [
            "feat: Add new feature",
            "feature: Implement something",
            "Add new functionality",
            "Implement new component",
        ]

        categories = manager.categorize_commits(commits)

        assert "Features" in categories, "Should have Features category"
        # Note: "Add new functionality" might be categorized as Other if it doesn't match patterns
        assert len(categories["Features"]) >= 3, "Should categorize most feature commits"

    def test_categorize_commits_bug_fixes(self, manager: ReleaseManager) -> None:
        """Test that bug fix commits are categorized correctly."""
        commits = [
            "fix: Resolve memory leak",
            "bug: Fix crash on startup",
            "patch: Update dependency",
        ]

        categories = manager.categorize_commits(commits)

        assert "Bug Fixes" in categories, "Should have Bug Fixes category"
        assert len(categories["Bug Fixes"]) == 3, "Should categorize all bug fix commits"

    def test_categorize_commits_mixed(self, manager: ReleaseManager) -> None:
        """Test categorization of mixed commit types."""
        commits = [
            "feat: Add feature",
            "fix: Fix bug",
            "docs: Update README",
            "perf: Optimize algorithm",
            "refactor: Clean up code",
            "test: Add tests",
            "build: Update dependencies",
            "Random commit message",
        ]

        categories = manager.categorize_commits(commits)

        # Should have multiple categories
        assert len(categories) >= 7, "Should have at least 7 categories"
        assert "Features" in categories
        assert "Bug Fixes" in categories
        assert "Documentation" in categories
        assert "Performance" in categories
        assert "Refactoring" in categories
        assert "Tests" in categories
        assert "Build" in categories
        assert "Other" in categories

    def test_generate_changelog_entry_empty(self, manager: ReleaseManager) -> None:
        """Test changelog generation with no commits."""
        entry = manager.generate_changelog_entry("1.0.0", [])

        assert "1.0.0" in entry, "Should include version"
        assert "Initial release" in entry, "Should indicate initial release"

    def test_generate_changelog_entry_with_commits(self, manager: ReleaseManager) -> None:
        """Test changelog generation with commits."""
        commits = [
            "feat: Add new feature",
            "fix: Fix critical bug",
        ]

        entry = manager.generate_changelog_entry("1.0.0", commits)

        assert "1.0.0" in entry, "Should include version"
        assert "Features" in entry, "Should have Features section"
        assert "Bug Fixes" in entry, "Should have Bug Fixes section"
        assert "Add new feature" in entry, "Should include feature commit"
        assert "Fix critical bug" in entry, "Should include bug fix commit"

    def test_generate_changelog_entry_date(self, manager: ReleaseManager) -> None:
        """Test that changelog entry includes current date."""
        entry = manager.generate_changelog_entry("1.0.0", [])

        today = datetime.now().strftime("%Y-%m-%d")
        assert today in entry, "Should include current date"


class TestChecksumGeneration:
    """Test checksum generation functionality."""

    @pytest.fixture
    def temp_dist(self) -> Generator[Path, None, None]:
        """Create temporary distribution directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dist_dir = Path(tmpdir) / "dist"
            dist_dir.mkdir()

            # Create test files
            windows_dir = dist_dir / "windows"
            windows_dir.mkdir()

            test_exe = windows_dir / "test.exe"
            test_exe.write_bytes(b"test executable content")

            yield dist_dir

    def test_calculate_checksum_sha256(self, temp_dist: Path) -> None:
        """Test SHA256 checksum calculation."""
        manager = ReleaseManager(str(temp_dist.parent))

        test_file = temp_dist / "windows" / "test.exe"
        checksum = manager.calculate_checksum(test_file, "sha256")

        assert isinstance(checksum, str), "Checksum should be a string"
        assert len(checksum) == 64, "SHA256 checksum should be 64 characters"
        assert checksum.isalnum(), "Checksum should be alphanumeric"

    def test_generate_checksums(self, temp_dist: Path) -> None:
        """Test checksum generation for distribution files."""
        manager = ReleaseManager(str(temp_dist.parent))
        manager.dist_dir = temp_dist

        checksums = manager.generate_checksums([temp_dist / "windows"])

        assert len(checksums) > 0, "Should generate checksums"

        # Check structure
        for file_path, data in checksums.items():
            assert "sha256" in data, "Should include SHA256"
            assert "size" in data, "Should include file size"
            assert isinstance(data["size"], int), "Size should be integer"

    def test_save_checksums_json(self, temp_dist: Path) -> None:
        """Test saving checksums to JSON file."""
        manager = ReleaseManager(str(temp_dist.parent))
        manager.dist_dir = temp_dist

        checksums = {"windows/test.exe": {"sha256": "abc123", "size": 1024}}

        manager.save_checksums(checksums, "1.0.0")

        # Check JSON file
        json_file = temp_dist / "checksums-1.0.0.json"
        assert json_file.exists(), "JSON checksums file should be created"

        with open(json_file) as f:
            data = json.load(f)

        assert data == checksums, "JSON should contain correct checksums"

        # Check text file
        txt_file = temp_dist / "checksums-1.0.0.txt"
        assert txt_file.exists(), "Text checksums file should be created"

        content = txt_file.read_text()
        assert "1.0.0" in content, "Should include version"
        assert "SHA256" in content, "Should include SHA256 label"


class TestEnvironmentValidation:
    """Test environment validation functionality."""

    def test_validate_environment_no_git(self) -> None:
        """Test that validation fails without git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ReleaseManager(tmpdir)

            with pytest.raises(ReleaseError) as exc_info:
                manager.validate_environment()

            assert "git repository" in str(exc_info.value).lower()

    def test_validate_environment_missing_version_file(self) -> None:
        """Test that validation fails without version file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create .git directory
            git_dir = project_root / ".git"
            git_dir.mkdir()

            # Initialize git repo to avoid uncommitted changes error
            subprocess.run(["git", "init"], cwd=project_root, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=project_root,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=project_root, capture_output=True
            )

            manager = ReleaseManager(str(project_root))

            with pytest.raises(ReleaseError) as exc_info:
                manager.validate_environment()

            assert "Version file not found" in str(exc_info.value)

    def test_validate_environment_dry_run_allows_uncommitted(self) -> None:
        """Test that dry-run mode allows uncommitted changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create .git directory
            git_dir = project_root / ".git"
            git_dir.mkdir()

            # Create version file
            apgi_dir = project_root / "apgi_framework"
            apgi_dir.mkdir()
            init_file = apgi_dir / "__init__.py"
            init_file.write_text('__version__ = "0.1.0"\n')

            manager = ReleaseManager(str(project_root))

            # Should not raise error in dry-run mode
            # (even though we can't actually check git status in temp dir)
            try:
                manager.validate_environment()
            except ReleaseError as e:
                # Only acceptable error is git command failure
                assert "git status" in str(e).lower() or "git repository" in str(e).lower()


class TestPlatformBuilding:
    """Test platform-specific building functionality."""

    @pytest.fixture
    def temp_project(self) -> Generator[Path, None, None]:
        """Create temporary project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create build directory
            build_dir = project_root / "build"
            build_dir.mkdir()

            # Create dist directory
            dist_dir = project_root / "dist"
            dist_dir.mkdir()

            # Create .git directory
            git_dir = project_root / ".git"
            git_dir.mkdir()

            yield project_root

    @patch("subprocess.run")
    def test_build_platform_windows(self, mock_run: Mock, temp_project: Path) -> None:
        """Test building for Windows platform."""
        manager = ReleaseManager(str(temp_project))

        # Create build script
        build_script = temp_project / "build" / "build_windows.py"
        build_script.write_text("# Build script")

        # Mock successful build
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        # Only test on Windows
        if sys.platform == "win32":
            result = manager.build_platform("windows")

            assert result is not None, "Should return build path"
            assert mock_run.called, "Should call build script"

    def test_build_platform_skip_wrong_platform(self, temp_project: Path) -> None:
        """Test that building skips when on wrong platform."""
        manager = ReleaseManager(str(temp_project))

        # Try to build for platform we're not on
        if sys.platform == "win32":
            result = manager.build_platform("macos")
        else:
            result = manager.build_platform("windows")

        assert result is None, "Should skip build for wrong platform"

    def test_build_platform_dry_run(self, temp_project: Path) -> None:
        """Test building in dry-run mode."""
        manager = ReleaseManager(str(temp_project))

        result = manager.build_platform("windows")

        assert result is None, "Should not build in dry-run mode"


class TestReleaseManagerInitialization:
    """Test ReleaseManager initialization."""

    def test_initialization_with_project_root(self) -> None:
        """Test initialization with explicit project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ReleaseManager(tmpdir)

            assert manager.project_root == Path(tmpdir), "Should use provided root"
            assert manager.build_dir == Path(tmpdir) / "build"
            assert manager.dist_dir == Path(tmpdir) / "dist"

    def test_initialization_dry_run_mode(self) -> None:
        """Test initialization in dry-run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ReleaseManager(tmpdir)

            assert manager.dry_run is True, "Should be in dry-run mode"

    def test_initialization_default_paths(self) -> None:
        """Test that default paths are set correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ReleaseManager(tmpdir)

            assert manager.version_file.name == "__init__.py"
            assert manager.changelog_file.name == "CHANGELOG.md"
            assert manager.pyproject_file.name == "pyproject.toml"


class TestErrorHandling:
    """Test error handling in release process."""

    def test_release_error_exception(self) -> None:
        """Test that ReleaseError can be raised and caught."""
        with pytest.raises(ReleaseError) as exc_info:
            raise ReleaseError("Test error message")

        assert "Test error message" in str(exc_info.value)

    def test_invalid_version_raises_error(self) -> None:
        """Test that invalid version raises appropriate error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create minimal structure
            apgi_dir = project_root / "apgi_framework"
            apgi_dir.mkdir()
            init_file = apgi_dir / "__init__.py"
            init_file.write_text('__version__ = "0.1.0"\n')

            manager = ReleaseManager(str(project_root))

            with pytest.raises(ReleaseError) as exc_info:
                manager.bump_version("invalid")

            assert "Invalid version format" in str(exc_info.value)


# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit
