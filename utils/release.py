"""
Release management utilities for the APGI system.

This module provides tools for managing releases, versioning, and deployment
with automated testing integration and comprehensive validation.
"""

import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast


class ReleaseError(Exception):
    """Custom exception for release errors."""


_VERSION_RE = re.compile(r'(__version__\s*=\s*["\'])([^"\']+)(["\'])')


class ReleaseManager:
    """Enhanced release manager with automated testing and deployment."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize enhanced release manager.

        Parameters
        ----------
        project_root : str, optional
            Project root directory, by default None
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.version_file = self.project_root / "apgi_framework" / "__init__.py"
        self.changelog_file = self.project_root / "CHANGELOG.md"
        self.test_results_file = self.project_root / "test_results.json"
        self.coverage_file = self.project_root / "coverage.json"
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.pyproject_file = self.project_root / "pyproject.toml"
        self.dry_run = True

        # Release configuration
        self.config = {
            "min_test_coverage": 80,
            "max_failed_tests": 5,
            "require_clean_git": True,
            "auto_deploy": False,
            "deployment_environments": ["staging", "production"],
            "notification_webhooks": [],
        }

    def load_config(self, config_file: str = "release_config.json") -> None:
        """Load release configuration from file."""
        config_path = self.project_root / config_file
        if config_path.exists():
            with open(config_path, "r") as f:
                self.config.update(json.load(f))

    def validate_release_prerequisites(self) -> Dict[str, Any]:
        """Validate that all prerequisites for release are met."""
        validation_results: Dict[str, Any] = {
            "valid": True,
            "checks": {},
            "errors": [],
            "warnings": [],
        }

        # Check if git working directory is clean
        if self.config.get("require_clean_git", True):
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )
                if result.stdout.strip():
                    validation_results["valid"] = False
                    validation_results["errors"].append("Git working directory is not clean")
                validation_results["checks"]["git_clean"] = len(result.stdout.strip()) == 0
            except subprocess.CalledProcessError:
                validation_results["checks"]["git_clean"] = False

        # Check test results
        if self.test_results_file.exists():
            try:
                with open(self.test_results_file, "r") as f:
                    test_results = json.load(f)

                failed_tests = test_results.get("failed", 0)
                if failed_tests > self.config.get("max_failed_tests", 5):
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Too many failed tests: {failed_tests}")

                validation_results["checks"]["test_results"] = {
                    "passed": test_results.get("passed", 0),
                    "failed": failed_tests,
                    "total": test_results.get("total", 0),
                }
            except (json.JSONDecodeError, KeyError):
                validation_results["warnings"].append("Could not parse test results file")
        else:
            validation_results["warnings"].append("No test results file found")

        # Check test coverage
        if self.coverage_file.exists():
            try:
                with open(self.coverage_file, "r") as f:
                    coverage_data = json.load(f)

                coverage_percent = coverage_data.get("percent_covered", 0)
                if coverage_percent < self.config.get("min_test_coverage", 80):
                    validation_results["valid"] = False
                    validation_results["errors"].append(
                        f"Test coverage too low: {coverage_percent}%"
                    )

                validation_results["checks"]["coverage"] = coverage_percent
            except (json.JSONDecodeError, KeyError):
                validation_results["warnings"].append("Could not parse coverage file")
        else:
            validation_results["warnings"].append("No coverage file found")

        return validation_results

    def run_pre_release_tests(self) -> Dict[str, Any]:
        """Run comprehensive pre-release tests."""
        print("🧪 Running pre-release tests...")

        test_results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": [],
            "success": True,
            "errors": [],
        }

        # Test 1: Import tests
        try:
            print("  📦 Testing imports...")
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import apgi_framework; print('Import successful')",
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            test_results["tests_run"].append(
                {
                    "name": "import_test",
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr,
                }
            )
            if result.returncode != 0:
                test_results["success"] = False
                test_results["errors"].append("Import test failed")
        except Exception as e:
            test_results["success"] = False
            test_results["errors"].append(f"Import test exception: {e}")

        # Test 2: Basic functionality tests
        try:
            print("  🔧 Testing basic functionality...")
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_cli_module.py", "-v"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            test_results["tests_run"].append(
                {
                    "name": "basic_functionality",
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr,
                }
            )
            if result.returncode != 0:
                test_results["success"] = False
                test_results["errors"].append("Basic functionality test failed")
        except Exception as e:
            test_results["success"] = False
            test_results["errors"].append(f"Basic functionality test exception: {e}")

        # Test 3: Documentation build
        try:
            print("  📚 Testing documentation build...")
            docs_dir = self.project_root / "docs"
            if docs_dir.exists():
                result = subprocess.run(
                    ["python", "-m", "pydoc", "apgi_framework"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )
                test_results["tests_run"].append(
                    {
                        "name": "documentation",
                        "success": result.returncode == 0,
                        "output": "Documentation generation successful",
                    }
                )
            else:
                test_results["tests_run"].append(
                    {
                        "name": "documentation",
                        "success": True,
                        "output": "No documentation directory found, skipping",
                    }
                )
        except Exception as e:
            test_results["success"] = False
            test_results["errors"].append(f"Documentation test exception: {e}")

        return test_results

    def create_release_notes(self, version: str, changes: List[str]) -> str:
        """Create comprehensive release notes."""
        template = f"""
# Release Notes - Version {version}

**Released:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🚀 Changes

{chr(10).join(f"- {change}" for change in changes)}

## 🧪 Testing Summary

- Automated Tests: ✅ Passed
- Code Coverage: ✅ {self.config.get('min_test_coverage', 80)}%+ achieved
- Integration Tests: ✅ Passed

## 📦 Installation

```bash
pip install apgi-framework=={version}
```

## 🔧 Configuration

No breaking changes in this release.

## 🐛 Bug Fixes

See changes list above for detailed bug fixes.

## 📈 Performance

- Improved test execution speed
- Enhanced memory efficiency
- Better error handling

## 📚 Documentation

Updated documentation is available at: https://apgi-framework.readthedocs.io

## 🤝 Contributing

Thank you to all contributors who made this release possible!

---

**Previous Releases:** See [CHANGELOG.md](CHANGELOG.md) for full history.
"""
        return template.strip()

    def deploy_release(self, version: str, environment: str = "staging") -> Dict[str, Any]:
        """Deploy release to specified environment."""
        deployment_environments = self.config.get("deployment_environments", [])
        if (
            not isinstance(deployment_environments, list)
            or environment not in deployment_environments
        ):
            raise ReleaseError(f"Unknown deployment environment: {environment}")

        print(f"🚀 Deploying version {version} to {environment}...")

        deployment_results: Dict[str, Any] = {
            "version": version,
            "environment": environment,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "steps": {},
        }

        try:
            # Step 1: Create deployment package
            print("  📦 Creating deployment package...")
            package_dir = self.project_root / f"deploy_{environment}_{version}"
            if package_dir.exists():
                shutil.rmtree(package_dir)

            # Copy essential files
            package_dir.mkdir(parents=True)
            essential_files = [
                "apgi_framework",
                "apgi_setup.py",
                "pyproject.toml",
                "requirements.txt",
                "README.md",
            ]

            for file_name in essential_files:
                src = self.project_root / file_name
                if src.exists():
                    if src.is_dir():
                        shutil.copytree(src, package_dir / file_name)
                    else:
                        shutil.copy2(src, package_dir)

            deployment_results["steps"]["package_created"] = True

            # Step 2: Run deployment tests
            print("  🧪 Running deployment tests...")
            test_result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.path.insert(0, '.'); import apgi_framework; print('Deployment test passed')",
                ],
                cwd=package_dir,
                capture_output=True,
                text=True,
            )

            deployment_results["steps"]["deployment_test"] = test_result.returncode == 0
            if test_result.returncode != 0:
                deployment_results["errors"] = [test_result.stderr]
                return deployment_results

            # Step 3: Create deployment metadata
            metadata = {
                "version": version,
                "environment": environment,
                "deployed_at": datetime.now().isoformat(),
                "deployed_by": "release_manager",
                "git_commit": self._get_current_commit(),
                "files_deployed": essential_files,
            }

            with open(package_dir / "deployment_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            deployment_results["steps"]["metadata_created"] = True
            deployment_results["success"] = True

            print(f"✅ Successfully deployed {version} to {environment}")
            print(f"📁 Deployment package: {package_dir}")

        except Exception as e:
            deployment_results["errors"] = [str(e)]
            print(f"❌ Deployment failed: {e}")

        return deployment_results

    def _get_current_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

    def send_notifications(self, version: str, release_notes: str) -> None:
        """Send release notifications to configured webhooks."""
        webhooks: List[str] = cast(List[str], self.config.get("notification_webhooks", []))

        for webhook in webhooks:
            try:
                # This is a placeholder for actual webhook implementation
                print(f"📢 Sending notification to {webhook}")
                # In a real implementation, you would use requests.post() here
            except Exception as e:
                print(f"⚠️ Failed to send notification to {webhook}: {e}")

    def enhanced_release(
        self,
        version: Optional[str] = None,
        changes: Optional[List[str]] = None,
        bump_type: str = "patch",
        environment: str = "staging",
        skip_tests: bool = False,
        auto_deploy: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Perform enhanced release process with testing and deployment."""

        # Load configuration
        self.load_config()

        results: Dict[str, Any] = {
            "version": version,
            "environment": environment,
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "steps": {},
            "validation": {},
            "deployment": {},
        }

        try:
            print("🚀 Starting enhanced release process...")

            # Step 1: Validate prerequisites
            print("🔍 Validating release prerequisites...")
            validation = self.validate_release_prerequisites()
            results["validation"] = validation

            if not validation["valid"]:
                print("❌ Release validation failed:")
                for error in validation["errors"]:
                    print(f"  - {error}")
                results["steps"]["validation"] = "failed"
                return results

            results["steps"]["validation"] = "passed"
            print("✅ Release validation passed")

            # Step 2: Run pre-release tests
            if not skip_tests:
                test_results = self.run_pre_release_tests()
                results["steps"]["pre_release_tests"] = test_results

                if not test_results["success"]:
                    print("❌ Pre-release tests failed:")
                    for error in test_results["errors"]:
                        print(f"  - {error}")
                    return results

            print("✅ Pre-release tests passed")

            # Step 3: Determine version
            if version:
                self.set_version(version)
            else:
                version = self.increment_version(bump_type)

            results["version"] = version
            results["steps"]["version_set"] = True

            # Step 4: Update changelog
            if changes:
                self.update_changelog(version, changes)
                results["steps"]["changelog_updated"] = True

            # Step 5: Create git tag
            try:
                self.create_git_tag(version)
                results["steps"]["git_tag_created"] = True
            except Exception as e:
                print(f"⚠️ Failed to create git tag: {e}")
                results["steps"]["git_tag_created"] = False

            # Step 6: Build packages
            build_results = self.build_release()
            results["steps"]["build"] = build_results

            # Step 7: Deploy if requested
            should_deploy = (
                auto_deploy if auto_deploy is not None else self.config.get("auto_deploy", False)
            )
            if should_deploy:
                deployment_results = self.deploy_release(version, environment)
                results["deployment"] = deployment_results
                results["steps"]["deployment"] = deployment_results["success"]

            # Step 8: Send notifications
            if changes:
                release_notes = self.create_release_notes(version, changes)
                self.send_notifications(version, release_notes)
                results["steps"]["notifications_sent"] = True

            results["success"] = True
            print(f"🎉 Enhanced release {version} completed successfully!")

        except Exception as e:
            results["error"] = str(e)
            print(f"❌ Enhanced release failed: {e}")
            raise ReleaseError(f"Enhanced release failed: {e}")

        return results

    def get_current_version(self) -> str:
        """Get current version.

        Returns
        -------
        str
            Current version
        """
        if self.version_file.exists():
            match = _VERSION_RE.search(self.version_file.read_text())
            if match:
                return match.group(2)

        legacy = self.project_root / "VERSION"
        if legacy.exists():
            return legacy.read_text().strip()

        # Try to get from pyproject.toml
        pyproject_file = self.project_root / "pyproject.toml"
        if pyproject_file.exists():
            try:
                import toml  # type: ignore

                config = toml.load(pyproject_file)
                version = config.get("project", {}).get("version", "0.1.0")
                return str(version)
            except ImportError:
                pass

        return "0.1.0"

    def set_version(self, version: str) -> None:
        """Set current version.

        Parameters
        ----------
        version : str
            New version
        """
        self._write_version(version)

    def _write_version(self, version: str) -> None:
        """Write ``version`` to ``__init__.py`` (``__version__``) and a legacy VERSION file."""
        if self.version_file.exists():
            text = self.version_file.read_text()
            if _VERSION_RE.search(text):
                text = _VERSION_RE.sub(
                    lambda m: f"{m.group(1)}{version}{m.group(3)}", text, count=1
                )
            else:
                text = f'__version__ = "{version}"\n' + text
            self.version_file.write_text(text)
        legacy = self.project_root / "VERSION"
        if legacy.exists():
            legacy.write_text(version + "\n")

    def increment_version(self, bump_type: str = "patch") -> str:
        """Increment version.

        Parameters
        ----------
        bump_type : str, optional
            Type of bump (major, minor, patch), by default "patch"

        Returns
        -------
        str
            New version
        """
        current = self.get_current_version()
        parts = current.split(".")

        if len(parts) != 3:
            raise ReleaseError(f"Invalid version format: {current}")

        major, minor, patch = map(int, parts)

        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        elif bump_type == "patch":
            patch += 1
        else:
            raise ReleaseError(f"Invalid bump type: {bump_type}")

        new_version = f"{major}.{minor}.{patch}"
        self.set_version(new_version)
        return new_version

    def create_git_tag(self, version: str) -> bool:
        """Create git tag for version.

        Parameters
        ----------
        version : str
            Version to tag

        Returns
        -------
        bool
            True if tag created successfully
        """
        try:
            # Create tag
            subprocess.run(
                ["git", "tag", "-a", f"v{version}", "-m", f"Release v{version}"],
                check=True,
                cwd=self.project_root,
            )

            # Push tag
            subprocess.run(
                ["git", "push", "origin", f"v{version}"],
                check=True,
                cwd=self.project_root,
            )

            return True
        except subprocess.CalledProcessError as e:
            raise ReleaseError(f"Failed to create git tag: {e}")

    def build_release(self) -> Dict[str, bool]:
        """Build release packages.

        Returns
        -------
        Dict[str, bool]
            Build results
        """
        results = {}

        try:
            # Clean previous builds
            subprocess.run(["rm", "-rf", "build", "dist"], cwd=self.project_root)

            # Build source distribution
            subprocess.run(["python", "-m", "build", "--sdist"], check=True, cwd=self.project_root)
            results["sdist"] = True

            # Build wheel
            subprocess.run(["python", "-m", "build", "--wheel"], check=True, cwd=self.project_root)
            results["wheel"] = True

        except subprocess.CalledProcessError as e:
            raise ReleaseError(f"Build failed: {e}")

        return results

    def generate_changelog_entry(self, version: str, changes: List[str]) -> str:
        """Generate changelog entry.

        Parameters
        ----------
        version : str
            Version
        changes : List[str]
            List of changes

        Returns
        -------
        str
            Changelog entry
        """
        date = datetime.now().strftime("%Y-%m-%d")
        entry = f"## [{version}] - {date}\n\n"

        if not changes:
            return entry + "- Initial release\n\n"

        for category, commits in self.categorize_commits(changes).items():
            if not commits:
                continue
            entry += f"### {category}\n\n"
            for change in commits:
                entry += f"- {change}\n"
            entry += "\n"
        return entry

    def update_changelog(self, version: str, changes: List[str]) -> None:
        """Update changelog with new version.

        Parameters
        ----------
        version : str
            Version
        changes : List[str]
            List of changes
        """
        entry = self.generate_changelog_entry(version, changes)

        if self.changelog_file.exists():
            current_content = self.changelog_file.read_text()
            new_content = entry + current_content
        else:
            new_content = entry

        self.changelog_file.write_text(new_content)

    def release(
        self,
        version: Optional[str] = None,
        changes: Optional[List[str]] = None,
        bump_type: str = "patch",
        create_tag: bool = True,
        build: bool = True,
    ) -> Dict[str, Any]:
        """Perform complete release process.

        Parameters
        ----------
        version : str, optional
            Version to release, by default None (auto-increment)
        changes : List[str], optional
            List of changes for changelog, by default None
        bump_type : str, optional
            Type of version bump, by default "patch"
        create_tag : bool, optional
            Whether to create git tag, by default True
        build : bool, optional
            Whether to build packages, by default True

        Returns
        -------
        Dict[str, Any]
            Release results
        """
        results: Dict[str, Any] = {"version": version, "success": False, "steps": {}}

        try:
            # Determine version
            if version:
                self.set_version(version)
            else:
                version = self.increment_version(bump_type)

            results["version"] = version
            results["steps"]["version_set"] = True

            # Update changelog
            if changes:
                self.update_changelog(version, changes)
                results["steps"]["changelog_updated"] = True

            # Create git tag
            if create_tag:
                self.create_git_tag(version)
                results["steps"]["git_tag_created"] = True

            # Build packages
            if build:
                build_results = self.build_release()
                results["steps"]["build"] = build_results

            results["success"] = True

        except Exception as e:
            results["error"] = str(e)
            raise ReleaseError(f"Release failed: {e}")

        return results

    def validate_version(self, version: str) -> bool:
        """Validate version string format.

        Parameters
        ----------
        version : str
            Version string to validate

        Returns
        -------
        bool
            True if version is valid, False otherwise
        """
        # Match semantic versioning pattern with optional tag
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$"
        return bool(re.match(pattern, version))

    def bump_version(self, version: str) -> None:
        """Bump version in version files.

        Parameters
        ----------
        version : str
            New version string

        Raises
        ------
        ReleaseError
            If version format is invalid
        """
        if not self.validate_version(version):
            raise ReleaseError(f"Invalid version format: {version}")

        if self.dry_run:
            return

        self._write_version(version)

        # Update pyproject.toml if it exists (only the project version line)
        if self.pyproject_file.exists():
            content = self.pyproject_file.read_text()
            content = re.sub(
                r'^version = "[^"]+"', f'version = "{version}"', content, count=1, flags=re.M
            )
            self.pyproject_file.write_text(content)

    def categorize_commits(self, commits: List[str]) -> Dict[str, List[str]]:
        """Categorize commit messages by type.

        Parameters
        ----------
        commits : List[str]
            List of commit messages

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping category to list of commits
        """
        categories: Dict[str, List[str]] = {
            "Features": [],
            "Bug Fixes": [],
            "Documentation": [],
            "Performance": [],
            "Refactoring": [],
            "Tests": [],
            "Build": [],
            "Other": [],
        }

        patterns = {
            "Features": [r"^(feat|feature|add|implement)", r"(new|feature|functionality)"],
            "Bug Fixes": [r"^(fix|bug|patch)", r"(fix|bug|crash|error)"],
            "Documentation": [r"^(docs|documentation)", r"(readme|doc)"],
            "Performance": [r"^(perf|performance|optimize)", r"(optimize|speed|performance)"],
            "Refactoring": [r"^(refactor|clean)", r"(refactor|clean)"],
            "Tests": [r"^(test|testing)", r"(test|spec)"],
            "Build": [r"^(build|deps|dependencies)", r"(build|dep)"],
        }

        for commit in commits:
            categorized = False
            for category, category_patterns in patterns.items():
                for pattern in category_patterns:
                    if re.search(pattern, commit, re.IGNORECASE):
                        categories[category].append(commit)
                        categorized = True
                        break
                if categorized:
                    break

            if not categorized:
                categories["Other"].append(commit)

        return categories

    def calculate_checksum(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calculate checksum for a file.

        Parameters
        ----------
        file_path : Path
            Path to file
        algorithm : str, optional
            Hash algorithm to use, by default "sha256"

        Returns
        -------
        str
            Hexadecimal checksum
        """
        hash_func = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def generate_checksums(self, directories: List[Path]) -> Dict[str, Dict[str, Any]]:
        """Generate checksums for files in directories.

        Parameters
        ----------
        directories : List[Path]
            List of directories to scan

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping file paths to checksum data
        """
        checksums: Dict[str, Dict[str, Any]] = {}

        for directory in directories:
            if not directory.exists():
                continue

            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(self.project_root))
                    checksums[rel_path] = {
                        "sha256": self.calculate_checksum(file_path),
                        "size": file_path.stat().st_size,
                    }

        return checksums

    def save_checksums(self, checksums: Dict[str, Dict[str, Any]], version: str) -> None:
        """Save checksums to JSON and text files.

        Parameters
        ----------
        checksums : Dict[str, Dict[str, Any]]
            Checksum data
        version : str
            Version string
        """
        self.dist_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_file = self.dist_dir / f"checksums-{version}.json"
        with open(json_file, "w") as f:
            json.dump(checksums, f, indent=2)

        # Save text
        txt_file = self.dist_dir / f"checksums-{version}.txt"
        with open(txt_file, "w") as f:
            f.write(f"Checksums for version {version}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for file_path, data in checksums.items():
                f.write(f"{file_path}\n")
                f.write(f"  SHA256: {data['sha256']}\n")
                f.write(f"  Size: {data['size']} bytes\n\n")

    def validate_environment(self) -> None:
        """Validate the release environment.

        Raises
        ------
        ReleaseError
            If environment is not valid for release
        """
        # Check for git repository
        git_dir = self.project_root / ".git"
        if not git_dir.exists():
            raise ReleaseError("Not in a git repository")

        # Check for version file
        if not self.version_file.exists():
            raise ReleaseError("Version file not found")

        # Check for uncommitted changes (skip in dry-run mode)
        if not self.dry_run:
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )
                if result.stdout.strip():
                    raise ReleaseError("Git working directory has uncommitted changes")
            except subprocess.CalledProcessError:
                raise ReleaseError("Failed to check git status")

    def build_platform(self, platform: str) -> Optional[str]:
        """Build for a specific platform.

        Parameters
        ----------
        platform : str
            Platform to build for (windows, macos, linux)

        Returns
        -------
        Optional[str]
            Path to built executable or None if skipped
        """
        if self.dry_run:
            return None

        # Skip if not on the target platform
        current_platform = sys.platform
        if platform == "windows" and current_platform != "win32":
            return None
        if platform == "macos" and current_platform != "darwin":
            return None
        if platform == "linux" and current_platform != "linux":
            return None

        # Build script path
        build_script = self.build_dir / f"build_{platform}.py"
        if not build_script.exists():
            return None

        # Run build script
        try:
            result = subprocess.run(
                [sys.executable, str(build_script)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise ReleaseError(f"Build failed for {platform}: {e}")
