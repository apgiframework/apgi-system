"""CI/CD Integration Component for APGI Framework Test Enhancement.

This module provides automated test execution capabilities for CI/CD pipelines,
including change impact analysis, pre-commit hooks, and pipeline integration.

Requirements: 8.1, 8.2, 8.6
"""

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..utils.ast_analyzer import ASTAnalyzer
from ..utils.file_utils import FileUtils
from .activity_logger import ActivityLevel, ActivityType, get_activity_logger


@dataclass
class ChangeImpact:
    """Represents the impact of code changes on test execution."""

    changed_files: List[str]
    affected_modules: Set[str]
    required_tests: Set[str]
    impact_score: float
    analysis_timestamp: datetime


@dataclass
class CIConfiguration:
    """Configuration for CI/CD integration."""

    pipeline_type: str  # 'github', 'gitlab', 'jenkins', 'azure', 'generic'
    test_subset_strategy: str  # 'all', 'changed', 'impact', 'critical'
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_minutes: int = 30
    coverage_threshold: float = 0.8
    critical_test_patterns: Optional[List[str]] = None
    pre_commit_enabled: bool = True
    notification_channels: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.critical_test_patterns is None:
            self.critical_test_patterns = [
                "**/test_core_*.py",
                "**/test_security_*.py",
                "**/test_data_*.py",
            ]
        if self.notification_channels is None:
            self.notification_channels = []


@dataclass
class ExecutionResult:
    """Result of CI test execution."""

    execution_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    coverage_percentage: float
    execution_time_seconds: float
    failed_test_details: List[Dict[str, Any]]
    pipeline_context: Dict[str, Any]


class ChangeAnalyzer:
    """Analyzes code changes to determine test impact."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.ast_analyzer = ASTAnalyzer()
        self.file_utils = FileUtils()
        self.logger = logging.getLogger(__name__)

    def analyze_git_changes(self, base_ref: str = "HEAD~1") -> ChangeImpact:
        """Analyze git changes to determine test impact."""
        try:
            # Get changed files from git
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{base_ref}..HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode != 0:
                # Check if it's because HEAD~1 doesn't exist (shallow repo)
                if "ambiguous argument" in result.stderr and base_ref == "HEAD~1":
                    self.logger.warning("HEAD~1 not available, falling back to HEAD")
                    # Fall back to comparing working tree to HEAD
                    result = subprocess.run(
                        ["git", "diff", "--name-only", "HEAD"],
                        capture_output=True,
                        text=True,
                        cwd=self.project_root,
                    )
                    if result.returncode != 0:
                        self.logger.warning(f"Git diff HEAD failed: {result.stderr}")
                        return self._fallback_analysis()
                else:
                    self.logger.warning(f"Git diff failed: {result.stderr}")
                    return self._fallback_analysis()

            changed_files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
            return self._analyze_file_changes(changed_files)

        except Exception as e:
            self.logger.error(f"Error analyzing git changes: {e}")
            return self._fallback_analysis()

    def analyze_file_changes(self, changed_files: List[str]) -> ChangeImpact:
        """Analyze specific file changes to determine test impact."""
        return self._analyze_file_changes(changed_files)

    def _analyze_file_changes(self, changed_files: List[str]) -> ChangeImpact:
        """Internal method to analyze file changes."""
        affected_modules = set()
        required_tests = set()
        impact_score = 0.0

        for file_path in changed_files:
            if not file_path.endswith(".py"):
                continue

            full_path = self.project_root / file_path
            if not full_path.exists():
                continue

            # Determine module impact
            module_name = self._get_module_name(file_path)
            affected_modules.add(module_name)

            # Find related tests
            related_tests = self._find_related_tests(file_path)
            required_tests.update(related_tests)

            # Calculate impact score based on file importance
            file_impact = self._calculate_file_impact(file_path)
            impact_score += file_impact

        # Normalize impact score
        impact_score = min(impact_score / len(changed_files) if changed_files else 0, 1.0)

        return ChangeImpact(
            changed_files=changed_files,
            affected_modules=affected_modules,
            required_tests=required_tests,
            impact_score=impact_score,
            analysis_timestamp=datetime.now(),
        )

    def _get_module_name(self, file_path: str) -> str:
        """Extract module name from file path."""
        path_parts = Path(file_path).parts
        if "apgi_framework" in path_parts:
            idx = path_parts.index("apgi_framework")
            if idx + 1 < len(path_parts):
                return path_parts[idx + 1]
        return "unknown"

    def _find_related_tests(self, file_path: str) -> Set[str]:
        """Find tests related to a changed file."""
        related_tests = set()

        # Direct test file
        test_file = self._get_test_file_path(file_path)
        if test_file and test_file.exists():
            related_tests.add(str(test_file))

        # Module-level tests
        module_name = self._get_module_name(file_path)
        module_tests = self._find_module_tests(module_name)
        related_tests.update(module_tests)

        # Integration tests that might import this module
        integration_tests = self._find_integration_tests(file_path)
        related_tests.update(integration_tests)

        return related_tests

    def _get_test_file_path(self, source_file: str) -> Optional[Path]:
        """Get the corresponding test file for a source file."""
        source_path = Path(source_file)

        # Try different test file naming conventions
        test_patterns = [
            f"test_{source_path.stem}.py",
            f"{source_path.stem}_test.py",
            f"test_{source_path.stem}_properties.py",
        ]

        tests_dir = self.project_root / "tests"
        for pattern in test_patterns:
            test_file = tests_dir / pattern
            if test_file.exists():
                return test_file

        return None

    def _find_module_tests(self, module_name: str) -> Set[str]:
        """Find all tests for a specific module."""
        tests: Set[str] = set()
        tests_dir = self.project_root / "tests"

        if not tests_dir.exists():
            return tests

        for test_file in tests_dir.glob(f"test_{module_name}*.py"):
            tests.add(str(test_file))

        return tests

    def _find_integration_tests(self, file_path: str) -> Set[str]:
        """Find integration tests that might be affected by file changes."""
        integration_tests: Set[str] = set()
        tests_dir = self.project_root / "tests"

        if not tests_dir.exists():
            return integration_tests

        # Look for integration test files
        for test_file in tests_dir.glob("test_*integration*.py"):
            integration_tests.add(str(test_file))

        # Look for tests that import the changed module
        module_path = str(Path(file_path).with_suffix(""))
        module_import = module_path.replace("/", ".").replace("\\", ".")

        for test_file in tests_dir.glob("test_*.py"):
            try:
                content = test_file.read_text()
                if module_import in content or Path(file_path).stem in content:
                    integration_tests.add(str(test_file))
            except Exception:
                continue

        return integration_tests

    def _calculate_file_impact(self, file_path: str) -> float:
        """Calculate the impact score for a file based on its importance."""
        impact_weights = {
            "core": 1.0,
            "security": 0.9,
            "data": 0.8,
            "analysis": 0.7,
            "clinical": 0.7,
            "neural": 0.6,
            "gui": 0.4,
            "utils": 0.3,
            "tests": 0.2,
        }

        module_name = self._get_module_name(file_path)
        base_impact = impact_weights.get(module_name, 0.5)

        # Increase impact for critical files
        if any(critical in file_path.lower() for critical in ["__init__", "main", "config"]):
            base_impact *= 1.2

        return min(base_impact, 1.0)

    def _fallback_analysis(self) -> ChangeImpact:
        """Fallback analysis when git analysis fails."""
        return ChangeImpact(
            changed_files=[],
            affected_modules=set(),
            required_tests=set(),
            impact_score=1.0,  # Run all tests as fallback
            analysis_timestamp=datetime.now(),
        )


class PreCommitHookManager:
    """Manages pre-commit hooks for critical test execution."""

    def __init__(self, project_root: str, config: CIConfiguration):
        self.project_root = Path(project_root)
        self.config = config
        self.logger = logging.getLogger(__name__)

    def install_hooks(self) -> bool:
        """Install pre-commit hooks."""
        try:
            hooks_dir = self.project_root / ".git" / "hooks"
            hooks_dir.mkdir(exist_ok=True)

            # Create pre-commit hook script
            hook_script = self._generate_hook_script()
            hook_file = hooks_dir / "pre-commit"

            hook_file.write_text(hook_script)
            hook_file.chmod(0o755)

            self.logger.info("Pre-commit hooks installed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to install pre-commit hooks: {e}")
            return False

    def uninstall_hooks(self) -> bool:
        """Uninstall pre-commit hooks."""
        try:
            hook_file = self.project_root / ".git" / "hooks" / "pre-commit"
            if hook_file.exists():
                hook_file.unlink()
                self.logger.info("Pre-commit hooks uninstalled")
            return True

        except Exception as e:
            self.logger.error(f"Failed to uninstall pre-commit hooks: {e}")
            return False

    def _generate_hook_script(self) -> str:
        """Generate the pre-commit hook script."""
        return f"""#!/bin/bash
# APGI Framework Pre-commit Hook
# Auto-generated by CIIntegrator

set -e

echo "Running APGI Framework pre-commit tests..."

# Change to project root
cd "{self.project_root}"

# Run critical tests
python -m pytest {' '.join(self.config.critical_test_patterns or [])} \\
    --tb=short \\
    --maxfail=5 \\
    --timeout={self.config.timeout_minutes * 60} \\
    -q

echo "Pre-commit tests passed!"
"""


class CIIntegrator:
    """Main CI/CD integration component."""

    def __init__(self, project_root: str, config: Optional[CIConfiguration] = None):
        self.project_root = Path(project_root)
        self.config = config or CIConfiguration(pipeline_type="generic", test_subset_strategy="all")
        self.change_analyzer = ChangeAnalyzer(project_root)
        self.hook_manager = PreCommitHookManager(project_root, self.config)
        self.logger = logging.getLogger(__name__)

        # Ensure required directories exist
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Setup required directories for CI integration."""
        ci_dir = self.project_root / ".ci"
        ci_dir.mkdir(exist_ok=True)

        reports_dir = ci_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        logs_dir = ci_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

    def analyze_changes(self, base_ref: str = "HEAD~1") -> ChangeImpact:
        """Analyze code changes for test impact."""
        return self.change_analyzer.analyze_git_changes(base_ref)

    def execute_ci_tests(self, change_impact: Optional[ChangeImpact] = None) -> ExecutionResult:
        """Execute tests based on CI configuration and change impact."""
        execution_id = self._generate_execution_id()
        start_time = datetime.now()

        # Set up activity logging
        activity_logger = get_activity_logger()
        activity_logger.set_context(
            execution_id=execution_id, component="ci_integrator", environment="ci"
        )

        activity_logger.log_ci_integration(
            pipeline_type=self.config.pipeline_type,
            event="test_execution_start",
            details={
                "execution_id": execution_id,
                "test_subset_strategy": self.config.test_subset_strategy,
                "change_impact_score": (change_impact.impact_score if change_impact else None),
            },
        )

        try:
            # Determine which tests to run
            if change_impact and self.config.test_subset_strategy == "impact":
                test_files = list(change_impact.required_tests)
                activity_logger.log_activity(
                    ActivityType.TEST_DISCOVERY,
                    ActivityLevel.INFO,
                    f"Using impact analysis: {len(test_files)} tests required",
                    data={"impact_score": change_impact.impact_score},
                )
            elif self.config.test_subset_strategy == "critical":
                test_files = self._get_critical_tests()
                activity_logger.log_activity(
                    ActivityType.TEST_DISCOVERY,
                    ActivityLevel.INFO,
                    f"Using critical tests: {len(test_files)} tests",
                    data={"strategy": "critical"},
                )
            else:
                test_files = self._get_all_tests()
                activity_logger.log_activity(
                    ActivityType.TEST_DISCOVERY,
                    ActivityLevel.INFO,
                    f"Using all tests: {len(test_files)} tests",
                    data={"strategy": "all"},
                )

            if not test_files:
                self.logger.warning("No tests found to execute")
                return self._create_empty_result(execution_id, start_time)

            # Execute tests
            result = self._run_pytest(test_files, execution_id)

            # Generate reports
            self._generate_ci_reports(result, execution_id)

            # Log CI completion
            activity_logger.log_ci_integration(
                pipeline_type=self.config.pipeline_type,
                event="test_execution_complete",
                details={
                    "execution_id": execution_id,
                    "total_tests": result.total_tests,
                    "failed_tests": result.failed_tests,
                    "coverage_percentage": result.coverage_percentage,
                    "duration_seconds": result.execution_time_seconds,
                },
            )

            return result

        except Exception as e:
            self.logger.error(f"CI test execution failed: {e}")
            activity_logger.log_error(
                "ci_integrator",
                e,
                {
                    "execution_id": execution_id,
                    "pipeline_type": self.config.pipeline_type,
                },
            )
            return self._create_error_result(execution_id, start_time, str(e))

    def setup_pipeline_integration(self) -> Dict[str, str]:
        """Setup CI/CD pipeline integration files."""
        integration_files = {}

        if self.config.pipeline_type == "github":
            integration_files.update(self._setup_github_actions())
        elif self.config.pipeline_type == "gitlab":
            integration_files.update(self._setup_gitlab_ci())
        elif self.config.pipeline_type == "jenkins":
            integration_files.update(self._setup_jenkins())
        elif self.config.pipeline_type == "azure":
            integration_files.update(self._setup_azure_pipelines())

        # Generic CI script
        integration_files["ci_script.sh"] = self._generate_ci_script()

        return integration_files

    def install_pre_commit_hooks(self) -> bool:
        """Install pre-commit hooks."""
        if not self.config.pre_commit_enabled:
            self.logger.info("Pre-commit hooks disabled in configuration")
            return True

        return self.hook_manager.install_hooks()

    def _generate_execution_id(self) -> str:
        """Generate unique execution ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{timestamp}_{os.getpid()}"
        hash_suffix = hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest()[
            :8
        ]  # id suffix, not security
        return f"ci_{timestamp}_{hash_suffix}"

    def _get_critical_tests(self) -> List[str]:
        """Get list of critical test files."""
        critical_tests: List[str] = []
        tests_dir = self.project_root / "tests"

        for pattern in self.config.critical_test_patterns or []:
            critical_tests.extend(str(p) for p in tests_dir.glob(pattern))

        return critical_tests

    def _get_all_tests(self) -> List[str]:
        """Get list of all test files."""
        tests_dir = self.project_root / "tests"
        return [str(p) for p in tests_dir.glob("test_*.py")]

    def _run_pytest(self, test_files: List[str], execution_id: str) -> ExecutionResult:
        """Run pytest on specified test files."""
        start_time = datetime.now()

        # Prepare pytest command
        cmd = [
            "python",
            "-m",
            "pytest",
            "--tb=short",
            "--json-report",
            f"--json-report-file=.ci/reports/{execution_id}_report.json",
            "--cov=apgi_framework",
            f"--cov-report=html:.ci/reports/{execution_id}_coverage",
            f"--cov-report=json:.ci/reports/{execution_id}_coverage.json",
            "--maxfail=10",
        ]

        if self.config.parallel_execution:
            cmd.extend(["-n", str(self.config.max_workers)])

        cmd.extend(test_files)

        # Execute pytest
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=self.config.timeout_minutes * 60,
            )

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Parse results
            return self._parse_pytest_results(
                result, execution_id, start_time, end_time, execution_time
            )

        except subprocess.TimeoutExpired:
            self.logger.error("Test execution timed out")
            return self._create_error_result(execution_id, start_time, "Timeout")
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            return self._create_error_result(execution_id, start_time, str(e))

    def _parse_pytest_results(
        self,
        result: subprocess.CompletedProcess,
        execution_id: str,
        start_time: datetime,
        end_time: datetime,
        execution_time: float,
    ) -> ExecutionResult:
        """Parse pytest results into ExecutionResult."""
        # Try to load JSON report
        report_file = self.project_root / ".ci" / "reports" / f"{execution_id}_report.json"
        coverage_file = self.project_root / ".ci" / "reports" / f"{execution_id}_coverage.json"

        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        failed_test_details = []
        coverage_percentage = 0.0

        try:
            if report_file.exists():
                with open(report_file) as f:
                    report_data = json.load(f)

                summary = report_data.get("summary", {})
                total_tests = summary.get("total", 0)
                passed_tests = summary.get("passed", 0)
                failed_tests = summary.get("failed", 0)
                skipped_tests = summary.get("skipped", 0)

                # Extract failure details
                for test in report_data.get("tests", []):
                    if test.get("outcome") == "failed":
                        failed_test_details.append(
                            {
                                "name": test.get("nodeid", ""),
                                "error": test.get("call", {}).get("longrepr", ""),
                                "duration": test.get("duration", 0),
                            }
                        )
        except Exception as e:
            self.logger.warning(f"Failed to parse test report: {e}")

        try:
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                coverage_percentage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
        except Exception as e:
            self.logger.warning(f"Failed to parse coverage report: {e}")

        return ExecutionResult(
            execution_id=execution_id,
            start_time=start_time,
            end_time=end_time,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            coverage_percentage=coverage_percentage,
            execution_time_seconds=execution_time,
            failed_test_details=failed_test_details,
            pipeline_context=self._get_pipeline_context(),
        )

    def _get_pipeline_context(self) -> Dict[str, Any]:
        """Get CI/CD pipeline context information."""
        context = {
            "pipeline_type": self.config.pipeline_type,
            "timestamp": datetime.now().isoformat(),
        }

        # Add environment-specific context
        env_vars = [
            "GITHUB_SHA",
            "GITHUB_REF",
            "GITHUB_ACTOR",
            "GITLAB_CI",
            "CI_COMMIT_SHA",
            "CI_COMMIT_REF_NAME",
            "JENKINS_URL",
            "BUILD_NUMBER",
            "JOB_NAME",
            "AZURE_HTTP_USER_AGENT",
            "BUILD_BUILDID",
        ]

        for var in env_vars:
            if var in os.environ:
                context[var.lower()] = os.environ[var]

        return context

    def _create_empty_result(self, execution_id: str, start_time: datetime) -> ExecutionResult:
        """Create empty test result."""
        return ExecutionResult(
            execution_id=execution_id,
            start_time=start_time,
            end_time=datetime.now(),
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            coverage_percentage=0.0,
            execution_time_seconds=0.0,
            failed_test_details=[],
            pipeline_context=self._get_pipeline_context(),
        )

    def _create_error_result(
        self, execution_id: str, start_time: datetime, error: str
    ) -> ExecutionResult:
        """Create error test result."""
        return ExecutionResult(
            execution_id=execution_id,
            start_time=start_time,
            end_time=datetime.now(),
            total_tests=0,
            passed_tests=0,
            failed_tests=1,
            skipped_tests=0,
            coverage_percentage=0.0,
            execution_time_seconds=0.0,
            failed_test_details=[{"name": "CI_EXECUTION_ERROR", "error": error, "duration": 0}],
            pipeline_context=self._get_pipeline_context(),
        )

    def _generate_ci_reports(self, result: ExecutionResult, execution_id: str) -> None:
        """Generate CI-specific reports."""
        reports_dir = self.project_root / ".ci" / "reports"

        # Generate summary report
        summary_file = reports_dir / f"{execution_id}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        # Generate failure report if there are failures
        if result.failed_tests > 0:
            failure_file = reports_dir / f"{execution_id}_failures.json"
            with open(failure_file, "w") as f:
                json.dump(
                    {
                        "execution_id": execution_id,
                        "failed_tests": result.failed_test_details,
                        "summary": {
                            "total_failures": result.failed_tests,
                            "coverage_below_threshold": result.coverage_percentage
                            < self.config.coverage_threshold,
                        },
                    },
                    f,
                    indent=2,
                )

    def _setup_github_actions(self) -> Dict[str, str]:
        """Setup GitHub Actions workflow."""
        workflow = """
name: APGI Framework CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 2
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-json-report pytest-xdist
    
    - name: Run APGI CI Tests
      run: |
        python -c "
        from apgi_framework.testing.ci_integrator import CIIntegrator, CIConfiguration
        config = CIConfiguration(pipeline_type='github')
        ci = CIIntegrator('.', config)
        impact = ci.analyze_changes('HEAD~1')
        result = ci.execute_ci_tests(impact)
        exit(1 if result.failed_tests > 0 else 0)
        "
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: .ci/reports/
"""
        return {".github/workflows/ci.yml": workflow}

    def _setup_gitlab_ci(self) -> Dict[str, str]:
        """Setup GitLab CI configuration."""
        config = """
stages:
  - test
  - report

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip/
    - venv/

test:
  stage: test
  image: python:3.9
  before_script:
    - python -m venv venv
    - source venv/bin/activate
    - pip install --upgrade pip
    - pip install -r requirements.txt
    - pip install pytest pytest-cov pytest-json-report pytest-xdist
  script:
    - python -c "
      from apgi_framework.testing.ci_integrator import CIIntegrator, CIConfiguration;
      config = CIConfiguration(pipeline_type='gitlab');
      ci = CIIntegrator('.', config);
      impact = ci.analyze_changes('HEAD~1');
      result = ci.execute_ci_tests(impact);
      exit(1 if result.failed_tests > 0 else 0)
      "
  artifacts:
    when: always
    paths:
      - .ci/reports/
    expire_in: 1 week
"""
        return {".gitlab-ci.yml": config}

    def _setup_jenkins(self) -> Dict[str, str]:
        """Setup Jenkins pipeline."""
        pipeline = """
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'python -m venv venv'
                sh '. venv/bin/activate && pip install --upgrade pip'
                sh '. venv/bin/activate && pip install -r requirements.txt'
                sh '. venv/bin/activate && pip install pytest pytest-cov pytest-json-report pytest-xdist'
            }
        }
        
        stage('Test') {
            steps {
                sh '''
                . venv/bin/activate
                python -c "
                from apgi_framework.testing.ci_integrator import CIIntegrator, CIConfiguration
                config = CIConfiguration(pipeline_type='jenkins')
                ci = CIIntegrator('.', config)
                impact = ci.analyze_changes('HEAD~1')
                result = ci.execute_ci_tests(impact)
                exit(1 if result.failed_tests > 0 else 0)
                "
                '''
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: '.ci/reports/**/*', fingerprint: true
        }
    }
}
"""
        return {"Jenkinsfile": pipeline}

    def _setup_azure_pipelines(self) -> Dict[str, str]:
        """Setup Azure Pipelines configuration."""
        config = """
trigger:
- main
- develop

pool:
  vmImage: 'ubuntu-latest'

variables:
  pythonVersion: '3.9'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '$(pythonVersion)'
  displayName: 'Use Python $(pythonVersion)'

- script: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install pytest pytest-cov pytest-json-report pytest-xdist
  displayName: 'Install dependencies'

- script: |
    python -c "
    from apgi_framework.testing.ci_integrator import CIIntegrator, CIConfiguration
    config = CIConfiguration(pipeline_type='azure')
    ci = CIIntegrator('.', config)
    impact = ci.analyze_changes('HEAD~1')
    result = ci.execute_ci_tests(impact)
    exit(1 if result.failed_tests > 0 else 0)
    "
  displayName: 'Run APGI CI Tests'

- task: PublishTestResults@2
  condition: always()
  inputs:
    testResultsFiles: '.ci/reports/*_report.json'
    testRunTitle: 'APGI Framework Tests'

- task: PublishCodeCoverageResults@1
  inputs:
    codeCoverageTool: 'Cobertura'
    summaryFileLocation: '.ci/reports/*_coverage.xml'
"""
        return {"azure-pipelines.yml": config}

    def _generate_ci_script(self) -> str:
        """Generate generic CI script."""
        return f"""#!/bin/bash
# APGI Framework CI Script
# Generic CI/CD integration script

set -e

echo "Starting APGI Framework CI execution..."

# Setup Python environment
if [ ! -d "venv" ]; then
    python -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest pytest-cov pytest-json-report pytest-xdist

# Run CI tests
python -c "
from apgi_framework.testing.ci_integrator import CIIntegrator, CIConfiguration
config = CIConfiguration(
    pipeline_type='generic',
    test_subset_strategy='{self.config.test_subset_strategy}',
    parallel_execution={str(self.config.parallel_execution).lower()},
    max_workers={self.config.max_workers},
    timeout_minutes={self.config.timeout_minutes},
    coverage_threshold={self.config.coverage_threshold}
)
ci = CIIntegrator('.', config)
impact = ci.analyze_changes('HEAD~1')
result = ci.execute_ci_tests(impact)
print(f'Tests: {{result.total_tests}}, Passed: {{result.passed_tests}}, Failed: {{result.failed_tests}}')
print(f'Coverage: {{result.coverage_percentage:.1f}}%')
exit(1 if result.failed_tests > 0 or result.coverage_percentage < {self.config.coverage_threshold} else 0)
"

echo "CI execution completed!"
"""
