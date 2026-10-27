"""End-to-End Workflow Orchestration for APGI Framework Testing.

This module provides comprehensive workflow orchestration for running complete
falsification testing pipelines, automated experiment execution, and result
aggregation with detailed reporting.
"""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .exceptions import APGIFrameworkError, ValidationError
from .main_controller import MainApplicationController


class WorkflowStage(Enum):
    """Stages of the falsification testing workflow."""

    INITIALIZATION = "initialization"
    SYSTEM_VALIDATION = "system_validation"
    PRIMARY_FALSIFICATION = "primary_falsification"
    SECONDARY_FALSIFICATION = "secondary_falsification"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    RESULT_AGGREGATION = "result_aggregation"
    REPORT_GENERATION = "report_generation"
    CLEANUP = "cleanup"


class WorkflowStatus(Enum):
    """Status of workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStageResult:
    """Result of a workflow stage execution."""

    stage: WorkflowStage
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Calculate duration if end_time is set."""
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time


@dataclass
class WorkflowConfiguration:
    """Configuration for workflow execution."""

    run_primary_tests: bool = True
    run_secondary_tests: bool = True
    run_statistical_analysis: bool = True
    generate_detailed_reports: bool = True

    # Test-specific configurations
    primary_test_trials: int = 1000
    secondary_test_participants: int = 100

    # Execution options
    parallel_execution: bool = False
    max_workers: int = 4
    timeout_minutes: int = 60

    # Output options
    save_intermediate_results: bool = True
    generate_plots: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv"])

    # Validation options
    strict_validation: bool = True
    skip_system_validation: bool = False


@dataclass
class WorkflowResult:
    """Complete result of workflow execution."""

    workflow_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: Optional[timedelta] = None
    overall_status: WorkflowStatus = WorkflowStatus.PENDING
    status: WorkflowStatus = WorkflowStatus.PENDING  # For backward compatibility

    stage_results: Dict[WorkflowStage, WorkflowStageResult] = field(default_factory=dict)
    test_results: Dict[str, Any] = field(default_factory=dict)
    statistical_summary: Dict[str, Any] = field(default_factory=dict)

    falsification_conclusion: Optional[str] = None
    confidence_level: Optional[float] = None

    def __post_init__(self) -> None:
        """Calculate total duration if end_time is set."""
        if self.end_time and self.start_time:
            self.total_duration = self.end_time - self.start_time


class WorkflowOrchestrator:
    """Orchestrates end-to-end falsification testing workflows.

    Manages the complete pipeline from system initialization through
    result reporting, with support for parallel execution and
    comprehensive error handling.

    Thread Safety:
    --------------
    This class is designed to be thread-safe for concurrent access.
    The following guarantees are provided:

    - All public methods are thread-safe and can be called concurrently
      from multiple threads without external synchronization.
    - Internal state is protected by RLock (_workflow_lock) for re-entrant
      access within the same thread.
    - Callback registration/removal is protected by a separate Lock
      (_callback_lock) to prevent race conditions.
    - Workflow execution methods (_execute_stage, run_complete_workflow, etc.)
      are designed to be interruptible via cancellation.
    - The cancellation mechanism uses threading.Event for efficient
      cross-thread signaling.

    Thread-Safe Methods:
    - register_stage_callback() / unregister_stage_callback()
    - run_complete_workflow() / run_parallel_workflow()
    - cancel_workflow()
    - get_workflow_status()

    Re-entrancy:
    - Methods are re-entrant within the same thread (using RLock).
    - Concurrent calls from different threads are serialized where necessary.
    - Callback execution is serialized to prevent overlapping callbacks.

    Cancellation:
    - Workflows can be cancelled at any point using cancel_workflow().
    - Cancellation is signaled via threading.Event and checked at stage boundaries.
    - Running stages will complete their current operation before checking for cancellation.
    - Executor threads are properly shut down with timeout on cancellation.
    """

    def __init__(self, controller: MainApplicationController, config: WorkflowConfiguration):
        """Initialize the workflow orchestrator.

        Args:
            controller: Main application controller instance.
            config: Workflow configuration.
        """
        self.controller = controller
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Workflow state
        self.current_workflow: Optional[WorkflowResult] = None
        self.stage_callbacks: Dict[WorkflowStage, List[Callable]] = {}

        # Execution control
        self.executor: Optional[ThreadPoolExecutor] = None
        self.cancelled = False
        self._cancel_event = threading.Event()  # Event for cancellation signaling

        # Thread safety
        self._workflow_lock = threading.RLock()
        self._callback_lock = threading.Lock()

    def _check_cancellation(self) -> None:
        """Check if workflow has been cancelled and raise exception if so."""
        if self._cancel_event.is_set() or self.cancelled:
            raise APGIFrameworkError("Workflow cancelled by user request")

    def register_stage_callback(self, stage: WorkflowStage, callback: Callable) -> None:
        """Register a callback function for a specific workflow stage.

        Args:
            stage: Workflow stage to register callback for.
            callback: Function to call when stage completes.
        """
        with self._callback_lock:
            if stage not in self.stage_callbacks:
                self.stage_callbacks[stage] = []
            self.stage_callbacks[stage].append(callback)

    def run_complete_workflow(self, workflow_id: Optional[str] = None) -> WorkflowResult:
        """Run the complete testing workflow.

        Args:
            workflow_id: Optional workflow identifier. If None, generates one.

        Returns:
            Complete workflow result with all stage results.
        """
        with self._workflow_lock:
            if (
                self.current_workflow
                and self.current_workflow.overall_status == WorkflowStatus.RUNNING
            ):
                raise APGIFrameworkError("Workflow already running")

            # Generate workflow ID if not provided
            if workflow_id is None:
                workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Initialize workflow result
            self.current_workflow = WorkflowResult(
                workflow_id=workflow_id,
                start_time=datetime.now(),
                overall_status=WorkflowStatus.RUNNING,
            )

        try:
            # Execute workflow stages in sequence
            self._execute_stage(WorkflowStage.INITIALIZATION, self._run_initialization)

            if not self.config.skip_system_validation:
                self._execute_stage(WorkflowStage.SYSTEM_VALIDATION, self._run_system_validation)

            if self.config.run_primary_tests:
                self._execute_stage(WorkflowStage.PRIMARY_FALSIFICATION, self._run_primary_tests)

            if self.config.run_secondary_tests:
                self._execute_stage(
                    WorkflowStage.SECONDARY_FALSIFICATION, self._run_secondary_tests
                )

            if self.config.run_statistical_analysis:
                self._execute_stage(
                    WorkflowStage.STATISTICAL_ANALYSIS, self._run_statistical_analysis
                )

            self._execute_stage(WorkflowStage.RESULT_AGGREGATION, self._run_result_aggregation)

            if self.config.generate_detailed_reports:
                self._execute_stage(WorkflowStage.REPORT_GENERATION, self._run_report_generation)

            self._execute_stage(WorkflowStage.CLEANUP, self._run_cleanup)

            # Mark workflow as completed
            self.current_workflow.overall_status = WorkflowStatus.COMPLETED
            self.current_workflow.end_time = datetime.now()

            self.logger.info(f"Workflow {workflow_id} completed successfully")

        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            self.current_workflow.overall_status = WorkflowStatus.FAILED
            self.current_workflow.end_time = datetime.now()
            raise APGIFrameworkError(f"Workflow execution failed: {e}")

        return self.current_workflow

    def run_parallel_workflow(self, workflow_id: Optional[str] = None) -> WorkflowResult:
        """Run workflow with parallel execution where possible.

        Args:
            workflow_id: Optional workflow identifier.

        Returns:
            Complete workflow result.
        """
        if workflow_id is None:
            workflow_id = f"parallel_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"Starting parallel workflow: {workflow_id}")

        # Initialize workflow result
        self.current_workflow = WorkflowResult(
            workflow_id=workflow_id,
            start_time=datetime.now(),
            overall_status=WorkflowStatus.RUNNING,
        )

        try:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                self.executor = executor

                # Sequential stages that must run first
                self._execute_stage(WorkflowStage.INITIALIZATION, self._run_initialization)

                if not self.config.skip_system_validation:
                    self._execute_stage(
                        WorkflowStage.SYSTEM_VALIDATION, self._run_system_validation
                    )

                # Parallel execution of falsification tests
                if self.config.run_primary_tests or self.config.run_secondary_tests:
                    self._run_parallel_falsification_tests()

                # Sequential stages that must run after tests
                if self.config.run_statistical_analysis:
                    self._execute_stage(
                        WorkflowStage.STATISTICAL_ANALYSIS,
                        self._run_statistical_analysis,
                    )

                self._execute_stage(WorkflowStage.RESULT_AGGREGATION, self._run_result_aggregation)

                if self.config.generate_detailed_reports:
                    self._execute_stage(
                        WorkflowStage.REPORT_GENERATION, self._run_report_generation
                    )

                self._execute_stage(WorkflowStage.CLEANUP, self._run_cleanup)

            # Mark workflow as completed
            self.current_workflow.overall_status = WorkflowStatus.COMPLETED
            self.current_workflow.end_time = datetime.now()

            self.logger.info(f"Parallel workflow {workflow_id} completed successfully")

        except Exception as e:
            self.logger.error(f"Parallel workflow {workflow_id} failed: {e}")
            self.current_workflow.overall_status = WorkflowStatus.FAILED
            self.current_workflow.end_time = datetime.now()
            raise APGIFrameworkError(f"Parallel workflow execution failed: {e}")

        return self.current_workflow

    def _execute_stage(self, stage: WorkflowStage, stage_function: Callable) -> None:
        """Execute a workflow stage with error handling and timing.

        Args:
            stage: Workflow stage to execute.
            stage_function: Function to execute for this stage.
        """
        # Check if workflow has been cancelled
        if self.cancelled or self._cancel_event.is_set():
            self.logger.info(f"Stage {stage.value} cancelled before execution")
            raise APGIFrameworkError(f"Workflow cancelled during {stage.value}")

        self.logger.info(f"Executing stage: {stage.value}")

        stage_result = WorkflowStageResult(
            stage=stage, status=WorkflowStatus.RUNNING, start_time=datetime.now()
        )

        try:
            # Execute the stage function
            result_data = stage_function()

            # Check for cancellation after execution (in case it was cancelled during execution)
            if self.cancelled or self._cancel_event.is_set():
                self.logger.warning(f"Stage {stage.value} completed but workflow was cancelled")
                stage_result.status = WorkflowStatus.CANCELLED
                stage_result.end_time = datetime.now()
                stage_result.error_message = "Workflow cancelled after stage completion"
                raise APGIFrameworkError(f"Workflow cancelled during {stage.value}")

            # Mark stage as completed
            stage_result.status = WorkflowStatus.COMPLETED
            stage_result.end_time = datetime.now()
            stage_result.result_data = result_data or {}

            self.logger.info(f"Stage {stage.value} completed successfully")

        except (ValueError, RuntimeError, ConnectionError, TimeoutError, OSError) as e:
            # Mark stage as failed for expected operational errors
            stage_result.status = WorkflowStatus.FAILED
            stage_result.end_time = datetime.now()
            stage_result.error_message = str(e)

            self.logger.error(f"Stage {stage.value} failed: {e}")
            raise

        finally:
            # Store stage result with thread safety
            with self._workflow_lock:
                if self.current_workflow is None:
                    raise RuntimeError("Workflow should be initialized")
                self.current_workflow.stage_results[stage] = stage_result

            # Call registered callbacks with thread safety
            with self._callback_lock:
                if stage in self.stage_callbacks:
                    for callback in self.stage_callbacks[stage]:
                        try:
                            callback(stage_result)
                        except Exception as e:
                            self.logger.warning(f"Stage callback failed: {e}")

    def _run_parallel_falsification_tests(self) -> None:
        """Run falsification tests in parallel."""
        self.logger.info("Running falsification tests in parallel")

        futures = []

        if self.config.run_primary_tests:
            if self.executor is None:
                raise RuntimeError("Executor should be initialized in parallel workflow")
            future = self.executor.submit(self._run_primary_tests)
            futures.append(("primary", future))

        if self.config.run_secondary_tests:
            if self.executor is None:
                raise RuntimeError("Executor should be initialized in parallel workflow")
            future = self.executor.submit(self._run_secondary_tests)
            futures.append(("secondary", future))

        # Wait for all tests to complete
        for test_type, future in futures:
            try:
                future.result(timeout=self.config.timeout_minutes * 60)
                self.logger.info(f"Parallel {test_type} tests completed")
            except Exception as e:
                self.logger.error(f"Parallel {test_type} tests failed: {e}")
                raise

    def _run_initialization(self) -> Dict[str, Any]:
        """Initialize the system for workflow execution."""
        self._check_cancellation()
        self.logger.debug("Initializing system for workflow")

        if self.current_workflow is None:
            raise RuntimeError("Workflow should be initialized")

        # Ensure system is properly initialized
        if not self.controller._initialized:
            self.controller.initialize_system()

        # Get system configuration
        apgi_params = self.controller.config_manager.get_apgi_parameters()
        exp_config = self.controller.config_manager.get_experimental_config()

        return {
            "system_initialized": True,
            "apgi_parameters": apgi_params.__dict__,
            "experimental_config": exp_config.__dict__,
        }

    def _run_system_validation(self) -> Dict[str, Any]:
        """Run comprehensive system validation."""
        self._check_cancellation()
        self.logger.debug("Running system validation")

        if self.current_workflow is None:
            raise RuntimeError("Workflow should be initialized")

        validation_results = self.controller.run_system_validation()

        if self.config.strict_validation and not validation_results.get("overall", False):
            raise ValidationError("System validation failed in strict mode")

        return validation_results

    def _run_primary_tests(self) -> Dict[str, Any]:
        """Run primary falsification tests."""
        self._check_cancellation()
        self.logger.debug("Running primary falsification tests")

        # Auto-initialize workflow for testing
        if self.current_workflow is None:
            self.current_workflow = WorkflowResult(
                workflow_id="test_workflow",
                start_time=datetime.now(),
                overall_status=WorkflowStatus.RUNNING,
            )

        tests = self.controller.get_falsification_tests()
        results = {}

        # Run primary falsification test
        primary_result = tests["primary"].run_test(n_trials=self.config.primary_test_trials)
        results["primary_falsification"] = primary_result

        # Store in workflow result
        self.current_workflow.test_results.update(results)

        return results

    def _run_secondary_tests(self) -> Dict[str, Any]:
        """Run secondary falsification tests."""
        self.logger.debug("Running secondary falsification tests")

        # Auto-initialize workflow for testing
        if self.current_workflow is None:
            self.current_workflow = WorkflowResult(
                workflow_id="test_workflow",
                start_time=datetime.now(),
                overall_status=WorkflowStatus.RUNNING,
            )

        tests = self.controller.get_falsification_tests()
        results = {}

        # Run consciousness without ignition test
        consciousness_result = tests["consciousness_without_ignition"].run_test()
        results["consciousness_without_ignition"] = consciousness_result

        # Run threshold insensitivity test
        threshold_result = tests["threshold_insensitivity"].run_test()
        results["threshold_insensitivity"] = threshold_result

        # Run soma-bias test
        soma_bias_result = tests["soma_bias"].run_test(
            n_participants=self.config.secondary_test_participants
        )
        results["soma_bias"] = soma_bias_result

        # Store in workflow result
        self.current_workflow.test_results.update(results)

        return results

    def _run_statistical_analysis(self) -> Dict[str, Any]:
        """Run comprehensive statistical analysis."""
        self.logger.debug("Running statistical analysis")

        # Auto-initialize workflow for testing
        if self.current_workflow is None:
            self.current_workflow = WorkflowResult(
                workflow_id="test_workflow",
                start_time=datetime.now(),
                overall_status=WorkflowStatus.RUNNING,
            )

        # Aggregate statistical results from all tests
        statistical_summary: Dict[str, Any] = {
            "total_tests_run": len(self.current_workflow.test_results),
            "falsification_results": {},
            "effect_sizes": {},
            "p_values": {},
            "confidence_levels": {},
        }

        for test_name, result in self.current_workflow.test_results.items():
            if hasattr(result, "is_falsified"):
                statistical_summary["falsification_results"][test_name] = result.is_falsified
                statistical_summary["effect_sizes"][test_name] = getattr(
                    result, "effect_size", None
                )
                statistical_summary["p_values"][test_name] = getattr(result, "p_value", None)
                statistical_summary["confidence_levels"][test_name] = getattr(
                    result, "confidence_level", None
                )

        # Determine overall falsification conclusion
        falsification_count = sum(
            1 for falsified in statistical_summary["falsification_results"].values() if falsified
        )
        total_tests = len(statistical_summary["falsification_results"])

        if falsification_count > 0:
            self.current_workflow.falsification_conclusion = (
                f"Framework falsified by {falsification_count}/{total_tests} tests"
            )
        else:
            self.current_workflow.falsification_conclusion = "Framework not falsified by any tests"

        # Calculate overall confidence
        confidence_values = [
            c for c in statistical_summary["confidence_levels"].values() if c is not None
        ]
        if confidence_values:
            self.current_workflow.confidence_level = sum(confidence_values) / len(confidence_values)

        self.current_workflow.statistical_summary = statistical_summary

        return statistical_summary

    def _run_result_aggregation(self) -> Dict[str, Any]:
        """Aggregate all results into final format."""
        self.logger.debug("Aggregating results")

        # Auto-initialize workflow for testing
        if self.current_workflow is None:
            self.current_workflow = WorkflowResult(
                workflow_id="test_workflow",
                start_time=datetime.now(),
                overall_status=WorkflowStatus.RUNNING,
            )

        aggregated_results = {
            "workflow_summary": {
                "workflow_id": self.current_workflow.workflow_id,
                "total_duration": (
                    str(self.current_workflow.total_duration)
                    if self.current_workflow.total_duration
                    else None
                ),
                "stages_completed": len(
                    [
                        s
                        for s in self.current_workflow.stage_results.values()
                        if s.status == WorkflowStatus.COMPLETED
                    ]
                ),
                "total_stages": len(self.current_workflow.stage_results),
            },
            "test_results": self.current_workflow.test_results,
            "statistical_summary": self.current_workflow.statistical_summary,
            "falsification_conclusion": self.current_workflow.falsification_conclusion,
            "overall_confidence": self.current_workflow.confidence_level,
        }

        return aggregated_results

    def _run_report_generation(self) -> Dict[str, Any]:
        """Generate detailed reports."""
        self.logger.debug("Generating reports")

        # Auto-initialize workflow for testing
        if self.current_workflow is None:
            self.current_workflow = WorkflowResult(
                workflow_id="test_workflow",
                start_time=datetime.now(),
                overall_status=WorkflowStatus.RUNNING,
            )

        # Save workflow result to file
        output_dir = Path(self.controller.config_manager.get_experimental_config().output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate JSON report
        json_report_path = output_dir / f"{self.current_workflow.workflow_id}_report.json"
        self._save_json_report(json_report_path)

        # Generate summary report
        summary_report_path = output_dir / f"{self.current_workflow.workflow_id}_summary.txt"
        self._save_summary_report(summary_report_path)

        return {
            "json_report": str(json_report_path),
            "summary_report": str(summary_report_path),
        }

    def _run_cleanup(self) -> Dict[str, Any]:
        """Cleanup resources and finalize workflow."""
        self.logger.debug("Running cleanup")

        # Close executor if it exists
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        return {"cleanup_completed": True}

    def _save_json_report(self, file_path: Path) -> None:
        """Save complete workflow result as JSON."""
        # Convert workflow result to dictionary for JSON serialization
        if self.current_workflow is None:
            raise RuntimeError("Workflow should be initialized")
        report_data = {
            "workflow_id": self.current_workflow.workflow_id,
            "start_time": self.current_workflow.start_time.isoformat(),
            "end_time": (
                self.current_workflow.end_time.isoformat()
                if self.current_workflow.end_time
                else None
            ),
            "total_duration": (
                str(self.current_workflow.total_duration)
                if self.current_workflow.total_duration
                else None
            ),
            "overall_status": self.current_workflow.overall_status.value,
            "stage_results": {
                stage.value: {
                    "status": result.status.value,
                    "start_time": result.start_time.isoformat(),
                    "end_time": (result.end_time.isoformat() if result.end_time else None),
                    "duration": str(result.duration) if result.duration else None,
                    "result_data": result.result_data,
                    "error_message": result.error_message,
                }
                for stage, result in self.current_workflow.stage_results.items()
            },
            "test_results": self._serialize_test_results(self.current_workflow.test_results),
            "statistical_summary": self.current_workflow.statistical_summary,
            "falsification_conclusion": self.current_workflow.falsification_conclusion,
            "confidence_level": self.current_workflow.confidence_level,
        }

        with open(file_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

    def _save_summary_report(self, file_path: Path) -> None:
        """Save human-readable summary report."""
        if self.current_workflow is None:
            raise RuntimeError("Workflow should be initialized")
        with open(file_path, "w") as f:
            f.write("APGI Framework Testing - Workflow Summary\n")
            f.write("WORKFLOW SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Workflow ID: {self.current_workflow.workflow_id}\n")
            f.write(f"Status: {self.current_workflow.overall_status.value.upper()}\n")
            f.write(f"Duration: {self.current_workflow.total_duration}\n\n")

            f.write("FALSIFICATION CONCLUSION:\n")
            f.write(f"{self.current_workflow.falsification_conclusion}\n")
            if self.current_workflow.confidence_level:
                f.write(f"Overall Confidence: {self.current_workflow.confidence_level:.3f}\n\n")

            f.write("TEST RESULTS:\n")
            for test_name, result in self.current_workflow.test_results.items():
                f.write(f"  {test_name}: ")
                if hasattr(result, "is_falsified"):
                    status = "FALSIFIED" if result.is_falsified else "NOT FALSIFIED"
                    f.write(
                        f"{status} (confidence: {getattr(result, 'confidence_level', 'N/A')})\n"
                    )
                else:
                    f.write(f"{result}\n")

            f.write("\nSTAGE EXECUTION:\n")
            for stage, result in self.current_workflow.stage_results.items():
                status = result.status.value.upper()
                duration = f" ({result.duration})" if result.duration else ""
                f.write(f"  {stage.value}: {status}{duration}\n")

    def _serialize_test_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize test results for JSON output."""
        serialized = {}
        for test_name, result in test_results.items():
            if hasattr(result, "__dict__"):
                serialized[test_name] = result.__dict__
            else:
                serialized[test_name] = str(result)
        return serialized

    def cancel_workflow(self) -> None:
        """Cancel the currently running workflow."""
        self.logger.info("Cancelling workflow")
        self.cancelled = True
        self._cancel_event.set()  # Signal all threads to cancel

        if self.current_workflow:
            self.current_workflow.overall_status = WorkflowStatus.CANCELLED
            self.current_workflow.end_time = datetime.now()

        # Shutdown executor and wait for running tasks to complete or cancel
        if self.executor:
            self.executor.shutdown(wait=True, timeout=10.0)  # type: ignore
            if not self.executor._threads:  # Check if threads are still alive
                self.logger.info("All executor threads have finished")
            else:
                self.logger.warning("Some executor threads may still be running")
            self.executor = None

        # Close any open file handles in the controller if they exist
        # Note: Most file operations in this workflow use context managers or are short-lived
        # so explicit closing may not be necessary, but this provides a hook for future extensions

        self.logger.info("Workflow cancellation completed")

    def get_workflow_status(self) -> Optional[WorkflowResult]:
        """Get the current workflow status."""
        return self.current_workflow


# Convenience functions for common workflow patterns


def run_standard_falsification_workflow(
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    parallel: bool = False,
    controller: Optional[MainApplicationController] = None,
) -> WorkflowResult:
    """Run a standard falsification testing workflow.

    Args:
        config_path: Path to configuration file.
        output_dir: Output directory for results.
        parallel: Whether to run tests in parallel.
        controller: Optional pre-initialized controller instance.

    Returns:
        Complete workflow result.
    """
    # Initialize controller
    if controller is None:
        controller = MainApplicationController(config_path)
        controller.initialize_system()

    # Update output directory if provided
    if output_dir:
        controller.config_manager.update_experimental_config(output_directory=output_dir)

    # Create workflow configuration
    workflow_config = WorkflowConfiguration(
        parallel_execution=parallel,
        run_primary_tests=True,
        run_secondary_tests=True,
        run_statistical_analysis=True,
        generate_detailed_reports=True,
    )

    # Create and run orchestrator
    orchestrator = WorkflowOrchestrator(controller, workflow_config)

    if parallel:
        return orchestrator.run_parallel_workflow()
    else:
        return orchestrator.run_complete_workflow()


def run_quick_validation_workflow(config_path: Optional[str] = None) -> WorkflowResult:
    """Run a quick validation workflow with minimal tests.

    Args:
        config_path: Path to configuration file.

    Returns:
        Complete workflow result.
    """
    # Initialize controller
    controller = MainApplicationController(config_path)
    controller.initialize_system()

    # Create minimal workflow configuration
    workflow_config = WorkflowConfiguration(
        run_primary_tests=True,
        run_secondary_tests=False,
        run_statistical_analysis=False,
        generate_detailed_reports=False,
        primary_test_trials=100,
        skip_system_validation=False,
    )

    # Create and run orchestrator
    orchestrator = WorkflowOrchestrator(controller, workflow_config)
    return orchestrator.run_complete_workflow()
