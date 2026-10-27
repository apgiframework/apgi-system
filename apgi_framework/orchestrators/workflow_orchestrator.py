"""Workflow Orchestrator for coordinating multi-domain workflows.

Coordinates research, clinical, and falsification workflows by delegating
to specialized orchestrators.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from .analysis_orchestrator import AnalysisConfig, AnalysisOrchestrator
from .data_orchestrator import DataConfig, DataOrchestrator
from .equation_orchestrator import EquationConfig, EquationOrchestrator

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Supported workflow types."""

    RESEARCH = "research"
    CLINICAL = "clinical"
    FALSIFICATION = "falsification"
    CUSTOM = "custom"


@dataclass
class WorkflowConfig:
    """Configuration for workflow orchestrator."""

    equation_config: EquationConfig = None
    data_config: DataConfig = None
    analysis_config: AnalysisConfig = None

    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.equation_config is None:
            self.equation_config = EquationConfig()
        if self.data_config is None:
            self.data_config = DataConfig()
        if self.analysis_config is None:
            self.analysis_config = AnalysisConfig()


class WorkflowOrchestrator:
    """Orchestrates multi-domain workflows.

    Responsibilities:
    - Coordinate equation, data, and analysis orchestrators
    - Execute research workflows
    - Execute clinical workflows
    - Execute falsification workflows
    - Manage workflow state and transitions
    """

    def __init__(self, config: Optional[WorkflowConfig] = None):
        """Initialize the workflow orchestrator.

        Args:
            config: Configuration for workflow orchestrator
        """
        self.config = config or WorkflowConfig()

        # Initialize sub-orchestrators
        self.equation_orchestrator = EquationOrchestrator(self.config.equation_config)
        self.data_orchestrator = DataOrchestrator(self.config.data_config)
        self.analysis_orchestrator = AnalysisOrchestrator(self.config.analysis_config)

        self._initialized = False
        self._active_workflows: Dict[str, Dict[str, Any]] = {}

        logger.info("WorkflowOrchestrator initialized")

    def initialize(self) -> None:
        """Initialize all sub-orchestrators."""
        if self._initialized:
            logger.warning("WorkflowOrchestrator already initialized")
            return

        try:
            self.equation_orchestrator.initialize()
            self.data_orchestrator.initialize()
            self.analysis_orchestrator.initialize()
            self._initialized = True
            logger.info("WorkflowOrchestrator initialization complete")
        except Exception as e:
            logger.error("Failed to initialize WorkflowOrchestrator: %s", e)
            self.shutdown()
            raise

    def execute_research_workflow(
        self, workflow_id: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a research workflow.

        Args:
            workflow_id: Identifier for the workflow
            parameters: Workflow parameters

        Returns:
            Workflow results
        """
        if not self._initialized:
            raise RuntimeError("WorkflowOrchestrator not initialized")

        logger.info("Starting research workflow: %s", workflow_id)

        try:
            # Step 1: Validate and prepare data
            data = parameters.get("data", {})
            if not self.data_orchestrator.validate_data(data):
                raise ValueError("Invalid input data")

            # Step 2: Execute equations
            equation_results = self.equation_orchestrator.execute_equation(
                parameters.get("equation_id"), parameters.get("equation_params", {})
            )

            # Step 3: Store results
            self.data_orchestrator.store_data(f"{workflow_id}_results", equation_results)

            # Step 4: Run analysis
            analysis_results = self.analysis_orchestrator.execute_analysis(
                workflow_id, equation_results
            )

            # Step 5: Generate report
            report_path = self.analysis_orchestrator.generate_report(analysis_results)

            results = {
                "workflow_id": workflow_id,
                "status": "completed",
                "equation_results": equation_results,
                "analysis_results": analysis_results,
                "report_path": report_path,
            }

            self._active_workflows[workflow_id] = results
            logger.info("Research workflow completed: %s", workflow_id)
            return results

        except Exception as e:
            logger.error("Research workflow failed: %s - %s", workflow_id, e)
            raise

    def execute_clinical_workflow(
        self, workflow_id: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a clinical workflow.

        Args:
            workflow_id: Identifier for the workflow
            parameters: Workflow parameters

        Returns:
            Workflow results
        """
        if not self._initialized:
            raise RuntimeError("WorkflowOrchestrator not initialized")

        logger.info("Starting clinical workflow: %s", workflow_id)

        try:
            # Clinical workflow: similar to research but with clinical-specific steps
            results = self.execute_research_workflow(workflow_id, parameters)
            results["workflow_type"] = "clinical"
            logger.info("Clinical workflow completed: %s", workflow_id)
            return results

        except Exception as e:
            logger.error("Clinical workflow failed: %s - %s", workflow_id, e)
            raise

    def execute_falsification_workflow(
        self, workflow_id: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a falsification workflow.

        Args:
            workflow_id: Identifier for the workflow
            parameters: Workflow parameters

        Returns:
            Workflow results
        """
        if not self._initialized:
            raise RuntimeError("WorkflowOrchestrator not initialized")

        logger.info("Starting falsification workflow: %s", workflow_id)

        try:
            # Falsification workflow: test model predictions against data
            results = self.execute_research_workflow(workflow_id, parameters)
            results["workflow_type"] = "falsification"
            logger.info("Falsification workflow completed: %s", workflow_id)
            return results

        except Exception as e:
            logger.error("Falsification workflow failed: %s - %s", workflow_id, e)
            raise

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the status of a workflow.

        Args:
            workflow_id: Identifier for the workflow

        Returns:
            Workflow status information
        """
        if workflow_id not in self._active_workflows:
            raise ValueError(f"Workflow '{workflow_id}' not found")

        return self._active_workflows[workflow_id]

    def shutdown(self) -> None:
        """Shutdown all sub-orchestrators and cleanup resources."""
        logger.info("Shutting down WorkflowOrchestrator")

        self.equation_orchestrator.shutdown()
        self.data_orchestrator.shutdown()
        self.analysis_orchestrator.shutdown()

        self._active_workflows.clear()
        self._initialized = False

        logger.info("WorkflowOrchestrator shutdown complete")
