"""Analysis Orchestrator for statistical and computational analysis.

Handles analysis pipeline execution, model fitting, and result generation.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for analysis orchestrator."""

    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_gpu_acceleration: bool = False
    result_caching: bool = True


class AnalysisOrchestrator:
    """Orchestrates analysis operations.

    Responsibilities:
    - Initialize analysis engines
    - Execute analysis pipelines
    - Manage statistical tests
    - Generate analysis reports
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize the analysis orchestrator.

        Args:
            config: Configuration for analysis orchestrator
        """
        self.config = config or AnalysisConfig()
        self._analysis_engine: Optional[Any] = None
        self._statistical_tester: Optional[Any] = None
        self._report_generator: Optional[Any] = None
        self._initialized = False

        logger.info("AnalysisOrchestrator initialized with config: %s", self.config)

    def initialize(self) -> None:
        """Initialize analysis systems."""
        if self._initialized:
            logger.warning("AnalysisOrchestrator already initialized")
            return

        try:
            self._initialize_analysis_systems()
            self._initialized = True
            logger.info("AnalysisOrchestrator initialization complete")
        except Exception as e:
            logger.error("Failed to initialize AnalysisOrchestrator: %s", e)
            raise

    def _initialize_analysis_systems(self) -> None:
        """Initialize analysis engines and utilities."""
        from apgi_framework.analysis.analysis_engine import AnalysisEngine
        from apgi_framework.analysis.statistical_tester import StatisticalTester
        from apgi_framework.data.report_generator import ReportGenerator

        self._analysis_engine = AnalysisEngine()
        self._statistical_tester = StatisticalTester()
        self._report_generator = ReportGenerator()

        logger.debug("Analysis systems initialized")

    def execute_analysis(self, analysis_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an analysis pipeline.

        Args:
            analysis_id: Identifier for the analysis
            data: Input data for analysis

        Returns:
            Analysis results
        """
        if not self._initialized:
            raise RuntimeError("AnalysisOrchestrator not initialized")

        logger.info("Starting analysis: %s", analysis_id)
        results = self._analysis_engine.execute(analysis_id, data)
        logger.info("Analysis complete: %s", analysis_id)
        return results

    def run_statistical_test(self, test_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a statistical test.

        Args:
            test_name: Name of the statistical test
            data: Data for the test

        Returns:
            Test results
        """
        if not self._initialized:
            raise RuntimeError("AnalysisOrchestrator not initialized")

        logger.debug("Running statistical test: %s", test_name)
        results = self._statistical_tester.run_test(test_name, data)
        return results

    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate analysis report.

        Args:
            analysis_results: Results from analysis

        Returns:
            Path to generated report
        """
        if not self._initialized:
            raise RuntimeError("AnalysisOrchestrator not initialized")

        report_path = self._report_generator.generate(analysis_results)
        logger.info("Report generated: %s", report_path)
        return report_path

    def shutdown(self) -> None:
        """Shutdown the orchestrator and cleanup resources."""
        self._initialized = False
        logger.info("AnalysisOrchestrator shutdown complete")
