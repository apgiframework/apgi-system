"""Exception classes for the APGI Framework.

This module defines the hierarchy of exceptions used throughout the framework
to handle various error conditions in mathematical calculations, simulations,
statistical analysis, and experimental validation.
"""

from typing import Any, Optional


class APGIFrameworkError(Exception):
    """Base exception for all APGI Framework errors."""

    def __init__(
        self, message: str = "An error occurred in the APGI Framework", **context: Any
    ) -> None:
        """Initialize the exception.

        Args:
            message: Error message
            **context: Additional contextual information
        """
        super().__init__(message)
        self.context = context


class MathematicalError(APGIFrameworkError, ValueError):
    """Errors in mathematical calculations and computations."""

    def __init__(
        self,
        message: str = "Mathematical calculation error",
        operation: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Initialize the mathematical error.

        Args:
            message: Error message
            operation: The mathematical operation that failed
            **context: Additional context
        """
        if operation:
            message = f"Mathematical error in operation '{operation}': {message}"
        super().__init__(message, operation=operation, **context)


class SimulationError(APGIFrameworkError):
    """Errors in neural signature simulation and generation."""

    def __init__(
        self,
        message: str = "Neural simulation error",
        simulation_type: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Initialize the simulation error.

        Args:
            message: Error message
            simulation_type: Type of simulation that failed
            **context: Additional context
        """
        if simulation_type:
            message = f"Simulation error in {simulation_type}: {message}"
        super().__init__(message, simulation_type=simulation_type, **context)


class StatisticalError(APGIFrameworkError):
    """Errors in statistical analysis and validation."""

    def __init__(
        self,
        message: str = "Statistical analysis error",
        test_type: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Initialize the statistical error.

        Args:
            message: Error message
            test_type: Type of statistical test that failed
            **context: Additional context
        """
        if test_type:
            message = f"Statistical error in {test_type} test: {message}"
        super().__init__(message, test_type=test_type, **context)


class ValidationError(APGIFrameworkError):
    """Errors in experimental validation and control mechanisms."""

    def __init__(
        self,
        message: str = "Validation error",
        parameter: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Initialize the validation error.

        Args:
            message: Error message
            parameter: Parameter that failed validation
            **context: Additional context
        """
        if parameter:
            message = f"Validation error for parameter '{parameter}': {message}"
        super().__init__(message, parameter=parameter, **context)


class ConfigurationError(APGIFrameworkError):
    """Errors in configuration management and parameter validation."""

    def __init__(
        self,
        message: str = "Configuration error",
        config_file: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Initialize the configuration error.

        Args:
            message: Error message
            config_file: Configuration file that caused the error
            **context: Additional context
        """
        if config_file:
            message = f"Configuration error in {config_file}: {message}"
        super().__init__(message, config_file=config_file, **context)


class DataError(APGIFrameworkError):
    """Errors in data management, storage, and retrieval."""

    def __init__(
        self,
        message: str = "Data management error",
        data_source: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Initialize the data error.

        Args:
            message: Error message
            data_source: Data source that caused the error
            **context: Additional context
        """
        if data_source:
            message = f"Data error from {data_source}: {message}"
        super().__init__(message, data_source=data_source, **context)


class DataManagementError(APGIFrameworkError):
    """Errors in integrated data management operations."""

    def __init__(
        self,
        message: str = "Data management operation error",
        operation: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Initialize the data management error.

        Args:
            message: Error message
            operation: Data operation that failed
            **context: Additional context
        """
        if operation:
            message = f"Data management error in {operation}: {message}"
        super().__init__(message, operation=operation, **context)


class VisualizationError(APGIFrameworkError):
    """Errors in visualization and plotting operations."""

    def __init__(
        self,
        message: str = "Visualization error",
        plot_type: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Initialize the visualization error.

        Args:
            message: Error message
            plot_type: Type of plot that failed
            **context: Additional context
        """
        if plot_type:
            message = f"Visualization error in {plot_type}: {message}"
        super().__init__(message, plot_type=plot_type, **context)


class DashboardError(APGIFrameworkError):
    """Errors in dashboard and monitoring operations."""

    def __init__(
        self,
        message: str = "Dashboard error",
        component: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Initialize the dashboard error.

        Args:
            message: Error message
            component: Dashboard component that failed
            **context: Additional context
        """
        if component:
            message = f"Dashboard error in {component}: {message}"
        super().__init__(message, component=component, **context)


class PersistenceError(APGIFrameworkError):
    """Errors in data persistence and storage operations."""

    def __init__(
        self,
        message: str = "Persistence error",
        storage_type: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Initialize the persistence error.

        Args:
            message: Error message
            storage_type: Type of storage that failed
            **context: Additional context
        """
        if storage_type:
            message = f"Persistence error in {storage_type}: {message}"
        super().__init__(message, storage_type=storage_type, **context)


class ReportGenerationError(APGIFrameworkError):
    """Errors in report generation operations."""

    def __init__(
        self,
        message: str = "Report generation error",
        report_type: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Initialize the report generation error.

        Args:
            message: Error message
            report_type: Type of report that failed to generate
            **context: Additional context
        """
        if report_type:
            message = f"Report generation error for {report_type}: {message}"
        super().__init__(message, report_type=report_type, **context)


class DataExportError(APGIFrameworkError):
    """Errors in data export operations."""

    def __init__(
        self,
        message: str = "Data export error",
        export_format: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Initialize the data export error.

        Args:
            message: Error message
            export_format: Format of export that failed
            **context: Additional context
        """
        if export_format:
            message = f"Data export error for {export_format}: {message}"
        super().__init__(message, export_format=export_format, **context)


class AnalysisError(APGIFrameworkError):
    """Errors in data analysis operations."""

    def __init__(
        self,
        message: str = "Data analysis error",
        analysis_type: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Initialize the analysis error.

        Args:
            message: Error message
            analysis_type: Type of analysis that failed
            **context: Additional context
        """
        if analysis_type:
            message = f"Analysis error in {analysis_type}: {message}"
        super().__init__(message, analysis_type=analysis_type, **context)


class ProcessingError(APGIFrameworkError):
    """Errors in results processing operations."""

    def __init__(
        self,
        message: str = "Results processing error",
        processing_step: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Initialize the processing error.

        Args:
            message: Error message
            processing_step: Processing step that failed
            **context: Additional context
        """
        if processing_step:
            message = f"Processing error at step '{processing_step}': {message}"
        super().__init__(message, processing_step=processing_step, **context)
