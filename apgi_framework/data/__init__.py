"""Data management and storage components for the APGI Framework.

This module provides comprehensive data storage, versioning, management,
reporting, visualization, and dashboard capabilities for experimental data,
results, and metadata.
"""

from typing import Any, Dict, Optional

from .allen_integration import AllenLoader, AllenVisualCodingDataset
from .crcns_integration import CRCNSAC1Dataset, CRCNSLoader, CRCNSV1Dataset
from .dashboard import (
    DashboardServer,
    ExperimentComparator,
    ExperimentMonitor,
    create_dashboard,
)
from .data_exporter import DataExporter
from .data_manager import IntegratedDataManager, create_data_manager
from .data_models import (
    BackupInfo,
    DataVersion,
    ExperimentalDataset,
    ExperimentMetadata,
)
from .data_validator import DataValidator
from .experiment_tracker import ExperimentTracker
from .huggingface_integration import HuggingFaceLoader, NeuromorphicDataset
from .migration_manager import MigrationManager, create_migration_manager
from .parameter_estimation_dao import (
    ParameterEstimationDAO,
    create_parameter_estimation_dao,
)
from .parameter_estimation_models import (
    BehavioralResponse,
    DetectionTrialResult,
    HeartbeatTrialResult,
    ModelFitMetrics,
    OddballTrialResult,
    ParameterDistribution,
    ParameterEstimates,
    QualityMetrics,
    ReliabilityMetrics,
    SessionData,
    StimulusModality,
    TaskType,
    TrialData,
)
from .parameter_estimation_schema import (
    ParameterEstimationSchema,
    create_parameter_estimation_schema,
)
from .persistence_layer import PersistenceLayer

# New reporting and visualization components
from .report_generator import FalsificationReport, ReportGenerator, ReportSection
from .storage_manager import StorageManager

# New external dataset integration modules
from .unified_data_loader import (
    LoadedDataset,
    UnifiedDataLoader,
    load_allen_dataset,
    load_crcns_dataset,
    load_huggingface_dataset,
)
from .visualizer import APGIVisualizer, InteractiveVisualizer

# Import example data loading functionality


# Mock classes for testing
class DataProcessor:
    """Mock data processor for testing purposes."""

    def __init__(self) -> None:
        self.processed_data: Dict[str, Any] = {}

    def process_data(self, raw_data: Any) -> Dict[str, Any]:
        """Process raw experimental data."""
        data_id = f"processed_{hash(str(raw_data)) % 10000:04d}"
        result: Dict[str, Any] = {
            "data_id": data_id,
            "processed": True,
            "raw_data": raw_data,
            "processing_timestamp": "2024-01-01T00:00:00Z",
        }
        self.processed_data[data_id] = result
        return result

    def get_processed_data(self, data_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve processed data by ID."""
        return self.processed_data.get(data_id)


class MultiModalProcessor:
    """Mock multi-modal processor for testing purposes."""

    def __init__(self) -> None:
        self.modalities: Dict[str, Any] = {}

    def add_modality(self, modality_name: str, data: Any) -> None:
        """Add a data modality."""
        self.modalities[modality_name] = data

    def process_modalities(self) -> Dict[str, Any]:
        """Process all modalities together."""
        integrated_data = {
            "integrated": True,
            "modalities": list(self.modalities.keys()),
            "integration_timestamp": "2024-01-01T00:00:00Z",
        }
        return integrated_data

    def get_modality_data(self, modality_name: str) -> Optional[Any]:
        """Get data for a specific modality."""
        return self.modalities.get(modality_name)


__all__ = [
    # Core data management
    "StorageManager",
    "ExperimentalDataset",
    "ExperimentMetadata",
    "DataVersion",
    "BackupInfo",
    "PersistenceLayer",
    "DataValidator",
    "ExperimentTracker",
    # Parameter estimation models
    "TaskType",
    "StimulusModality",
    "BehavioralResponse",
    "QualityMetrics",
    "TrialData",
    "DetectionTrialResult",
    "HeartbeatTrialResult",
    "OddballTrialResult",
    "ParameterDistribution",
    "ModelFitMetrics",
    "ReliabilityMetrics",
    "ParameterEstimates",
    "SessionData",
    # Parameter estimation database
    "ParameterEstimationSchema",
    "create_parameter_estimation_schema",
    "ParameterEstimationDAO",
    "create_parameter_estimation_dao",
    "MigrationManager",
    "create_migration_manager",
    # Reporting and visualization
    "ReportGenerator",
    "FalsificationReport",
    "ReportSection",
    "DataExporter",
    "APGIVisualizer",
    "InteractiveVisualizer",
    # Dashboard and monitoring
    "DashboardServer",
    "ExperimentMonitor",
    "ExperimentComparator",
    "create_dashboard",
    # Integrated management
    "IntegratedDataManager",
    "create_data_manager",
    # External dataset integration
    "UnifiedDataLoader",
    "LoadedDataset",
    "load_crcns_dataset",
    "load_allen_dataset",
    "load_huggingface_dataset",
    "CRCNSLoader",
    "CRCNSV1Dataset",
    "CRCNSAC1Dataset",
    "AllenLoader",
    "AllenVisualCodingDataset",
    "HuggingFaceLoader",
    "NeuromorphicDataset",
    # Mock classes for testing
    "DataProcessor",
    "MultiModalProcessor",
]
