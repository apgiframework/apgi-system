"""Data models for the APGI Framework data management system.

Defines the core data structures for experimental datasets, metadata,
versioning, and backup information.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataVersion:
    """Represents a version of experimental data."""

    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version_number: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    description: str = ""
    parent_version: Optional[str] = None
    checksum: str = ""
    size_bytes: int = 0


@dataclass
class ExperimentMetadata:
    """Metadata for experimental datasets."""

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_name: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    researcher: str = ""
    institution: str = ""

    # Experimental parameters
    n_participants: int = 0
    n_trials: int = 0
    conditions: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Data characteristics
    data_format: str = "hdf5"  # hdf5, sqlite, csv
    file_paths: List[str] = field(default_factory=list)
    total_size_mb: float = 0.0

    # Version information
    current_version: str = "1.0.0"
    version_history: List[DataVersion] = field(default_factory=list)

    # Tags and categories
    tags: List[str] = field(default_factory=list)
    category: str = "falsification_test"

    # Quality metrics
    data_quality_score: float = 1.0
    completeness_percentage: float = 100.0
    validation_status: str = "pending"  # pending, validated, failed


@dataclass
class BackupInfo:
    """Information about data backups."""

    backup_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    backup_path: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    backup_type: str = "full"  # full, incremental, differential
    size_bytes: int = 0
    checksum: str = ""
    compression_ratio: float = 1.0
    retention_days: int = 30
    status: str = "active"  # active, archived, deleted


@dataclass
class ExperimentalDataset:
    """Complete experimental dataset with data and metadata."""

    metadata: ExperimentMetadata
    data: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[Dict[str, Any]] = None
    processed_data: Optional[Dict[str, Any]] = None
    analysis_results: Optional[Dict[str, Any]] = None

    # File system information
    storage_path: Optional[Path] = None
    backup_info: List[BackupInfo] = field(default_factory=list)

    # Access control
    access_permissions: Dict[str, List[str]] = field(default_factory=dict)
    is_locked: bool = False
    lock_reason: str = ""

    def __post_init__(self) -> None:
        """Initialize dataset after creation."""
        if not self.metadata.experiment_id:
            self.metadata.experiment_id = str(uuid.uuid4())

        if not self.metadata.created_at:
            self.metadata.created_at = datetime.now()

        self.metadata.updated_at = datetime.now()


@dataclass
class QueryFilter:
    """Filter criteria for querying experimental datasets."""

    experiment_ids: Optional[List[str]] = None
    researcher: Optional[str] = None
    institution: Optional[str] = None
    date_range: Optional[tuple[datetime, datetime]] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    min_participants: Optional[int] = None
    max_participants: Optional[int] = None
    conditions: Optional[List[str]] = None
    validation_status: Optional[str] = None
    data_format: Optional[str] = None


@dataclass
class StorageStats:
    """Statistics about data storage usage."""

    total_datasets: int = 0
    total_size_mb: float = 0.0
    total_backups: int = 0
    backup_size_mb: float = 0.0
    oldest_dataset: Optional[datetime] = None
    newest_dataset: Optional[datetime] = None
    average_dataset_size_mb: float = 0.0
    storage_efficiency: float = 1.0  # compression ratio

    # By category
    datasets_by_category: Dict[str, int] = field(default_factory=dict)
    size_by_category: Dict[str, float] = field(default_factory=dict)

    # Health metrics
    corrupted_datasets: int = 0
    missing_backups: int = 0
    validation_failures: int = 0


@dataclass
class ParticipantData:
    """Data structure for participant information."""

    participant_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    experiment_id: str = ""

    # Demographics
    age: Optional[int] = None
    gender: Optional[str] = None
    handedness: Optional[str] = None  # left, right, ambidextrous

    # Session information
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    completion_status: str = "incomplete"  # incomplete, complete, aborted

    # Performance metrics
    accuracy: float = 0.0
    reaction_time_ms: float = 0.0
    num_trials: int = 0
    num_correct: int = 0

    # Physiological data references
    eeg_file: Optional[str] = None
    pupillometry_file: Optional[str] = None
    physiological_file: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)

    def complete_session(self, end_time: Optional[datetime] = None) -> None:
        """Mark session as complete."""
        self.end_time = end_time or datetime.now()
        self.completion_status = "complete"


@dataclass
class ExperimentData:
    """Data structure for complete experiment information."""

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_name: str = ""
    description: str = ""

    # Study information
    researcher: str = ""
    institution: str = ""
    study_phase: str = ""  # pilot, main, replication

    # Experimental parameters
    n_participants: int = 0
    n_trials_per_participant: int = 0
    conditions: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Participants
    participants: List[ParticipantData] = field(default_factory=list)

    # Data files
    data_files: Dict[str, str] = field(default_factory=dict)  # participant_id -> file_path

    # Status
    status: str = "created"  # created, running, paused, completed, archived

    # Quality metrics
    overall_accuracy: float = 0.0
    mean_reaction_time_ms: float = 0.0
    data_quality_score: float = 1.0

    def add_participant(self, participant: ParticipantData) -> None:
        """Add a participant to the experiment."""
        participant.experiment_id = self.experiment_id
        self.participants.append(participant)
        self.n_participants = len(self.participants)

    def start_experiment(self) -> None:
        """Mark experiment as started."""
        self.status = "running"
        self.started_at = datetime.now()

    def complete_experiment(self) -> None:
        """Mark experiment as completed."""
        self.status = "completed"
        self.completed_at = datetime.now()
        self._calculate_aggregate_metrics()

    def _calculate_aggregate_metrics(self) -> None:
        """Calculate overall metrics from participant data."""
        if not self.participants:
            return

        total_accuracy = sum(p.accuracy for p in self.participants)
        total_rt = sum(p.reaction_time_ms for p in self.participants)
        total_trials = sum(p.num_trials for p in self.participants)

        self.overall_accuracy = total_accuracy / len(self.participants)
        self.mean_reaction_time_ms = total_rt / len(self.participants)
        self.n_trials_per_participant = total_trials // len(self.participants)
