"""Persistence layer for the APGI Framework data management system.

Provides unified interface for data storage using SQLite and HDF5 backends.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import h5py
import numpy as np
import pandas as pd

from ..exceptions import APGIFrameworkError

logger = logging.getLogger(__name__)
from ..security.secure_pickle import safe_pickle_dump, safe_pickle_load
from .data_models import (
    BackupInfo,
    DataVersion,
    ExperimentalDataset,
    ExperimentMetadata,
)


class PersistenceError(APGIFrameworkError):
    """Errors in data persistence operations."""


class PersistenceLayer:
    """Unified persistence layer supporting SQLite and HDF5 storage.

    Provides data storage, retrieval, versioning, and backup capabilities
    with automatic format detection and conversion.
    """

    def __init__(self, storage_path: Union[str, Path], backend: str = "hdf5"):
        """Initialize persistence layer with specified backend."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.backend = backend

        # Backend availability flags
        self.postgresql_available = False
        self.hdf5_available = True
        self.sqlite_available = True

        # Create storage directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.storage_path / "metadata"
        self.data_path = self.storage_path / "data"
        self.backup_path = self.storage_path / "backups"

        for path in [self.metadata_path, self.data_path, self.backup_path]:
            path.mkdir(exist_ok=True)

        # Initialize backends
        if backend == "hdf5":
            self._init_hdf5()
        elif backend == "hybrid":
            # Hybrid uses HDF5 for data and SQLite for metadata
            self._init_hdf5()
            self._init_sqlite()
        elif backend == "postgresql":
            self._init_postgresql()
        elif backend == "sqlite":
            self._init_sqlite()
        else:
            raise PersistenceError(f"Unknown backend: {backend}")

    def _validate_experiment_id(self, experiment_id: str) -> None:
        """Validate experiment_id to prevent path traversal."""
        if not re.match(r"^[a-zA-Z0-9_\-]+$", experiment_id):
            raise PersistenceError(
                f"Invalid experiment_id: {experiment_id}. Must contain only alphanumeric characters, underscores, and hyphens"
            )

    def _init_sqlite(self) -> None:
        """Initialize SQLite database for metadata storage."""
        self.db_path = self.metadata_path / "experiments.db"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    experiment_name TEXT,
                    description TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    researcher TEXT,
                    institution TEXT,
                    n_participants INTEGER,
                    n_trials INTEGER,
                    conditions TEXT,
                    parameters TEXT,
                    data_format TEXT,
                    file_paths TEXT,
                    total_size_mb REAL,
                    current_version TEXT,
                    tags TEXT,
                    category TEXT,
                    data_quality_score REAL,
                    completeness_percentage REAL,
                    validation_status TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    version_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    version_number TEXT,
                    created_at TEXT,
                    created_by TEXT,
                    description TEXT,
                    parent_version TEXT,
                    checksum TEXT,
                    size_bytes INTEGER,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS backups (
                    backup_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    backup_path TEXT,
                    created_at TEXT,
                    backup_type TEXT,
                    size_bytes INTEGER,
                    checksum TEXT,
                    compression_ratio REAL,
                    retention_days INTEGER,
                    status TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)

            conn.commit()

    def _init_hdf5(self) -> None:
        """Initialize HDF5 storage structure."""
        self.hdf5_path = self.data_path / "experiments.h5"

        # Create initial HDF5 structure if it doesn't exist
        if not self.hdf5_path.exists():
            with h5py.File(self.hdf5_path, "w") as f:
                f.attrs["created_at"] = datetime.now().isoformat()
                f.attrs["version"] = "1.0.0"
                f.attrs["description"] = "APGI Framework Experimental Data Storage"

    def _init_postgresql(self) -> None:
        """Initialize PostgreSQL database for large-scale data storage."""
        try:
            # Import psycopg2 only when needed (BUG-015)
            import psycopg2
            from psycopg2.extras import Json

            # Store as instance attributes for use in other methods
            self._psycopg2 = psycopg2
            self._psycopg2_extras_Json = Json
            self._psycopg2_extras_RealDictCursor = psycopg2.extras.RealDictCursor

            # Get database connection parameters from environment or config
            self.db_host = os.getenv("POSTGRES_HOST", "localhost")
            self.db_port = int(os.getenv("POSTGRES_PORT", "5432"))
            self.db_name = os.getenv("POSTGRES_DB", "apgi_experiments")
            self.db_user = os.getenv("POSTGRES_USER", "apgi_user")
            self.db_password = os.getenv("POSTGRES_PASSWORD", "")

            # Test connection
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
            )
            conn.close()

            # Mark PostgreSQL as available
            self.postgresql_available = True

            # Create tables if they don't exist
            self._create_postgresql_tables()

        except (ImportError, Exception) as e:
            # Graceful degradation: Fall back to SQLite if PostgreSQL unavailable (BUG-012)
            self.postgresql_available = False
            logger.warning(f"PostgreSQL unavailable, falling back to SQLite: {e}")
            self._init_sqlite()

    def _create_postgresql_tables(self) -> None:
        """Create necessary tables in PostgreSQL database."""
        if not self.postgresql_available or self._psycopg2 is None:
            return

        with self._psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
        ) as conn:
            with conn.cursor() as cursor:
                # Experiments table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS experiments (
                        experiment_id VARCHAR(255) PRIMARY KEY,
                        experiment_name VARCHAR(255),
                        description TEXT,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        researcher VARCHAR(255),
                        institution VARCHAR(255),
                        n_participants INTEGER,
                        n_trials INTEGER,
                        conditions JSONB,
                        parameters JSONB,
                        data_format VARCHAR(50),
                        file_paths JSONB,
                        total_size_mb REAL,
                        current_version VARCHAR(50),
                        tags JSONB,
                        category VARCHAR(100),
                        data_quality_score REAL,
                        completeness_percentage REAL,
                        validation_status VARCHAR(50)
                    )
                """)

                # Versions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS versions (
                        version_id VARCHAR(255) PRIMARY KEY,
                        experiment_id VARCHAR(255) REFERENCES experiments(experiment_id),
                        version_number VARCHAR(50),
                        created_at TIMESTAMP,
                        created_by VARCHAR(255),
                        description TEXT,
                        parent_version VARCHAR(255),
                        checksum VARCHAR(255),
                        size_bytes BIGINT
                    )
                """)

                # Backups table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS backups (
                        backup_id VARCHAR(255) PRIMARY KEY,
                        experiment_id VARCHAR(255) REFERENCES experiments(experiment_id),
                        backup_path TEXT,
                        created_at TIMESTAMP,
                        backup_type VARCHAR(50),
                        size_bytes BIGINT,
                        checksum VARCHAR(255),
                        compression_ratio REAL,
                        retention_days INTEGER,
                        status VARCHAR(50)
                    )
                """)

                # Data chunks table for large datasets
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_chunks (
                        chunk_id SERIAL PRIMARY KEY,
                        experiment_id VARCHAR(255) REFERENCES experiments(experiment_id),
                        data_type VARCHAR(50),
                        chunk_index INTEGER,
                        total_chunks INTEGER,
                        data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Indexes for performance
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_data_chunks_experiment ON data_chunks(experiment_id, data_type)"
                )

            conn.commit()

    def store_dataset(self, dataset: ExperimentalDataset) -> str:
        """Store experimental dataset with metadata.

        Args:
            dataset: ExperimentalDataset to store

        Returns:
            str: Experiment ID of stored dataset
        """
        try:
            # Store metadata in SQLite
            self._store_metadata(dataset.metadata)

            # Store data based on backend
            if self.backend == "sqlite":
                self._store_data_sqlite(dataset)
            elif self.backend == "hdf5":
                self._store_data_hdf5(dataset)
            elif self.backend == "hybrid":
                # Store large numerical data in HDF5, metadata in SQLite
                self._store_data_hdf5(dataset)
            elif self.backend == "postgresql":
                self._store_data_postgresql(dataset)

            # Create initial version
            version = DataVersion(
                version_number="1.0.0",
                description="Initial dataset version",
                checksum=self._calculate_checksum(dataset),
            )
            self._store_version(dataset.metadata.experiment_id, version)

            return dataset.metadata.experiment_id

        except (IOError, OSError, ValueError, KeyError, TypeError) as e:
            raise PersistenceError(f"Failed to store dataset: {str(e)}")

    def load_dataset(
        self, experiment_id: str, version: Optional[str] = None
    ) -> ExperimentalDataset:
        """Load experimental dataset by ID.

        Args:
            experiment_id: Experiment identifier
            version: Specific version to load (latest if None)

        Returns:
            ExperimentalDataset: Loaded dataset
        """
        try:
            # Load metadata
            metadata = self._load_metadata(experiment_id)
            if not metadata:
                raise PersistenceError(f"Experiment {experiment_id} not found")

            # Load data based on backend
            data: dict[str, Any] | None = {}
            if self.backend == "sqlite":
                data = self._load_data_sqlite(experiment_id, version)
            elif self.backend in ["hdf5", "hybrid"]:
                data = self._load_data_hdf5(experiment_id, version)
            elif self.backend == "postgresql":
                data = self._load_data_postgresql(experiment_id, version)
            else:
                raise PersistenceError(f"Unsupported backend for loading: {self.backend}")

            # Ensure data is not None before using .get()
            if data is None:
                raise PersistenceError(f"Failed to load data for experiment {experiment_id}")

            # Load backup information
            backup_info = self._load_backup_info(experiment_id)

            dataset = ExperimentalDataset(
                metadata=metadata,
                data=data.get("data", {}),
                raw_data=data.get("raw_data"),
                processed_data=data.get("processed_data"),
                analysis_results=data.get("analysis_results"),
                backup_info=backup_info,
            )

            return dataset

        except (IOError, OSError, ValueError, KeyError, TypeError) as e:
            raise PersistenceError(f"Failed to load dataset {experiment_id}: {str(e)}")

    def store_data(
        self,
        experiment_id: str,
        data: Union[Dict[str, Any], pd.DataFrame],
        format: str = "hdf5",
    ) -> None:
        """Store data for an experiment (convenience method)."""
        self._validate_experiment_id(experiment_id)
        try:
            file_path = self.data_path / f"{experiment_id}.h5"
            with h5py.File(file_path, "w") as f:
                if isinstance(data, pd.DataFrame):
                    # Store DataFrame in HDF5
                    self._store_dataframe_hdf5(f, "data", data)
                elif isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, np.ndarray):
                            f.create_dataset(key, data=value)
                        elif isinstance(value, pd.DataFrame):
                            self._store_dataframe_hdf5(f, key, value)
                        else:
                            try:
                                f.attrs[key] = value
                            except (TypeError, ValueError):
                                # Fallback to JSON serialization for complex objects
                                f.attrs[key] = json.dumps(value, default=str)
                else:
                    raise PersistenceError(f"Unsupported data type: {type(data)}")
        except (IOError, OSError, ValueError, TypeError) as e:
            raise PersistenceError(f"Failed to store data: {str(e)}")

    def _store_dataframe_hdf5(self, group: h5py.Group, key: str, df: pd.DataFrame) -> None:
        """Store a pandas DataFrame in HDF5."""
        df_group = group.create_group(key)
        # Store column names
        df_group.attrs["columns"] = json.dumps(list(df.columns), default=str)
        df_group.attrs["index"] = json.dumps(list(df.index), default=str)
        df_group.attrs["dtype"] = str(df.dtypes.iloc[0]) if len(df.columns) > 0 else "object"

        # Store each column as a separate dataset
        for col in df.columns:
            col_data = df[col].values
            if col_data.dtype == "O":  # Object dtype - convert to strings
                col_data = np.array([str(x) for x in col_data])
            try:
                df_group.create_dataset(col, data=col_data)
            except (TypeError, ValueError):
                # Fallback: store as JSON
                df_group.attrs[col] = json.dumps(df[col].tolist(), default=str)

    def store_numpy_array(self, experiment_id: str, dataset_name: str, array: np.ndarray) -> None:
        """Store a numpy array for an experiment."""
        self._validate_experiment_id(experiment_id)
        try:
            file_path = self.data_path / f"{experiment_id}.h5"
            with h5py.File(file_path, "a") as f:
                if dataset_name in f:
                    del f[dataset_name]
                f.create_dataset(dataset_name, data=array)
        except (IOError, OSError, ValueError) as e:
            raise PersistenceError(f"Failed to store numpy array: {str(e)}")

    def create_version(
        self, experiment_id: str, version_number: str, description: str = ""
    ) -> DataVersion:
        """Create a new version for an experiment."""
        self._validate_experiment_id(experiment_id)
        try:
            data_version = DataVersion(
                version_id=f"{experiment_id}_{version_number}",
                version_number=version_number,
                description=description,
            )
            self._store_version(experiment_id, data_version)
            # Attach experiment_id for compatibility with tests
            setattr(data_version, "experiment_id", experiment_id)
            return data_version
        except (IOError, OSError, ValueError, TypeError) as e:
            raise PersistenceError(f"Failed to create version: {str(e)}")

    def list_versions(self, experiment_id: str) -> List[DataVersion]:
        """List all versions for an experiment."""
        self._validate_experiment_id(experiment_id)
        try:
            return self._load_versions(experiment_id)
        except (IOError, OSError, ValueError) as e:
            raise PersistenceError(f"Failed to list versions: {str(e)}")

    def get_version(self, experiment_id: str, version_number: str) -> Optional[DataVersion]:
        """Get a specific version for an experiment."""
        self._validate_experiment_id(experiment_id)
        try:
            versions = self._load_versions(experiment_id)
            for version in versions:
                if version.version_number == version_number:
                    return version
            return None
        except (IOError, OSError, ValueError, AttributeError) as e:
            raise PersistenceError(f"Failed to get version: {str(e)}")

    def store_backup(self, experiment_id: str, backup_type: str = "full") -> BackupInfo:
        """Create a backup for an experiment (alias for create_backup)."""
        return self.create_backup(experiment_id, backup_type)

    def restore_backup(self, backup_id: str) -> str:
        """Restore an experiment from a backup."""
        try:
            # First find the backup info from database to get the backup path
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT backup_path, experiment_id FROM backups WHERE backup_id = ?",
                    (backup_id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise PersistenceError(f"Backup {backup_id} not found in database")
                backup_path = Path(row["backup_path"])
                experiment_id = row["experiment_id"]

            if not backup_path.exists():
                raise PersistenceError(f"Backup directory not found: {backup_path}")

            # Restore experiment-specific data file if it exists
            exp_backup_file = backup_path / f"{experiment_id}.h5"
            if exp_backup_file.exists():
                target_file = self.data_path / f"{experiment_id}.h5"
                shutil.copy2(exp_backup_file, target_file)

            # Restore main experiments file if it exists
            main_backup = backup_path / "experiments.h5"
            if main_backup.exists() and self.hdf5_path.exists():
                # For safety, don't overwrite the whole file - individual experiment groups should be handled separately
                pass

            # Restore database if needed
            db_backup = backup_path / "experiments.db"
            if db_backup.exists():
                shutil.copy2(db_backup, self.db_path)

            return str(experiment_id)
        except (IOError, OSError, ValueError, FileNotFoundError) as e:
            raise PersistenceError(f"Failed to restore backup: {str(e)}")

    def save_metadata(
        self, experiment_id: str, metadata: Union[Dict[str, Any], ExperimentMetadata]
    ) -> None:
        """Save metadata for an experiment."""
        self._validate_experiment_id(experiment_id)
        try:
            # Save to JSON file for backward compatibility
            metadata_file = self.metadata_path / f"{experiment_id}.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            # Also save to SQLite so list_experiments works
            if isinstance(metadata, dict):
                metadata_obj = ExperimentMetadata(
                    experiment_id=experiment_id,
                    experiment_name=metadata.get("experiment_name", ""),
                    description=metadata.get("description", ""),
                    created_at=metadata.get("created_at", datetime.now()),
                    updated_at=metadata.get("updated_at", datetime.now()),
                    researcher=metadata.get("researcher", ""),
                    institution=metadata.get("institution", ""),
                    n_participants=metadata.get("n_participants", 0),
                    n_trials=metadata.get("n_trials", 0),
                    conditions=metadata.get("conditions", []),
                    parameters=metadata.get("parameters", {}),
                    data_format=metadata.get("data_format", "hdf5"),
                    file_paths=metadata.get("file_paths", []),
                    total_size_mb=metadata.get("total_size_mb", 0.0),
                    current_version=metadata.get("current_version", "1.0.0"),
                    tags=metadata.get("tags", []),
                    category=metadata.get("category", ""),
                    data_quality_score=metadata.get("data_quality_score", 0.0),
                    completeness_percentage=metadata.get("completeness_percentage", 0.0),
                    validation_status=metadata.get("validation_status", "unknown"),
                )
            else:
                metadata_obj = metadata

            self._store_metadata(metadata_obj)
        except (IOError, OSError, ValueError, TypeError) as e:
            raise PersistenceError(f"Failed to save metadata: {str(e)}")

    def get_metadata(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Get metadata for an experiment."""
        self._validate_experiment_id(experiment_id)
        try:
            # First try to load from SQLite database
            metadata = self._load_metadata(experiment_id)
            if metadata:
                return metadata

            # Fallback to JSON file if exists
            metadata_file = self.metadata_path / f"{experiment_id}.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                    # Convert dict to ExperimentMetadata
                    if isinstance(data, dict):
                        # Handle datetime fields
                        for dt_field in ["created_at", "updated_at"]:
                            if dt_field in data and isinstance(data[dt_field], str):
                                data[dt_field] = datetime.fromisoformat(data[dt_field])
                        return ExperimentMetadata(**data)
                return None
            return None
        except (IOError, OSError, ValueError, KeyError) as e:
            raise PersistenceError(f"Failed to get metadata: {str(e)}")

    def update_metadata(self, experiment_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for an experiment."""
        self._validate_experiment_id(experiment_id)
        try:
            existing_metadata = self.get_metadata(experiment_id)
            existing: Dict[str, Any]
            if existing_metadata is None:
                existing = {}
            else:
                # Convert ExperimentMetadata to dict
                existing = {
                    "experiment_id": existing_metadata.experiment_id,
                    "experiment_name": existing_metadata.experiment_name,
                    "description": existing_metadata.description,
                    "created_at": existing_metadata.created_at,
                    "updated_at": existing_metadata.updated_at,
                    "researcher": existing_metadata.researcher,
                    "institution": existing_metadata.institution,
                    "n_participants": existing_metadata.n_participants,
                    "n_trials": existing_metadata.n_trials,
                    "conditions": existing_metadata.conditions,
                    "parameters": existing_metadata.parameters,
                    "data_format": existing_metadata.data_format,
                    "file_paths": existing_metadata.file_paths,
                    "total_size_mb": existing_metadata.total_size_mb,
                    "current_version": existing_metadata.current_version,
                    "tags": existing_metadata.tags,
                    "category": existing_metadata.category,
                    "data_quality_score": existing_metadata.data_quality_score,
                    "completeness_percentage": existing_metadata.completeness_percentage,
                    "validation_status": existing_metadata.validation_status,
                }
            existing.update(metadata)
            self.save_metadata(experiment_id, existing)
        except (IOError, OSError, ValueError, TypeError, AttributeError) as e:
            raise PersistenceError(f"Failed to update metadata: {str(e)}")

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment and all its data."""
        self._validate_experiment_id(experiment_id)
        try:
            # Delete data file
            data_file = self.data_path / f"{experiment_id}.h5"
            if data_file.exists():
                data_file.unlink()

            # Delete metadata file
            metadata_file = self.metadata_path / f"{experiment_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()

            # Delete from SQLite
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
                conn.commit()
        except (IOError, OSError, sqlite3.Error) as e:
            raise PersistenceError(f"Failed to delete experiment: {str(e)}")

    def delete_version(self, version_id: str) -> None:
        """Delete a specific version by its version_id."""
        try:
            # Extract experiment_id from version_id (format: experiment_id_version)
            # Find and delete the version from the database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM versions WHERE version_id = ?", (version_id,))
                conn.commit()
        except (IOError, OSError, sqlite3.Error) as e:
            raise PersistenceError(f"Failed to delete version: {str(e)}")

    def retrieve_data(self, experiment_id: str, format: str = "hdf5") -> Optional[pd.DataFrame]:
        """Retrieve data for an experiment."""
        self._validate_experiment_id(experiment_id)
        try:
            file_path = self.data_path / f"{experiment_id}.h5"
            if not file_path.exists():
                raise PersistenceError(f"Experiment {experiment_id} not found")

            # Load from HDF5
            with h5py.File(file_path, "r") as f:
                # Check if data is stored as a DataFrame group
                if "data" in f and isinstance(f["data"], h5py.Group):
                    df_group = f["data"]
                    # Load columns from attrs
                    columns = json.loads(df_group.attrs.get("columns", "[]"))
                    # Load data from datasets
                    data_dict = {}
                    for col in columns:
                        if col in df_group:
                            data_dict[col] = df_group[col][:]
                        elif col in df_group.attrs:
                            # Column stored as JSON
                            data_dict[col] = json.loads(df_group.attrs[col])
                    return pd.DataFrame(data_dict)
                else:
                    # Legacy format: simple key-value storage
                    data_dict = {}
                    for key in f.keys():
                        data_dict[key] = f[key][:]
                    for key in f.attrs.keys():
                        data_dict[key] = f.attrs[key]

                    if data_dict:
                        return pd.DataFrame(data_dict)
                    raise PersistenceError(f"No data found for experiment {experiment_id}")
        except PersistenceError:
            raise
        except (IOError, OSError, ValueError, KeyError) as e:
            raise PersistenceError(f"Failed to retrieve data: {str(e)}")

    def retrieve_numpy_array(self, experiment_id: str, dataset_name: str) -> Optional[np.ndarray]:
        """Retrieve a numpy array for an experiment."""
        self._validate_experiment_id(experiment_id)
        try:
            file_path = self.data_path / f"{experiment_id}.h5"
            if not file_path.exists():
                return None

            with h5py.File(file_path, "r") as f:
                if dataset_name in f:
                    return f[dataset_name][:]  # type: ignore
                return None
        except (IOError, OSError, KeyError) as e:
            raise PersistenceError(f"Failed to retrieve numpy array: {str(e)}")

    def export_to_csv(self, experiment_id: str, file_path: str) -> None:
        """Export experiment data to CSV."""
        self._validate_experiment_id(experiment_id)
        try:
            dataset = self.load_dataset(experiment_id)
            if dataset.data:
                df = pd.DataFrame(dataset.data)
                df.to_csv(file_path, index=False)
        except (IOError, OSError, ValueError) as e:
            raise PersistenceError(f"Failed to export to CSV: {str(e)}")

    def export_to_json(self, experiment_id: str, file_path: str) -> None:
        """Export experiment data to JSON."""
        self._validate_experiment_id(experiment_id)
        try:
            dataset = self.load_dataset(experiment_id)
            with open(file_path, "w") as f:
                json.dump(dataset.data, f, indent=2, default=str)
        except (IOError, OSError, ValueError, TypeError) as e:
            raise PersistenceError(f"Failed to export to JSON: {str(e)}")

    def import_from_csv(self, file_path: str, experiment_id: str) -> None:
        """Import experiment data from CSV."""
        self._validate_experiment_id(experiment_id)
        try:
            df = pd.read_csv(file_path)
            self.store_data(experiment_id, {str(k): v for k, v in df.to_dict("list").items()})
        except (
            IOError,
            OSError,
            ValueError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as e:
            raise PersistenceError(f"Failed to import from CSV: {str(e)}")

    def list_backups(self, experiment_id: str) -> List[BackupInfo]:
        """List all backups for an experiment."""
        self._validate_experiment_id(experiment_id)
        try:
            return self._load_backup_info(experiment_id)
        except (IOError, OSError, ValueError) as e:
            raise PersistenceError(f"Failed to list backups: {str(e)}")

    def export_data(self, experiment_id: str, file_path: str, format: str = "csv") -> None:
        """Export experiment data to a file."""
        self._validate_experiment_id(experiment_id)
        try:
            data = self.retrieve_data(experiment_id)
            if data is None:
                raise PersistenceError(f"No data found for experiment {experiment_id}")

            export_path = Path(file_path)
            if format.lower() == "csv":
                data.to_csv(export_path, index=False)
            elif format.lower() == "json":
                data.to_json(export_path, orient="records", indent=2)
            else:
                raise PersistenceError(f"Unsupported export format: {format}")
        except (IOError, OSError, ValueError, TypeError) as e:
            raise PersistenceError(f"Failed to export data: {str(e)}")

    def import_data(self, experiment_id: str, file_path: str, format: str = "csv") -> None:
        """Import experiment data from a file."""
        self._validate_experiment_id(experiment_id)
        try:
            import_path = Path(file_path)
            if format.lower() == "csv":
                data = pd.read_csv(import_path)
            elif format.lower() == "json":
                data = pd.read_json(import_path)
            else:
                raise PersistenceError(f"Unsupported import format: {format}")

            self.store_data(experiment_id, data)
        except (
            IOError,
            OSError,
            ValueError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as e:
            raise PersistenceError(f"Failed to import data: {str(e)}")

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            total_size = 0
            experiment_count = 0

            for file in self.data_path.glob("*.h5"):
                total_size += file.stat().st_size
                experiment_count += 1

            return {
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "total_experiments": experiment_count,
                "backend": self.backend,
                "data_path": str(self.data_path),
            }
        except (IOError, OSError) as e:
            raise PersistenceError(f"Failed to get storage stats: {str(e)}")

    def get_experiment_size(self, experiment_id: str) -> int:
        """Get the size of an experiment in bytes."""
        self._validate_experiment_id(experiment_id)
        try:
            data_file = self.data_path / f"{experiment_id}.h5"
            if data_file.exists():
                return data_file.stat().st_size
            return 0
        except (IOError, OSError) as e:
            raise PersistenceError(f"Failed to get experiment size: {str(e)}")

    def _store_metadata(self, metadata: ExperimentMetadata) -> None:
        """Store experiment metadata in SQLite or PostgreSQL."""
        if self.backend == "postgresql":
            self._store_metadata_postgresql(metadata)
        else:
            self._store_metadata_sqlite(metadata)

    def _store_metadata_sqlite(self, metadata: ExperimentMetadata) -> None:
        """Store experiment metadata in SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """,
                (
                    metadata.experiment_id,
                    metadata.experiment_name,
                    metadata.description,
                    metadata.created_at.isoformat(),
                    metadata.updated_at.isoformat(),
                    metadata.researcher,
                    metadata.institution,
                    metadata.n_participants,
                    metadata.n_trials,
                    json.dumps(metadata.conditions),
                    json.dumps(metadata.parameters),
                    metadata.data_format,
                    json.dumps(metadata.file_paths),
                    metadata.total_size_mb,
                    metadata.current_version,
                    json.dumps(metadata.tags),
                    metadata.category,
                    metadata.data_quality_score,
                    metadata.completeness_percentage,
                    metadata.validation_status,
                ),
            )
            conn.commit()

    def _store_metadata_postgresql(self, metadata: ExperimentMetadata) -> None:
        """Store experiment metadata in PostgreSQL."""
        if not self.postgresql_available or self._psycopg2 is None:
            return

        with self._psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO experiments VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) ON CONFLICT (experiment_id) DO UPDATE SET
                        experiment_name = EXCLUDED.experiment_name,
                        description = EXCLUDED.description,
                        updated_at = EXCLUDED.updated_at,
                        researcher = EXCLUDED.researcher,
                        institution = EXCLUDED.institution,
                        n_participants = EXCLUDED.n_participants,
                        n_trials = EXCLUDED.n_trials,
                        conditions = EXCLUDED.conditions,
                        parameters = EXCLUDED.parameters,
                        data_format = EXCLUDED.data_format,
                        file_paths = EXCLUDED.file_paths,
                        total_size_mb = EXCLUDED.total_size_mb,
                        current_version = EXCLUDED.current_version,
                        tags = EXCLUDED.tags,
                        category = EXCLUDED.category,
                        data_quality_score = EXCLUDED.data_quality_score,
                        completeness_percentage = EXCLUDED.completeness_percentage,
                        validation_status = EXCLUDED.validation_status
                """,
                    (
                        metadata.experiment_id,
                        metadata.experiment_name,
                        metadata.description,
                        metadata.created_at,
                        metadata.updated_at,
                        metadata.researcher,
                        metadata.institution,
                        metadata.n_participants,
                        metadata.n_trials,
                        self._psycopg2_extras_Json(metadata.conditions),
                        self._psycopg2_extras_Json(metadata.parameters),
                        metadata.data_format,
                        self._psycopg2_extras_Json(metadata.file_paths),
                        metadata.total_size_mb,
                        metadata.current_version,
                        self._psycopg2_extras_Json(metadata.tags),
                        metadata.category,
                        metadata.data_quality_score,
                        metadata.completeness_percentage,
                        metadata.validation_status,
                    ),
                )
            conn.commit()

    def _load_metadata(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Load experiment metadata from SQLite or PostgreSQL."""
        if self.backend == "postgresql":
            return self._load_metadata_postgresql(experiment_id)
        else:
            return self._load_metadata_sqlite(experiment_id)

    def _load_metadata_sqlite(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Load experiment metadata from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            return ExperimentMetadata(
                experiment_id=row["experiment_id"],
                experiment_name=row["experiment_name"],
                description=row["description"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                researcher=row["researcher"],
                institution=row["institution"],
                n_participants=row["n_participants"],
                n_trials=row["n_trials"],
                conditions=json.loads(row["conditions"]),
                parameters=json.loads(row["parameters"]),
                data_format=row["data_format"],
                file_paths=json.loads(row["file_paths"]),
                total_size_mb=row["total_size_mb"],
                current_version=row["current_version"],
                tags=json.loads(row["tags"]),
                category=row["category"],
                data_quality_score=row["data_quality_score"],
                completeness_percentage=row["completeness_percentage"],
                validation_status=row["validation_status"],
            )

    def _load_metadata_postgresql(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Load experiment metadata from PostgreSQL."""
        if not self.postgresql_available or self._psycopg2 is None:
            return None

        with self._psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
        ) as conn:
            with conn.cursor(cursor_factory=self._psycopg2_extras_RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM experiments WHERE experiment_id = %s",
                    (experiment_id,),
                )
                row = cursor.fetchone()

                if not row:
                    return None

                return ExperimentMetadata(
                    experiment_id=row["experiment_id"],
                    experiment_name=row["experiment_name"],
                    description=row["description"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    researcher=row["researcher"],
                    institution=row["institution"],
                    n_participants=row["n_participants"],
                    n_trials=row["n_trials"],
                    conditions=row["conditions"],
                    parameters=row["parameters"],
                    data_format=row["data_format"],
                    file_paths=row["file_paths"],
                    total_size_mb=row["total_size_mb"],
                    current_version=row["current_version"],
                    tags=row["tags"],
                    category=row["category"],
                    data_quality_score=row["data_quality_score"],
                    completeness_percentage=row["completeness_percentage"],
                    validation_status=row["validation_status"],
                )

    def _store_data_hdf5(self, dataset: ExperimentalDataset) -> None:
        """Store dataset in HDF5 format."""
        experiment_id = dataset.metadata.experiment_id
        self._validate_experiment_id(experiment_id)

        with h5py.File(self.hdf5_path, "a") as f:
            # Create experiment group
            if experiment_id in f:
                del f[experiment_id]

            exp_group = f.create_group(experiment_id)
            exp_group.attrs["created_at"] = datetime.now().isoformat()

            # Store different data types
            for data_type, data_dict in [
                ("data", dataset.data),
                ("raw_data", dataset.raw_data),
                ("processed_data", dataset.processed_data),
                ("analysis_results", dataset.analysis_results),
            ]:
                if data_dict:
                    type_group = exp_group.create_group(data_type)
                    self._store_dict_hdf5(type_group, data_dict)

    def _store_dict_hdf5(self, group: h5py.Group, data_dict: Dict[str, Any]) -> None:
        """Recursively store dictionary data in HDF5."""
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                # Store numpy arrays directly
                try:
                    group.create_dataset(key, data=value)
                except (TypeError, ValueError):
                    # Fallback to JSON for complex arrays
                    group.attrs[key] = json.dumps(value.tolist(), default=str)
            elif isinstance(value, list):
                # Handle lists carefully - check if they can be converted to homogeneous array
                try:
                    # Try to create a homogeneous numpy array
                    arr = np.array(value)
                    if arr.dtype == "O":  # Object dtype means mixed types
                        # Store as JSON string instead
                        group.attrs[key] = json.dumps(value, default=str)
                    else:
                        # Homogeneous array, store directly
                        group.create_dataset(key, data=arr)
                except (TypeError, ValueError):
                    # Fallback to JSON serialization
                    group.attrs[key] = json.dumps(value, default=str)
            elif isinstance(value, dict):
                # Create subgroup for nested dictionaries
                subgroup = group.create_group(key)
                self._store_dict_hdf5(subgroup, value)
            elif isinstance(value, pd.DataFrame):
                # Store DataFrame as dataset with metadata
                df_group = group.create_group(key)
                df_group.create_dataset("values", data=value.values)
                df_group.create_dataset("columns", data=[str(c) for c in value.columns])
                df_group.create_dataset("index", data=[str(i) for i in value.index])
            else:
                # Store scalar values as attributes or datasets
                try:
                    group.attrs[key] = value
                except (TypeError, ValueError):
                    # Fallback to JSON serialization for complex objects
                    group.attrs[key] = json.dumps(value, default=str)

    def _load_data_hdf5(self, experiment_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Load dataset from HDF5 format."""
        self._validate_experiment_id(experiment_id)
        with h5py.File(self.hdf5_path, "r") as f:
            if experiment_id not in f:
                return {}

            exp_group = f[experiment_id]
            result = {}

            for data_type in ["data", "raw_data", "processed_data", "analysis_results"]:
                if data_type in exp_group:
                    result[data_type] = self._load_dict_hdf5(exp_group[data_type])

            return result

    def _load_dict_hdf5(self, group: h5py.Group) -> Dict[str, Any]:
        """Recursively load dictionary data from HDF5."""
        result = {}

        # Load attributes
        for key, value in group.attrs.items():
            try:
                # Try to parse as JSON if it's a string
                if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                    result[key] = json.loads(value)
                else:
                    result[key] = value
            except (json.JSONDecodeError, TypeError):
                result[key] = value

        # Load datasets and subgroups
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                result[key] = item[()]
            elif isinstance(item, h5py.Group):
                # Check if it's a DataFrame
                if "values" in item and "columns" in item and "index" in item:
                    result[key] = pd.DataFrame(
                        data=item["values"][()],
                        columns=item["columns"][()],
                        index=item["index"][()],
                    )
                else:
                    result[key] = self._load_dict_hdf5(item)

        return result

    def _store_data_sqlite(self, dataset: ExperimentalDataset) -> None:
        """Store dataset in SQLite format (for smaller datasets)."""
        experiment_id = dataset.metadata.experiment_id
        self._validate_experiment_id(experiment_id)

        # Serialize data to JSON/pickle for SQLite storage
        data_file = self.data_path / f"{experiment_id}.pkl"

        # Verify containment within data_path
        if not data_file.resolve().is_relative_to(self.data_path.resolve()):
            raise PersistenceError(f"Path traversal detected: {data_file}")

        data_to_store = {
            "data": dataset.data,
            "raw_data": dataset.raw_data,
            "processed_data": dataset.processed_data,
            "analysis_results": dataset.analysis_results,
        }

        safe_pickle_dump(data_to_store, data_file)

    def _store_data_postgresql(self, dataset: ExperimentalDataset) -> None:
        """Store dataset in PostgreSQL format for large-scale data handling."""
        experiment_id = dataset.metadata.experiment_id
        self._validate_experiment_id(experiment_id)

        with self._psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
        ) as conn:
            with conn.cursor() as cursor:
                # Store data in chunks for large datasets
                chunk_size = 100  # Adjust based on data size
                data_types = ["data", "raw_data", "processed_data", "analysis_results"]

                for data_type in data_types:
                    data_dict = getattr(dataset, data_type, {}) or {}

                    if not data_dict:
                        continue

                    # Convert data to serializable format
                    serializable_data = self._prepare_data_for_postgresql(data_dict)

                    # Split into chunks if too large
                    items = list(serializable_data.items())
                    total_chunks = (len(items) + chunk_size - 1) // chunk_size

                    for i in range(0, len(items), chunk_size):
                        chunk_items = dict(items[i : i + chunk_size])
                        chunk_index = i // chunk_size

                        cursor.execute(
                            """
                            INSERT INTO data_chunks (experiment_id, data_type, chunk_index, total_chunks, data)
                            VALUES (%s, %s, %s, %s, %s)
                        """,
                            (
                                experiment_id,
                                data_type,
                                chunk_index,
                                total_chunks,
                                self._psycopg2_extras_Json(chunk_items),
                            ),
                        )

            conn.commit()

    def _prepare_data_for_postgresql(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for PostgreSQL storage by converting numpy arrays and complex objects."""
        prepared = {}

        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                # Convert numpy arrays to lists
                prepared[key] = value.tolist()
            elif isinstance(value, dict):
                # Recursively prepare nested dictionaries
                prepared[key] = self._prepare_data_for_postgresql(value)
            elif isinstance(value, list):
                # Handle lists of arrays or complex objects
                prepared[key] = [
                    item.tolist() if isinstance(item, np.ndarray) else item for item in value
                ]
            else:
                # Store scalar values and other serializable objects
                prepared[key] = value

        return prepared

    def _load_data_sqlite(
        self, experiment_id: str, version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load dataset from SQLite format."""
        self._validate_experiment_id(experiment_id)
        data_file = self.data_path / f"{experiment_id}.pkl"

        # Verify containment within data_path
        if not data_file.resolve().is_relative_to(self.data_path.resolve()):
            raise PersistenceError(f"Path traversal detected: {data_file}")

        if not data_file.exists():
            return {}

        return safe_pickle_load(data_file)  # type: ignore

    def _load_data_postgresql(
        self, experiment_id: str, version: Optional[str] = None
    ) -> dict[str, Any] | None:
        """Load dataset from PostgreSQL format."""
        if not self.postgresql_available or self._psycopg2 is None:
            return None

        self._validate_experiment_id(experiment_id)

        with self._psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
        ) as conn:
            with conn.cursor(cursor_factory=self._psycopg2_extras_RealDictCursor) as cursor:
                # Load data by type
                data_types = ["data", "raw_data", "processed_data", "analysis_results"]
                result = {}

                for data_type in data_types:
                    cursor.execute(
                        """
                        SELECT data, chunk_index
                        FROM data_chunks
                        WHERE experiment_id = %s AND data_type = %s
                        ORDER BY chunk_index
                    """,
                        (experiment_id, data_type),
                    )

                    chunks = cursor.fetchall()
                    if chunks:
                        # Reassemble chunks
                        type_data = {}
                        for chunk in chunks:
                            chunk_data = chunk["data"]
                            if isinstance(chunk_data, str):
                                chunk_data = json.loads(chunk_data)
                            type_data.update(chunk_data)

                        result[data_type] = type_data

                return result

    def _store_version(self, experiment_id: str, version: DataVersion) -> None:
        """Store version information."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    version.version_id,
                    experiment_id,
                    version.version_number,
                    version.created_at.isoformat(),
                    version.created_by,
                    version.description,
                    version.parent_version or "",
                    version.checksum,
                    version.size_bytes,
                ),
            )
            conn.commit()

    def _load_versions(self, experiment_id: str) -> List[DataVersion]:
        """Load all versions for an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM versions WHERE experiment_id = ?", (experiment_id,)
            )

            versions = []
            for row in cursor.fetchall():
                version = DataVersion(
                    version_id=row["version_id"],
                    version_number=row["version_number"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    created_by=row["created_by"],
                    description=row["description"],
                    parent_version=row["parent_version"],
                    checksum=row["checksum"],
                    size_bytes=row["size_bytes"],
                )
                # Attach experiment_id for compatibility with tests
                setattr(version, "experiment_id", experiment_id)
                versions.append(version)

            return versions

    def _save_versions(self, experiment_id: str, versions: List[DataVersion]) -> None:
        """Save versions for an experiment (replaces existing)."""
        with sqlite3.connect(self.db_path) as conn:
            # Delete existing versions
            conn.execute("DELETE FROM versions WHERE experiment_id = ?", (experiment_id,))
            # Insert new versions
            for version in versions:
                conn.execute(
                    """
                    INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        version.version_id,
                        experiment_id,
                        version.version_number,
                        version.created_at.isoformat(),
                        version.created_by,
                        version.description,
                        version.parent_version,
                        version.checksum,
                        version.size_bytes,
                    ),
                )
            conn.commit()

    def _load_backup_info(self, experiment_id: str) -> List[BackupInfo]:
        """Load backup information for experiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM backups WHERE experiment_id = ?", (experiment_id,))

            backups = []
            for row in cursor.fetchall():
                backup = BackupInfo(
                    backup_id=row["backup_id"],
                    backup_path=row["backup_path"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    backup_type=row["backup_type"],
                    size_bytes=row["size_bytes"],
                    checksum=row["checksum"],
                    compression_ratio=row["compression_ratio"],
                    retention_days=row["retention_days"],
                    status=row["status"],
                )
                backups.append(backup)

            return backups

    def _calculate_checksum(self, dataset: ExperimentalDataset) -> str:
        """Calculate checksum for dataset integrity verification."""
        # Create a string representation of the dataset for hashing
        data_str = json.dumps(
            {
                "metadata": dataset.metadata.experiment_id,
                "data_keys": list(dataset.data.keys()) if dataset.data else [],
                "timestamp": datetime.now().isoformat(),
            },
            sort_keys=True,
        )

        return hashlib.sha256(data_str.encode()).hexdigest()

    def create_backup(
        self, experiment_id: str, backup_type: str = "full", retention_days: int = 30
    ) -> BackupInfo:
        """Create backup of experimental dataset."""
        try:
            # Create backup directory
            backup_dir = (
                self.backup_path / f"{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            backup_dir.mkdir(exist_ok=True)

            # Copy data files
            if self.backend in ["hdf5", "hybrid"]:
                # Copy main experiments file
                if self.hdf5_path.exists():
                    shutil.copy2(self.hdf5_path, backup_dir / "experiments.h5")

            # Copy experiment-specific data file if it exists
            exp_data_file = self.data_path / f"{experiment_id}.h5"
            if exp_data_file.exists():
                shutil.copy2(exp_data_file, backup_dir / f"{experiment_id}.h5")

            if self.backend in ["sqlite", "hybrid"]:
                shutil.copy2(self.db_path, backup_dir / "experiments.db")

                # Copy pickle files if they exist
                data_file = self.data_path / f"{experiment_id}.pkl"
                if data_file.exists():
                    shutil.copy2(data_file, backup_dir / f"{experiment_id}.pkl")

            # Calculate backup size and checksum
            backup_size = sum(f.stat().st_size for f in backup_dir.rglob("*") if f.is_file())
            backup_checksum = self._calculate_backup_checksum(backup_dir)

            # Create backup info
            backup_info = BackupInfo(
                backup_path=str(backup_dir),
                backup_type=backup_type,
                size_bytes=backup_size,
                checksum=backup_checksum,
                retention_days=retention_days,
            )
            # Attach experiment_id for compatibility with tests
            setattr(backup_info, "experiment_id", experiment_id)

            # Store backup info in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO backups VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        backup_info.backup_id,
                        experiment_id,
                        backup_info.backup_path,
                        backup_info.created_at.isoformat(),
                        backup_info.backup_type,
                        backup_info.size_bytes,
                        backup_info.checksum,
                        backup_info.compression_ratio,
                        backup_info.retention_days,
                        backup_info.status,
                    ),
                )
                conn.commit()

            return backup_info

        except (IOError, OSError, sqlite3.Error) as e:
            raise PersistenceError(f"Failed to create backup for {experiment_id}: {str(e)}")

    def _calculate_backup_checksum(self, backup_dir: Path) -> str:
        """Calculate checksum for backup directory."""
        checksums = []
        for file_path in sorted(backup_dir.rglob("*")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    file_checksum = hashlib.sha256(f.read()).hexdigest()
                    checksums.append(f"{file_path.name}:{file_checksum}")

        combined = "|".join(checksums)
        return hashlib.sha256(combined.encode()).hexdigest()

    def list_experiments(self, limit: Optional[int] = None) -> List[str]:
        """List all experiment IDs."""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT experiment_id FROM experiments ORDER BY created_at DESC"
            if limit:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query)
            return [row[0] for row in cursor.fetchall()]

    def list_experiments_with_metadata(
        self, limit: Optional[int] = None
    ) -> List[ExperimentMetadata]:
        """List all experiments with their metadata."""
        experiment_ids = self.list_experiments(limit)
        metadata_list = []
        for exp_id in experiment_ids:
            metadata = self._load_metadata(exp_id)
            if metadata:
                metadata_list.append(metadata)
        return metadata_list

    def delete_experiment_with_backups(
        self, experiment_id: str, include_backups: bool = False
    ) -> None:
        """Delete experiment and optionally its backups."""
        try:
            # Delete from SQLite
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
                conn.execute("DELETE FROM versions WHERE experiment_id = ?", (experiment_id,))

                if include_backups:
                    # Get backup paths before deletion
                    cursor = conn.execute(
                        "SELECT backup_path FROM backups WHERE experiment_id = ?",
                        (experiment_id,),
                    )
                    backup_paths = [row[0] for row in cursor.fetchall()]

                    # Delete backup records
                    conn.execute("DELETE FROM backups WHERE experiment_id = ?", (experiment_id,))

                    # Delete backup directories
                    for backup_path in backup_paths:
                        backup_dir = Path(backup_path)
                        if backup_dir.exists():
                            shutil.rmtree(backup_dir)

                conn.commit()

            # Delete from HDF5
            if self.backend in ["hdf5", "hybrid"]:
                with h5py.File(self.hdf5_path, "a") as f:
                    if experiment_id in f:
                        del f[experiment_id]

            # Delete pickle file
            data_file = self.data_path / f"{experiment_id}.pkl"
            if data_file.exists():
                data_file.unlink()

        except (IOError, OSError, sqlite3.Error) as e:
            raise PersistenceError(f"Failed to delete experiment {experiment_id}: {str(e)}")

    def flush_all(self) -> None:
        """Flush all pending operations and ensure data persistence.

        Forces all cached data to be written to disk and commits
        any pending database transactions.
        """
        try:
            # Flush SQLite connections
            if self.backend in ["sqlite", "hybrid"]:
                with sqlite3.connect(self.db_path) as conn:
                    conn.commit()

            # Flush HDF5 files
            if self.backend in ["hdf5", "hybrid"]:
                # HDF5 files are automatically flushed when closed
                # Force flush by opening and closing
                with h5py.File(self.hdf5_path, "a") as f:
                    f.flush()

        except (IOError, OSError, sqlite3.Error) as e:
            raise PersistenceError(f"Failed to flush persistence layer: {str(e)}")

    def stream_dataset(
        self,
        experiment_id: str,
        data_type: str = "processed_data",
        chunk_size: int = 1000,
    ) -> Iterator[Dict[str, Any]]:
        """Stream dataset data in chunks to handle large datasets efficiently.

        Args:
            experiment_id: Experiment identifier
            data_type: Type of data to stream ('raw_data', 'processed_data', 'analysis_results')
            chunk_size: Number of items per chunk

        Yields:
            Dict containing chunk of data with metadata
        """
        try:
            if self.backend not in ["hdf5", "hybrid"]:
                raise PersistenceError("Streaming only supported for HDF5 backend")

            with h5py.File(self.hdf5_path, "r") as f:
                if experiment_id not in f:
                    raise PersistenceError(f"Experiment {experiment_id} not found")

                exp_group = f[experiment_id]
                if data_type not in exp_group:
                    raise PersistenceError(f"Data type {data_type} not found in experiment")

                data_group = exp_group[data_type]

                # Get total number of datasets/items
                total_items = len(list(data_group.keys()))

                for i in range(0, total_items, chunk_size):
                    chunk_data = {}
                    chunk_end = min(i + chunk_size, total_items)

                    # Load chunk of data
                    for j in range(i, chunk_end):
                        key = list(data_group.keys())[j]
                        item = data_group[key]

                        if isinstance(item, h5py.Dataset):
                            chunk_data[key] = item[()]
                        elif isinstance(item, h5py.Group):
                            chunk_data[key] = self._load_dict_hdf5(item)

                    yield {
                        "chunk_start": i,
                        "chunk_end": chunk_end,
                        "total_items": total_items,
                        "data": chunk_data,
                        "is_last_chunk": chunk_end >= total_items,
                    }

        except (IOError, OSError, KeyError, IndexError) as e:
            raise PersistenceError(f"Failed to stream dataset {experiment_id}: {str(e)}")

    def get_dataset_info(self, experiment_id: str) -> Dict[str, Any]:
        """Get dataset information without loading full data.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dict with dataset metadata and structure info
        """
        try:
            metadata = self._load_metadata(experiment_id)
            if not metadata:
                raise PersistenceError(f"Experiment {experiment_id} not found")

            info: Dict[str, Any] = {
                "experiment_id": experiment_id,
                "metadata": {
                    "name": metadata.experiment_name,
                    "created_at": metadata.created_at.isoformat(),
                    "n_participants": metadata.n_participants,
                    "n_trials": metadata.n_trials,
                    "total_size_mb": metadata.total_size_mb,
                    "data_format": metadata.data_format,
                },
                "data_structure": {},
                "estimated_memory_mb": 0,
            }

            # Get data structure info without loading
            if self.backend in ["hdf5", "hybrid"]:
                with h5py.File(self.hdf5_path, "r") as f:
                    if experiment_id in f:
                        exp_group = f[experiment_id]
                        for data_type in [
                            "data",
                            "raw_data",
                            "processed_data",
                            "analysis_results",
                        ]:
                            if data_type in exp_group:
                                type_group = exp_group[data_type]
                                type_info = self._analyze_group_structure(type_group)
                                info["data_structure"][data_type] = type_info
                                info["estimated_memory_mb"] += type_info.get("estimated_size_mb", 0)

            return info

        except (IOError, OSError, ValueError, KeyError) as e:
            raise PersistenceError(f"Failed to get dataset info for {experiment_id}: {str(e)}")

    def _analyze_group_structure(self, group: h5py.Group) -> Dict[str, Any]:
        """Analyze HDF5 group structure and estimate memory usage.

        Args:
            group: HDF5 group to analyze

        Returns:
            Dict with structure information and size estimates
        """
        structure: Dict[str, Any] = {
            "datasets": [],
            "subgroups": [],
            "estimated_size_mb": 0,
            "total_items": 0,
        }

        try:
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    dataset_info = {
                        "name": key,
                        "shape": item.shape,
                        "dtype": str(item.dtype),
                        "size_mb": (item.size * item.dtype.itemsize) / (1024 * 1024),
                    }
                    structure["datasets"].append(dataset_info)
                    structure["estimated_size_mb"] += dataset_info["size_mb"]
                    structure["total_items"] += 1

                elif isinstance(item, h5py.Group):
                    subgroup_info = self._analyze_group_structure(item)
                    subgroup_info["name"] = key
                    structure["subgroups"].append(subgroup_info)
                    structure["estimated_size_mb"] += subgroup_info["estimated_size_mb"]
                    structure["total_items"] += subgroup_info["total_items"]

        except Exception as e:
            # Log error but continue with partial info
            structure["error"] = str(e)

        return structure

    def load_dataset_memory_aware(
        self,
        experiment_id: str,
        data_type: str = "processed_data",
        chunk_size: int = 1000,
        max_memory_mb: float = 500.0,
    ) -> Iterator[Dict[str, Any]]:
        """Load dataset with memory-aware chunking.

        Args:
            experiment_id: Experiment identifier
            data_type: Type of data to load
            chunk_size: Maximum items per chunk
            max_memory_mb: Maximum memory to use for loading

        Returns:
            Iterator yielding data chunks
        """
        # Get dataset info first
        info = self.get_dataset_info(experiment_id)
        data_info = info["data_structure"].get(data_type, {})

        if data_info.get("estimated_size_mb", 0) > max_memory_mb:
            # Use streaming for large datasets
            yield from self.stream_dataset(experiment_id, data_type, chunk_size)
        else:
            # Load normally for smaller datasets
            dataset = self.load_dataset(experiment_id)
            data = getattr(dataset, data_type, {}) or {}

            # Yield as single chunk
            yield {
                "chunk_start": 0,
                "chunk_end": len(data),
                "total_items": len(data),
                "data": data,
                "is_last_chunk": True,
            }

    def process_large_dataset(
        self,
        experiment_id: str,
        processor_func: Callable[[Any], Any],
        data_type: str = "processed_data",
        chunk_size: int = 1000,
    ) -> List[Any]:
        """Process large datasets in chunks to avoid memory issues.

        Args:
            experiment_id: Experiment identifier
            processor_func: Function to process each chunk
            data_type: Type of data to process
            chunk_size: Size of chunks to process

        Returns:
            List of processing results from each chunk
        """
        results = []

        try:
            for chunk in self.stream_dataset(experiment_id, data_type, chunk_size):
                # Process chunk
                chunk_result = processor_func(chunk["data"])
                results.append(chunk_result)

                # Optional: yield intermediate results for very long processing
                if len(results) % 10 == 0:  # Every 10 chunks
                    # Could yield here if needed for progress updates
                    pass

        except Exception as e:
            raise PersistenceError(f"Failed to process large dataset {experiment_id}: {str(e)}")

        return results
