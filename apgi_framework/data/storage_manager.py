"""Storage manager for the APGI Framework data management system.

Provides high-level interface for data storage, retrieval, querying,
and management operations with automatic validation and backup.
"""

import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from ..exceptions import APGIFrameworkError
from ..logging.standardized_logging import get_logger
from .data_models import (
    BackupInfo,
    DataVersion,
    ExperimentalDataset,
    ExperimentMetadata,
    QueryFilter,
    StorageStats,
)
from .data_validator import DataValidator, ValidationError
from .persistence_layer import PersistenceError, PersistenceLayer


def _safe_extract(tar: "tarfile.TarFile", destination: Path) -> None:
    """Extract ``tar`` into ``destination``, refusing anything that could escape it.

    Rejects absolute paths, ``..`` components, links, devices and other special members, so a
    crafted backup cannot write outside the temporary directory (path traversal, CWE-22).
    """
    root = destination.resolve()
    members = []
    for member in tar.getmembers():
        target = (root / member.name).resolve()
        if root != target and root not in target.parents:
            raise StorageError(f"unsafe path in backup archive: {member.name!r}")
        if not (member.isreg() or member.isdir()):
            raise StorageError(f"unsupported member type in backup archive: {member.name!r}")
        members.append(member)
    try:
        tar.extractall(root, members=members, filter="data")  # nosec B202 - members validated above
    except TypeError:  # Python without extraction filters
        tar.extractall(root, members=members)  # nosec B202 - members validated above


class StorageError(APGIFrameworkError):
    """Errors in storage management operations."""


class StorageManager:
    """High-level storage manager for experimental datasets.

    Provides unified interface for storing, retrieving, querying, and managing
    experimental data with automatic validation, versioning, and backup.
    """

    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
        backend: str = "hdf5",
        auto_validate: bool = True,
        auto_backup: bool = True,
    ):
        """Initialize storage manager.

        Args:
            storage_path: Base path for data storage (default: ./data)
            backend: Storage backend ('sqlite', 'hdf5', or 'hybrid')
            auto_validate: Whether to automatically validate data on storage
            auto_backup: Whether to automatically create backups
        """
        if storage_path is None:
            storage_path = "./data"
        self.storage_path = Path(storage_path)
        self.backend = backend
        self.auto_validate = auto_validate
        self.auto_backup = auto_backup

        # Initialize components
        self.persistence = PersistenceLayer(storage_path, backend)
        self.validator = DataValidator(strict_mode=True)

        # Setup logging
        self.logger = get_logger(__name__)

        # Storage statistics
        self._stats_cache: Optional[StorageStats] = None
        self._stats_cache_time: Optional[datetime] = None
        self._stats_cache_duration = timedelta(minutes=5)

    def store_dataset(
        self,
        dataset: ExperimentalDataset,
        validate: Optional[bool] = None,
        create_backup: Optional[bool] = None,
    ) -> str:
        """Store experimental dataset with validation and backup.

        Args:
            dataset: ExperimentalDataset to store
            validate: Override auto_validate setting
            create_backup: Override auto_backup setting

        Returns:
            str: Experiment ID of stored dataset
        """
        validate = validate if validate is not None else self.auto_validate
        create_backup = create_backup if create_backup is not None else self.auto_backup

        try:
            # Validate dataset if requested
            if validate:
                validation_result = self.validator.validate_dataset(dataset)

                if not validation_result["is_valid"]:
                    error_msg = f"Dataset validation failed: {validation_result['errors']}"
                    self.logger.error(error_msg)
                    raise StorageError(error_msg)

                # Log warnings
                if validation_result["warnings"]:
                    self.logger.warning(
                        f"Dataset validation warnings: {validation_result['warnings']}"
                    )

            # Store dataset
            experiment_id = self.persistence.store_dataset(dataset)
            self.logger.info(f"Stored dataset {experiment_id}")

            # Create backup if requested
            if create_backup:
                try:
                    backup_info = self.persistence.create_backup(experiment_id)
                    self.logger.info(f"Created backup {backup_info.backup_id} for {experiment_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to create backup for {experiment_id}: {str(e)}")

            # Invalidate stats cache
            self._invalidate_stats_cache()  # type: ignore[no-untyped-call]

            return experiment_id

        except (PersistenceError, ValidationError) as e:
            raise StorageError(f"Failed to store dataset: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error storing dataset: {str(e)}")
            raise StorageError(f"Unexpected error storing dataset: {str(e)}")

    def load_dataset(
        self, experiment_id: str, version: Optional[str] = None, validate: bool = False
    ) -> ExperimentalDataset:
        """Load experimental dataset by ID.

        Args:
            experiment_id: Experiment identifier
            version: Specific version to load (latest if None)
            validate: Whether to validate loaded dataset

        Returns:
            ExperimentalDataset: Loaded dataset
        """
        try:
            dataset = self.persistence.load_dataset(experiment_id, version)
            self.logger.info(f"Loaded dataset {experiment_id}")

            # Validate if requested
            if validate:
                validation_result = self.validator.validate_dataset(dataset)

                if not validation_result["is_valid"]:
                    self.logger.warning(
                        f"Loaded dataset {experiment_id} failed validation: {validation_result['errors']}"
                    )

                if validation_result["warnings"]:
                    self.logger.warning(
                        f"Loaded dataset {experiment_id} validation warnings: {validation_result['warnings']}"
                    )

            return dataset

        except PersistenceError as e:
            raise StorageError(f"Failed to load dataset {experiment_id}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading dataset {experiment_id}: {str(e)}")
            raise StorageError(f"Unexpected error loading dataset {experiment_id}: {str(e)}")

    def query_datasets(
        self, filter_criteria: Optional[QueryFilter] = None, limit: Optional[int] = None
    ) -> List[ExperimentMetadata]:
        """Query datasets based on filter criteria.

        Args:
            filter_criteria: QueryFilter with search criteria
            limit: Maximum number of results to return

        Returns:
            List[ExperimentMetadata]: Matching experiment metadata
        """
        try:
            # Get all experiment IDs
            experiment_ids = self.persistence.list_experiments(limit)

            # Load metadata for filtering
            metadata_list = []
            for exp_id in experiment_ids:
                try:
                    dataset = self.persistence.load_dataset(exp_id)
                    metadata_list.append(dataset.metadata)
                except Exception as e:
                    self.logger.warning(f"Failed to load metadata for {exp_id}: {str(e)}")
                    continue

            # Apply filters
            if filter_criteria:
                metadata_list = self._apply_filters(metadata_list, filter_criteria)

            return metadata_list

        except Exception as e:
            self.logger.error(f"Failed to query datasets: {str(e)}")
            raise StorageError(f"Failed to query datasets: {str(e)}")

    def _apply_filters(
        self, metadata_list: List[ExperimentMetadata], filters: QueryFilter
    ) -> List[ExperimentMetadata]:
        """Apply query filters to metadata list."""
        filtered = metadata_list

        # Filter by experiment IDs
        if filters.experiment_ids:
            filtered = [m for m in filtered if m.experiment_id in filters.experiment_ids]

        # Filter by researcher
        if filters.researcher:
            filtered = [m for m in filtered if filters.researcher.lower() in m.researcher.lower()]

        # Filter by institution
        if filters.institution:
            filtered = [m for m in filtered if filters.institution.lower() in m.institution.lower()]

        # Filter by date range
        if filters.date_range:
            start_date, end_date = filters.date_range
            filtered = [m for m in filtered if start_date <= m.created_at <= end_date]

        # Filter by tags
        if filters.tags:
            filtered = [m for m in filtered if any(tag in m.tags for tag in filters.tags)]

        # Filter by category
        if filters.category:
            filtered = [m for m in filtered if m.category == filters.category]

        # Filter by participant count
        if filters.min_participants:
            filtered = [m for m in filtered if m.n_participants >= filters.min_participants]

        if filters.max_participants:
            filtered = [m for m in filtered if m.n_participants <= filters.max_participants]

        # Filter by conditions
        if filters.conditions:
            filtered = [
                m for m in filtered if any(cond in m.conditions for cond in filters.conditions)
            ]

        # Filter by validation status
        if filters.validation_status:
            filtered = [m for m in filtered if m.validation_status == filters.validation_status]

        # Filter by data format
        if filters.data_format:
            filtered = [m for m in filtered if m.data_format == filters.data_format]

        return filtered

    def update_dataset(
        self, experiment_id: str, updates: Dict[str, Any], create_version: bool = True
    ) -> str:
        """Update existing dataset with new data or metadata.

        Args:
            experiment_id: Experiment to update
            updates: Dictionary of updates to apply
            create_version: Whether to create new version

        Returns:
            str: Version ID if new version created, else experiment ID
        """
        try:
            # Load existing dataset
            dataset = self.load_dataset(experiment_id)

            # Apply updates
            if "metadata" in updates:
                for key, value in updates["metadata"].items():
                    if hasattr(dataset.metadata, key):
                        setattr(dataset.metadata, key, value)

            if "data" in updates:
                dataset.data.update(updates["data"])

            if "raw_data" in updates:
                dataset.raw_data = updates["raw_data"]

            if "processed_data" in updates:
                dataset.processed_data = updates["processed_data"]

            if "analysis_results" in updates:
                dataset.analysis_results = updates["analysis_results"]

            # Update timestamp
            dataset.metadata.updated_at = datetime.now()

            # Store updated dataset
            self.persistence.store_dataset(dataset)

            # Create new version if requested
            if create_version:
                version = DataVersion(
                    version_number=self._increment_version(dataset.metadata.current_version),
                    description=f"Updated: {', '.join(updates.keys())}",
                    checksum=self.persistence._calculate_checksum(dataset),
                )
                self.persistence._store_version(experiment_id, version)
                dataset.metadata.current_version = version.version_number

                self.logger.info(f"Created version {version.version_number} for {experiment_id}")
                return version.version_id

            self.logger.info(f"Updated dataset {experiment_id}")
            return experiment_id

        except Exception as e:
            self.logger.error(f"Failed to update dataset {experiment_id}: {str(e)}")
            raise StorageError(f"Failed to update dataset {experiment_id}: {str(e)}")

    def delete_dataset(
        self, experiment_id: str, include_backups: bool = False, confirm: bool = False
    ) -> bool:
        """Delete experimental dataset.

        Args:
            experiment_id: Experiment to delete
            include_backups: Whether to delete backups too
            confirm: Confirmation flag for safety

        Returns:
            bool: True if deleted successfully
        """
        if not confirm:
            raise StorageError("Dataset deletion requires explicit confirmation")

        try:
            if include_backups:
                # Call the version that supports backup deletion
                if hasattr(self.persistence, "delete_experiment_with_backups"):
                    self.persistence.delete_experiment_with_backups(experiment_id, include_backups)
                else:
                    self.persistence.delete_experiment(experiment_id)
            else:
                # Call the simple version
                self.persistence.delete_experiment(experiment_id)
            self.logger.info(f"Deleted dataset {experiment_id} (backups: {include_backups})")

            # Invalidate stats cache
            self._invalidate_stats_cache()  # type: ignore[no-untyped-call]

            return True

        except Exception as e:
            self.logger.error(f"Failed to delete dataset {experiment_id}: {str(e)}")
            raise StorageError(f"Failed to delete dataset {experiment_id}: {str(e)}")

    def create_backup(self, experiment_id: str, backup_type: str = "manual") -> BackupInfo:
        """Create backup of experimental dataset.

        Args:
            experiment_id: Experiment to backup
            backup_type: Type of backup ('manual', 'scheduled', 'auto')

        Returns:
            BackupInfo: Information about created backup
        """
        try:
            backup_info = self.persistence.create_backup(experiment_id, backup_type)
            self.logger.info(
                f"Created {backup_type} backup {backup_info.backup_id} for {experiment_id}"
            )
            return backup_info

        except Exception as e:
            self.logger.error(f"Failed to create backup for {experiment_id}: {str(e)}")
            raise StorageError(f"Failed to create backup for {experiment_id}: {str(e)}")

    def restore_from_backup(self, backup_id: str) -> str:
        """Restore dataset from backup.

        Args:
            backup_id: Backup to restore from

        Returns:
            str: Experiment ID of restored dataset
        """
        try:
            # Load backup information from persistence layer
            backup_info = None
            experiment_ids = self.persistence.list_experiments()

            # Find the backup across all experiments
            for exp_id in experiment_ids:
                try:
                    dataset = self.persistence.load_dataset(exp_id)
                    for backup in dataset.backup_info:
                        if backup.backup_id == backup_id:
                            backup_info = backup
                            break
                    if backup_info:
                        break
                except Exception:
                    continue

            if not backup_info:
                raise StorageError(f"Backup {backup_id} not found")

            # Check if backup file exists
            backup_path = Path(backup_info.backup_path)
            if not backup_path.exists():
                raise StorageError(f"Backup file not found: {backup_path}")

            # Restore based on backup type
            if backup_info.backup_type == "full":
                restored_dataset = self._restore_full_backup(backup_path)
            elif backup_info.backup_type == "incremental":
                restored_dataset = self._restore_incremental_backup(backup_path, backup_info)
            else:
                raise StorageError(f"Unsupported backup type: {backup_info.backup_type}")

            # Generate new experiment ID for restored dataset
            original_id = restored_dataset.metadata.experiment_id
            restored_dataset.metadata.experiment_id = (
                f"{original_id}_restored_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            restored_dataset.metadata.created_at = datetime.now()
            restored_dataset.metadata.updated_at = datetime.now()
            restored_dataset.metadata.description = (
                f"Restored from backup {backup_id}: {restored_dataset.metadata.description}"
            )

            # Store the restored dataset
            new_experiment_id = self.store_dataset(
                restored_dataset, validate=True, create_backup=False
            )

            self.logger.info(f"Restored dataset from backup {backup_id} as {new_experiment_id}")
            return new_experiment_id

        except Exception as e:
            self.logger.error(f"Failed to restore from backup {backup_id}: {str(e)}")
            raise StorageError(f"Failed to restore from backup {backup_id}: {str(e)}")

    def _restore_full_backup(self, backup_path: Path) -> ExperimentalDataset:
        """Restore dataset from full backup."""
        import tarfile
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract backup
            with tarfile.open(backup_path, "r:gz") as tar:
                _safe_extract(tar, temp_path)

            # Find the dataset file
            dataset_files = list(temp_path.glob("*.pkl"))
            if not dataset_files:
                raise StorageError("No dataset file found in backup")

            # Load the dataset securely
            from ..security.secure_pickle import safe_pickle_load

            dataset = safe_pickle_load(dataset_files[0])

            return cast(ExperimentalDataset, dataset)

    def _restore_incremental_backup(
        self, backup_path: Path, backup_info: BackupInfo
    ) -> ExperimentalDataset:
        """Restore dataset from incremental backup."""
        # For incremental backups, we would need to apply changes to a base dataset
        # This is a simplified implementation
        return self._restore_full_backup(backup_path)

    def get_storage_stats(self, refresh: bool = False) -> StorageStats:
        """Get storage statistics and usage information.

        Args:
            refresh: Whether to refresh cached statistics

        Returns:
            StorageStats: Current storage statistics
        """
        # Check cache
        if not refresh and self._stats_cache and self._stats_cache_time:
            if datetime.now() - self._stats_cache_time < self._stats_cache_duration:
                return self._stats_cache

        try:
            stats = StorageStats()

            # Get all experiments
            experiment_ids = self.persistence.list_experiments()
            stats.total_datasets = len(experiment_ids)

            # Calculate statistics
            total_size = 0.0
            oldest_date: Optional[datetime] = None
            newest_date: Optional[datetime] = None
            category_counts: Dict[str, int] = {}
            category_sizes: Dict[str, float] = {}

            for exp_id in experiment_ids:
                try:
                    dataset = self.persistence.load_dataset(exp_id)
                    metadata = dataset.metadata

                    # Size calculation
                    size_mb = metadata.total_size_mb
                    total_size += size_mb

                    # Date tracking
                    if oldest_date is None or metadata.created_at < oldest_date:
                        oldest_date = metadata.created_at

                    if newest_date is None or metadata.created_at > newest_date:
                        newest_date = metadata.created_at

                    # Category statistics
                    category = metadata.category
                    category_counts[category] = category_counts.get(category, 0) + 1
                    category_sizes[category] = category_sizes.get(category, 0.0) + size_mb

                except Exception as e:
                    self.logger.warning(f"Failed to get stats for {exp_id}: {str(e)}")
                    continue

            # Update statistics
            stats.total_size_mb = total_size
            stats.oldest_dataset = oldest_date
            stats.newest_dataset = newest_date
            stats.datasets_by_category = category_counts
            stats.size_by_category = category_sizes

            if stats.total_datasets > 0:
                stats.average_dataset_size_mb = total_size / stats.total_datasets

            # Calculate storage efficiency (would need compression info)
            stats.storage_efficiency = 1.0  # Placeholder

            # Cache results
            self._stats_cache = stats
            self._stats_cache_time = datetime.now()

            return stats

        except Exception as e:
            self.logger.error(f"Failed to calculate storage stats: {str(e)}")
            raise StorageError(f"Failed to calculate storage stats: {str(e)}")

    def cleanup_old_backups(self, retention_days: int = 30) -> int:
        """Clean up old backups based on retention policy.

        Args:
            retention_days: Number of days to retain backups

        Returns:
            int: Number of backups cleaned up
        """
        try:
            cleanup_count = 0

            # This would query backup records and remove old ones
            # Implementation depends on backup storage format

            self.logger.info(f"Cleaned up {cleanup_count} old backups")
            return cleanup_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {str(e)}")
            raise StorageError(f"Failed to cleanup old backups: {str(e)}")

    def validate_all_datasets(self) -> Dict[str, Any]:
        """Validate all stored datasets and return summary.

        Returns:
            Dict with validation summary and results
        """
        try:
            experiment_ids = self.persistence.list_experiments()

            validation_summary: Dict[str, Any] = {
                "total_datasets": len(experiment_ids),
                "valid_datasets": 0,
                "invalid_datasets": 0,
                "validation_errors": {},
                "validation_warnings": {},
                "average_quality_score": 0.0,
            }

            total_quality = 0.0

            for exp_id in experiment_ids:
                try:
                    dataset = self.load_dataset(exp_id)
                    validation_result = self.validator.validate_dataset(dataset)

                    if validation_result["is_valid"]:
                        validation_summary["valid_datasets"] = (
                            validation_summary["valid_datasets"] + 1
                        )
                    else:
                        validation_summary["invalid_datasets"] = (
                            validation_summary["invalid_datasets"] + 1
                        )
                        validation_summary["validation_errors"][exp_id] = validation_result[
                            "errors"
                        ]

                    if validation_result["warnings"]:
                        validation_summary["validation_warnings"][exp_id] = validation_result[
                            "warnings"
                        ]

                    total_quality += validation_result["quality_score"]

                except Exception as e:
                    validation_summary["invalid_datasets"] = (
                        validation_summary["invalid_datasets"] + 1
                    )
                    validation_summary["validation_errors"][exp_id] = [
                        f"Failed to validate: {str(e)}"
                    ]

            if validation_summary["total_datasets"] > 0:
                validation_summary["average_quality_score"] = (
                    total_quality / validation_summary["total_datasets"]
                )

            self.logger.info(f"Validated {validation_summary['total_datasets']} datasets")
            return validation_summary

        except Exception as e:
            self.logger.error(f"Failed to validate all datasets: {str(e)}")
            raise StorageError(f"Failed to validate all datasets: {str(e)}")

    def _increment_version(self, current_version: str) -> str:
        """Increment version number."""
        try:
            parts = current_version.split(".")
            if len(parts) == 3:
                major, minor, patch = map(int, parts)
                return f"{major}.{minor}.{patch + 1}"
            else:
                return "1.0.1"
        except (ValueError, TypeError, AttributeError):
            return "1.0.1"

    def _invalidate_stats_cache(self) -> None:
        """Invalidate cached statistics."""
        self._stats_cache = None
        self._stats_cache_time = None

    def flush_all(self) -> bool:
        """Flush all pending data operations and ensure data persistence.

        Returns:
            bool: True if flush successful
        """
        try:
            # Flush persistence layer
            self.persistence.flush_all()

            # Invalidate caches
            self._invalidate_stats_cache()  # type: ignore[no-untyped-call]

            self.logger.info("Flushed all pending data operations")
            return True

        except Exception as e:
            self.logger.error(f"Failed to flush data operations: {str(e)}")
            raise StorageError(f"Failed to flush data operations: {str(e)}")

    def export_metadata(self, output_path: Union[str, Path], format: str = "json") -> Path:
        """Export all experiment metadata to file.

        Args:
            output_path: Path for exported file
            format: Export format ('json', 'csv')

        Returns:
            Path: Path to exported file
        """
        try:
            output_path = Path(output_path)

            # Get all metadata
            metadata_list = self.query_datasets()

            if format.lower() == "json":
                import json

                # Convert to serializable format
                export_data = []
                for metadata in metadata_list:
                    data = {
                        "experiment_id": metadata.experiment_id,
                        "experiment_name": metadata.experiment_name,
                        "description": metadata.description,
                        "created_at": metadata.created_at.isoformat(),
                        "updated_at": metadata.updated_at.isoformat(),
                        "researcher": metadata.researcher,
                        "institution": metadata.institution,
                        "n_participants": metadata.n_participants,
                        "n_trials": metadata.n_trials,
                        "conditions": metadata.conditions,
                        "parameters": metadata.parameters,
                        "data_format": metadata.data_format,
                        "total_size_mb": metadata.total_size_mb,
                        "current_version": metadata.current_version,
                        "tags": metadata.tags,
                        "category": metadata.category,
                        "data_quality_score": metadata.data_quality_score,
                        "completeness_percentage": metadata.completeness_percentage,
                        "validation_status": metadata.validation_status,
                    }
                    export_data.append(data)

                with open(output_path, "w") as f:
                    json.dump(export_data, f, indent=2)

            elif format.lower() == "csv":
                import pandas as pd

                # Convert to DataFrame
                df_data = []
                for metadata in metadata_list:
                    row = {
                        "experiment_id": metadata.experiment_id,
                        "experiment_name": metadata.experiment_name,
                        "description": metadata.description,
                        "created_at": metadata.created_at,
                        "updated_at": metadata.updated_at,
                        "researcher": metadata.researcher,
                        "institution": metadata.institution,
                        "n_participants": metadata.n_participants,
                        "n_trials": metadata.n_trials,
                        "conditions": str(metadata.conditions),
                        "data_format": metadata.data_format,
                        "total_size_mb": metadata.total_size_mb,
                        "current_version": metadata.current_version,
                        "tags": str(metadata.tags),
                        "category": metadata.category,
                        "data_quality_score": metadata.data_quality_score,
                        "completeness_percentage": metadata.completeness_percentage,
                        "validation_status": metadata.validation_status,
                    }
                    df_data.append(row)

                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False)

            else:
                raise StorageError(f"Unsupported export format: {format}")

            self.logger.info(f"Exported metadata to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to export metadata: {str(e)}")
            raise StorageError(f"Failed to export metadata: {str(e)}")

    # Convenience methods for simpler API
    def save(self, data: dict, filepath: Union[str, Path]) -> str:
        """Save data to file (JSON format).

        Args:
            data: Dictionary to save
            filepath: Path for output file

        Returns:
            Path to saved file
        """
        import json

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Saved data to {filepath}")
        return str(filepath)

    def load(self, filepath: Union[str, Path]) -> dict:
        """Load data from file (JSON format).

        Args:
            filepath: Path to file

        Returns:
            Loaded data dictionary
        """
        import json

        filepath = Path(filepath)

        with open(filepath, "r") as f:
            data = json.load(f)

        return cast(Dict[Any, Any], data)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check if file exists.

        Args:
            filepath: Path to check

        Returns:
            True if file exists
        """
        return Path(filepath).exists()

    def delete(self, filepath: Union[str, Path]) -> bool:
        """Delete file.

        Args:
            filepath: Path to delete

        Returns:
            True if deleted successfully
        """
        filepath = Path(filepath)
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def list_files(self, directory: Union[str, Path], pattern: str = "*") -> list:
        """List files in directory matching pattern.

        Args:
            directory: Directory to list
            pattern: Glob pattern to match

        Returns:
            List of file paths
        """
        directory = Path(directory)
        if not directory.exists():
            return []

        return [str(f) for f in directory.glob(pattern) if f.is_file()]

    def get_file_info(self, filepath: Union[str, Path]) -> dict:
        """Get file information.

        Args:
            filepath: Path to file

        Returns:
            Dictionary with file info
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return {"size": 0, "modified": None, "exists": False}

        stat = filepath.stat()
        return {
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "exists": True,
        }

    def batch_save(self, datasets: dict, directory: Union[str, Path]) -> dict:
        """Save multiple datasets to directory.

        Args:
            datasets: Dictionary mapping filenames to data
            directory: Target directory

        Returns:
            Dictionary mapping filenames to success status
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        results = {}
        for filename, data in datasets.items():
            try:
                filepath = directory / f"{filename}.json"
                self.save(data, filepath)
                results[filename] = True
            except Exception as e:
                self.logger.warning(f"Failed to save {filename}: {e}")
                results[filename] = False

        return results
