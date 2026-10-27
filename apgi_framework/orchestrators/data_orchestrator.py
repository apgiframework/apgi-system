"""Data Orchestrator for data management and persistence.

Handles data initialization, validation, storage, and retrieval operations.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from api.config import DEFAULT_BATCH_SIZE, DEFAULT_CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data orchestrator."""

    enable_caching: bool = True
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS
    batch_size: int = DEFAULT_BATCH_SIZE
    enable_compression: bool = False


class DataOrchestrator:
    """Orchestrates data management operations.

    Responsibilities:
    - Initialize data storage systems
    - Manage data validation
    - Handle data persistence
    - Coordinate data exports
    """

    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize the data orchestrator.

        Args:
            config: Configuration for data orchestrator
        """
        self.config = config or DataConfig()
        self._data_manager: Optional[Any] = None
        self._storage_manager: Optional[Any] = None
        self._initialized = False

        logger.info("DataOrchestrator initialized with config: %s", self.config)

    def initialize(self) -> None:
        """Initialize data management systems."""
        if self._initialized:
            logger.warning("DataOrchestrator already initialized")
            return

        try:
            self._initialize_data_systems()
            self._initialized = True
            logger.info("DataOrchestrator initialization complete")
        except Exception as e:
            logger.error("Failed to initialize DataOrchestrator: %s", e)
            raise

    def _initialize_data_systems(self) -> None:
        """Initialize data management and storage systems."""
        from apgi_framework.data.data_manager import (
            IntegratedDataManager,  # type: ignore[attr-defined]
        )
        from apgi_framework.data.storage_manager import StorageManager

        self._data_manager = IntegratedDataManager()
        self._storage_manager = StorageManager()

        logger.debug("Data systems initialized")

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate data structure and content.

        Args:
            data: Data to validate

        Returns:
            True if data is valid
        """
        if not self._initialized:
            raise RuntimeError("DataOrchestrator not initialized")

        return self._data_manager.validate(data)

    def store_data(self, data_id: str, data: Dict[str, Any]) -> None:
        """Store data persistently.

        Args:
            data_id: Identifier for the data
            data: Data to store
        """
        if not self._initialized:
            raise RuntimeError("DataOrchestrator not initialized")

        if not self.validate_data(data):
            raise ValueError("Invalid data structure")

        self._storage_manager.store(data_id, data)
        logger.debug("Data stored: %s", data_id)

    def retrieve_data(self, data_id: str) -> Dict[str, Any]:
        """Retrieve stored data.

        Args:
            data_id: Identifier for the data

        Returns:
            Retrieved data
        """
        if not self._initialized:
            raise RuntimeError("DataOrchestrator not initialized")

        data = self._storage_manager.retrieve(data_id)
        logger.debug("Data retrieved: %s", data_id)
        return data

    def export_data(self, data_id: str, format: str) -> str:
        """Export data in specified format.

        Args:
            data_id: Identifier for the data
            format: Export format (json, csv, parquet, etc.)

        Returns:
            Path to exported file
        """
        if not self._initialized:
            raise RuntimeError("DataOrchestrator not initialized")

        data = self.retrieve_data(data_id)
        export_path = self._storage_manager.export(data, format)
        logger.info("Data exported: %s -> %s", data_id, export_path)
        return export_path

    def shutdown(self) -> None:
        """Shutdown the orchestrator and cleanup resources."""
        if self._data_manager:
            self._data_manager.close()
        if self._storage_manager:
            self._storage_manager.close()
        self._initialized = False
        logger.info("DataOrchestrator shutdown complete")
