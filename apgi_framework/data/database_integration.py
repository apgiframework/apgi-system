"""Database Integration for Large Dataset Management
===============================================

This module provides enhanced database integration for handling large datasets
in the APGI Framework, including PostgreSQL optimization, data chunking, and
streaming capabilities.

Features:
- PostgreSQL backend with connection pooling
- Large dataset chunking and streaming
- Optimized queries for big data
- Data compression and archiving
- Background processing for large operations
"""

import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional, Union

try:
    import numpy as np
    import pandas as pd
    from psycopg2.extras import Json
    from psycopg2.pool import ThreadedConnectionPool

    PGSQL_AVAILABLE = True
except ImportError as e:
    PGSQL_AVAILABLE = False
    warnings.warn(f"PostgreSQL not available: {e}", ImportWarning)

from ..config.manager import get_config_manager
from ..exceptions import APGIFrameworkError
from .storage_manager import StorageManager


class DatabaseIntegrationError(APGIFrameworkError):
    """Database integration specific errors."""


class LargeDatasetHandler:
    """Handles large datasets with chunking and streaming capabilities.

    Provides methods for processing datasets that are too large to fit in memory
    by using database operations and streaming.
    """

    def __init__(self, storage_manager: StorageManager):
        """Initialize large dataset handler.

        Args:
            storage_manager: Existing storage manager instance
        """
        self.storage_manager = storage_manager
        self.config = get_config_manager()
        self.logger = logging.getLogger(__name__)

        # Database configuration
        self.db_config = self.config.get_database_config()
        self._pool: Optional[ThreadedConnectionPool] = None

        # Chunking configuration
        self.chunk_size = 10000  # Number of records per chunk
        self.max_memory_mb = 500  # Maximum memory usage per operation

        # Initialize database if available
        if PGSQL_AVAILABLE and get_config_manager().is_feature_enabled("database_integration"):
            self._init_database()

    def _init_database(self) -> None:
        """Initialize PostgreSQL database connection pool."""
        try:
            # Build connection string
            if self.db_config.username and self.db_config.password:
                conn_string = (
                    f"postgresql://{self.db_config.username}:{self.db_config.password}"
                    f"@{self.db_config.host}:{self.db_config.port}/{self.db_config.database}"
                )
            else:
                # Try environment variable
                conn_string = os.getenv("DATABASE_URL", "")
                if not conn_string:
                    self.logger.warning(
                        "Database credentials not configured. Using SQLite fallback."
                    )
                    return

            # Create connection pool
            self._pool = ThreadedConnectionPool(
                minconn=1, maxconn=self.db_config.pool_size, dsn=conn_string
            )

            # Initialize database schema
            self._init_database_schema()

            self.logger.info("Database integration initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise DatabaseIntegrationError(f"Database initialization failed: {e}")

    def _init_database_schema(self) -> None:
        """Initialize database schema for large dataset handling."""
        if not self._pool:
            return

        with self._pool.getconn() as conn:
            with conn.cursor() as cursor:
                # Create experiments table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS experiments (
                        experiment_id VARCHAR(255) PRIMARY KEY,
                        experiment_name TEXT NOT NULL,
                        description TEXT,
                        researcher TEXT,
                        institution TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        n_participants INTEGER,
                        n_trials INTEGER,
                        conditions JSONB,
                        parameters JSONB,
                        data_format TEXT,
                        total_size_mb FLOAT,
                        status TEXT DEFAULT 'active',
                        metadata JSONB
                    )
                """)

                # Create large datasets table for chunked data
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS large_datasets (
                        id SERIAL PRIMARY KEY,
                        experiment_id VARCHAR(255) REFERENCES experiments(experiment_id),
                        chunk_number INTEGER NOT NULL,
                        total_chunks INTEGER NOT NULL,
                        data_type VARCHAR(50) NOT NULL,
                        data_json JSONB,
                        data_binary BYTEA,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        size_bytes INTEGER,
                        compressed BOOLEAN DEFAULT FALSE,
                        UNIQUE(experiment_id, chunk_number, data_type)
                    )
                """)

                # Create indexes for performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_large_datasets_experiment 
                    ON large_datasets(experiment_id)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_large_datasets_created_at 
                    ON large_datasets(created_at)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_experiments_created_at 
                    ON experiments(created_at)
                """)

                # Create data processing queue
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_processing_queue (
                        id SERIAL PRIMARY KEY,
                        experiment_id VARCHAR(255),
                        operation_type VARCHAR(50),
                        status VARCHAR(20) DEFAULT 'pending',
                        priority INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        error_message TEXT,
                        parameters JSONB
                    )
                """)

                conn.commit()

    def store_large_dataset(
        self,
        experiment_id: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame, np.ndarray],
        data_type: str = "raw_data",
        compress: bool = True,
    ) -> bool:
        """Store large dataset in chunks using database.

        Args:
            experiment_id: Experiment identifier
            data: Data to store (dict, list, DataFrame, or array)
            data_type: Type of data (raw_data, processed_data, etc.)
            compress: Whether to compress the data

        Returns:
            bool: True if successful
        """
        if not self._pool:
            self.logger.warning("Database not available, falling back to file storage")
            return False

        try:
            # Convert data to appropriate format
            if isinstance(data, pd.DataFrame):
                return self._store_dataframe(experiment_id, data, data_type, compress)
            elif isinstance(data, np.ndarray):
                return self._store_numpy_array(experiment_id, data, data_type, compress)
            else:
                # isinstance(data, (list, dict)) - all other Union types
                return self._store_json_data(experiment_id, data, data_type, compress)

        except Exception as e:
            self.logger.error(f"Failed to store large dataset: {e}")
            raise DatabaseIntegrationError(f"Large dataset storage failed: {e}")

    def _store_dataframe(
        self, experiment_id: str, df: pd.DataFrame, data_type: str, compress: bool
    ) -> bool:
        """Store pandas DataFrame in chunks."""
        try:
            total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size

            if self._pool is None:
                raise DatabaseIntegrationError("Database pool not initialized")
            with self._pool.getconn() as conn:
                with conn.cursor() as cursor:
                    for chunk_idx, chunk_start in enumerate(range(0, len(df), self.chunk_size)):
                        chunk_end = min(chunk_start + self.chunk_size, len(df))
                        chunk_df = df.iloc[chunk_start:chunk_end]

                        # Convert to JSON for storage
                        data_json = chunk_df.to_dict("records")

                        # Store chunk
                        cursor.execute(
                            """
                            INSERT INTO large_datasets 
                            (experiment_id, chunk_number, total_chunks, data_type, 
                             data_json, created_at, size_bytes, compressed)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                            (
                                experiment_id,
                                chunk_idx,
                                total_chunks,
                                data_type,
                                Json(data_json),
                                datetime.now(),
                                len(str(data_json)),
                                compress,
                            ),
                        )

                    conn.commit()

            self.logger.info(f"Stored DataFrame for {experiment_id} in {total_chunks} chunks")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store DataFrame: {e}")
            return False

    def _store_numpy_array(
        self, experiment_id: str, array: np.ndarray, data_type: str, compress: bool
    ) -> bool:
        """Store numpy array in database."""
        try:
            # Flatten array and get metadata
            flattened = array.flatten()
            total_chunks = (len(flattened) + self.chunk_size - 1) // self.chunk_size

            if self._pool is None:
                raise DatabaseIntegrationError("Database pool not initialized")
            with self._pool.getconn() as conn:
                with conn.cursor() as cursor:
                    for chunk_idx in range(total_chunks):
                        start_idx = chunk_idx * self.chunk_size
                        end_idx = min(start_idx + self.chunk_size, len(flattened))
                        chunk_data = flattened[start_idx:end_idx]

                        # Store as binary data
                        cursor.execute(
                            """
                            INSERT INTO large_datasets 
                            (experiment_id, chunk_number, total_chunks, data_type, 
                             data_binary, created_at, size_bytes, compressed)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                            (
                                experiment_id,
                                chunk_idx,
                                total_chunks,
                                data_type,
                                chunk_data.tobytes(),
                                datetime.now(),
                                len(chunk_data.tobytes()),
                                compress,
                            ),
                        )

                    conn.commit()

            self.logger.info(f"Stored numpy array for {experiment_id} in {total_chunks} chunks")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store numpy array: {e}")
            return False

    def _store_json_data(
        self,
        experiment_id: str,
        data: Union[Dict, List],
        data_type: str,
        compress: bool,
    ) -> bool:
        """Store JSON data in chunks."""
        try:
            # Convert to list of records if it's a dict
            if isinstance(data, dict):
                data = [data]

            total_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size

            if self._pool is None:
                raise DatabaseIntegrationError("Database pool not initialized")
            with self._pool.getconn() as conn:
                with conn.cursor() as cursor:
                    for chunk_idx in range(total_chunks):
                        start_idx = chunk_idx * self.chunk_size
                        end_idx = min(start_idx + self.chunk_size, len(data))
                        chunk_data = data[start_idx:end_idx]

                        cursor.execute(
                            """
                            INSERT INTO large_datasets 
                            (experiment_id, chunk_number, total_chunks, data_type, 
                             data_json, created_at, size_bytes, compressed)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                            (
                                experiment_id,
                                chunk_idx,
                                total_chunks,
                                data_type,
                                Json(chunk_data),
                                datetime.now(),
                                len(str(chunk_data)),
                                compress,
                            ),
                        )

                    conn.commit()

            self.logger.info(f"Stored JSON data for {experiment_id} in {total_chunks} chunks")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store JSON data: {e}")
            return False

    def load_large_dataset(
        self,
        experiment_id: str,
        data_type: str = "raw_data",
        chunk_size: Optional[int] = None,
    ) -> Generator[Any, None, None]:
        """Load large dataset in chunks to avoid memory issues.

        Args:
            experiment_id: Experiment identifier
            data_type: Type of data to load
            chunk_size: Override default chunk size

        Yields:
            Data chunks in the original format
        """
        if not self._pool:
            self.logger.warning("Database not available")
            return

        try:
            chunk_size = chunk_size or self.chunk_size

            if self._pool is None:
                raise DatabaseIntegrationError("Database pool not initialized")
            with self._pool.getconn() as conn:
                with conn.cursor() as cursor:
                    # Get all chunks for this experiment and data type
                    cursor.execute(
                        """
                        SELECT chunk_number, data_json, data_binary, total_chunks
                        FROM large_datasets
                        WHERE experiment_id = %s AND data_type = %s
                        ORDER BY chunk_number
                    """,
                        (experiment_id, data_type),
                    )

                    chunks = cursor.fetchall()

                    if not chunks:
                        self.logger.warning(f"No data found for {experiment_id}, type {data_type}")
                        return

                    # Reconstruct data from chunks
                    if chunks[0][2]:  # data_binary (numpy array)
                        # Handle numpy array
                        all_data = []
                        for chunk_data in chunks:
                            chunk_bytes = chunk_data[2]
                            chunk_array = np.frombuffer(chunk_bytes, dtype=np.float64)
                            all_data.append(chunk_array)

                        # Combine all chunks
                        combined_array = np.concatenate(all_data)
                        yield combined_array

                    elif chunks[0][1]:  # data_json (DataFrame or JSON)
                        # Handle JSON data
                        all_records = []
                        for chunk_data in chunks:
                            chunk_records = chunk_data[1]
                            all_records.extend(chunk_records)

                        # Try to convert to DataFrame first
                        try:
                            df = pd.DataFrame(all_records)
                            yield df
                        except (ValueError, TypeError):
                            # Fall back to raw JSON
                            yield all_records

        except Exception as e:
            self.logger.error(f"Failed to load large dataset: {e}")
            raise DatabaseIntegrationError(f"Large dataset loading failed: {e}")

    def query_large_datasets(
        self,
        filter_criteria: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query large datasets with filtering.

        Args:
            filter_criteria: Dictionary of filter conditions
            limit: Maximum number of results to return

        Returns:
            List of dataset metadata
        """
        if not self._pool:
            self.logger.warning("Database not available")
            return []

        try:
            if self._pool is None:
                raise DatabaseIntegrationError("Database pool not initialized")
            with self._pool.getconn() as conn:
                with conn.cursor() as cursor:
                    query = """
                        SELECT DISTINCT e.experiment_id, e.experiment_name, 
                               e.description, e.researcher, e.institution,
                               e.created_at, e.updated_at, e.n_participants,
                               e.n_trials, e.conditions, e.parameters,
                               e.data_format, e.total_size_mb, e.status,
                               COUNT(ld.id) as chunk_count,
                               SUM(ld.size_bytes) as total_bytes
                        FROM experiments e
                        LEFT JOIN large_datasets ld ON e.experiment_id = ld.experiment_id
                        WHERE e.experiment_id IN (
                            SELECT DISTINCT experiment_id FROM large_datasets
                        )
                    """

                    params = []

                    # Add filter conditions
                    if filter_criteria:
                        if "researcher" in filter_criteria:
                            query += " AND e.researcher ILIKE %s"
                            params.append(f"%{filter_criteria['researcher']}%")

                        if "date_range" in filter_criteria:
                            query += " AND e.created_at BETWEEN %s AND %s"
                            params.extend(filter_criteria["date_range"])

                        if "status" in filter_criteria:
                            query += " AND e.status = %s"
                            params.append(filter_criteria["status"])

                    query += " GROUP BY e.experiment_id, e.experiment_name, e.description, e.researcher, e.institution, e.created_at, e.updated_at, e.n_participants, e.n_trials, e.conditions, e.parameters, e.data_format, e.total_size_mb, e.status"

                    query += " ORDER BY e.created_at DESC"

                    if limit:
                        query += " LIMIT %s"
                        params.append(str(limit))

                    cursor.execute(query, params)

                    results = []
                    for row in cursor.fetchall():
                        results.append(
                            {
                                "experiment_id": row[0],
                                "experiment_name": row[1],
                                "description": row[2],
                                "researcher": row[3],
                                "institution": row[4],
                                "created_at": row[5],
                                "updated_at": row[6],
                                "n_participants": row[7],
                                "n_trials": row[8],
                                "conditions": row[9],
                                "parameters": row[10],
                                "data_format": row[11],
                                "total_size_mb": row[12],
                                "status": row[13],
                                "chunk_count": row[14],
                                "total_bytes": row[15],
                            }
                        )

                    return results

        except Exception as e:
            self.logger.error(f"Failed to query large datasets: {e}")
            return []

    def cleanup_old_data(self, retention_days: int = 30) -> int:
        """Clean up old data from database.

        Args:
            retention_days: Number of days to retain data

        Returns:
            int: Number of records cleaned up
        """
        if not self._pool:
            self.logger.warning("Database not available")
            return 0

        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            if self._pool is None:
                raise DatabaseIntegrationError("Database pool not initialized")
            with self._pool.getconn() as conn:
                with conn.cursor() as cursor:
                    # Delete old large dataset chunks
                    cursor.execute(
                        """
                        DELETE FROM large_datasets 
                        WHERE created_at < %s
                    """,
                        (cutoff_date,),
                    )

                    chunk_count = cursor.rowcount

                    # Clean up experiments without data
                    cursor.execute(
                        """
                        DELETE FROM experiments 
                        WHERE experiment_id NOT IN (
                            SELECT DISTINCT experiment_id FROM large_datasets
                        ) AND created_at < %s
                    """,
                        (cutoff_date,),
                    )

                    exp_count = cursor.rowcount

                    conn.commit()

            total_cleaned: int = chunk_count + exp_count
            self.logger.info(f"Cleaned up {total_cleaned} old database records")
            return total_cleaned

        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
            return 0

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and performance metrics.

        Returns:
            Dictionary with database statistics
        """
        if not self._pool:
            return {"status": "not_available"}

        try:
            if self._pool is None:
                raise DatabaseIntegrationError("Database pool not initialized")
            with self._pool.getconn() as conn:
                with conn.cursor() as cursor:
                    # Get table statistics
                    cursor.execute("""
                        SELECT 
                            (SELECT COUNT(*) FROM experiments) as experiments_count,
                            (SELECT COUNT(*) FROM large_datasets) as chunks_count,
                            (SELECT SUM(size_bytes) FROM large_datasets) as total_bytes,
                            (SELECT AVG(size_bytes) FROM large_datasets) as avg_chunk_size,
                            (SELECT COUNT(DISTINCT experiment_id) FROM large_datasets) as datasets_with_chunks
                    """)

                    stats = cursor.fetchone()

                    # Get connection pool stats
                    pool_stats = {
                        "min_connections": self._pool.minconn,
                        "max_connections": self._pool.maxconn,
                        "current_connections": (
                            self._pool._used.maxlen if hasattr(self._pool, "_used") else 0
                        ),
                    }

                    return {
                        "status": "active",
                        "experiments_count": stats[0],
                        "chunks_count": stats[1],
                        "total_bytes": stats[2] or 0,
                        "avg_chunk_size": stats[3] or 0,
                        "datasets_with_chunks": stats[4] or 0,
                        "pool_stats": pool_stats,
                    }

        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {"status": "error", "error": str(e)}

    def close(self) -> None:
        """Close database connections."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            self.logger.info("Database connections closed")


# Utility function to enable database integration
def enable_database_integration(storage_manager: StorageManager) -> LargeDatasetHandler:
    """Enable database integration for large dataset handling.

    Args:
        storage_manager: Existing storage manager

    Returns:
        LargeDatasetHandler: Handler for large datasets
    """
    # Enable feature flag
    config = get_config_manager()
    config.enable_feature("database_integration")

    # Create and return handler
    return LargeDatasetHandler(storage_manager)
