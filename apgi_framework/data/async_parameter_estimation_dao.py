"""Async Data Access Object (DAO) for parameter estimation functionality.

Provides async CRUD operations for parameter estimation sessions, trials, and estimates
using asyncio thread pool executor for non-blocking database operations.
"""

import asyncio
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from ..exceptions import APGIFrameworkError
from .parameter_estimation_dao import ParameterEstimationDAO
from .parameter_estimation_models import (
    ParameterEstimates,
    SessionData,
    TrialData,
)


class AsyncParameterEstimationDAOError(APGIFrameworkError):
    """Errors in async parameter estimation data access operations."""


class AsyncParameterEstimationDAO:
    """Async Data Access Object for parameter estimation functionality.

    Provides high-level async CRUD operations for sessions, trials, and parameter
    estimates by wrapping the synchronous DAO with asyncio thread pool execution.
    """

    def __init__(self, db_path: Path, max_workers: int = 4):
        """Initialize async DAO with database connection.

        Args:
            db_path: Path to SQLite database file
            max_workers: Maximum number of threads in the executor pool
        """
        self.db_path = Path(db_path)
        self._sync_dao: Optional[ParameterEstimationDAO] = None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = asyncio.Lock()

    async def _get_sync_dao(self) -> ParameterEstimationDAO:
        """Lazy initialization of synchronous DAO in thread-safe manner."""
        if self._sync_dao is None:
            async with self._lock:
                if self._sync_dao is None:
                    # Run synchronous initialization in executor
                    loop = asyncio.get_event_loop()
                    self._sync_dao = await loop.run_in_executor(
                        self._executor, lambda: ParameterEstimationDAO(self.db_path)
                    )
        return self._sync_dao

    async def create_session(self, session_data: SessionData) -> str:
        """Create a new parameter estimation session asynchronously.

        Args:
            session_data: SessionData object to store

        Returns:
            str: Session ID of created session
        """
        dao = await self._get_sync_dao()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, dao.create_session, session_data)

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve session by ID asynchronously.

        Args:
            session_id: Session identifier

        Returns:
            SessionData: Session data or None if not found
        """
        dao = await self._get_sync_dao()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, dao.get_session, session_id)

    async def update_session(self, session_data: SessionData) -> None:
        """Update existing session asynchronously.

        Args:
            session_data: Updated session data
        """
        dao = await self._get_sync_dao()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, dao.update_session, session_data)

    async def list_sessions(
        self, participant_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[str]:
        """List session IDs asynchronously, optionally filtered by participant.

        Args:
            participant_id: Filter by participant ID
            limit: Maximum number of sessions to return

        Returns:
            List[str]: List of session IDs
        """
        dao = await self._get_sync_dao()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, lambda: dao.list_sessions(participant_id, limit)
        )

    async def create_trial(self, trial_data: TrialData) -> str:
        """Create a new trial record asynchronously.

        Args:
            trial_data: Trial data to store

        Returns:
            str: Trial ID of created trial
        """
        dao = await self._get_sync_dao()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, dao.create_trial, trial_data)

    async def create_trials_batch(self, trials: List[TrialData]) -> List[str]:
        """Create multiple trial records asynchronously in a single transaction.

        Args:
            trials: List of trial data objects

        Returns:
            List[str]: List of created trial IDs
        """
        dao = await self._get_sync_dao()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, dao.create_trials_batch, trials)

    async def get_detection_trials(self, session_id: str) -> List:
        """Get all detection trials for a session asynchronously.

        Args:
            session_id: Session identifier

        Returns:
            List[DetectionTrialResult]: List of detection trials
        """
        dao = await self._get_sync_dao()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, dao.get_detection_trials, session_id)

    async def get_heartbeat_trials(self, session_id: str) -> List:
        """Get all heartbeat trials for a session asynchronously.

        Args:
            session_id: Session identifier

        Returns:
            List[HeartbeatTrialResult]: List of heartbeat trials
        """
        dao = await self._get_sync_dao()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, dao.get_heartbeat_trials, session_id)

    async def get_oddball_trials(self, session_id: str) -> List:
        """Get all oddball trials for a session asynchronously.

        Args:
            session_id: Session identifier

        Returns:
            List[OddballTrialResult]: List of oddball trials
        """
        dao = await self._get_sync_dao()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, dao.get_oddball_trials, session_id)

    async def create_parameter_estimates(self, estimates: ParameterEstimates) -> str:
        """Store parameter estimates for a session asynchronously.

        Args:
            estimates: ParameterEstimates to store

        Returns:
            str: Estimate ID
        """
        dao = await self._get_sync_dao()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, dao.create_parameter_estimates, estimates)

    async def get_parameter_estimates(self, session_id: str) -> Optional[ParameterEstimates]:
        """Retrieve parameter estimates for a session asynchronously.

        Args:
            session_id: Session identifier

        Returns:
            ParameterEstimates: Parameter estimates or None if not found
        """
        dao = await self._get_sync_dao()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, dao.get_parameter_estimates, session_id)

    async def close(self) -> None:
        """Clean up resources and shutdown executor."""
        self._executor.shutdown(wait=True)
        self._sync_dao = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class AsyncParameterEstimationDAOWithPooling:
    """Async DAO with connection pooling support for high-concurrency scenarios.

    Maintains a pool of sync DAO instances for improved parallel performance.
    """

    def __init__(
        self,
        db_path: Path,
        pool_size: int = 5,
        max_workers: int = 10,
    ):
        """Initialize pooled async DAO.

        Args:
            db_path: Path to SQLite database file
            pool_size: Number of DAO instances to maintain in pool
            max_workers: Maximum number of threads in the executor pool
        """
        self.db_path = Path(db_path)
        self._pool_size = pool_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._dao_pool: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self._pool_lock = asyncio.Lock()
        self._initialized = False

    async def _initialize_pool(self) -> None:
        """Initialize the DAO pool with instances."""
        if self._initialized:
            return

        async with self._pool_lock:
            if self._initialized:
                return

            loop = asyncio.get_event_loop()
            for _ in range(self._pool_size):
                dao = await loop.run_in_executor(
                    self._executor, lambda: ParameterEstimationDAO(self.db_path)
                )
                await self._dao_pool.put(dao)

            self._initialized = True

    async def _acquire_dao(self) -> ParameterEstimationDAO:
        """Acquire a DAO instance from the pool."""
        await self._initialize_pool()
        return await self._dao_pool.get()

    def _release_dao(self, dao: ParameterEstimationDAO) -> None:
        """Release a DAO instance back to the pool."""
        try:
            self._dao_pool.put_nowait(dao)
        except asyncio.QueueFull:
            # Pool is full, instance will be garbage collected
            pass

    async def execute_with_dao(self, operation, *args, **kwargs):
        """Execute an operation with a pooled DAO instance.

        Args:
            operation: Callable that takes a DAO as first argument
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Result of the operation
        """
        dao = await self._acquire_dao()
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor, lambda: operation(dao, *args, **kwargs)
            )
        finally:
            self._release_dao(dao)

    async def close(self) -> None:
        """Clean up resources and shutdown executor."""
        self._executor.shutdown(wait=True)
        # Clear the pool
        while not self._dao_pool.empty():
            try:
                self._dao_pool.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Deprecation warning for the old sync DAO
warnings.warn(
    "ParameterEstimationDAO is deprecated. Use AsyncParameterEstimationDAO instead.",
    DeprecationWarning,
    stacklevel=2,
)
