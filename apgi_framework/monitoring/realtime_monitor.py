"""
Real-time Monitoring System for APGI Framework

Provides live data streaming, WebSocket support, and real-time visualization
updates for monitoring experiments and system status.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import threading

    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None  # type: ignore[assignment]
    threading = None  # type: ignore[assignment]

try:
    from apgi_framework.logging.standardized_logging import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore [assignment]


@dataclass
class MonitoringData:
    """Data structure for monitoring information."""

    timestamp: float
    experiment_id: str
    data_type: str
    value: Any
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SystemStatus:
    """System status information."""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    experiment_count: int
    timestamp: float


class RealtimeDataStreamer:
    """
    Real-time data streaming manager.

    Handles WebSocket connections, data broadcasting, and client management
    for real-time monitoring capabilities.
    """

    def __init__(self, port: int = 8765):
        """
        Initialize real-time data streamer.

        Args:
            port: WebSocket server port
        """
        self.port = port
        self.clients: set = set()
        self.data_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.server: Optional[Any] = None

        # Data subscribers
        self.subscribers: Dict[str, set] = {}

        logger.info(f"RealtimeDataStreamer initialized on port {port}")

    async def register_client(self, websocket: Any) -> None:
        """Register a new WebSocket client."""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")

        try:
            # Send initial status
            await self.send_to_client(
                websocket,
                {
                    "type": "connection",
                    "message": "Connected to APGI real-time monitoring",
                    "timestamp": time.time(),
                },
            )

            # Keep connection alive and handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    await self.send_to_client(
                        websocket, {"type": "error", "message": "Invalid JSON format"}
                    )

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        finally:
            self.clients.discard(websocket)

    async def handle_client_message(self, websocket: Any, data: Dict[str, Any]) -> None:
        """Handle incoming messages from clients."""
        message_type = data.get("type")

        if message_type == "subscribe":
            # Subscribe to specific data types
            data_types = data.get("data_types", [])
            for data_type in data_types:
                if data_type not in self.subscribers:
                    self.subscribers[data_type] = set()
                self.subscribers[data_type].add(websocket)

            await self.send_to_client(
                websocket, {"type": "subscription_confirmed", "data_types": data_types}
            )

        elif message_type == "unsubscribe":
            # Unsubscribe from data types
            data_types = data.get("data_types", [])
            for data_type in data_types:
                if data_type in self.subscribers:
                    self.subscribers[data_type].discard(websocket)

            await self.send_to_client(
                websocket,
                {"type": "unsubscription_confirmed", "data_types": data_types},
            )

    async def send_to_client(self, websocket: Any, data: Dict[str, Any]) -> None:
        """Send data to a specific client."""
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            self.clients.discard(websocket)

    async def broadcast_data(self, data: MonitoringData) -> None:
        """Broadcast monitoring data to all subscribed clients."""
        message = {
            "type": "data",
            "timestamp": data.timestamp,
            "experiment_id": data.experiment_id,
            "data_type": data.data_type,
            "value": data.value,
            "metadata": data.metadata,
        }

        # Send to all clients subscribed to this data type
        if data.data_type in self.subscribers:
            for client in self.subscribers[data.data_type]:
                if client in self.clients:
                    await self.send_to_client(client, message)

        # Also send to all clients (for general monitoring)
        disconnected_clients = set()
        for client in self.clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)

        # Remove disconnected clients
        self.clients -= disconnected_clients

    async def broadcast_system_status(self, status: SystemStatus) -> None:
        """Broadcast system status to all clients."""
        message = {
            "type": "system_status",
            "timestamp": status.timestamp,
            "status": asdict(status),
        }

        disconnected_clients = set()
        for client in self.clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)

        self.clients -= disconnected_clients

    async def start_server(self, max_retries: int = 5, initial_delay: float = 1.0) -> bool:
        """Start the WebSocket server with retry logic and exponential backoff."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets not available. Install websockets package.")
            return False

        for attempt in range(max_retries + 1):
            try:
                self.server = await websockets.serve(self.register_client, "localhost", self.port)
                self.is_running = True
                logger.info(f"Real-time monitoring server started on ws://localhost:{self.port}")
                return True
            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        f"Failed to start WebSocket server after {max_retries + 1} attempts: {e}"
                    )
                    return False

                delay = initial_delay * (2**attempt)  # Exponential backoff
                logger.warning(
                    f"Failed to start WebSocket server (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                logger.info(f"Retrying in {delay:.1f} seconds...")
                await asyncio.sleep(delay)

        return False

    async def stop_server(self) -> None:
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.is_running = False
            logger.info("Real-time monitoring server stopped")

    def add_monitoring_data(self, data: MonitoringData) -> None:
        """Add monitoring data to the queue for broadcasting."""
        if self.is_running:
            try:
                asyncio.create_task(self.broadcast_data(data))
            except RuntimeError:
                # No event loop running, use thread
                if threading:
                    threading.Thread(target=lambda: asyncio.run(self.broadcast_data(data))).start()


class RealtimeMonitor:
    """
    High-level real-time monitoring interface.

    Provides easy-to-use interface for real-time monitoring
    with automatic data collection and broadcasting.
    """

    def __init__(self, enable_websocket: bool = True, websocket_port: int = 8765) -> None:
        """
        Initialize real-time monitor.

        Args:
            enable_websocket: Enable WebSocket server
            websocket_port: Port for WebSocket server
        """
        self.enable_websocket = enable_websocket and WEBSOCKETS_AVAILABLE

        # Check if WebSocket is requested but not available
        if enable_websocket and not WEBSOCKETS_AVAILABLE:
            logger.error(
                "WebSocket monitoring requested but websockets package not available. "
                "Install with: pip install websockets"
            )
            raise ImportError("websockets package required for WebSocket functionality")

        self.websocket_port = websocket_port

        # Initialize components
        if self.enable_websocket:
            self.streamer: Optional[RealtimeDataStreamer] = RealtimeDataStreamer(websocket_port)
        else:
            self.streamer: Optional[RealtimeDataStreamer] = None  # type: ignore
            logger.info("WebSocket monitoring disabled")

        # Data collection
        self.monitoring_data: List[MonitoringData] = []
        self.max_data_points = 1000  # Keep last 1000 data points

        # System monitoring
        self.system_status = SystemStatus(
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            active_connections=0,
            experiment_count=0,
            timestamp=time.time(),
        )

        logger.info("RealtimeMonitor initialized")

    async def start(self) -> bool:
        """Start the real-time monitoring system."""
        if self.streamer:
            success = await self.streamer.start_server()
            if not success:
                logger.error("Failed to start real-time monitoring")
                return False

        # Start system monitoring task
        asyncio.create_task(self._monitor_system())

        logger.info("Real-time monitoring started")
        return True

    async def stop(self) -> None:
        """Stop the real-time monitoring system."""
        if self.streamer:
            await self.streamer.stop_server()

        logger.info("Real-time monitoring stopped")

    def log_experiment_data(
        self,
        experiment_id: str,
        data_type: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log experiment data for real-time monitoring.

        Args:
            experiment_id: Experiment identifier
            data_type: Type of data (e.g., 'eeg', 'response_time', 'accuracy')
            value: Data value
            metadata: Additional metadata
        """
        data = MonitoringData(
            timestamp=time.time(),
            experiment_id=experiment_id,
            data_type=data_type,
            value=value,
            metadata=metadata or {},
        )

        # Add to local storage
        self.monitoring_data.append(data)

        # Keep only recent data points
        if len(self.monitoring_data) > self.max_data_points:
            self.monitoring_data = self.monitoring_data[-self.max_data_points :]

        # Broadcast via WebSocket
        if self.streamer:
            self.streamer.add_monitoring_data(data)

        logger.debug(f"Logged experiment data: {experiment_id} - {data_type} = {value}")

    def log_system_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Log system event for monitoring.

        Args:
            event_type: Type of system event
            details: Event details
        """
        self.log_experiment_data(
            experiment_id="system",
            data_type=event_type,
            value=details,
            metadata={"source": "system_monitor"},
        )

    async def _monitor_system(self) -> None:
        """Monitor system resources and status with retry logic for network failures."""
        try:
            import psutil

            consecutive_failures = 0
            max_consecutive_failures = 3
            base_delay = 5  # Base delay in seconds

            while True:
                try:
                    # Get system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage("/")

                    # Update system status
                    self.system_status = SystemStatus(
                        cpu_usage=cpu_percent,
                        memory_usage=memory.percent,
                        disk_usage=disk.percent,
                        active_connections=(len(self.streamer.clients) if self.streamer else 0),
                        experiment_count=len(
                            set(
                                d.experiment_id
                                for d in self.monitoring_data
                                if d.experiment_id != "system"
                            )
                        ),
                        timestamp=time.time(),
                    )

                    # Broadcast system status with retry
                    if self.streamer:
                        await self._broadcast_system_status_with_retry(self.system_status)

                    # Reset failure count on success
                    consecutive_failures = 0

                    # Wait before next update
                    await asyncio.sleep(base_delay)

                except (OSError, ConnectionError, asyncio.TimeoutError) as e:
                    consecutive_failures += 1
                    delay = min(base_delay * (2**consecutive_failures), 300)  # Max 5 minutes
                    logger.warning(
                        f"Network error in system monitoring (failure {consecutive_failures}/{max_consecutive_failures}): {e}"
                    )
                    logger.info(f"Retrying system monitoring in {delay} seconds...")

                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(
                            "Too many consecutive network failures, stopping system monitoring"
                        )
                        break

                    await asyncio.sleep(delay)

                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                    await asyncio.sleep(base_delay * 2)  # Wait longer on general error

        except ImportError:
            logger.warning("psutil not available - system monitoring disabled")
        except Exception as e:
            logger.error(f"Fatal system monitoring error: {e}")

    async def _broadcast_system_status_with_retry(
        self, status: SystemStatus, max_retries: int = 3
    ) -> None:
        """Broadcast system status with retry logic."""
        for attempt in range(max_retries):
            try:
                if self.streamer is not None:
                    await self.streamer.broadcast_system_status(status)
                return
            except (OSError, ConnectionError, asyncio.TimeoutError) as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to broadcast system status after {max_retries} attempts: {e}"
                    )
                    raise
                delay = 0.5 * (2**attempt)  # Exponential backoff
                logger.warning(f"Failed to broadcast system status (attempt {attempt + 1}): {e}")
                await asyncio.sleep(delay)

    def get_recent_data(
        self,
        experiment_id: Optional[str] = None,
        data_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[MonitoringData]:
        """
        Get recent monitoring data.

        Args:
            experiment_id: Filter by experiment ID
            data_type: Filter by data type
            limit: Maximum number of data points to return

        Returns:
            List of monitoring data
        """
        data = self.monitoring_data

        # Apply filters
        if experiment_id:
            data = [d for d in data if d.experiment_id == experiment_id]

        if data_type:
            data = [d for d in data if d.data_type == data_type]

        # Sort by timestamp (newest first) and limit
        data.sort(key=lambda x: x.timestamp, reverse=True)
        return data[:limit]

    def get_current_status(self) -> SystemStatus:
        """Get current system status."""
        return self.system_status

    def export_data(self, file_path: Path, experiment_id: Optional[str] = None) -> None:
        """
        Export monitoring data to file.

        Args:
            file_path: Output file path
            experiment_id: Optional experiment ID filter
        """
        data = self.get_recent_data(experiment_id=experiment_id, limit=10000)

        export_data = []
        for item in data:
            export_data.append(
                {
                    "timestamp": item.timestamp,
                    "experiment_id": item.experiment_id,
                    "data_type": item.data_type,
                    "value": item.value,
                    "metadata": item.metadata,
                }
            )

        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(export_data)} data points to {file_path}")


# Global monitor instance
_global_monitor: Optional[RealtimeMonitor] = None


def get_global_monitor() -> RealtimeMonitor:
    """Get or create the global monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RealtimeMonitor()
    return _global_monitor


def log_experiment_data(
    experiment_id: str,
    data_type: str,
    value: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Convenience function to log experiment data.

    Args:
        experiment_id: Experiment identifier
        data_type: Type of data
        value: Data value
        metadata: Additional metadata
    """
    monitor = get_global_monitor()
    monitor.log_experiment_data(experiment_id, data_type, value, metadata)


def log_system_event(event_type: str, details: Dict[str, Any]) -> None:
    """
    Convenience function to log system events.

    Args:
        event_type: Type of system event
        details: Event details
    """
    monitor = get_global_monitor()
    monitor.log_system_event(event_type, details)


async def start_monitoring(enable_websocket: bool = True, websocket_port: int = 8765) -> bool:
    """
    Start the global monitoring system.

    Args:
        enable_websocket: Enable WebSocket server
        websocket_port: Port for WebSocket server

    Returns:
        True if started successfully
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RealtimeMonitor(enable_websocket, websocket_port)

    result = await _global_monitor.start()
    return bool(result)


async def stop_monitoring() -> None:
    """Stop the global monitoring system."""
    if _global_monitor:
        await _global_monitor.stop()
