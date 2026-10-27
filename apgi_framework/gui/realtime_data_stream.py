"""Real-time data streaming infrastructure for APGI Framework monitoring.

Provides WebSocket-based streaming for EEG, pupillometry, cardiac, and parameter data.
"""

import asyncio
import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    from dataclasses import dataclass

    import numpy as np
    import websockets
    from websockets.server import ServerConnection

    _websockets_available = True
except ImportError:
    # Fallback for environments without websockets
    websockets = None  # type: ignore
    ServerConnection = Any  # type: ignore
    _websockets_available = False

from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)


@dataclass
class StreamDataPoint:
    """Single data point for streaming."""

    timestamp: float
    source: str  # 'eeg', 'pupil', 'cardiac', 'parameters'
    channel: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EEGStreamData:
    """EEG streaming data packet."""

    timestamp: float
    channels: List[float]
    quality_score: float
    artifact_rate: float
    bad_channels: List[str]
    p3b_amplitude: Optional[float] = None
    hep_amplitude: Optional[float] = None


@dataclass
class PupilStreamData:
    """Pupillometry streaming data packet."""

    timestamp: float
    pupil_diameter: float
    blink_rate: float
    tracking_loss: int
    data_quality: float
    data_loss: float


@dataclass
class CardiacStreamData:
    """Cardiac streaming data packet."""

    timestamp: float
    heart_rate: float
    hrv: float
    rr_interval: float
    signal_quality: float
    rpeak_confidence: float


@dataclass
class ParameterStreamData:
    """Parameter estimate streaming data packet."""

    timestamp: float
    parameter_name: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    converged: bool
    r_hat: Optional[float] = None


class RealTimeDataStreamer:
    """Real-time data streaming server and manager.

    Provides WebSocket streaming for real-time monitoring dashboard.
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        """Initialize real-time data streamer.

        Args:
            host: WebSocket server host
            port: WebSocket server port
        """
        self.host = host
        self.port = port
        self.clients: set = set()
        self.is_running = False
        self.server: Optional[Any] = None
        self._loop = None  # Store event loop reference for cleanup

        # Data buffers for each stream type
        self.eeg_buffer: queue.Queue = queue.Queue(maxsize=1000)
        self.pupil_buffer: queue.Queue = queue.Queue(maxsize=1000)
        self.cardiac_buffer: queue.Queue = queue.Queue(maxsize=1000)
        self.parameter_buffer: queue.Queue = queue.Queue(maxsize=1000)
        self.data_buffer: Dict[str, Any] = {}

        # Streaming thread
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stream_active = False

        # Data generators for simulation
        self._data_generators: Dict[str, Callable] = {}

        logger.info(f"Real-time data streamer initialized on {host}:{port}")

    async def register_client(self, websocket: ServerConnection, path: str) -> None:
        """Register new WebSocket client."""
        self.clients.add(websocket)
        # websockets.ServerConnection doesn't have type stubs, use type ignore
        logger.info(f"Client connected: {websocket.remote_address}")  # type: ignore[attr-defined]

        try:
            await websocket.send(  # type: ignore[attr-defined]
                json.dumps(
                    {
                        "type": "welcome",
                        "message": "Connected to APGI Framework real-time stream",
                        "timestamp": time.time(),
                    }
                )
            )

            # Keep connection alive and handle client messages
            async for message in websocket:  # type: ignore[attr-defined]
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(  # type: ignore[attr-defined]
                        json.dumps({"type": "error", "message": "Invalid JSON format"})
                    )

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")  # type: ignore[attr-defined]  # noqa: E501
        finally:
            self.clients.discard(websocket)

    async def handle_client_message(self, websocket: Any, data: Dict[str, Any]) -> None:
        """Handle messages from clients."""
        message_type = data.get("type")

        if message_type == "subscribe":
            # Client wants to subscribe to specific data streams
            streams = data.get("streams", [])
            response = {
                "type": "subscription_ack",
                "streams": streams,
                "timestamp": time.time(),
            }
            await websocket.send(json.dumps(response))

        elif message_type == "get_status":
            # Client requests current status
            status = {
                "type": "status",
                "is_running": self.is_running,
                "client_count": len(self.clients),
                "timestamp": time.time(),
            }
            await websocket.send(json.dumps(status))

    async def broadcast_data(self, data_type: str, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected clients."""
        if not self.clients:
            return

        message = {
            "type": "data",
            "data_type": data_type,
            "data": data,
            "timestamp": time.time(),
        }

        # Send to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        self.clients -= disconnected

    def start_server(self) -> bool:
        """Start the WebSocket server."""
        if websockets is None:
            logger.error("websockets library not available")
            return False

        if self.is_running:
            logger.warning("Server already running")
            return True

        try:
            # Start server in a separate thread
            self._stop_event.clear()
            self._stream_thread = threading.Thread(target=self._run_server, daemon=True)
            self._stream_thread.start()

            # Wait for server to start
            time.sleep(0.5)
            self.is_running = True

            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False

    def _run_server(self) -> None:
        """Run the WebSocket server."""
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Create server
            server_coro = websockets.serve(self.register_client, self.host, self.port)  # type: ignore[arg-type]
            self.server = loop.run_until_complete(server_coro)

            # Start streaming loop
            streaming_task = asyncio.ensure_future(self._streaming_loop())

            logger.info(f"WebSocket server started on {self.host}:{self.port}")
            loop.run_until_complete(streaming_task)

        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            # Clean up
            if self.server:
                self.server.close()
                loop.run_until_complete(self.server.wait_closed())
            loop.close()

    async def _streaming_loop(self) -> None:
        """Main streaming loop for data generation and broadcasting."""
        while not self._stop_event.is_set():
            try:
                # Generate and broadcast simulated data
                await self._generate_and_broadcast_data()
                await asyncio.sleep(0.1)  # 10 Hz update rate

            except Exception as e:
                logger.error(f"Streaming loop error: {e}")
                await asyncio.sleep(1)

    async def _generate_and_broadcast_data(self) -> None:
        """Generate simulated data and broadcast to clients."""
        current_time = time.time()

        # Generate EEG data
        if not self.eeg_buffer.full():
            eeg_data = self._generate_eeg_data(current_time)
            await self.broadcast_data("eeg", asdict(eeg_data))

        # Generate pupillometry data
        if not self.pupil_buffer.full():
            pupil_data = self._generate_pupil_data(current_time)
            await self.broadcast_data("pupil", asdict(pupil_data))

        # Generate cardiac data
        if not self.cardiac_buffer.full():
            cardiac_data = self._generate_cardiac_data(current_time)
            await self.broadcast_data("cardiac", asdict(cardiac_data))

        # Generate parameter data (less frequently)
        if int(current_time * 2) % 5 == 0:  # Every 2.5 seconds
            param_data = self._generate_parameter_data(current_time)
            await self.broadcast_data("parameters", asdict(param_data))

    def _generate_eeg_data(self, timestamp: float) -> EEGStreamData:
        """Generate simulated EEG data."""
        # Simulate 32 channels of EEG data
        channels = []
        for i in range(32):
            # Add some realistic EEG-like signals
            base_signal = np.sin(2 * np.pi * 10 * timestamp + i * 0.1)  # 10 Hz alpha
            noise = np.random.normal(0, 0.1)
            channels.append(base_signal + noise)

        # Simulate quality metrics
        quality_score = np.random.beta(8, 2)  # Usually good quality
        artifact_rate = (
            np.random.beta(1, 9) if np.random.random() < 0.1 else 0
        )  # Occasional artifacts

        # Simulate bad channels
        bad_channels = []
        if np.random.random() < 0.05:  # 5% chance of bad channel
            bad_channels.append(f"Channel_{np.random.randint(1, 33)}")

        # Occasionally detect neural signatures
        p3b_amplitude = None
        hep_amplitude = None
        if np.random.random() < 0.02:  # 2% chance
            p3b_amplitude = np.random.normal(5.0, 2.0)
            hep_amplitude = np.random.normal(3.0, 1.5)

        return EEGStreamData(
            timestamp=timestamp,
            channels=channels,
            quality_score=quality_score,
            artifact_rate=artifact_rate,
            bad_channels=bad_channels,
            p3b_amplitude=p3b_amplitude,
            hep_amplitude=hep_amplitude,
        )

    def _generate_pupil_data(self, timestamp: float) -> PupilStreamData:
        """Generate simulated pupillometry data."""
        # Simulate pupil diameter with realistic variation
        base_diameter = 4.0 + 2.0 * np.sin(2 * np.pi * 0.1 * timestamp)  # Slow oscillation
        pupil_diameter = base_diameter + np.random.normal(0, 0.2)
        pupil_diameter = max(2.0, min(8.0, pupil_diameter))  # Clamp to realistic range

        # Simulate blink rate
        blink_rate = 15 + np.random.normal(0, 3)
        blink_rate = max(5, min(30, blink_rate))

        # Simulate tracking
        tracking_loss = np.random.poisson(0.1)  # Usually 0, occasionally 1
        data_quality = np.random.beta(9, 1) if tracking_loss == 0 else np.random.beta(3, 2)
        data_loss = tracking_loss * 0.1

        return PupilStreamData(
            timestamp=timestamp,
            pupil_diameter=pupil_diameter,
            blink_rate=blink_rate,
            tracking_loss=tracking_loss,
            data_quality=data_quality,
            data_loss=data_loss,
        )

    def _generate_cardiac_data(self, timestamp: float) -> CardiacStreamData:
        """Generate simulated cardiac data."""
        # Simulate heart rate with realistic variation
        base_hr = 70 + 10 * np.sin(2 * np.pi * 0.05 * timestamp)  # Respiratory sinus arrhythmia
        heart_rate = base_hr + np.random.normal(0, 3)
        heart_rate = max(50, min(100, heart_rate))

        # Calculate RR interval
        rr_interval = 60000 / heart_rate  # ms

        # Simulate HRV
        hrv = 30 + np.random.normal(0, 10)
        hrv = max(10, min(100, hrv))

        # Signal quality
        signal_quality = np.random.beta(9, 1)
        rpeak_confidence = np.random.beta(19, 1)  # Usually high

        return CardiacStreamData(
            timestamp=timestamp,
            heart_rate=heart_rate,
            hrv=hrv,
            rr_interval=rr_interval,
            signal_quality=signal_quality,
            rpeak_confidence=rpeak_confidence,
        )

    def _generate_parameter_data(self, timestamp: float) -> ParameterStreamData:
        """Generate simulated parameter estimation data."""
        # Choose a parameter to update
        parameters = ["theta0", "pi_i", "beta"]
        param_name = np.random.choice(parameters)

        # Simulate converging parameter estimates
        if param_name == "theta0":
            mean = np.random.normal(0.5, 0.05)
            std = max(0.01, np.random.normal(0.1, 0.02))
        elif param_name == "pi_i":
            mean = np.random.normal(1.0, 0.1)
            std = max(0.05, np.random.normal(0.2, 0.05))
        else:  # beta
            mean = np.random.normal(0.0, 0.02)
            std = max(0.01, np.random.normal(0.05, 0.01))

        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std

        # Simulate convergence
        converged = np.random.random() < 0.7  # 70% chance converged
        r_hat = np.random.normal(1.0, 0.01) if converged else np.random.normal(1.1, 0.05)
        r_hat = max(1.0, r_hat)

        return ParameterStreamData(
            timestamp=timestamp,
            parameter_name=param_name,
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            converged=converged,
            r_hat=r_hat,
        )

    def stop_server(self) -> None:
        """Stop the WebSocket server."""
        if not self.is_running:
            return

        self._stop_event.set()
        self.is_running = False

        if self._stream_thread:
            self._stream_thread.join(timeout=2)

        # Close all client connections and server
        if self.server and self._loop and not self._loop.is_closed():
            try:
                # Close all client connections first
                for client in list(self.clients):
                    try:
                        if hasattr(client, "close"):
                            asyncio.run_coroutine_threadsafe(client.close(), self._loop)
                    except Exception as e:
                        logger.warning(f"Failed to close client connection: {e}")

                # Close the server
                if hasattr(self.server, "close"):
                    asyncio.run_coroutine_threadsafe(self.server.close(), self._loop)
                elif hasattr(self.server, "ws_server") and hasattr(self.server.ws_server, "close"):
                    asyncio.run_coroutine_threadsafe(self.server.ws_server.close(), self._loop)

                logger.info("WebSocket server and connections closed properly")
            except Exception as e:
                logger.warning(f"WebSocket cleanup warning: {e}")

        # Clear clients set
        self.clients.clear()

        logger.info("WebSocket server stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current server status."""
        return {
            "is_running": self.is_running,
            "host": self.host,
            "port": self.port,
            "client_count": len(self.clients),
            "eeg_buffer_size": self.eeg_buffer.qsize(),
            "pupil_buffer_size": self.pupil_buffer.qsize(),
            "cardiac_buffer_size": self.cardiac_buffer.qsize(),
            "parameter_buffer_size": self.parameter_buffer.qsize(),
        }


# Global instance for easy access
_streamer_instance: Optional[RealTimeDataStreamer] = None


def get_streamer() -> RealTimeDataStreamer:
    """Get or create global streamer instance."""
    global _streamer_instance
    if _streamer_instance is None:
        _streamer_instance = RealTimeDataStreamer()
    return _streamer_instance


def start_realtime_streaming() -> bool:
    """Start real-time streaming server."""
    streamer = get_streamer()
    result = streamer.start_server()
    return bool(result)


def stop_realtime_streaming() -> None:
    """Stop real-time streaming server."""
    streamer = get_streamer()
    streamer.stop_server()
