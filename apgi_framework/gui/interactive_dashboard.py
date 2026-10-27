"""Interactive Web Dashboard for APGI Framework.

Provides real-time monitoring, interactive visualizations, and experiment management
through a modern web interface using Flask and WebSocket for real-time updates.
"""

from typing import Any, Callable, Dict, Optional

# Pre-define Flask components to avoid redefinition issues
Flask: Optional[Any] = None
jsonify: Optional[Callable] = None
render_template: Optional[Callable] = None
request: Optional[Any] = None
SocketIO: Optional[Any] = None
emit: Optional[Callable] = None

# Import Flask components - successful imports will overwrite the None values
try:
    from flask import Flask, jsonify, render_template, request  # type: ignore
    from flask_socketio import SocketIO, emit  # type: ignore
except ImportError:
    import logging

    logging.warning("flask_socketio not available. Running in limited mode.")

import functools
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from apgi_framework.logging.standardized_logging import get_logger

logger = get_logger(__name__)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from ..config import get_config_manager
    from ..data.dashboard import ExperimentComparator, ExperimentMonitor
    from ..data.visualizer import InteractiveVisualizer
    from ..logging.standardized_logging import get_logger
except ImportError:
    # Handle relative import when run directly
    from apgi_framework.config import get_config_manager
    from apgi_framework.data.dashboard import ExperimentComparator, ExperimentMonitor
    from apgi_framework.data.visualizer import InteractiveVisualizer
    from apgi_framework.logging.standardized_logging import get_logger


class InteractiveWebDashboard:
    """Interactive web-based dashboard for real-time experiment monitoring.

    Features:
    - Real-time data streaming via WebSocket
    - Interactive plots with Plotly
    - Experiment management and comparison
    - Live parameter monitoring
    - Export capabilities
    """

    def __init__(self, port: int = 8050, host: str = "localhost"):
        """Initialize interactive dashboard.

        Args:
            port: Web server port
            host: Server host
        """
        self.port = port
        self.host = host

        # Flask app setup
        template_dir = Path(__file__).parent / "templates"
        static_dir = Path(__file__).parent / "static"
        self.app = Flask(  # type: ignore
            __name__,
            template_folder=str(template_dir),
            static_folder=str(static_dir) if static_dir.exists() else None,
        )
        self.app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())

        # WebSocket setup
        self.socketio = SocketIO(  # type: ignore
            self.app,
            cors_allowed_origins=["http://localhost:5000", "http://127.0.0.1:5000"],
        )

        # Core components
        self.experiment_monitor = ExperimentMonitor()  # type: ignore
        self.comparator = ExperimentComparator()  # type: ignore
        self.visualizer = InteractiveVisualizer()  # type: ignore
        self.config_manager = get_config_manager()  # type: ignore

        # Dashboard state
        self.is_running = False
        self.connected_clients: set[str] = set()
        self.update_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Concurrency control - async/parallel request handling
        self.max_concurrent_requests = 20
        self.request_semaphore = threading.Semaphore(self.max_concurrent_requests)
        self.request_executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="dashboard_request_"
        )
        self.request_queue: list[tuple] = []
        self.queue_lock = threading.Lock()

        # Data cache - unified data structure
        self.data_cache: Dict[str, Any] = {}
        self.last_update = datetime.now()

        # Client request tracking for rate limiting
        self.client_request_counts: Dict[str, list[datetime]] = {}
        self.rate_limit_lock = threading.Lock()
        self.rate_limit_per_minute = 60  # Max requests per client per minute

        # Setup routes and WebSocket events
        self._setup_routes()  # type: ignore
        self._setup_socketio_events()  # type: ignore

        self.logger = get_logger(__name__)
        self.logger.info(f"InteractiveWebDashboard initialized for {host}:{port}")

    def _async_route(self, func: Callable) -> Callable:
        """Decorator to make Flask routes async with concurrency control."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Acquire semaphore for concurrency control
            if not self.request_semaphore.acquire(timeout=5.0):
                assert jsonify is not None
                return (
                    jsonify({"success": False, "error": "Server busy, try again later"}),
                    503,
                )

            try:
                # Check rate limiting
                client_id = request.remote_addr if request else "unknown"
                if not self._check_rate_limit(client_id):
                    assert jsonify is not None
                    return (
                        jsonify({"success": False, "error": "Rate limit exceeded"}),
                        429,
                    )

                # Submit to thread pool for parallel processing
                future = self.request_executor.submit(func, *args, **kwargs)
                return future.result(timeout=30.0)  # 30s timeout per request
            finally:
                self.request_semaphore.release()

        return wrapper

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = datetime.now()
        with self.rate_limit_lock:
            if client_id not in self.client_request_counts:
                self.client_request_counts[client_id] = []

            # Remove old requests (older than 1 minute)
            cutoff = now - timedelta(minutes=1)
            self.client_request_counts[client_id] = [
                t for t in self.client_request_counts[client_id] if t > cutoff
            ]

            # Check limit
            if len(self.client_request_counts[client_id]) >= self.rate_limit_per_minute:
                return False

            # Record this request
            self.client_request_counts[client_id].append(now)
            return True

    def _setup_routes(self) -> None:
        """Setup Flask routes with async/concurrent handling."""
        assert self.logger is not None  # for mypy

        @self.app.route("/")
        @self._async_route
        def index() -> Any:
            """Main dashboard page."""
            assert render_template is not None  # for mypy
            return render_template("dashboard.html")

        @self.app.route("/api/experiments")
        @self._async_route
        def get_experiments() -> Any:
            """Get all experiments data (async with concurrency control)."""
            assert self.experiment_monitor is not None  # for mypy
            try:
                experiments = self.experiment_monitor.get_all_experiments()

                # Validate experiments data structure
                if experiments:
                    for exp in experiments:
                        if not isinstance(exp, dict):
                            self.logger.warning(f"Invalid experiment data type: {type(exp)}")
                            continue
                        required_fields = ["id", "name", "status", "timestamp"]
                        missing_fields = [field for field in required_fields if field not in exp]
                        if missing_fields:
                            self.logger.warning(
                                f"Experiment {exp.get('id', 'unknown')} missing fields: {missing_fields}"
                            )

                # If no experiments, provide sample data for demonstration
                if not experiments:
                    self.logger.info("No experiments found, providing sample data")
                    experiments = self._get_sample_experiments()

                assert jsonify is not None  # for mypy
                return jsonify(
                    {
                        "success": True,
                        "data": experiments,
                        "timestamp": datetime.now().isoformat(),
                        "count": len(experiments),
                        "message": (
                            "Sample data"
                            if not self.experiment_monitor.get_all_experiments()
                            else "Real data"
                        ),
                    }
                )
            except Exception as e:
                assert jsonify is not None  # for mypy
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/experiment/<experiment_id>")
        @self._async_route
        def get_experiment(experiment_id: str) -> Any:
            """Get specific experiment data (async with concurrency control)."""
            assert self.experiment_monitor is not None  # for mypy
            assert self.visualizer is not None  # for mypy
            try:
                experiment = self.experiment_monitor.get_experiment_status(experiment_id)
                if not experiment:
                    assert jsonify is not None  # for mypy
                    return (
                        jsonify({"success": False, "error": "Experiment not found"}),
                        404,
                    )

                # Add visualization data
                viz_data = self.visualizer.create_interactive_dashboard_data(
                    experiment.get("results", []), experiment.get("trials", [])
                )

                assert jsonify is not None  # for mypy
                return jsonify(
                    {
                        "success": True,
                        "data": experiment,
                        "visualizations": viz_data,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                assert jsonify is not None  # for mypy
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/compare", methods=["POST"])
        @self._async_route
        def compare_experiments() -> Any:
            """Compare multiple experiments (async with concurrency control)."""
            assert request is not None  # for mypy
            try:
                assert request.get_json is not None  # for mypy
                data = request.get_json()
                experiment_ids = data.get("experiments", [])

                if len(experiment_ids) < 2:
                    assert jsonify is not None  # for mypy
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "At least 2 experiments required for comparison",
                            }
                        ),
                        400,
                    )

                # Get experiment data
                experiments = {}
                for exp_id in experiment_ids:
                    exp_data = self.experiment_monitor.get_experiment_status(exp_id)
                    if exp_data:
                        experiments[exp_id] = exp_data

                if len(experiments) < 2:
                    assert jsonify is not None  # for mypy
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "Not enough valid experiments found",
                            }
                        ),
                        400,
                    )

                # Generate comparison
                assert self.comparator is not None  # for mypy
                comparison = self.comparator.compare_experiments(experiments)

                assert jsonify is not None  # for mypy
                return jsonify(
                    {
                        "success": True,
                        "data": comparison,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                assert jsonify is not None  # for mypy
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/config")
        @self._async_route
        def get_config() -> Any:
            """Get current configuration (async with concurrency control)."""
            assert self.config_manager is not None  # for mypy
            try:
                config_data = {
                    "apgi_parameters": self.config_manager.get_apgi_parameters().__dict__,
                    "experimental_config": self.config_manager.get_experimental_config().__dict__,
                    "performance_thresholds": self.config_manager.get_performance_thresholds().__dict__,
                    "stimulus_parameters": self.config_manager.get_stimulus_params().__dict__,
                }

                assert jsonify is not None  # for mypy
                return jsonify(
                    {
                        "success": True,
                        "data": config_data,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                if jsonify is not None:
                    return jsonify({"success": False, "error": str(e)}), 500
                else:
                    return {"success": False, "error": str(e)}, 500

        @self.app.route("/api/export/<experiment_id>")
        @self._async_route
        def export_experiment(experiment_id: str) -> Any:
            """Export experiment data (async with concurrency control)."""
            try:
                format_type = request.args.get("format", "json")  # type: ignore[union-attr]
                experiment = self.experiment_monitor.get_experiment_status(experiment_id)

                if not experiment:
                    return (
                        jsonify({"success": False, "error": "Experiment not found"}),  # type: ignore[misc]
                        404,
                    )

                # Export logic here
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if format_type == "json":
                    filename = f"experiment_{experiment_id}_{timestamp}.json"
                    filepath = Path("exports") / filename
                    filepath.parent.mkdir(exist_ok=True)

                    with open(filepath, "w") as f:
                        json.dump(experiment, f, indent=2, default=str)

                    return jsonify(  # type: ignore[misc]
                        {
                            "success": True,
                            "filename": filename,
                            "filepath": str(filepath),
                        }
                    )
                else:
                    return (
                        jsonify(  # type: ignore[misc]
                            {
                                "success": False,
                                "error": f"Unsupported format: {format_type}",
                            }
                        ),
                        400,
                    )

            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500  # type: ignore[misc]

    def _setup_socketio_events(self) -> None:
        """Setup WebSocket events for real-time updates."""

        @self.socketio.on("connect")
        def handle_connect() -> None:
            """Handle client connection."""
            self.connected_clients.add(request.sid)  # type: ignore[union-attr]
            self.logger.info(f"Client connected: {request.sid}")  # type: ignore[union-attr]

            # Send initial data
            try:
                experiments = self.experiment_monitor.get_all_experiments()
                if not experiments:
                    self.logger.info("No experiments found, providing sample data for WebSocket")
                    experiments = self._get_sample_experiments()

                emit(  # type: ignore[misc]
                    "initial_data",
                    {
                        "experiments": experiments,
                        "config": self._get_config_summary(),
                        "timestamp": datetime.now().isoformat(),
                        "message": (
                            "Sample data"
                            if not self.experiment_monitor.get_all_experiments()
                            else "Real data"
                        ),
                    },
                )
            except Exception as e:
                self.logger.error(f"Error sending initial data: {e}")
                emit("error", {"message": f"Failed to load initial data: {e}"})  # type: ignore[misc]

        @self.socketio.on("disconnect")
        def handle_disconnect() -> None:
            """Handle client disconnection."""
            try:
                self.connected_clients.discard(request.sid)  # type: ignore[union-attr]
                self.logger.info(f"Client disconnected: {request.sid}")  # type: ignore[union-attr]
            except Exception as e:
                self.logger.error(f"Error handling client disconnection: {e}")

        @self.socketio.on("request_update")
        def handle_update_request() -> None:
            """Handle manual update request."""
            try:
                experiments = self.experiment_monitor.get_all_experiments()
                if not experiments:
                    self.logger.info("No experiments found, providing sample data for update")
                    experiments = self._get_sample_experiments()

                update_data = {
                    "experiments": experiments,
                    "timestamp": datetime.now().isoformat(),
                    "message": (
                        "Sample data"
                        if not self.experiment_monitor.get_all_experiments()
                        else "Real data"
                    ),
                }
                emit("update", update_data)  # type: ignore[misc]
            except Exception as e:
                self.logger.error(f"Error handling update request: {e}")
                emit("error", {"message": f"Failed to update data: {e}"})  # type: ignore[misc]

    def start_dashboard(self) -> None:
        """Start the interactive dashboard."""
        if self.is_running:
            self.logger.warning("Dashboard is already running")
            return

        try:
            self.is_running = True
            self.stop_event.clear()

            # Start experiment monitoring
            self.experiment_monitor.start_monitoring()

            # Start real-time update thread
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()

            # Start Flask server
            self.logger.info(f"Starting interactive dashboard at http://{self.host}:{self.port}")
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=False,
                allow_unsafe_werkzeug=True,
            )

        except Exception as e:
            self.logger.error(f"Failed to start dashboard: {e}")
            self.stop_dashboard()
            raise

    def stop_dashboard(self) -> None:
        """Stop the interactive dashboard."""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()

        # Stop experiment monitoring
        self.experiment_monitor.stop_monitoring()

        # Wait for update thread
        if self.update_thread:
            self.update_thread.join(timeout=2.0)

        self.logger.info("Interactive dashboard stopped")

    def _update_loop(self) -> None:
        """Background thread for real-time updates."""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Only check for updates if there are connected clients
                if self.connected_clients:
                    # Check for updates
                    current_experiments = self.experiment_monitor.get_all_experiments()

                    # Compare with cached data
                    if current_experiments != self.data_cache.get("experiments"):
                        self.data_cache["experiments"] = current_experiments
                        self.last_update = datetime.now()

                        # Broadcast update to all connected clients
                        update_data = {
                            "experiments": current_experiments,
                            "timestamp": self.last_update.isoformat(),
                        }
                        self.socketio.emit("update", update_data)

                    # Sleep for 1 second when clients are connected
                    time.sleep(1.0)
                else:
                    # Sleep longer when no clients are connected to reduce CPU usage
                    time.sleep(5.0)

            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                time.sleep(5.0)  # Wait longer on error

    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for dashboard."""
        try:
            return {
                "apgi": self.config_manager.get_apgi_parameters().__dict__,
                "experimental": self.config_manager.get_experimental_config().__dict__,
            }
        except Exception as e:
            self.logger.error(f"Error getting config summary: {e}")
            return {}

    def _get_sample_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Get sample experiment data for demonstration purposes."""
        return {
            "sample_primary": {
                "id": "sample_primary",
                "metadata": {
                    "name": "Primary Falsification Test",
                    "description": "Sample primary falsification test",
                    "total_trials": 1000,
                    "test_type": "primary",
                },
                "start_time": datetime.now() - timedelta(hours=2),
                "status": "completed",
                "progress": 100.0,
                "current_trial": 1000,
                "total_trials": 1000,
                "results": [
                    {
                        "trial": 1,
                        "falsified": True,
                        "confidence": 0.95,
                        "response_time": 1.23,
                    },
                    {
                        "trial": 2,
                        "falsified": False,
                        "confidence": 0.12,
                        "response_time": 0.87,
                    },
                ],
                "trials": [],
                "last_update": datetime.now() - timedelta(hours=1),
                "statistics": {
                    "trials_per_second": 2.5,
                    "estimated_completion": None,
                    "falsifications_detected": 234,
                    "success_rate": 76.6,
                },
            },
            "sample_consciousness": {
                "id": "sample_consciousness",
                "metadata": {
                    "name": "Consciousness Without Ignition Test",
                    "description": "Sample consciousness test",
                    "total_trials": 500,
                    "test_type": "consciousness-without-ignition",
                },
                "start_time": datetime.now() - timedelta(minutes=30),
                "status": "running",
                "progress": 65.0,
                "current_trial": 325,
                "total_trials": 500,
                "results": [],
                "trials": [],
                "last_update": datetime.now() - timedelta(minutes=5),
                "statistics": {
                    "trials_per_second": 1.8,
                    "estimated_completion": datetime.now() + timedelta(minutes=97),
                    "falsifications_detected": 87,
                    "success_rate": 73.2,
                },
            },
        }

    def get_dashboard_url(self) -> str:
        """Get dashboard URL."""
        return f"http://{self.host}:{self.port}"


def create_interactive_dashboard(
    port: int = 8050, host: str = "localhost"
) -> Optional[InteractiveWebDashboard]:
    """Create and configure an interactive dashboard.

    Args:
        port: Port number for the dashboard
        host: Host address for the dashboard

    Returns:
        Configured dashboard instance or None if Flask not available
    """
    if Flask is None:
        logger.error("Flask and flask-socketio are required for the interactive dashboard.")
        logger.info("Please install them with: pip install flask flask-socketio")
        return None

    dashboard = InteractiveWebDashboard(port=port, host=host)  # type: ignore
    return dashboard


if __name__ == "__main__":
    if Flask is None:
        logger.error("Flask and flask-socketio are required for the interactive dashboard.")
        logger.info("Please install them with: pip install flask flask-socketio")
        sys.exit(1)

    dashboard = create_interactive_dashboard()  # type: ignore
    if dashboard is not None:
        logger.info(f"Starting interactive dashboard at {dashboard.get_dashboard_url()}")
        dashboard.run()  # type: ignore[attr-defined]
    else:
        sys.exit(1)
