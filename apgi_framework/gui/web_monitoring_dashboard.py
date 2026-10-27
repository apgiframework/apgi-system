#!/usr/bin/env python3
"""Web-based Real-time Monitoring Dashboard for APGI Framework
===========================================================

This module provides a web interface for real-time monitoring of APGI experiments,
complementing the GUI dashboard with browser-based access.

Features:
- WebSocket-based real-time updates
- Interactive visualizations using Plotly
- Multi-modal data display (EEG, pupillometry, cardiac)
- Parameter estimation tracking
- Quality alerts and notifications
"""

import threading
import time
import warnings
from typing import Any, Dict, List, Optional, Union

try:
    from flask import Flask, jsonify, render_template_string
    from flask_socketio import SocketIO, emit

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    warnings.warn("Flask not available. Web dashboard features will be limited.")

# Plotly is used in the frontend HTML template, not in Python code
PLOTLY_AVAILABLE = True  # Assume available for frontend

from ..logging.standardized_logging import get_logger
from .monitoring_dashboard import QualityAlertSystem

logger = get_logger(__name__)


class WebMonitoringDashboard:
    """Web-based real-time monitoring dashboard using Flask and SocketIO."""

    def __init__(self, host: str = "localhost", port: int = 5000):
        """Initialize the web dashboard.

        Parameters
        ----------
        host : str
            Host address for the web server
        port : int
            Port number for the web server
        """
        if not FLASK_AVAILABLE:
            raise ImportError(
                "Flask and Flask-SocketIO are required for the web dashboard. "
                "Install them with: pip install flask flask-socketio"
            )

        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins=["http://localhost:5000", "http://127.0.0.1:5000"],
        )

        # Data storage for web clients
        self.current_data: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]] = {
            "eeg": {},
            "pupil": {},
            "cardiac": {},
            "parameters": {},
            "alerts": [],
        }

        # Initialize alert system
        self.alert_system = WebQualityAlertSystem()

        # Setup routes and socket handlers
        self._setup_routes()
        self._setup_socket_handlers()

        logger.info(f"WebMonitoringDashboard initialized on {host}:{port}")

    def _setup_routes(self) -> None:
        """Setup Flask routes."""

        @self.app.route("/")
        def index() -> Any:
            """Serve the main dashboard page."""
            return render_template_string(self._get_dashboard_html())

        @self.app.route("/api/data")
        def get_data() -> Any:
            """API endpoint to get current data."""
            return self.current_data

        @self.app.route("/api/experiments")
        def get_experiments() -> Any:
            """API endpoint to get list of experiments."""
            return jsonify({"experiments": []})

        @self.app.route("/api/experiment/<experiment_id>")
        def get_experiment(experiment_id: str) -> Any:
            """API endpoint to get experiment details."""
            return jsonify({"id": experiment_id, "name": "", "status": "not_found"})

        @self.app.route("/api/config")
        def get_config() -> Any:
            """API endpoint to get configuration."""
            return jsonify({"config": self.current_data.get("parameters", {})})

    def _setup_socket_handlers(self) -> None:
        """Setup SocketIO event handlers."""

        @self.socketio.on("connect")
        def handle_connect() -> None:
            """Handle client connection."""
            logger.info("Web client connected")
            emit("initial_data", self.current_data)

        @self.socketio.on("disconnect")
        def handle_disconnect() -> None:
            """Handle client disconnection."""
            logger.info("Web client disconnected")

        @self.socketio.on("subscribe")
        def handle_subscribe(data: Any) -> None:
            """Handle subscription requests."""
            streams = data.get("streams", [])
            logger.info(f"Client subscribed to streams: {streams}")
            emit("subscription_confirmed", {"streams": streams})

    def _get_dashboard_html(self) -> str:
        """Get the HTML template for the dashboard."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APGI Real-time Monitoring</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        .panel {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .panel h3 {
            margin-top: 0;
            color: #333;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }
        .alert-list {
            max-height: 200px;
            overflow-y: auto;
            background: #f9f9f9;
            padding: 10px;
            border-radius: 4px;
        }
        .alert {
            padding: 5px;
            margin: 5px 0;
            border-radius: 3px;
        }
        .alert.info { background: #cce7ff; }
        .alert.warning { background: #ffe4b3; }
        .alert.error { background: #ffcccc; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 APGI Real-time Monitoring Dashboard</h1>
            <p>Live monitoring of consciousness experiments</p>
        </div>

        <div class="dashboard-grid">
            <div class="panel">
                <h3>EEG Monitoring</h3>
                <div class="metric">
                    <span>Signal Quality:</span>
                    <span id="eeg-quality">N/A</span>
                </div>
                <div class="metric">
                    <span>Artifact Rate:</span>
                    <span id="eeg-artifacts">N/A</span>
                </div>
                <div id="eeg-plot"></div>
            </div>

            <div class="panel">
                <h3>Pupillometry</h3>
                <div class="metric">
                    <span>Pupil Diameter:</span>
                    <span id="pupil-diameter">N/A</span>
                </div>
                <div class="metric">
                    <span>Blink Rate:</span>
                    <span id="blink-rate">N/A</span>
                </div>
                <div id="pupil-plot"></div>
            </div>

            <div class="panel">
                <h3>Cardiac Monitoring</h3>
                <div class="metric">
                    <span>Heart Rate:</span>
                    <span id="heart-rate">N/A</span>
                </div>
                <div class="metric">
                    <span>HRV:</span>
                    <span id="hrv">N/A</span>
                </div>
                <div id="cardiac-plot"></div>
            </div>

            <div class="panel">
                <h3>Parameter Estimates</h3>
                <div class="metric">
                    <span>θ₀ (Ignition):</span>
                    <span id="theta0">N/A</span>
                </div>
                <div class="metric">
                    <span>Πᵢ (Precision):</span>
                    <span id="pi_i">N/A</span>
                </div>
                <div class="metric">
                    <span>β (Bias):</span>
                    <span id="beta">N/A</span>
                </div>
                <div id="param-plot"></div>
            </div>

            <div class="panel" style="grid-column: span 2;">
                <h3>Quality Alerts</h3>
                <div id="alert-list" class="alert-list"></div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        const plots = {};

        socket.on('connect', function() {
            console.log('Connected to server');
        });

        socket.on('initial_data', function(data) {
            updateDashboard(data);
        });

        socket.on('data_update', function(data) {
            updateDashboard(data);
        });

        function updateDashboard(data) {
            // Update EEG data
            if (data.eeg) {
                document.getElementById('eeg-quality').textContent =
                    data.eeg.quality_score ? data.eeg.quality_score.toFixed(2) : 'N/A';
                document.getElementById('eeg-artifacts').textContent =
                    data.eeg.artifact_rate ? (data.eeg.artifact_rate * 100).toFixed(1) + '%' : 'N/A';
            }

            // Update pupil data
            if (data.pupil) {
                document.getElementById('pupil-diameter').textContent =
                    data.pupil.diameter ? data.pupil.diameter.toFixed(2) + ' mm' : 'N/A';
                document.getElementById('blink-rate').textContent =
                    data.pupil.blink_rate ? data.pupil.blink_rate.toFixed(1) + ' bpm' : 'N/A';
            }

            // Update cardiac data
            if (data.cardiac) {
                document.getElementById('heart-rate').textContent =
                    data.cardiac.heart_rate ? data.cardiac.heart_rate.toFixed(1) + ' BPM' : 'N/A';
                document.getElementById('hrv').textContent =
                    data.cardiac.hrv ? data.cardiac.hrv.toFixed(1) + ' ms' : 'N/A';
            }

            // Update parameters
            if (data.parameters) {
                document.getElementById('theta0').textContent =
                    data.parameters.theta0 ? data.parameters.theta0.mean.toFixed(3) : 'N/A';
                document.getElementById('pi_i').textContent =
                    data.parameters.pi_i ? data.parameters.pi_i.mean.toFixed(3) : 'N/A';
                document.getElementById('beta').textContent =
                    data.parameters.beta ? data.parameters.beta.mean.toFixed(3) : 'N/A';
            }

            // Update alerts
            if (data.alerts) {
                const alertList = document.getElementById('alert-list');
                alertList.innerHTML = '';
                data.alerts.slice(-10).forEach(alert => {
                    const alertDiv = document.createElement('div');
                    alertDiv.className = `alert ${alert.level}`;
                    alertDiv.textContent = `[${alert.timestamp}] ${alert.source}: ${alert.message}`;
                    alertList.appendChild(alertDiv);
                });
            }

            // Update plots if data available
            updatePlots(data);
        }

        function updatePlots(data) {
            // This would update Plotly plots with real-time data
            // Implementation would depend on the specific plot requirements
        }
    </script>
</body>
</html>
        """

    def update_eeg_data(self, data: Dict[str, Any]) -> None:
        """Update EEG data and broadcast to web clients.

        Parameters
        ----------
        data : dict
            EEG data dictionary
        """
        self.current_data["eeg"] = data
        self.socketio.emit("data_update", {"eeg": data})

        # Check for alerts
        if "quality_score" in data and "artifact_rate" in data:
            self.alert_system.check_eeg_quality(data["quality_score"], data["artifact_rate"])

    def update_pupil_data(self, data: Dict[str, Any]) -> None:
        """Update pupillometry data and broadcast to web clients.

        Parameters
        ----------
        data : dict
            Pupillometry data dictionary
        """
        self.current_data["pupil"] = data
        self.socketio.emit("data_update", {"pupil": data})

        # Check for alerts
        if "data_loss" in data and "tracking_loss" in data:
            self.alert_system.check_pupil_quality(data["data_loss"], data["tracking_loss"])

    def update_cardiac_data(self, data: Dict[str, Any]) -> None:
        """Update cardiac data and broadcast to web clients.

        Parameters
        ----------
        data : dict
            Cardiac data dictionary
        """
        self.current_data["cardiac"] = data
        self.socketio.emit("data_update", {"cardiac": data})

        # Check for alerts
        if "signal_quality" in data and "rpeak_confidence" in data:
            self.alert_system.check_cardiac_quality(
                data["signal_quality"], data["rpeak_confidence"]
            )

    def update_parameter_data(self, data: Dict[str, Any]) -> None:
        """Update parameter estimate data and broadcast to web clients.

        Parameters
        ----------
        data : dict
            Parameter data dictionary
        """
        param_name = data.get("parameter_name", "")
        if param_name:
            self.current_data["parameters"][param_name] = data
            self.socketio.emit("data_update", {"parameters": {param_name: data}})

    def start_server(self) -> bool:
        """Start the web server.

        Returns
        -------
        bool
            True if server started successfully
        """
        try:
            # Start server in a separate thread
            self.server_thread = threading.Thread(
                target=lambda: self.socketio.run(
                    self.app, host=self.host, port=self.port, debug=False
                ),
                daemon=True,
            )
            self.server_thread.start()

            logger.info(f"Web dashboard server started on http://{self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start web dashboard server: {e}")
            return False

    def stop_server(self) -> None:
        """Stop the web server."""
        if hasattr(self, "server_thread"):
            logger.info("Stopping web dashboard server")
            # Note: Flask-SocketIO doesn't have a clean shutdown method in this version
            # The thread will be terminated when the main process exits


class WebQualityAlertSystem(QualityAlertSystem):
    """Web-compatible quality alert system that broadcasts alerts via WebSocket."""

    def __init__(self, dashboard: Optional[WebMonitoringDashboard] = None):
        """Initialize web alert system.

        Parameters
        ----------
        dashboard : WebMonitoringDashboard, optional
            Dashboard instance to broadcast alerts to
        """
        # Don't call super().__init__() since we don't need the tkinter components
        self.alerts: List[Dict[str, Any]] = []
        self.dashboard = dashboard

    def add_alert(self, level: str, message: str, source: str = "") -> None:
        """Add a quality alert and broadcast to web clients.

        Parameters
        ----------
        level : str
            Alert level ('info', 'warning', or 'error')
        message : str
            Alert message
        source : str
            Source of alert
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")

        alert = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "source": source,
        }

        self.alerts.append(alert)

        # Broadcast to web clients if dashboard available
        if self.dashboard:
            self.dashboard.current_data["alerts"] = self.alerts[-50:]  # Keep last 50 alerts
            self.dashboard.socketio.emit(
                "data_update", {"alerts": self.alerts[-10:]}
            )  # Send last 10

        # Log the alert
        if level == "warning":
            logger.warning(f"[{source}] {message}")
        elif level == "error":
            logger.error(f"[{source}] {message}")
        else:
            logger.info(f"[{source}] {message}")


def launch_web_dashboard(host: str = "localhost", port: int = 5000) -> WebMonitoringDashboard:
    """Launch the web-based monitoring dashboard.

    Parameters
    ----------
    host : str
        Host address
    port : int
        Port number

    Returns
    -------
    WebMonitoringDashboard
        The dashboard instance
    """
    dashboard = WebMonitoringDashboard(host=host, port=port)
    success = dashboard.start_server()

    if success:
        logger.info(f"🌐 Web dashboard launched at http://{host}:{port}")
        logger.info("Open this URL in your browser to monitor experiments in real-time")
    else:
        logger.info("❌ Failed to launch web dashboard")

    return dashboard


if __name__ == "__main__":
    # Launch the web dashboard
    dashboard = launch_web_dashboard()

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nStopping web dashboard...")
        dashboard.stop_server()
