"""Advanced Deployment Automation Manager for APGI Framework.

Provides comprehensive automated deployment with environment management,
health monitoring, and self-healing capabilities.
"""

import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..logging.standardized_logging import get_logger
from .deployment_validator import DeploymentValidator

logger = get_logger(__name__)


@dataclass
class DeploymentConfig:
    """Deployment configuration."""

    environment: str = "production"
    docker_image: str = "apgi-framework"
    docker_tag: str = "latest"
    container_name: str = "apgi-app"
    ports: Dict[str, int] = field(default_factory=lambda: {"8000": 8000})
    volumes: Dict[str, str] = field(default_factory=dict)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    health_check_interval: int = 30
    auto_restart: bool = True
    backup_enabled: bool = True
    monitoring_enabled: bool = True


@dataclass
class DeploymentStatus:
    """Current deployment status."""

    is_running: bool = False
    container_id: Optional[str] = None
    start_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None


class DeploymentAutomationManager:
    """Advanced deployment automation manager.

    Features:
    - Automated deployment and environment setup
    - Health monitoring and auto-recovery
    - Backup and restore management
    - Rolling updates
    - Multi-environment support
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize deployment automation manager.

        Args:
            config_path: Path to deployment configuration file
        """
        self.logger = logger
        self.config_path = config_path or Path("deployment_config.yaml")
        self.config = self._load_config()
        self.status = DeploymentStatus()
        self.validator = DeploymentValidator()

        # Monitoring thread
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None

        # Deployment history
        self.deployment_history: List[Dict[str, Any]] = []

    def _load_config(self) -> DeploymentConfig:
        """Load deployment configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    data = yaml.safe_load(f)
                return DeploymentConfig(**data)
            except Exception as e:
                self.logger.warning(f"Failed to load config, using defaults: {e}")

        return DeploymentConfig()

    def _save_config(self) -> None:
        """Save deployment configuration."""
        try:
            config_dict = {
                "environment": self.config.environment,
                "docker_image": self.config.docker_image,
                "docker_tag": self.config.docker_tag,
                "container_name": self.config.container_name,
                "ports": self.config.ports,
                "volumes": self.config.volumes,
                "environment_vars": self.config.environment_vars,
                "health_check_interval": self.config.health_check_interval,
                "auto_restart": self.config.auto_restart,
                "backup_enabled": self.config.backup_enabled,
                "monitoring_enabled": self.config.monitoring_enabled,
            }

            with open(self.config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)

        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    def deploy(self, force: bool = False) -> bool:
        """Perform automated deployment.

        Args:
            force: Force redeployment even if running

        Returns:
            True if deployment successful
        """
        self.logger.info(f"Starting automated deployment for {self.config.environment}")

        try:
            # Check if already running
            if self._is_container_running() and not force:
                self.logger.info("Container already running, use force=True to redeploy")
                return True

            # Validate deployment prerequisites
            validation_report = self.validator.validate_deployment()
            if not validation_report.overall_passed:
                self.logger.error("Deployment validation failed")
                self.logger.error(self.validator.generate_summary())
                return False

            # Stop existing container
            self._stop_container()

            # Create backup if enabled
            if self.config.backup_enabled:
                self._create_backup()

            # Build and deploy
            if not self._build_image():
                return False

            if not self._start_container():
                return False

            # Wait for health check
            if not self._wait_for_health_check():
                self.logger.error("Health check failed after deployment")
                return False

            # Start monitoring
            if self.config.monitoring_enabled:
                self._start_monitoring()

            # Record deployment
            self._record_deployment("success")

            self.logger.info("Deployment completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            self._record_deployment("failed", str(e))
            return False

    def _is_container_running(self) -> bool:
        """Check if container is running."""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"name={self.config.container_name}",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False

    def _stop_container(self) -> None:
        """Stop existing container."""
        try:
            self.logger.info("Stopping existing container...")
            subprocess.run(
                ["docker", "stop", self.config.container_name],
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["docker", "rm", self.config.container_name],
                capture_output=True,
                check=False,
            )
        except Exception as e:
            self.logger.warning(f"Error stopping container: {e}")

    def _build_image(self) -> bool:
        """Build Docker image."""
        try:
            self.logger.info(
                f"Building Docker image: {self.config.docker_image}:{self.config.docker_tag}"
            )

            cmd = [
                "docker",
                "build",
                "-t",
                f"{self.config.docker_image}:{self.config.docker_tag}",
                ".",
            ]

            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info("Docker image built successfully")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to build Docker image: {e}")
            self.logger.error(f"Build output: {e.stderr}")
            return False

    def _start_container(self) -> bool:
        """Start container."""
        try:
            self.logger.info("Starting container...")

            # Prepare volume mounts
            volumes = []
            default_volumes = {
                "./data": "/app/data",
                "./apgi_outputs": "/app/apgi_outputs",
                "./session_state": "/app/session_state",
                "./logs": "/app/logs",
            }

            all_volumes = {**default_volumes, **self.config.volumes}
            for host_path, container_path in all_volumes.items():
                volumes.extend(["-v", f"{host_path}:{container_path}"])

            # Prepare port mappings
            ports = []
            for host_port, container_port in self.config.ports.items():
                ports.extend(["-p", f"{host_port}:{container_port}"])

            # Prepare environment variables
            env_vars = []
            default_env = {
                "APGI_ENV": self.config.environment,
                "APGI_LOG_LEVEL": "INFO",
                "PYTHONPATH": "/app",
            }

            all_env = {**default_env, **self.config.environment_vars}
            for key, value in all_env.items():
                env_vars.extend(["-e", f"{key}={value}"])

            # Build docker run command
            cmd = [
                "docker",
                "run",
                "-d",
                "--name",
                self.config.container_name,
                "--restart",
                "unless-stopped",
                *volumes,
                *ports,
                *env_vars,
                f"{self.config.docker_image}:{self.config.docker_tag}",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Get container ID
            self.status.container_id = result.stdout.strip()
            self.status.is_running = True
            self.status.start_time = datetime.now()

            self.logger.info(f"Container started: {self.status.container_id}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start container: {e}")
            self.logger.error(f"Docker output: {e.stderr}")
            return False

    def _wait_for_health_check(self, timeout: int = 60) -> bool:
        """Wait for health check to pass."""
        self.logger.info("Waiting for health check...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if container is still running
                if not self._is_container_running():
                    self.logger.error("Container stopped during health check")
                    return False

                # Perform health check
                result = subprocess.run(
                    [
                        "docker",
                        "exec",
                        self.config.container_name,
                        "python",
                        "-c",
                        "import apgi_framework; print('OK')",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )

                if result.returncode == 0:
                    self.logger.info("Health check passed")
                    self.status.health_status = "healthy"
                    self.status.last_health_check = datetime.now()
                    return True

                time.sleep(2)

            except subprocess.TimeoutExpired:
                self.logger.warning("Health check timeout")
                continue
            except Exception as e:
                self.logger.warning(f"Health check error: {e}")
                time.sleep(2)

        self.logger.error("Health check timed out")
        return False

    def _start_monitoring(self) -> None:
        """Start monitoring thread."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        self.logger.info("Monitoring started")

    def _monitoring_loop(self) -> None:
        """Monitoring loop."""
        while self._monitoring_active:
            try:
                if self._is_container_running():
                    self._update_container_stats()
                    self._perform_health_check()
                else:
                    self.logger.warning("Container not running")
                    if self.config.auto_restart:
                        self.logger.info("Attempting auto-restart...")
                        if self._start_container():
                            self._wait_for_health_check()
                        else:
                            self.status.error_count += 1

                time.sleep(self.config.health_check_interval)

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(self.config.health_check_interval)

    def _update_container_stats(self) -> None:
        """Update container resource statistics."""
        try:
            # Get container stats
            result = subprocess.run(
                [
                    "docker",
                    "stats",
                    "--no-stream",
                    "--format",
                    "{{.CPUPerc}}\t{{.MemPerc}}",
                    self.config.container_name,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            if result.stdout.strip():
                cpu_str, mem_str = result.stdout.strip().split("\t")
                self.status.cpu_usage = float(cpu_str.rstrip("%"))
                self.status.memory_usage = float(mem_str.rstrip("%"))

        except Exception as e:
            self.logger.warning(f"Failed to update container stats: {e}")

    def _perform_health_check(self) -> None:
        """Perform health check."""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    self.config.container_name,
                    "python",
                    "-c",
                    "import apgi_framework; print('OK')",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            self.status.last_health_check = datetime.now()

            if result.returncode == 0:
                self.status.health_status = "healthy"
            else:
                self.status.health_status = "unhealthy"
                self.status.error_count += 1
                self.status.last_error = result.stderr

        except Exception as e:
            self.status.health_status = "error"
            self.status.error_count += 1
            self.status.last_error = str(e)

    def _create_backup(self) -> None:
        """Create deployment backup."""
        try:
            backup_dir = Path("backups") / f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Backup configuration
            if self.config_path.exists():
                import shutil

                shutil.copy2(self.config_path, backup_dir / "deployment_config.yaml")

            # Backup data directories
            for source in ["data", "apgi_outputs", "session_state"]:
                source_path = Path(source)
                if source_path.exists():
                    import shutil

                    shutil.copytree(source_path, backup_dir / source, dirs_exist_ok=True)

            self.logger.info(f"Backup created: {backup_dir}")

        except Exception as e:
            self.logger.error(f"Backup failed: {e}")

    def _record_deployment(self, status: str, error: Optional[str] = None) -> None:
        """Record deployment in history."""
        deployment_record = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.config.environment,
            "docker_tag": self.config.docker_tag,
            "status": status,
            "error": error,
            "container_id": self.status.container_id,
        }

        self.deployment_history.append(deployment_record)

        # Keep only last 50 deployments
        if len(self.deployment_history) > 50:
            self.deployment_history = self.deployment_history[-50:]

    def get_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            "is_running": self.status.is_running,
            "container_id": self.status.container_id,
            "start_time": (self.status.start_time.isoformat() if self.status.start_time else None),
            "last_health_check": (
                self.status.last_health_check.isoformat() if self.status.last_health_check else None
            ),
            "health_status": self.status.health_status,
            "cpu_usage": self.status.cpu_usage,
            "memory_usage": self.status.memory_usage,
            "error_count": self.status.error_count,
            "last_error": self.status.last_error,
            "monitoring_active": self._monitoring_active,
        }

    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        self.logger.info("Monitoring stopped")

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.stop_monitoring()
        self._stop_container()
        self.logger.info("Cleanup completed")
