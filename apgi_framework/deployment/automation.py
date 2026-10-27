"""Deployment Automation for APGI Framework.

Provides automated deployment, environment setup, dependency management,
and deployment validation for production and development environments.
"""

import json
import os
import shutil
import subprocess
import sys
import venv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from apgi_framework.logging.standardized_logging import get_logger

from ..config import get_config_manager
from ..exceptions import APGIFrameworkError
from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)


def _staging_dir() -> Path:
    """Per-user private staging directory (0700), instead of the shared, world-writable /tmp."""
    import tempfile

    path = (
        Path(tempfile.gettempdir())
        / f"apgi-deploy-{os.getuid() if hasattr(os, 'getuid') else 'user'}"
    )
    path.mkdir(mode=0o700, exist_ok=True)
    return path


class DeploymentError(APGIFrameworkError):
    """Deployment-related errors."""


@dataclass
class DeploymentConfig:
    """Configuration for deployment automation."""

    # Environment settings
    environment: str = "development"  # development, staging, production
    python_version: str = "3.9"

    # Deployment paths
    deploy_path: str = "deploy"
    backup_path: str = "backups"
    log_path: str = "logs"

    # Service configuration
    enable_dashboard: bool = True
    dashboard_port: int = 8050
    enable_api: bool = True
    api_port: int = 8000

    # Database settings
    database_url: Optional[str] = None
    enable_database_backup: bool = True

    # Security settings
    enable_ssl: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

    # Performance settings
    worker_processes: int = 4
    worker_connections: int = 1000
    enable_monitoring: bool = True

    # Auto-deployment settings
    enable_auto_deploy: bool = False
    deploy_branch: str = "main"
    health_check_url: str = "/health"
    health_check_interval: int = 30


class DeploymentManager:
    """Automated deployment manager for APGI Framework.

    Handles environment setup, dependency management, service deployment,
    monitoring, and rollback capabilities.
    """

    def __init__(self, config: Optional[DeploymentConfig] = None):
        """Initialize deployment manager.

        Args:
            config: Deployment configuration
        """
        self.config = config or DeploymentConfig()
        self.logger = get_logger(__name__)
        self.config_manager = get_config_manager()

        # Deployment state
        self.deployment_id: Optional[str] = None
        self.deployment_start_time: Optional[datetime] = None
        self.backup_created: bool = False

        # Initialize deployment paths
        self.deploy_dir = Path(self.config.deploy_path)
        self.backup_dir = Path(self.config.backup_path)
        self.log_dir = Path(self.config.log_path)

        # Create directories
        self._create_directories()

        self.logger.info(f"DeploymentManager initialized for {self.config.environment} environment")

    def _create_directories(self) -> None:
        """Create necessary directories."""
        for directory in [self.deploy_dir, self.backup_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def deploy(
        self, source_path: Optional[str] = None, create_backup: bool = True
    ) -> Dict[str, Any]:
        """Perform automated deployment.

        Args:
            source_path: Source code path (defaults to current directory)
            create_backup: Whether to create backup before deployment

        Returns:
            Deployment result dictionary
        """
        self.deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.deployment_start_time = datetime.now()

        self.logger.info(f"Starting deployment {self.deployment_id}")

        try:
            # Step 1: Create backup if requested
            if create_backup:
                self._create_backup()

            # Step 2: Setup environment
            self._setup_environment()

            # Step 3: Install dependencies
            self._install_dependencies()

            # Step 4: Configure services
            self._configure_services()

            # Step 5: Deploy application
            self._deploy_application(source_path)

            # Step 6: Start services
            self._start_services()

            # Step 7: Health check
            health_status = self._perform_health_check()

            # Step 8: Cleanup
            self._cleanup_deployment()

            if self.deployment_start_time is None:
                raise RuntimeError("Deployment start time should be initialized")

            deployment_time = (datetime.now() - self.deployment_start_time).total_seconds()

            result = {
                "deployment_id": self.deployment_id,
                "status": "success",
                "deployment_time": deployment_time,
                "health_status": health_status,
                "services": self._get_service_status(),
                "backup_created": self.backup_created,
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(
                f"Deployment {self.deployment_id} completed successfully in {deployment_time:.2f}s"
            )
            return result

        except Exception as e:
            self.logger.error(f"Deployment {self.deployment_id} failed: {e}")

            # Attempt rollback
            rollback_result = self._rollback()

            return {
                "deployment_id": self.deployment_id,
                "status": "failed",
                "error": str(e),
                "rollback_result": rollback_result,
                "timestamp": datetime.now().isoformat(),
            }

    def _create_backup(self) -> None:
        """Create backup of current deployment."""
        try:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.backup_dir / backup_name

            # Create backup of current application
            if self.deploy_dir.exists():
                shutil.copytree(self.deploy_dir, backup_path, dirs_exist_ok=True)
                self.backup_created = True
                self.logger.info(f"Backup created at {backup_path}")

        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
            self.backup_created = False

    def _setup_environment(self) -> None:
        """Setup deployment environment."""
        try:
            # Create virtual environment if it doesn't exist
            venv_path = self.deploy_dir / "venv"
            if not venv_path.exists():
                self.logger.info("Creating virtual environment...")
                venv.create(venv_path, with_pip=True)

            # Get python executable
            if sys.platform == "win32":
                python_exe = venv_path / "Scripts" / "python.exe"
                pip_exe = venv_path / "Scripts" / "pip.exe"
            else:
                python_exe = venv_path / "bin" / "python"
                pip_exe = venv_path / "bin" / "pip"

            self.python_exe = str(python_exe)
            self.pip_exe = str(pip_exe)

            # Upgrade pip
            self._run_command([self.pip_exe, "install", "--upgrade", "pip"])

            self.logger.info("Environment setup completed")

        except Exception as e:
            raise DeploymentError(f"Environment setup failed: {e}")

    def _install_dependencies(self) -> None:
        """Install application dependencies."""
        try:
            # Install requirements
            requirements_file = Path("requirements.txt")
            if requirements_file.exists():
                self.logger.info("Installing dependencies...")
                self._run_command([self.pip_exe, "install", "-r", str(requirements_file)])
            else:
                self.logger.warning("requirements.txt not found, installing basic dependencies")
                basic_deps = [
                    "flask",
                    "flask-socketio",
                    "numpy",
                    "scipy",
                    "matplotlib",
                    "plotly",
                    "pandas",
                    "scikit-learn",
                    "tkinter",
                ]
                for dep in basic_deps:
                    self._run_command([self.pip_exe, "install", dep])

            # Install APGI framework in development mode
            self._run_command([self.pip_exe, "install", "-e", "."])

            self.logger.info("Dependencies installed successfully")

        except Exception as e:
            raise DeploymentError(f"Dependency installation failed: {e}")

    def _configure_services(self) -> None:
        """Configure deployment services."""
        try:
            # Create service configuration files
            self._create_systemd_service()
            self._create_nginx_config()
            self._create_environment_file()

            self.logger.info("Services configured successfully")

        except Exception as e:
            raise DeploymentError(f"Service configuration failed: {e}")

    def _create_systemd_service(self) -> None:
        """Create systemd service file for APGI dashboard."""
        if sys.platform != "linux":
            return  # Only create systemd services on Linux

        service_content = f"""[Unit]
Description=APGI Framework Dashboard
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'www-data')}
WorkingDirectory={self.deploy_dir.absolute()}
Environment=PATH={self.deploy_dir.absolute()}/venv/bin
ExecStart={self.deploy_dir.absolute()}/venv/bin/python -m apgi_framework.gui.interactive_dashboard --port {self.config.dashboard_port}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

        service_file = _staging_dir() / "apgi-dashboard.service"
        with open(service_file, "w") as f:
            f.write(service_content)

        self.logger.info("Systemd service file created")

    def _create_nginx_config(self) -> None:
        """Create nginx configuration for reverse proxy."""
        if not self.config.enable_ssl:
            return

        nginx_config = f"""server {{
    listen 80;
    server_name localhost;
    return 301 https://$server_name$request_uri;
}}

server {{
    listen 443 ssl http2;
    server_name localhost;
    
    ssl_certificate {self.config.ssl_cert_path};
    ssl_certificate_key {self.config.ssl_key_path};
    
    location / {{
        proxy_pass http://localhost:{self.config.dashboard_port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
    
    location /socket.io/ {{
        proxy_pass http://localhost:{self.config.dashboard_port};
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}
"""

        config_file = _staging_dir() / "apgi-nginx.conf"
        with open(config_file, "w") as f:
            f.write(nginx_config)

        self.logger.info("Nginx configuration created")

    def _create_environment_file(self) -> None:
        """Create environment file for configuration."""
        env_content = f"""# APGI Framework Environment Configuration
APGI_ENVIRONMENT={self.config.environment}
APGI_DASHBOARD_PORT={self.config.dashboard_port}
APGI_API_PORT={self.config.api_port}
APGI_LOG_LEVEL=INFO
APGI_ENABLE_MONITORING={self.config.enable_monitoring}
"""

        if self.config.database_url:
            env_content += f"APGI_DATABASE_URL={self.config.database_url}\n"

        env_file = self.deploy_dir / ".env"
        with open(env_file, "w") as f:
            f.write(env_content)

        self.logger.info("Environment file created")

    def _deploy_application(self, source_path: Optional[str] = None) -> None:
        """Deploy application files."""
        try:
            source = Path(source_path) if source_path else Path.cwd()

            # Copy application files
            for item in source.iterdir():
                if item.name in [
                    "__pycache__",
                    ".git",
                    "node_modules",
                    ".pytest_cache",
                ]:
                    continue

                dest = self.deploy_dir / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir() and not dest.exists():
                    shutil.copytree(item, dest)

            self.logger.info("Application deployed successfully")

        except Exception as e:
            raise DeploymentError(f"Application deployment failed: {e}")

    def _start_services(self) -> None:
        """Start deployed services."""
        try:
            # Start dashboard service
            if self.config.enable_dashboard:
                self._start_dashboard_service()

            # Start API service
            if self.config.enable_api:
                self._start_api_service()

            self.logger.info("Services started successfully")

        except Exception as e:
            raise DeploymentError(f"Service startup failed: {e}")

    def _start_dashboard_service(self) -> None:
        """Start dashboard service."""
        try:
            # Start dashboard in background
            dashboard_script = self.deploy_dir / "start_dashboard.py"

            script_content = f"""#!/usr/bin/env python3
import sys
sys.path.insert(0, '{self.deploy_dir.absolute()}')

from apgi_framework.gui.interactive_dashboard import create_interactive_dashboard

dashboard = create_interactive_dashboard(port={self.config.dashboard_port})
dashboard.start_dashboard()
"""

            with open(dashboard_script, "w") as f:
                f.write(script_content)

            dashboard_script.chmod(0o755)

            # Start process
            subprocess.Popen(
                [self.python_exe, str(dashboard_script)],
                cwd=self.deploy_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.logger.info(f"Dashboard service started on port {self.config.dashboard_port}")

        except Exception as e:
            raise DeploymentError(f"Dashboard service startup failed: {e}")

    def _start_api_service(self) -> None:
        """Start API service."""
        # Placeholder for API service startup
        self.logger.info("API service startup not implemented yet")

    def _perform_health_check(self) -> Dict[str, Any]:
        """Perform health check on deployed services."""
        try:
            import time

            import requests

            health_url = (
                f"http://localhost:{self.config.dashboard_port}{self.config.health_check_url}"
            )

            # Wait for service to start
            max_wait = 60  # seconds
            wait_interval = 5

            for i in range(0, max_wait, wait_interval):
                try:
                    response = requests.get(health_url, timeout=10)
                    if response.status_code == 200:
                        return {
                            "status": "healthy",
                            "response_time": response.elapsed.total_seconds(),
                            "status_code": response.status_code,
                        }
                except requests.exceptions.RequestException:
                    pass

                time.sleep(wait_interval)

            return {"status": "unhealthy", "error": "Health check timeout"}

        except ImportError:
            # requests not available, perform basic check
            return {
                "status": "unknown",
                "error": "Health check not available (requests module missing)",
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _get_service_status(self) -> Dict[str, str]:
        """Get status of deployed services."""
        status = {}

        if self.config.enable_dashboard:
            try:
                import requests

                response = requests.get(
                    f"http://localhost:{self.config.dashboard_port}/", timeout=5
                )
                status["dashboard"] = "running" if response.status_code == 200 else "error"
            except (requests.RequestException, requests.Timeout, ConnectionError) as e:
                self.logger.warning(f"Dashboard health check failed: {e}")
                status["dashboard"] = "stopped"

        if self.config.enable_api:
            status["api"] = "not_implemented"

        return status

    def _cleanup_deployment(self) -> None:
        """Clean up after deployment."""
        try:
            # Clean up temporary files
            temp_files = [
                _staging_dir() / "apgi-dashboard.service",
                _staging_dir() / "apgi-nginx.conf",
            ]

            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()

            self.logger.info("Deployment cleanup completed")

        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

    def _rollback(self) -> Dict[str, Any]:
        """Rollback to previous deployment."""
        try:
            if not self.backup_created:
                return {"status": "failed", "error": "No backup available for rollback"}

            # Find latest backup
            backups = list(self.backup_dir.glob("backup_*"))
            if not backups:
                return {"status": "failed", "error": "No backups found"}

            latest_backup = max(backups, key=lambda x: x.stat().st_mtime)

            # Restore from backup
            if self.deploy_dir.exists():
                shutil.rmtree(self.deploy_dir)

            shutil.copytree(latest_backup, self.deploy_dir)

            self.logger.info(f"Rollback completed using backup {latest_backup.name}")

            return {
                "status": "success",
                "backup_used": latest_backup.name,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _run_command(self, command: List[str], cwd: Optional[str] = None) -> Tuple[str, str, int]:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.deploy_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                self.logger.error(f"Command failed: {' '.join(command)}")
                self.logger.error(f"Error output: {result.stderr}")

            return result.stdout, result.stderr, result.returncode

        except subprocess.TimeoutExpired:
            raise DeploymentError(f"Command timed out: {' '.join(command)}")
        except Exception as e:
            raise DeploymentError(f"Command execution failed: {e}")

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            "deployment_id": self.deployment_id,
            "environment": self.config.environment,
            "services": self._get_service_status(),
            "deploy_path": str(self.deploy_dir),
            "backup_count": len(list(self.backup_dir.glob("backup_*"))),
            "last_deployment": (
                self.deployment_start_time.isoformat() if self.deployment_start_time else None
            ),
        }

    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []

        for backup_path in self.backup_dir.glob("backup_*"):
            stat = backup_path.stat()
            backups.append(
                {
                    "name": backup_path.name,
                    "path": str(backup_path),
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

        return sorted(backups, key=lambda x: x["created"], reverse=True)

    def cleanup_old_backups(self, keep_count: int = 5) -> None:
        """Clean up old backups, keeping only the most recent ones."""
        try:
            backups = self.list_backups()

            if len(backups) <= keep_count:
                return

            # Remove oldest backups
            for backup in backups[keep_count:]:
                backup_path = Path(backup["path"])
                shutil.rmtree(backup_path)
                self.logger.info(f"Removed old backup: {backup['name']}")

        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")


def create_deployment_manager(
    config: Optional[DeploymentConfig] = None,
) -> DeploymentManager:
    """Create and configure a deployment manager.

    Args:
        config: Deployment configuration

    Returns:
        Configured deployment manager
    """
    return DeploymentManager(config)


# CLI interface for deployment
def main() -> None:
    """Command-line interface for deployment."""
    import argparse

    parser = argparse.ArgumentParser(description="APGI Framework Deployment Automation")
    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        default="development",
        help="Deployment environment",
    )
    parser.add_argument("--dashboard-port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--api-port", type=int, default=8000, help="API port")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")
    parser.add_argument("--source", help="Source code path")
    parser.add_argument("--status", action="store_true", help="Show deployment status")
    parser.add_argument("--list-backups", action="store_true", help="List available backups")
    parser.add_argument(
        "--cleanup-backups",
        type=int,
        metavar="COUNT",
        help="Clean up old backups, keeping COUNT most recent",
    )

    args = parser.parse_args()

    # Create deployment configuration
    config = DeploymentConfig(
        environment=args.environment,
        dashboard_port=args.dashboard_port,
        api_port=args.api_port,
    )

    # Create deployment manager
    manager = create_deployment_manager(config)

    try:
        if args.status:
            status = manager.get_deployment_status()
            logger.info(json.dumps(status, indent=2))

        elif args.list_backups:
            backups = manager.list_backups()
            logger.info(json.dumps(backups, indent=2))

        elif args.cleanup_backups:
            manager.cleanup_old_backups(args.cleanup_backups)
            logger.info(f"Cleaned up old backups, keeping {args.cleanup_backups} most recent")

        else:
            # Perform deployment
            result = manager.deploy(source_path=args.source, create_backup=not args.no_backup)
            logger.info(json.dumps(result, indent=2))

    except Exception as e:
        logger.info(f"Deployment failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
