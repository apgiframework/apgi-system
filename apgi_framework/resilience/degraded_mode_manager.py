"""Degraded Mode Operation Manager.

Enables graceful degradation when Redis or other critical services are unavailable.
Provides explicit feature downgrades and alerts.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set

from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)


class OperationMode(Enum):
    """System operation modes."""

    NORMAL = "normal"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    CRITICAL = "critical"


class FeatureStatus(Enum):
    """Feature availability status."""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class DegradedModeConfig:
    """Configuration for degraded mode operation."""

    # Redis fallback
    enable_redis_fallback: bool = True
    redis_fallback_ttl_seconds: int = 3600  # 1 hour
    redis_retry_interval_seconds: int = 30

    # Rate limiting fallback
    enable_rate_limit_fallback: bool = True
    rate_limit_fallback_strategy: str = "memory"  # "memory" or "none"

    # Session fallback
    enable_session_fallback: bool = True
    session_fallback_storage: str = "memory"  # "memory" or "disk"

    # Cache fallback
    enable_cache_fallback: bool = True
    cache_fallback_ttl_seconds: int = 300  # 5 minutes

    # Feature downgrades
    disable_features_on_degradation: Set[str] = field(
        default_factory=lambda: {"analytics", "recommendations"}
    )

    # Alerting
    alert_on_degradation: bool = True
    alert_callback: Optional[Callable] = None


@dataclass
class ServiceHealth:
    """Health status of a service."""

    service_name: str
    is_healthy: bool
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    last_recovery: Optional[datetime] = None


class DegradedModeManager:
    """Manages degraded mode operation."""

    def __init__(self, config: Optional[DegradedModeConfig] = None):
        """Initialize degraded mode manager.

        Args:
            config: Degraded mode configuration
        """
        self.config = config or DegradedModeConfig()
        self.mode = OperationMode.NORMAL
        self.services: Dict[str, ServiceHealth] = {}
        self.feature_status: Dict[str, FeatureStatus] = {}
        self.fallback_data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._monitoring = False

    def register_service(self, service_name: str) -> None:
        """Register a service for health monitoring.

        Args:
            service_name: Name of service
        """
        with self._lock:
            self.services[service_name] = ServiceHealth(
                service_name=service_name,
                is_healthy=True,
            )
            logger.info(f"Registered service for monitoring: {service_name}")

    def report_service_health(
        self,
        service_name: str,
        is_healthy: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Report service health status.

        Args:
            service_name: Name of service
            is_healthy: Whether service is healthy
            error_message: Optional error message
        """
        with self._lock:
            if service_name not in self.services:
                self.register_service(service_name)

            service = self.services[service_name]
            was_healthy = service.is_healthy

            service.is_healthy = is_healthy
            service.last_check = datetime.now(timezone.utc)
            service.error_message = error_message

            if is_healthy:
                service.consecutive_failures = 0
                if not was_healthy:
                    service.last_recovery = datetime.now(timezone.utc)
                    logger.info(f"Service recovered: {service_name}")
                    self._check_mode_transition()
            else:
                service.consecutive_failures += 1
                logger.warning(
                    f"Service unhealthy: {service_name} (failures: {service.consecutive_failures})",
                    extra={"error": error_message},
                )
                self._check_mode_transition()

    def _check_mode_transition(self) -> None:
        """Check if mode should transition based on service health."""
        unhealthy_services = [s for s in self.services.values() if not s.is_healthy]

        if not unhealthy_services:
            if self.mode != OperationMode.NORMAL:
                logger.info("Transitioning to NORMAL mode - all services healthy")
                self.mode = OperationMode.NORMAL
                self._restore_features()
        else:
            if self.mode == OperationMode.NORMAL:
                logger.warning(
                    f"Transitioning to DEGRADED mode - {len(unhealthy_services)} services unhealthy"
                )
                self.mode = OperationMode.DEGRADED
                self._downgrade_features()

                if self.config.alert_on_degradation and self.config.alert_callback:
                    self.config.alert_callback(
                        {
                            "mode": self.mode.value,
                            "unhealthy_services": [s.service_name for s in unhealthy_services],
                        }
                    )

    def _downgrade_features(self) -> None:
        """Downgrade features when entering degraded mode."""
        for feature in self.config.disable_features_on_degradation:
            self.feature_status[feature] = FeatureStatus.UNAVAILABLE
            logger.info(f"Feature disabled in degraded mode: {feature}")

    def _restore_features(self) -> None:
        """Restore features when returning to normal mode."""
        for feature in self.config.disable_features_on_degradation:
            self.feature_status[feature] = FeatureStatus.AVAILABLE
            logger.info(f"Feature restored: {feature}")

    def is_feature_available(self, feature_name: str) -> bool:
        """Check if feature is available.

        Args:
            feature_name: Name of feature

        Returns:
            True if feature is available
        """
        with self._lock:
            status = self.feature_status.get(feature_name, FeatureStatus.AVAILABLE)
            return status == FeatureStatus.AVAILABLE

    def get_mode(self) -> OperationMode:
        """Get current operation mode."""
        with self._lock:
            return self.mode

    def get_health_report(self) -> Dict[str, Any]:
        """Get system health report.

        Returns:
            Health status of all services
        """
        with self._lock:
            return {
                "mode": self.mode.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "services": [
                    {
                        "name": s.service_name,
                        "healthy": s.is_healthy,
                        "last_check": s.last_check.isoformat(),
                        "consecutive_failures": s.consecutive_failures,
                        "error_message": s.error_message,
                        "last_recovery": s.last_recovery.isoformat() if s.last_recovery else None,
                    }
                    for s in self.services.values()
                ],
                "features": {name: status.value for name, status in self.feature_status.items()},
            }

    def store_fallback_data(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store data for fallback use.

        Args:
            key: Data key
            value: Data value
            ttl_seconds: Optional time-to-live
        """
        with self._lock:
            self.fallback_data[key] = {
                "value": value,
                "stored_at": datetime.now(timezone.utc),
                "ttl_seconds": ttl_seconds,
            }

    def get_fallback_data(self, key: str) -> Optional[Any]:
        """Get fallback data.

        Args:
            key: Data key

        Returns:
            Data value or None if expired/missing
        """
        with self._lock:
            if key not in self.fallback_data:
                return None

            entry = self.fallback_data[key]
            if entry["ttl_seconds"]:
                age = (datetime.now(timezone.utc) - entry["stored_at"]).total_seconds()
                if age > entry["ttl_seconds"]:
                    del self.fallback_data[key]
                    return None

            return entry["value"]

    def clear_fallback_data(self) -> None:
        """Clear all fallback data."""
        with self._lock:
            self.fallback_data.clear()


# Global degraded mode manager instance
_degraded_mode_manager: Optional[DegradedModeManager] = None


def get_degraded_mode_manager(
    config: Optional[DegradedModeConfig] = None,
) -> DegradedModeManager:
    """Get or create global degraded mode manager."""
    global _degraded_mode_manager
    if _degraded_mode_manager is None:
        _degraded_mode_manager = DegradedModeManager(config)
    return _degraded_mode_manager
