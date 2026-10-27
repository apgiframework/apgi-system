"""Service Level Objective (SLO) Tracking and Alerting.

Tracks endpoint-level performance against SLO budgets (P50/P95/P99)
and generates alerts when thresholds are exceeded.
"""

import statistics
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SLOBudget:
    """SLO budget for an endpoint."""

    endpoint: str
    p50_ms: float  # Median response time
    p95_ms: float  # 95th percentile
    p99_ms: float  # 99th percentile
    error_rate_threshold: float = 0.05  # 5% error rate
    availability_threshold: float = 0.99  # 99% availability


@dataclass
class PerformanceMetric:
    """Single performance measurement."""

    endpoint: str
    response_time_ms: float
    status_code: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: bool = False


@dataclass
class SLOAlert:
    """Alert for SLO violation."""

    endpoint: str
    metric_type: str  # "p50", "p95", "p99", "error_rate", "availability"
    current_value: float
    threshold: float
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""


class SLOTracker:
    """Tracks SLO compliance for endpoints."""

    def __init__(self, window_size: int = 1000, alert_callback: Optional[Callable] = None):
        """Initialize SLO tracker.

        Args:
            window_size: Number of metrics to keep in rolling window
            alert_callback: Optional callback for alerts
        """
        self.window_size = window_size
        self.alert_callback = alert_callback
        self.metrics: Dict[str, Deque[PerformanceMetric]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.slo_budgets: Dict[str, SLOBudget] = {}
        self.alerts: List[SLOAlert] = []
        self._lock = threading.Lock()

    def register_slo(self, budget: SLOBudget) -> None:
        """Register SLO budget for endpoint.

        Args:
            budget: SLO budget configuration
        """
        with self._lock:
            self.slo_budgets[budget.endpoint] = budget
            logger.info(
                f"Registered SLO for {budget.endpoint}: "
                f"P50={budget.p50_ms}ms, P95={budget.p95_ms}ms, P99={budget.p99_ms}ms"
            )

    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record performance metric.

        Args:
            metric: Performance measurement
        """
        with self._lock:
            self.metrics[metric.endpoint].append(metric)

            # Check SLO compliance
            if metric.endpoint in self.slo_budgets:
                self._check_slo_compliance(metric.endpoint)

    def _check_slo_compliance(self, endpoint: str) -> None:
        """Check if endpoint meets SLO budgets."""
        budget = self.slo_budgets[endpoint]
        metrics = list(self.metrics[endpoint])

        if len(metrics) < 10:
            return  # Need minimum samples

        response_times = [m.response_time_ms for m in metrics]
        errors = [m for m in metrics if m.error]

        # Calculate percentiles
        sorted_times = sorted(response_times)
        p50 = statistics.median(sorted_times)
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]

        # Check P50
        if p50 > budget.p50_ms:
            alert = SLOAlert(
                endpoint=endpoint,
                metric_type="p50",
                current_value=p50,
                threshold=budget.p50_ms,
                severity=AlertSeverity.WARNING,
                message=f"P50 latency {p50:.2f}ms exceeds budget {budget.p50_ms}ms",
            )
            self._emit_alert(alert)

        # Check P95
        if p95 > budget.p95_ms:
            alert = SLOAlert(
                endpoint=endpoint,
                metric_type="p95",
                current_value=p95,
                threshold=budget.p95_ms,
                severity=AlertSeverity.WARNING,
                message=f"P95 latency {p95:.2f}ms exceeds budget {budget.p95_ms}ms",
            )
            self._emit_alert(alert)

        # Check P99
        if p99 > budget.p99_ms:
            alert = SLOAlert(
                endpoint=endpoint,
                metric_type="p99",
                current_value=p99,
                threshold=budget.p99_ms,
                severity=AlertSeverity.CRITICAL,
                message=f"P99 latency {p99:.2f}ms exceeds budget {budget.p99_ms}ms",
            )
            self._emit_alert(alert)

        # Check error rate
        error_rate = len(errors) / len(metrics) if metrics else 0
        if error_rate > budget.error_rate_threshold:
            alert = SLOAlert(
                endpoint=endpoint,
                metric_type="error_rate",
                current_value=error_rate,
                threshold=budget.error_rate_threshold,
                severity=AlertSeverity.CRITICAL,
                message=f"Error rate {error_rate:.2%} exceeds threshold {budget.error_rate_threshold:.2%}",
            )
            self._emit_alert(alert)

    def _emit_alert(self, alert: SLOAlert) -> None:
        """Emit SLO alert."""
        with self._lock:
            self.alerts.append(alert)

        logger.warning(
            f"SLO Alert: {alert.endpoint} - {alert.message}",
            extra={
                "endpoint": alert.endpoint,
                "metric_type": alert.metric_type,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "severity": alert.severity.value,
            },
        )

        if self.alert_callback:
            self.alert_callback(alert)

    def get_slo_report(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get SLO compliance report.

        Args:
            endpoint: Optional endpoint to filter by

        Returns:
            SLO compliance report
        """
        with self._lock:
            endpoints = [endpoint] if endpoint else list(self.slo_budgets.keys())
            report = {}

            for ep in endpoints:
                if ep not in self.slo_budgets:
                    continue

                budget = self.slo_budgets[ep]
                metrics = list(self.metrics[ep])

                if not metrics:
                    report[ep] = {
                        "status": "no_data",
                        "samples": 0,
                    }
                    continue

                response_times = [m.response_time_ms for m in metrics]
                errors = [m for m in metrics if m.error]
                sorted_times = sorted(response_times)

                p50 = statistics.median(sorted_times)
                p95 = sorted_times[int(len(sorted_times) * 0.95)]
                p99 = sorted_times[int(len(sorted_times) * 0.99)]
                error_rate = len(errors) / len(metrics)

                report[ep] = {
                    "status": (
                        "compliant"
                        if (
                            p50 <= budget.p50_ms
                            and p95 <= budget.p95_ms
                            and p99 <= budget.p99_ms
                            and error_rate <= budget.error_rate_threshold
                        )
                        else "violated"
                    ),
                    "samples": len(metrics),
                    "p50": {
                        "current": p50,
                        "budget": budget.p50_ms,
                        "compliant": p50 <= budget.p50_ms,
                    },
                    "p95": {
                        "current": p95,
                        "budget": budget.p95_ms,
                        "compliant": p95 <= budget.p95_ms,
                    },
                    "p99": {
                        "current": p99,
                        "budget": budget.p99_ms,
                        "compliant": p99 <= budget.p99_ms,
                    },
                    "error_rate": {
                        "current": error_rate,
                        "threshold": budget.error_rate_threshold,
                        "compliant": error_rate <= budget.error_rate_threshold,
                    },
                }

            return report

    def get_recent_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts
        """
        with self._lock:
            return [
                {
                    "endpoint": a.endpoint,
                    "metric_type": a.metric_type,
                    "current_value": a.current_value,
                    "threshold": a.threshold,
                    "severity": a.severity.value,
                    "timestamp": a.timestamp.isoformat(),
                    "message": a.message,
                }
                for a in self.alerts[-limit:]
            ]

    def clear_alerts(self) -> None:
        """Clear alert history."""
        with self._lock:
            self.alerts.clear()


# Global SLO tracker instance
_slo_tracker: Optional[SLOTracker] = None


def get_slo_tracker() -> SLOTracker:
    """Get or create global SLO tracker."""
    global _slo_tracker
    if _slo_tracker is None:
        _slo_tracker = SLOTracker()
    return _slo_tracker


def record_endpoint_metric(
    endpoint: str, response_time_ms: float, status_code: int, error: bool = False
) -> None:
    """Record endpoint performance metric.

    Args:
        endpoint: Endpoint path
        response_time_ms: Response time in milliseconds
        status_code: HTTP status code
        error: Whether request resulted in error
    """
    tracker = get_slo_tracker()
    metric = PerformanceMetric(
        endpoint=endpoint,
        response_time_ms=response_time_ms,
        status_code=status_code,
        error=error or status_code >= 400,
    )
    tracker.record_metric(metric)
