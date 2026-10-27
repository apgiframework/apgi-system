"""Incident Response Runbooks.

Provides runbooks for incident response paths tied to alerting middleware.
Enables rapid response to common operational issues.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class IncidentType(Enum):
    """Types of incidents."""

    PERFORMANCE_DEGRADATION = "performance_degradation"
    SERVICE_UNAVAILABLE = "service_unavailable"
    SECURITY_BREACH = "security_breach"
    DATA_LOSS = "data_loss"
    AUTHENTICATION_FAILURE = "authentication_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    DATABASE_ERROR = "database_error"
    REDIS_UNAVAILABLE = "redis_unavailable"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    DISK_FULL = "disk_full"


@dataclass
class RunbookStep:
    """Single step in incident runbook."""

    step_number: int
    title: str
    description: str
    action: str
    expected_outcome: str
    rollback_action: Optional[str] = None
    estimated_duration_seconds: int = 60


@dataclass
class IncidentRunbook:
    """Incident response runbook."""

    incident_type: IncidentType
    severity: IncidentSeverity
    title: str
    description: str
    detection_criteria: str
    steps: List[RunbookStep]
    escalation_contacts: List[str]
    post_incident_review: str
    estimated_resolution_time_minutes: int


class IncidentRunbookRegistry:
    """Registry of incident runbooks."""

    def __init__(self) -> None:
        """Initialize runbook registry."""
        self.runbooks: Dict[IncidentType, IncidentRunbook] = {}
        self._register_default_runbooks()

    def _register_default_runbooks(self) -> None:
        """Register default incident runbooks."""
        # Performance degradation runbook
        self.register_runbook(
            IncidentRunbook(
                incident_type=IncidentType.PERFORMANCE_DEGRADATION,
                severity=IncidentSeverity.WARNING,
                title="Performance Degradation Response",
                description="Respond to system performance degradation",
                detection_criteria="P95 latency exceeds SLO budget by >20%",
                steps=[
                    RunbookStep(
                        step_number=1,
                        title="Verify Performance Metrics",
                        description="Confirm performance degradation in monitoring dashboard",
                        action="Check Grafana dashboard for latency/throughput metrics",
                        expected_outcome="Confirm P95 latency spike",
                    ),
                    RunbookStep(
                        step_number=2,
                        title="Check Resource Utilization",
                        description="Verify CPU, memory, and disk usage",
                        action="Run: kubectl top nodes && kubectl top pods",
                        expected_outcome="Identify resource bottleneck",
                    ),
                    RunbookStep(
                        step_number=3,
                        title="Review Recent Deployments",
                        description="Check if recent deployment caused issue",
                        action="Review deployment history and recent code changes",
                        expected_outcome="Identify potential root cause",
                    ),
                    RunbookStep(
                        step_number=4,
                        title="Scale Resources if Needed",
                        description="Increase pod replicas or node capacity",
                        action="kubectl scale deployment api --replicas=5",
                        expected_outcome="Improved latency metrics",
                        rollback_action="kubectl scale deployment api --replicas=3",
                    ),
                    RunbookStep(
                        step_number=5,
                        title="Enable Degraded Mode if Necessary",
                        description="Disable non-critical features to reduce load",
                        action="Set DEGRADED_MODE=true in config",
                        expected_outcome="Reduced system load",
                        rollback_action="Set DEGRADED_MODE=false",
                    ),
                ],
                escalation_contacts=["on-call-engineer@company.com", "platform-team@company.com"],
                post_incident_review="Analyze root cause and implement preventive measures",
                estimated_resolution_time_minutes=15,
            )
        )

        # Redis unavailable runbook
        self.register_runbook(
            IncidentRunbook(
                incident_type=IncidentType.REDIS_UNAVAILABLE,
                severity=IncidentSeverity.CRITICAL,
                title="Redis Unavailability Response",
                description="Respond to Redis service unavailability",
                detection_criteria="Redis health check fails for >30 seconds",
                steps=[
                    RunbookStep(
                        step_number=1,
                        title="Verify Redis Status",
                        description="Confirm Redis is unavailable",
                        action="redis-cli ping",
                        expected_outcome="Connection refused or timeout",
                    ),
                    RunbookStep(
                        step_number=2,
                        title="Check Redis Logs",
                        description="Review Redis error logs",
                        action="kubectl logs -f redis-pod",
                        expected_outcome="Identify Redis error",
                    ),
                    RunbookStep(
                        step_number=3,
                        title="Restart Redis",
                        description="Restart Redis service",
                        action="kubectl restart pod redis-pod",
                        expected_outcome="Redis becomes available",
                    ),
                    RunbookStep(
                        step_number=4,
                        title="Enable Fallback Mode",
                        description="Switch to in-memory fallback",
                        action="System automatically enables degraded mode",
                        expected_outcome="Rate limiting and caching use memory backend",
                    ),
                    RunbookStep(
                        step_number=5,
                        title="Monitor Fallback Performance",
                        description="Monitor system performance in fallback mode",
                        action="Watch memory usage and error rates",
                        expected_outcome="System remains operational",
                    ),
                ],
                escalation_contacts=["database-team@company.com", "on-call-engineer@company.com"],
                post_incident_review="Implement Redis HA and monitoring improvements",
                estimated_resolution_time_minutes=10,
            )
        )

        # Security breach runbook
        self.register_runbook(
            IncidentRunbook(
                incident_type=IncidentType.SECURITY_BREACH,
                severity=IncidentSeverity.EMERGENCY,
                title="Security Breach Response",
                description="Respond to potential security breach",
                detection_criteria="Unauthorized access detected or suspicious activity",
                steps=[
                    RunbookStep(
                        step_number=1,
                        title="Isolate Affected Systems",
                        description="Isolate compromised systems from network",
                        action="Disable affected service pods",
                        expected_outcome="Breach contained",
                    ),
                    RunbookStep(
                        step_number=2,
                        title="Preserve Evidence",
                        description="Collect logs and forensic data",
                        action="Export logs to secure storage",
                        expected_outcome="Evidence preserved for investigation",
                    ),
                    RunbookStep(
                        step_number=3,
                        title="Notify Security Team",
                        description="Alert security team immediately",
                        action="Page security-team@company.com",
                        expected_outcome="Security team engaged",
                    ),
                    RunbookStep(
                        step_number=4,
                        title="Rotate Credentials",
                        description="Rotate all potentially compromised credentials",
                        action="Regenerate JWT secrets and database passwords",
                        expected_outcome="Credentials rotated",
                    ),
                    RunbookStep(
                        step_number=5,
                        title="Restore from Backup",
                        description="Restore systems from clean backup",
                        action="Restore database and application from backup",
                        expected_outcome="Systems restored to known good state",
                    ),
                ],
                escalation_contacts=[
                    "security-team@company.com",
                    "ciso@company.com",
                    "legal@company.com",
                ],
                post_incident_review="Conduct full security audit and implement preventive measures",
                estimated_resolution_time_minutes=30,
            )
        )

        # Database error runbook
        self.register_runbook(
            IncidentRunbook(
                incident_type=IncidentType.DATABASE_ERROR,
                severity=IncidentSeverity.CRITICAL,
                title="Database Error Response",
                description="Respond to database connectivity or query errors",
                detection_criteria="Database connection errors or query timeouts",
                steps=[
                    RunbookStep(
                        step_number=1,
                        title="Verify Database Status",
                        description="Check database health",
                        action="psql -c 'SELECT 1'",
                        expected_outcome="Database responds",
                    ),
                    RunbookStep(
                        step_number=2,
                        title="Check Connection Pool",
                        description="Verify connection pool status",
                        action="Check active connections: SELECT count(*) FROM pg_stat_activity",
                        expected_outcome="Connection pool status identified",
                    ),
                    RunbookStep(
                        step_number=3,
                        title="Review Slow Queries",
                        description="Identify slow or blocking queries",
                        action="Check pg_stat_statements for slow queries",
                        expected_outcome="Slow queries identified",
                    ),
                    RunbookStep(
                        step_number=4,
                        title="Kill Blocking Queries",
                        description="Terminate blocking queries if necessary",
                        action="SELECT pg_terminate_backend(pid) for blocking queries",
                        expected_outcome="Queries terminated",
                    ),
                    RunbookStep(
                        step_number=5,
                        title="Restart Database if Needed",
                        description="Restart database service",
                        action="kubectl restart pod postgres-pod",
                        expected_outcome="Database becomes responsive",
                    ),
                ],
                escalation_contacts=["database-team@company.com", "on-call-engineer@company.com"],
                post_incident_review="Analyze query performance and implement optimizations",
                estimated_resolution_time_minutes=20,
            )
        )

    def register_runbook(self, runbook: IncidentRunbook) -> None:
        """Register incident runbook.

        Args:
            runbook: Incident runbook
        """
        self.runbooks[runbook.incident_type] = runbook
        logger.info(f"Registered runbook for {runbook.incident_type.value}")

    def get_runbook(self, incident_type: IncidentType) -> Optional[IncidentRunbook]:
        """Get runbook for incident type.

        Args:
            incident_type: Type of incident

        Returns:
            Incident runbook or None
        """
        return self.runbooks.get(incident_type)

    def get_all_runbooks(self) -> Dict[IncidentType, IncidentRunbook]:
        """Get all registered runbooks."""
        return self.runbooks.copy()

    def export_runbooks_to_markdown(self, output_file: str = "incident_runbooks.md") -> None:
        """Export runbooks to markdown.

        Args:
            output_file: Output markdown file
        """
        with open(output_file, "w") as f:
            f.write("# Incident Response Runbooks\n\n")

            for incident_type, runbook in sorted(
                self.runbooks.items(), key=lambda x: x[1].severity.value
            ):
                f.write(f"## {runbook.title}\n\n")
                f.write(f"**Type:** {incident_type.value}\n")
                f.write(f"**Severity:** {runbook.severity.value}\n")
                f.write(f"**Description:** {runbook.description}\n")
                f.write(f"**Detection:** {runbook.detection_criteria}\n")
                f.write(
                    f"**Estimated Resolution:** {runbook.estimated_resolution_time_minutes} minutes\n\n"
                )

                f.write("### Steps\n\n")
                for step in runbook.steps:
                    f.write(f"#### Step {step.step_number}: {step.title}\n\n")
                    f.write(f"**Description:** {step.description}\n\n")
                    f.write(f"**Action:** `{step.action}`\n\n")
                    f.write(f"**Expected Outcome:** {step.expected_outcome}\n\n")
                    if step.rollback_action:
                        f.write(f"**Rollback:** `{step.rollback_action}`\n\n")

                f.write("### Escalation Contacts\n\n")
                for contact in runbook.escalation_contacts:
                    f.write(f"- {contact}\n")

                f.write("\n### Post-Incident Review\n\n")
                f.write(f"{runbook.post_incident_review}\n\n")
                f.write("---\n\n")

        logger.info(f"Exported runbooks to {output_file}")


# Global runbook registry
_runbook_registry: Optional[IncidentRunbookRegistry] = None


def get_runbook_registry() -> IncidentRunbookRegistry:
    """Get or create global runbook registry."""
    global _runbook_registry
    if _runbook_registry is None:
        _runbook_registry = IncidentRunbookRegistry()
    return _runbook_registry
