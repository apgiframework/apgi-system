"""Compliance Evidence Pack Generator.

Produces machine-readable control mapping for GDPR, HIPAA, SOC2, ISO 27001
with control owners, test evidence links, and residual risk register.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""

    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"


class ControlStatus(Enum):
    """Control implementation status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    TESTED = "tested"
    CERTIFIED = "certified"


class RiskLevel(Enum):
    """Residual risk levels."""

    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ControlMapping:
    """Mapping of control to implementation."""

    framework: ComplianceFramework
    control_id: str
    control_name: str
    description: str
    owner: str
    status: ControlStatus
    implementation_file: str
    test_file: Optional[str] = None
    test_evidence_link: Optional[str] = None
    last_tested: Optional[datetime] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["framework"] = self.framework.value
        data["status"] = self.status.value
        if self.last_tested:
            data["last_tested"] = self.last_tested.isoformat()
        return data


@dataclass
class ResidualRisk:
    """Residual risk item."""

    risk_id: str
    description: str
    framework: ComplianceFramework
    related_controls: List[str]
    risk_level: RiskLevel
    mitigation_strategy: str
    owner: str
    target_resolution_date: Optional[datetime] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["framework"] = self.framework.value
        data["risk_level"] = self.risk_level.value
        if self.target_resolution_date:
            data["target_resolution_date"] = self.target_resolution_date.isoformat()
        return data


class ComplianceEvidencePack:
    """Generates compliance evidence pack."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize evidence pack generator.

        Args:
            output_dir: Directory to write evidence files
        """
        self.output_dir = output_dir or Path("compliance_evidence")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.control_mappings: List[ControlMapping] = []
        self.residual_risks: List[ResidualRisk] = []

    def add_control_mapping(self, mapping: ControlMapping) -> None:
        """Add control mapping.

        Args:
            mapping: Control mapping
        """
        self.control_mappings.append(mapping)
        logger.info(f"Added control mapping: {mapping.framework.value}/{mapping.control_id}")

    def add_residual_risk(self, risk: ResidualRisk) -> None:
        """Add residual risk.

        Args:
            risk: Residual risk
        """
        self.residual_risks.append(risk)
        logger.info(f"Added residual risk: {risk.risk_id}")

    def generate_gdpr_controls(self) -> None:
        """Generate GDPR control mappings."""
        gdpr_controls = [
            ControlMapping(
                framework=ComplianceFramework.GDPR,
                control_id="A.32",
                control_name="Security of Processing",
                description="Implement appropriate technical and organizational measures",
                owner="Security Team",
                status=ControlStatus.IMPLEMENTED,
                implementation_file="apgi_framework/security/authentication.py",
                test_file="tests/unit/test_authentication.py",
                test_evidence_link="tests/unit/test_authentication.py::test_jwt_validation",
            ),
            ControlMapping(
                framework=ComplianceFramework.GDPR,
                control_id="A.33",
                control_name="Notification of Personal Data Breach",
                description="Notify supervisory authority without undue delay",
                owner="Compliance Team",
                status=ControlStatus.IMPLEMENTED,
                implementation_file="apgi_framework/compliance/compliance_framework.py",
                test_file="tests/unit/test_compliance.py",
            ),
            ControlMapping(
                framework=ComplianceFramework.GDPR,
                control_id="A.34",
                control_name="Data Protection Impact Assessment",
                description="Conduct DPIA for high-risk processing",
                owner="Data Protection Officer",
                status=ControlStatus.IN_PROGRESS,
                implementation_file="apgi_framework/compliance/dpia.py",
            ),
            ControlMapping(
                framework=ComplianceFramework.GDPR,
                control_id="A.35",
                control_name="Prior Consultation",
                description="Consult supervisory authority before processing",
                owner="Data Protection Officer",
                status=ControlStatus.IMPLEMENTED,
                implementation_file="apgi_framework/compliance/consultation.py",
            ),
        ]

        for control in gdpr_controls:
            self.add_control_mapping(control)

    def generate_hipaa_controls(self) -> None:
        """Generate HIPAA control mappings."""
        hipaa_controls = [
            ControlMapping(
                framework=ComplianceFramework.HIPAA,
                control_id="164.308(a)(1)",
                control_name="Security Management Process",
                description="Implement security management process",
                owner="Security Officer",
                status=ControlStatus.IMPLEMENTED,
                implementation_file="apgi_framework/security/authentication.py",
                test_file="tests/unit/test_hipaa_compliance.py",
            ),
            ControlMapping(
                framework=ComplianceFramework.HIPAA,
                control_id="164.308(a)(3)",
                control_name="Workforce Security",
                description="Implement workforce security procedures",
                owner="HR/Security",
                status=ControlStatus.IMPLEMENTED,
                implementation_file="apgi_framework/security/authentication.py",
            ),
            ControlMapping(
                framework=ComplianceFramework.HIPAA,
                control_id="164.312(a)(2)",
                control_name="Encryption and Decryption",
                description="Implement encryption for data at rest and in transit",
                owner="Security Team",
                status=ControlStatus.IMPLEMENTED,
                implementation_file="apgi_framework/security/encryption.py",
            ),
            ControlMapping(
                framework=ComplianceFramework.HIPAA,
                control_id="164.312(b)",
                control_name="Audit Controls",
                description="Implement audit logging and monitoring",
                owner="Compliance Team",
                status=ControlStatus.IMPLEMENTED,
                implementation_file="apgi_framework/logging/standardized_logging.py",
            ),
        ]

        for control in hipaa_controls:
            self.add_control_mapping(control)

    def generate_soc2_controls(self) -> None:
        """Generate SOC2 control mappings."""
        soc2_controls = [
            ControlMapping(
                framework=ComplianceFramework.SOC2,
                control_id="CC6.1",
                control_name="Logical Access Controls",
                description="Implement logical access controls",
                owner="Security Team",
                status=ControlStatus.IMPLEMENTED,
                implementation_file="apgi_framework/security/authentication.py",
                test_file="tests/unit/test_access_control.py",
            ),
            ControlMapping(
                framework=ComplianceFramework.SOC2,
                control_id="CC7.2",
                control_name="System Monitoring",
                description="Monitor system activity and performance",
                owner="Operations Team",
                status=ControlStatus.IMPLEMENTED,
                implementation_file="apgi_framework/monitoring.py",
            ),
            ControlMapping(
                framework=ComplianceFramework.SOC2,
                control_id="A1.1",
                control_name="Availability",
                description="System is available for operation and use",
                owner="Operations Team",
                status=ControlStatus.IMPLEMENTED,
                implementation_file="apgi_framework/resilience/degraded_mode_manager.py",
            ),
        ]

        for control in soc2_controls:
            self.add_control_mapping(control)

    def generate_iso27001_controls(self) -> None:
        """Generate ISO 27001 control mappings."""
        iso_controls = [
            ControlMapping(
                framework=ComplianceFramework.ISO27001,
                control_id="A.9.2.1",
                control_name="User Registration and De-registration",
                description="Manage user access lifecycle",
                owner="Security Team",
                status=ControlStatus.IMPLEMENTED,
                implementation_file="apgi_framework/security/authentication.py",
            ),
            ControlMapping(
                framework=ComplianceFramework.ISO27001,
                control_id="A.10.1.1",
                control_name="Cryptography Policy",
                description="Implement cryptography controls",
                owner="Security Team",
                status=ControlStatus.IMPLEMENTED,
                implementation_file="apgi_framework/security/encryption.py",
            ),
            ControlMapping(
                framework=ComplianceFramework.ISO27001,
                control_id="A.12.4.1",
                control_name="Event Logging",
                description="Record user activities and system events",
                owner="Compliance Team",
                status=ControlStatus.IMPLEMENTED,
                implementation_file="apgi_framework/logging/standardized_logging.py",
            ),
        ]

        for control in iso_controls:
            self.add_control_mapping(control)

    def generate_residual_risks(self) -> None:
        """Generate residual risk register."""
        risks = [
            ResidualRisk(
                risk_id="RISK-001",
                description="Redis unavailability impacts rate limiting",
                framework=ComplianceFramework.SOC2,
                related_controls=["CC6.1", "CC7.2"],
                risk_level=RiskLevel.MEDIUM,
                mitigation_strategy="Implement in-memory fallback with feature degradation",
                owner="Security Team",
            ),
            ResidualRisk(
                risk_id="RISK-002",
                description="Benchmark performance regression detection",
                framework=ComplianceFramework.SOC2,
                related_controls=["A1.1"],
                risk_level=RiskLevel.LOW,
                mitigation_strategy="Make benchmark tooling mandatory in CI",
                owner="DevOps Team",
            ),
            ResidualRisk(
                risk_id="RISK-003",
                description="PII exposure in logs",
                framework=ComplianceFramework.GDPR,
                related_controls=["A.32"],
                risk_level=RiskLevel.HIGH,
                mitigation_strategy="Implement PII masking in compliance middleware",
                owner="Security Team",
            ),
        ]

        for risk in risks:
            self.add_residual_risk(risk)

    def export_to_json(self, filename: str = "compliance_evidence.json") -> Path:
        """Export evidence pack to JSON.

        Args:
            filename: Output filename

        Returns:
            Path to exported file
        """
        output_file = self.output_dir / filename
        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "control_mappings": [m.to_dict() for m in self.control_mappings],
            "residual_risks": [r.to_dict() for r in self.residual_risks],
            "summary": {
                "total_controls": len(self.control_mappings),
                "controls_by_status": self._count_by_status(),
                "total_risks": len(self.residual_risks),
                "risks_by_level": self._count_risks_by_level(),
            },
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported compliance evidence to {output_file}")
        return output_file

    def _count_by_status(self) -> Dict[str, int]:
        """Count controls by status."""
        counts = {}
        for status in ControlStatus:
            count = sum(1 for m in self.control_mappings if m.status == status)
            if count > 0:
                counts[status.value] = count
        return counts

    def _count_risks_by_level(self) -> Dict[str, int]:
        """Count risks by level."""
        counts = {}
        for level in RiskLevel:
            count = sum(1 for r in self.residual_risks if r.risk_level == level)
            if count > 0:
                counts[level.value] = count
        return counts

    def generate_full_pack(self) -> Path:
        """Generate complete compliance evidence pack.

        Returns:
            Path to exported file
        """
        self.generate_gdpr_controls()
        self.generate_hipaa_controls()
        self.generate_soc2_controls()
        self.generate_iso27001_controls()
        self.generate_residual_risks()

        return self.export_to_json()
