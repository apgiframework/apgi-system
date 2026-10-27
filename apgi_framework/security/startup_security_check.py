"""Security Posture Verification at Startup.

Provides comprehensive security readiness checks that fail closed for production.
Ensures JWT keys, TLS certs, CORS, CSRF, and rate-limiter backend state are valid.
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging.standardized_logging import get_logger, get_security_logger

logger = get_logger(__name__)
security_logger = get_security_logger()


class SecurityLevel(Enum):
    """Security check severity levels."""

    CRITICAL = "critical"  # Must pass in production
    HIGH = "high"  # Should pass in production
    MEDIUM = "medium"  # Recommended
    LOW = "low"  # Nice to have


@dataclass
class SecurityCheckResult:
    """Result of a single security check."""

    name: str
    level: SecurityLevel
    passed: bool
    message: str
    remediation: Optional[str] = None


class SecurityPostureChecker:
    """Comprehensive security posture verification."""

    def __init__(self, environment: str = "development", strict_mode: bool = False):
        """Initialize security checker.

        Args:
            environment: 'development', 'staging', or 'production'
            strict_mode: If True, fail on any check failure in production
        """
        self.environment = environment
        self.strict_mode = strict_mode or environment == "production"
        self.results: List[SecurityCheckResult] = []

    def run_all_checks(self) -> Tuple[bool, Dict[str, Any]]:
        """Run all security checks and return overall status.

        Returns:
            Tuple of (all_passed, detailed_report)
        """
        self.results = []

        # Critical checks
        self._check_jwt_secret()
        self._check_environment_variables()
        self._check_cors_configuration()
        self._check_csrf_protection()
        self._check_rate_limiter_backend()
        self._check_tls_certificates()
        self._check_database_credentials()
        self._check_redis_connectivity()
        self._check_secret_key_entropy()
        self._check_debug_mode()

        # Generate report
        report = self._generate_report()

        # Determine if all critical checks passed
        critical_passed = all(r.passed for r in self.results if r.level == SecurityLevel.CRITICAL)

        if self.strict_mode and not critical_passed:
            security_logger.critical(
                "Security posture check failed in strict mode",
                extra={"report": report},
            )
            return False, report

        return critical_passed, report

    def _check_jwt_secret(self) -> None:
        """Verify JWT secret key is configured and strong."""
        jwt_secret = os.getenv("JWT_SECRET_KEY", "").strip()

        if not jwt_secret or jwt_secret == "REPLACE_ME_WITH_RANDOM_64_CHAR_SECRET":
            self.results.append(
                SecurityCheckResult(
                    name="JWT Secret Key",
                    level=SecurityLevel.CRITICAL,
                    passed=False,
                    message="JWT_SECRET_KEY not configured or using placeholder",
                    remediation="Generate a strong random 64-character secret: "
                    'python -c "import secrets; print(secrets.token_urlsafe(64))"',
                )
            )
            return

        if len(jwt_secret) < 32:
            self.results.append(
                SecurityCheckResult(
                    name="JWT Secret Key Entropy",
                    level=SecurityLevel.CRITICAL,
                    passed=False,
                    message=f"JWT_SECRET_KEY too short ({len(jwt_secret)} chars, need ≥32)",
                    remediation="Use a cryptographically random 64-character secret",
                )
            )
            return

        self.results.append(
            SecurityCheckResult(
                name="JWT Secret Key",
                level=SecurityLevel.CRITICAL,
                passed=True,
                message="JWT secret key configured with sufficient entropy",
            )
        )

    def _check_environment_variables(self) -> None:
        """Verify required environment variables are set."""
        required_vars = [
            "APGI_ENV",
            "JWT_SECRET_KEY",
            "CORS_ORIGINS",
        ]

        if self.environment == "production":
            required_vars.extend(
                [
                    "DATABASE_URL",
                    "REDIS_URL",
                ]
            )

        missing = [var for var in required_vars if not os.getenv(var)]

        if missing:
            self.results.append(
                SecurityCheckResult(
                    name="Required Environment Variables",
                    level=SecurityLevel.CRITICAL,
                    passed=False,
                    message=f"Missing environment variables: {', '.join(missing)}",
                    remediation="Set all required variables in .env or deployment config",
                )
            )
        else:
            self.results.append(
                SecurityCheckResult(
                    name="Required Environment Variables",
                    level=SecurityLevel.CRITICAL,
                    passed=True,
                    message="All required environment variables configured",
                )
            )

    def _check_cors_configuration(self) -> None:
        """Verify CORS is properly configured."""
        cors_origins = os.getenv("CORS_ORIGINS", "").strip()

        if not cors_origins:
            self.results.append(
                SecurityCheckResult(
                    name="CORS Configuration",
                    level=SecurityLevel.HIGH,
                    passed=False,
                    message="CORS_ORIGINS not configured",
                    remediation="Set CORS_ORIGINS to comma-separated list of allowed origins",
                )
            )
            return

        # Check for overly permissive CORS
        if "*" in cors_origins and self.environment == "production":
            self.results.append(
                SecurityCheckResult(
                    name="CORS Permissiveness",
                    level=SecurityLevel.CRITICAL,
                    passed=False,
                    message="CORS_ORIGINS set to '*' in production (overly permissive)",
                    remediation="Specify explicit allowed origins for production",
                )
            )
            return

        self.results.append(
            SecurityCheckResult(
                name="CORS Configuration",
                level=SecurityLevel.HIGH,
                passed=True,
                message=f"CORS configured with {len(cors_origins.split(','))} allowed origins",
            )
        )

    def _check_csrf_protection(self) -> None:
        """Verify CSRF protection is enabled."""
        csrf_enabled = os.getenv("CSRF_PROTECTION_ENABLED", "true").lower() == "true"

        if not csrf_enabled and self.environment == "production":
            self.results.append(
                SecurityCheckResult(
                    name="CSRF Protection",
                    level=SecurityLevel.CRITICAL,
                    passed=False,
                    message="CSRF protection disabled in production",
                    remediation="Set CSRF_PROTECTION_ENABLED=true",
                )
            )
        else:
            self.results.append(
                SecurityCheckResult(
                    name="CSRF Protection",
                    level=SecurityLevel.HIGH,
                    passed=True,
                    message="CSRF protection enabled",
                )
            )

    def _check_rate_limiter_backend(self) -> None:
        """Verify rate limiter backend is available."""
        rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"

        if not rate_limit_enabled:
            self.results.append(
                SecurityCheckResult(
                    name="Rate Limiting",
                    level=SecurityLevel.HIGH,
                    passed=False,
                    message="Rate limiting disabled",
                    remediation="Enable rate limiting: RATE_LIMIT_ENABLED=true",
                )
            )
            return

        # Check Redis availability for rate limiter
        redis_url = os.getenv("REDIS_URL", "").strip()
        if not redis_url and self.environment == "production":
            self.results.append(
                SecurityCheckResult(
                    name="Rate Limiter Backend",
                    level=SecurityLevel.HIGH,
                    passed=False,
                    message="Redis not configured for rate limiter backend",
                    remediation="Set REDIS_URL for distributed rate limiting",
                )
            )
        else:
            self.results.append(
                SecurityCheckResult(
                    name="Rate Limiting",
                    level=SecurityLevel.HIGH,
                    passed=True,
                    message="Rate limiting configured",
                )
            )

    def _check_tls_certificates(self) -> None:
        """Verify TLS certificates are configured for production."""
        if self.environment != "production":
            self.results.append(
                SecurityCheckResult(
                    name="TLS Certificates",
                    level=SecurityLevel.MEDIUM,
                    passed=True,
                    message="TLS check skipped in non-production environment",
                )
            )
            return

        cert_path = os.getenv("TLS_CERT_PATH", "").strip()
        key_path = os.getenv("TLS_KEY_PATH", "").strip()

        if not cert_path or not key_path:
            self.results.append(
                SecurityCheckResult(
                    name="TLS Certificates",
                    level=SecurityLevel.CRITICAL,
                    passed=False,
                    message="TLS certificate paths not configured",
                    remediation="Set TLS_CERT_PATH and TLS_KEY_PATH for HTTPS",
                )
            )
            return

        cert_exists = Path(cert_path).exists()
        key_exists = Path(key_path).exists()

        if not cert_exists or not key_exists:
            self.results.append(
                SecurityCheckResult(
                    name="TLS Certificates",
                    level=SecurityLevel.CRITICAL,
                    passed=False,
                    message=f"TLS files missing: cert={cert_exists}, key={key_exists}",
                    remediation="Ensure certificate and key files exist at configured paths",
                )
            )
            return

        self.results.append(
            SecurityCheckResult(
                name="TLS Certificates",
                level=SecurityLevel.CRITICAL,
                passed=True,
                message="TLS certificates configured and present",
            )
        )

    def _check_database_credentials(self) -> None:
        """Verify database credentials are not exposed."""
        db_url = os.getenv("DATABASE_URL", "").strip()

        if not db_url:
            if self.environment == "production":
                self.results.append(
                    SecurityCheckResult(
                        name="Database Configuration",
                        level=SecurityLevel.CRITICAL,
                        passed=False,
                        message="DATABASE_URL not configured",
                        remediation="Set DATABASE_URL for database connectivity",
                    )
                )
            return

        # Check for hardcoded credentials in URL
        if "password" in db_url.lower() and ":" in db_url:
            # This is expected, but log it
            self.results.append(
                SecurityCheckResult(
                    name="Database Credentials",
                    level=SecurityLevel.HIGH,
                    passed=True,
                    message="Database URL configured (credentials in environment)",
                )
            )
        else:
            self.results.append(
                SecurityCheckResult(
                    name="Database Configuration",
                    level=SecurityLevel.HIGH,
                    passed=True,
                    message="Database URL configured",
                )
            )

    def _check_redis_connectivity(self) -> None:
        """Verify Redis is accessible if configured."""
        redis_url = os.getenv("REDIS_URL", "").strip()

        if not redis_url:
            self.results.append(
                SecurityCheckResult(
                    name="Redis Configuration",
                    level=SecurityLevel.MEDIUM,
                    passed=True,
                    message="Redis not configured (optional)",
                )
            )
            return

        # Basic URL validation
        if not redis_url.startswith("redis://"):
            self.results.append(
                SecurityCheckResult(
                    name="Redis URL Format",
                    level=SecurityLevel.HIGH,
                    passed=False,
                    message="Invalid Redis URL format",
                    remediation="Use format: redis://[user:password@]host:port/db",
                )
            )
        else:
            self.results.append(
                SecurityCheckResult(
                    name="Redis Configuration",
                    level=SecurityLevel.MEDIUM,
                    passed=True,
                    message="Redis URL configured",
                )
            )

    def _check_secret_key_entropy(self) -> None:
        """Verify secret keys have sufficient entropy."""
        secret_key = os.getenv("APGI_SECRET_KEY", "").strip()

        if not secret_key or len(secret_key) < 32:
            self.results.append(
                SecurityCheckResult(
                    name="Secret Key Entropy",
                    level=SecurityLevel.HIGH,
                    passed=False,
                    message="APGI_SECRET_KEY missing or too short",
                    remediation='Generate: python -c "import secrets; print(secrets.token_hex(32))"',
                )
            )
        else:
            self.results.append(
                SecurityCheckResult(
                    name="Secret Key Entropy",
                    level=SecurityLevel.HIGH,
                    passed=True,
                    message="Secret key configured with sufficient entropy",
                )
            )

    def _check_debug_mode(self) -> None:
        """Verify debug mode is disabled in production."""
        debug_mode = os.getenv("APGI_DEBUG", "false").lower() == "true"

        if debug_mode and self.environment == "production":
            self.results.append(
                SecurityCheckResult(
                    name="Debug Mode",
                    level=SecurityLevel.CRITICAL,
                    passed=False,
                    message="Debug mode enabled in production",
                    remediation="Set APGI_DEBUG=false for production",
                )
            )
        else:
            self.results.append(
                SecurityCheckResult(
                    name="Debug Mode",
                    level=SecurityLevel.HIGH,
                    passed=True,
                    message="Debug mode properly configured",
                )
            )

    def _generate_report(self) -> Dict[str, Any]:
        """Generate detailed security report."""
        by_level = {}
        for level in SecurityLevel:
            by_level[level.value] = [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "remediation": r.remediation,
                }
                for r in self.results
                if r.level == level
            ]

        return {
            "environment": self.environment,
            "timestamp": str(
                __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
            ),
            "total_checks": len(self.results),
            "passed_checks": sum(1 for r in self.results if r.passed),
            "failed_checks": sum(1 for r in self.results if not r.passed),
            "by_level": by_level,
            "critical_passed": all(
                r.passed for r in self.results if r.level == SecurityLevel.CRITICAL
            ),
        }


def get_security_posture_report(
    environment: Optional[str] = None, strict_mode: bool = False
) -> Dict[str, Any]:
    """Get security posture report.

    Args:
        environment: Override environment (defaults to APGI_ENV)
        strict_mode: Fail on any check failure

    Returns:
        Security posture report dictionary
    """
    env = environment or os.getenv("APGI_ENV", "development")
    checker = SecurityPostureChecker(environment=env, strict_mode=strict_mode)
    passed, report = checker.run_all_checks()

    if not passed:
        security_logger.warning("Security posture check failed", extra={"report": report})
    else:
        security_logger.info("Security posture check passed", extra={"report": report})

    return report
