"""
API Configuration

Configuration settings for the APGI REST API.
"""

import os
import warnings
from typing import List, Optional

from dotenv import load_dotenv

from api import __version__

# Load environment variables from .env file
load_dotenv()

# Configuration constants
# JWT Token Settings
DEFAULT_JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
DEFAULT_JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7
MIN_JWT_KEY_LENGTH_DEV = 32
MIN_JWT_KEY_LENGTH_PROD = 64
MIN_JWT_CHARACTER_CLASSES = 3

# Rate Limiting Defaults
DEFAULT_RATE_LIMIT_PER_MINUTE = 60

# Alerting Thresholds
DEFAULT_ALERT_ERROR_RATE_THRESHOLD = 10
DEFAULT_ALERT_ERROR_RATE_WINDOW_MINUTES = 1
DEFAULT_ALERT_COOLDOWN_MINUTES = 5

# Database Pool Defaults
DEFAULT_DB_POOL_SIZE = 10
DEFAULT_DB_MAX_OVERFLOW = 20

# Data Orchestrator Defaults
DEFAULT_BATCH_SIZE = 100
DEFAULT_CACHE_TTL_SECONDS = 3600

# Performance Middleware Defaults
DEFAULT_REQUEST_DEDUP_MAX_SIZE = 1000
DEFAULT_REQUEST_DEDUP_TTL_SECONDS = 60
DEFAULT_REQUEST_DEDUP_ENABLED = False


class Settings:
    """
    API configuration settings.

    Settings can be overridden via environment variables or AWS SSM.
    """

    def _fetch_ssm_parameter(self, name: str) -> Optional[str]:
        """Fetch parameter from AWS SSM."""
        if os.getenv("USE_AWS_SSM", "false").lower() != "true":
            return None
        try:
            import boto3  # type: ignore[import-not-found]

            client = boto3.client("ssm", region_name=os.getenv("AWS_REGION", "us-east-1"))
            response = client.get_parameter(Name=name, WithDecryption=True)
            return response["Parameter"]["Value"]  # type: ignore[no-any-return]
        except Exception as e:
            warnings.warn(f"Failed to fetch {name} from AWS SSM: {str(e)}")
            return None

    def __init__(self) -> None:
        # API Settings
        self.api_title: str = "APGI System API"
        self.api_version: str = __version__
        self.api_description: str = "REST API for consciousness modeling"

        # Server Settings
        self.host: str = "0.0.0.0"
        self.port: int = 8000
        self.reload: bool = os.getenv("UVICORN_RELOAD", "true").lower() == "true"

        # HTTPS/TLS Settings
        self.https_enabled: bool = os.getenv("HTTPS_ENABLED", "true").lower() == "true"
        self.ssl_keyfile: Optional[str] = os.getenv("SSL_KEYFILE")
        self.ssl_certfile: Optional[str] = os.getenv("SSL_CERTFILE")

        # Database Settings
        self.database_url: str = (
            self._fetch_ssm_parameter("/apgi/database_url")
            or os.getenv("DATABASE_URL")
            or "postgresql://localhost/apgi_api?sslmode=require"
        )
        self.db_pool_size: int = int(os.getenv("DB_POOL_SIZE", str(DEFAULT_DB_POOL_SIZE)))
        self.db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", str(DEFAULT_DB_MAX_OVERFLOW)))
        self.db_echo_sql: bool = os.getenv("DB_ECHO_SQL", "false").lower() == "true"

        # Redis Settings
        self.redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # Celery Settings
        self.celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
        self.celery_result_backend: str = os.getenv(
            "CELERY_RESULT_BACKEND", "redis://localhost:6379/2"
        )

        # Authentication Settings
        self.environment: str = os.getenv("ENVIRONMENT", "production")

        # We now support AWS SSM for secrets to allow easy RS256 key management
        self.jwt_secret_key: Optional[str] = self._fetch_ssm_parameter(
            "/apgi/jwt_secret_key"
        ) or os.getenv("JWT_SECRET_KEY")
        self.jwt_private_key: Optional[str] = self._fetch_ssm_parameter(
            "/apgi/jwt_private_key"
        ) or os.getenv("JWT_PRIVATE_KEY")
        self.jwt_public_key: Optional[str] = self._fetch_ssm_parameter(
            "/apgi/jwt_public_key"
        ) or os.getenv("JWT_PUBLIC_KEY")

        self.jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "RS256")
        self.jwt_access_token_expire_minutes: int = DEFAULT_JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self.jwt_refresh_token_expire_days: int = DEFAULT_JWT_REFRESH_TOKEN_EXPIRE_DAYS

        # Rate Limiting Settings
        self.rate_limit_enabled: bool = True
        self.rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE

        # CORS Settings
        cors_origins_env = os.getenv("CORS_ORIGINS")
        if cors_origins_env:
            self.cors_origins: List[str] = cors_origins_env.split(",")
        else:
            self.cors_origins = [
                "http://localhost:3000",
                "http://localhost:8000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000",
            ]
        self.cors_allow_credentials: bool = (
            os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
        )
        self.cors_allow_methods: List[str] = (
            os.getenv("CORS_ALLOW_METHODS", "GET,POST,PUT,DELETE,OPTIONS").split(",")
            if os.getenv("CORS_ALLOW_METHODS")
            else ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        )
        self.cors_allow_headers: List[str] = (
            os.getenv("CORS_ALLOW_HEADERS", "Content-Type,Authorization,X-CSRF-Token").split(",")
            if os.getenv("CORS_ALLOW_HEADERS")
            else ["Content-Type", "Authorization", "X-CSRF-Token"]
        )

        # Logging Settings
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

        # Schema Validation Settings
        self.schema_validation_enabled: bool = (
            os.getenv("SCHEMA_VALIDATION_ENABLED", "true").lower() == "true"
        )
        self.schema_validation_fail_on_error: bool = (
            os.getenv("SCHEMA_VALIDATION_FAIL_ON_ERROR", "false").lower() == "true"
        )

        # Performance Middleware Settings
        self.request_dedup_enabled: bool = (
            os.getenv("REQUEST_DEDUP_ENABLED", "false").lower() == "true"
        )
        self.request_dedup_max_size: int = int(
            os.getenv("REQUEST_DEDUP_MAX_SIZE", str(DEFAULT_REQUEST_DEDUP_MAX_SIZE))
        )
        self.request_dedup_ttl_seconds: int = int(
            os.getenv("REQUEST_DEDUP_TTL_SECONDS", str(DEFAULT_REQUEST_DEDUP_TTL_SECONDS))
        )
        self.optimized_serialization_enabled: bool = (
            os.getenv("OPTIMIZED_SERIALIZATION_ENABLED", "true").lower() == "true"
        )

        # Alerting Settings
        self.alert_webhook_urls: List[str] = []  # Can be set via environment
        self.alert_enable_log_channel: bool = True
        self.alert_error_rate_threshold: int = int(
            os.getenv("ALERT_ERROR_RATE_THRESHOLD", str(DEFAULT_ALERT_ERROR_RATE_THRESHOLD))
        )
        self.alert_error_rate_window_minutes: int = int(
            os.getenv(
                "ALERT_ERROR_RATE_WINDOW_MINUTES", str(DEFAULT_ALERT_ERROR_RATE_WINDOW_MINUTES)
            )
        )
        self.alert_cooldown_minutes: int = int(
            os.getenv("ALERT_COOLDOWN_MINUTES", str(DEFAULT_ALERT_COOLDOWN_MINUTES))
        )

        # Validate security settings after initialization
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate critical security settings after initialization."""
        # Handle JWT secret key for development vs production
        if not self.jwt_secret_key:
            if self.environment.lower() == "development":
                # Development mode - provide a secure default with warning
                self.jwt_secret_key = "development-secret-key-change-in-production-32-chars-min"
                warnings.warn(
                    "DEVELOPMENT WARNING: JWT_SECRET_KEY not set, using development default. "
                    "This is insecure and should NOT be used in production. "
                    "Set JWT_SECRET_KEY environment variable for production deployment. "
                    "Copy .env.example to .env and configure your settings.",
                    UserWarning,
                )
            else:
                raise ValueError(
                    "CRITICAL: JWT_SECRET_KEY environment variable is not set. "
                    "This is required for secure JWT token generation. "
                    "Set a secure JWT_SECRET_KEY environment variable before starting the API. "
                    "For development, set ENVIRONMENT=development to use the development default."
                )

        # Check for known insecure default values
        insecure_defaults = [
            "your-secret-key-change-in-production",
            "your-secret-key-change-in-production-min-32-chars",
            "secret",
            "default-secret",
            "change-me",
            "insecure-key",
        ]

        if self.jwt_secret_key.lower() in [d.lower() for d in insecure_defaults]:
            if self.environment.lower() in ["production", "prod"]:
                raise ValueError(
                    "CRITICAL: JWT_SECRET_KEY is set to a known insecure default value in production. "
                    "This allows attackers to forge JWT tokens and bypass authentication. "
                    "Set a secure, random JWT_SECRET_KEY environment variable."
                )
            else:
                warnings.warn(
                    "DEVELOPMENT WARNING: JWT_SECRET_KEY is set to a known insecure default value. "
                    "This should be changed before any production deployment. "
                    "Set a secure JWT_SECRET_KEY environment variable.",
                    UserWarning,
                )

        # Validate minimum key length and entropy
        min_key_length = (
            MIN_JWT_KEY_LENGTH_PROD
            if self.environment.lower() in ["production", "prod"]
            else MIN_JWT_KEY_LENGTH_DEV
        )

        # Diversity check: must use multiple character classes
        has_upper = any(c.isupper() for c in self.jwt_secret_key)
        has_lower = any(c.islower() for c in self.jwt_secret_key)
        has_digit = any(c.isdigit() for c in self.jwt_secret_key)
        has_special = any(not c.isalnum() for c in self.jwt_secret_key)
        classes_used = sum([has_upper, has_lower, has_digit, has_special])

        if len(self.jwt_secret_key) < min_key_length or classes_used < MIN_JWT_CHARACTER_CLASSES:
            if self.environment.lower() in ["production", "prod"]:
                raise ValueError(
                    f"CRITICAL: JWT_SECRET_KEY is too weak for production. "
                    f"It must be at least {min_key_length} characters long and use "
                    f"at least {MIN_JWT_CHARACTER_CLASSES} character classes (uppercase, lowercase, digits, symbols)."
                )
            else:
                warnings.warn(
                    f"DEVELOPMENT WARNING: JWT_SECRET_KEY is weak. "
                    f"It should be at least {min_key_length} characters and use "
                    f"{MIN_JWT_CHARACTER_CLASSES}+ character classes.",
                    UserWarning,
                )

        # Validate CORS origins are explicitly configured
        if self.cors_origins == ["*"]:
            if self.cors_allow_credentials:
                raise ValueError(
                    "CRITICAL SECURITY: CORS origins are set to wildcard [*] with credentials enabled. "
                    "This allows any origin to access the API with credentials, enabling CSRF attacks. "
                    "Either set CORS_ORIGINS to specific allowed origins, or set CORS_ALLOW_CREDENTIALS=false."
                )
            else:
                warnings.warn(
                    "SECURITY WARNING: CORS origins are set to wildcard [*]. "
                    "This allows any origin to access the API. "
                    "Set CORS_ORIGINS environment variable to specific allowed origins for production."
                )

        # Check for localhost origins in production
        localhost_origins = {
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        }
        if self.environment.lower() in ["production", "prod"]:
            configured_localhost = set(self.cors_origins) & localhost_origins
            if configured_localhost:
                warnings.warn(
                    f"PRODUCTION WARNING: CORS_ORIGINS contains localhost origins: {', '.join(configured_localhost)}. "
                    "This may prevent legitimate production clients from accessing the API. "
                    "Remove localhost origins from CORS_ORIGINS in production deployments."
                )


# Global settings instance
settings = Settings()
