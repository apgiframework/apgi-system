"""
APGI REST API Main Application

FastAPI application providing RESTful access to the APGI System.
"""

import os
import socket
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Optional

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from apgi_framework.security.startup_security_check import get_security_posture_report
from api.config import settings
from api.database.connection import close_db, init_db
from api.exception_handlers import register_exception_handlers
from api.middleware.alerting import configure_alerting
from api.middleware.authentication import AuthenticationMiddleware
from api.middleware.body_cache import RequestBodyCachingMiddleware
from api.middleware.compliance import ComplianceMiddleware
from api.middleware.csrf import CSRFMiddleware
from api.middleware.deprecation import DeprecationMiddleware
from api.middleware.https_redirect import HTTPSRedirectMiddleware
from api.middleware.logging import (
    RequestLoggingMiddleware,
    StructuredLogger,
    configure_structured_logging,
)
from api.middleware.metrics import PrometheusMetricsMiddleware
from api.middleware.rate_limiting import RateLimitingMiddleware
from api.middleware.request_deduplication import RequestDeduplicationMiddleware
from api.middleware.request_size_limit import RequestSizeLimitMiddleware
from api.middleware.schema_validation import ResponseSchemaValidationMiddleware
from api.middleware.serialization import OptimizedSerializationMiddleware
from api.routes import (
    admin,
    auth,
    compliance_routes,
    export,
    health,
    metrics,
    sessions,
    state,
    tasks,
    users,
    version,
    webhooks,
)

from .middleware.security_headers import SecurityHeadersMiddleware

# Dependency checks moved to lifespan() to avoid import-time side effects


# Configure structured logging
configure_structured_logging(settings.log_level)
logger = StructuredLogger(__name__)


def run_dependency_checks() -> bool:
    """Run dependency checks safely without side effects.

    Returns True if checks pass or should be skipped.
    Raises RuntimeError in production if checks fail.
    """
    # Skip if explicitly disabled (for testing/embedding)
    if os.getenv("SKIP_DEPENDENCY_CHECKS", "").lower() in ("1", "true", "yes"):
        logger.info("Dependency checks skipped via SKIP_DEPENDENCY_CHECKS")
        return True

    try:
        from utils.dependency_checker import check_dependencies, check_security_dependencies

        # Run core dependency check
        summary = check_dependencies(core_only=True)
        if summary.get("overall_status") != "ready":
            logger.error(f"Core dependency check failed: {summary}")
            if os.getenv("ENVIRONMENT") == "production":
                raise RuntimeError("Critical core dependencies missing. Refusing to start.")
            return False

        # Run security dependency check strictly in production
        if os.getenv("ENVIRONMENT") == "production":
            security_results = check_security_dependencies()
            missing_security = [
                name for name, (available, _, _, _) in security_results.items() if not available
            ]
            if missing_security:
                logger.error(f"Security dependencies missing in production: {missing_security}")
                raise RuntimeError(
                    f"Security dependencies missing: {missing_security}. Refusing to start."
                )
    except ImportError:
        logger.warning("Dependency checker not available. Continuing anyway...")
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Error during dependency check: {e}. Continuing anyway...")

    return True


# Global Redis client
redis_client: Optional[redis.Redis] = None

# Global rate limiting middleware reference
rate_limiting_middleware: Optional[RateLimitingMiddleware] = None


def is_port_available(host: str, port: int) -> bool:
    """Check if a port is available on the given host."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    global redis_client

    # Startup
    logger.info("Application starting up", component="lifecycle")

    # Run dependency checks (moved from import-time to startup)
    if not run_dependency_checks():
        raise RuntimeError("Dependency checks failed. Cannot start application.")

    # Configure alerting system with PagerDuty and Slack
    configure_alerting(
        webhook_urls=settings.alert_webhook_urls,
        enable_log_channel=settings.alert_enable_log_channel,
        error_rate_threshold=settings.alert_error_rate_threshold,
        error_rate_window_minutes=settings.alert_error_rate_window_minutes,
        alert_cooldown_minutes=settings.alert_cooldown_minutes,
    )

    # Configure Degraded Mode Alerts to PagerDuty/Slack
    # We can fetch alert manager if we made it singleton, else just log that we would connect it.

    logger.info("Alerting system configured with PagerDuty/Slack integration", component="alerting")

    # Run Startup Security Checks
    logger.info("Running security posture checks", component="security")
    security_report = get_security_posture_report(strict_mode=True)
    if not security_report.get("critical_passed", True):
        logger.error(
            "Critical security checks failed!", component="security", details=security_report
        )
        if os.getenv("ENVIRONMENT") == "production":
            raise RuntimeError("Refusing to start in production with critical security failures.")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized", component="database")
    except Exception as e:
        logger.error("Failed to initialize database", component="database", error=str(e))
        raise

    # Initialize Redis client
    try:
        redis_client = redis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)  # type: ignore[no-untyped-call]
        if redis_client:
            await redis_client.ping()  # type: ignore[misc]
            logger.info("Redis client initialized", component="redis", url=settings.redis_url)

            # Update rate limiting middleware with Redis client
            # Note: FastAPI's add_middleware doesn't expose the instance directly,
            # so we set the client via a module-level reference maintained by the middleware
            from api.middleware.rate_limiting import set_global_redis_client

            set_global_redis_client(redis_client)
            logger.info(
                "Rate limiting middleware updated with Redis client", component="middleware"
            )

    except Exception as e:
        logger.error("Failed to initialize Redis", component="redis", error=str(e))
        raise

    # Initialize session routes with Redis client
    sessions.init_session_routes(redis_client)  # type: ignore[attr-defined]
    logger.info("Session routes initialized", component="routes")

    # Initialize task routes
    tasks.init_task_routes()
    logger.info("Task routes initialized", component="routes")

    # Initialize export routes with session manager
    session_mgr = sessions.get_session_manager()
    export.init_export_routes(session_mgr)
    logger.info("Export routes initialized", component="routes")

    # Initialize health routes with Redis client
    health.init_health_routes(redis_client)
    logger.info("Health routes initialized", component="routes")

    yield

    # Shutdown
    logger.info("Application shutting down", component="lifecycle")

    # Close Redis connection
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed", component="redis")

    # Close database connections
    close_db()
    logger.info("Database connections closed", component="database")


def create_app(test_mode: bool = False) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        test_mode: If True, disables authentication and CSRF middleware for testing

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    app = FastAPI(
        title="APGI System API",
        version=settings.api_version,
        description="REST API for Allostatic Precision-Gated Ignition consciousness modeling",
        docs_url="/docs" if settings.environment.lower() != "production" else None,
        redoc_url="/redoc" if settings.environment.lower() != "production" else None,
        openapi_url="/openapi.json" if settings.environment.lower() != "production" else None,
        lifespan=lifespan if not test_mode else None,
    )

    # Add request body caching middleware (very early, to capture body before consumption)
    app.add_middleware(RequestBodyCachingMiddleware)

    # Add request size limiting middleware (first, to catch large requests early)
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_size_mb=getattr(settings, "max_request_size_mb", 10),
        enabled=getattr(settings, "request_size_limit_enabled", True),
    )

    # Add optimized serialization middleware (early for format negotiation)
    if settings.optimized_serialization_enabled:
        app.add_middleware(OptimizedSerializationMiddleware)

    # Add request deduplication middleware (early to avoid duplicate processing)
    if settings.request_dedup_enabled:
        # RequestDeduplicationMiddleware automatically populates deduplication_manager in its __init__
        app.add_middleware(
            RequestDeduplicationMiddleware,
            max_size=settings.request_dedup_max_size,
            default_ttl_seconds=settings.request_dedup_ttl_seconds,
        )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # Add HTTPS redirect middleware - skip in test mode
    if not test_mode:
        app.add_middleware(HTTPSRedirectMiddleware, https_enabled=settings.https_enabled)

    # Add GZip compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add metrics middleware (early, to track all requests)
    app.add_middleware(PrometheusMetricsMiddleware)

    # Initialize SLO tracking for key endpoints
    from apgi_framework.performance.slo_tracker import SLOBudget, get_slo_tracker

    slo_tracker = get_slo_tracker()
    # Register SLO budgets for critical endpoints
    slo_tracker.register_slo(
        SLOBudget(
            endpoint="/v1/sessions",
            p50_ms=100,
            p95_ms=500,
            p99_ms=1000,
            error_rate_threshold=0.01,
        )
    )
    slo_tracker.register_slo(
        SLOBudget(
            endpoint="/v1/state",
            p50_ms=50,
            p95_ms=200,
            p99_ms=500,
            error_rate_threshold=0.005,
        )
    )
    slo_tracker.register_slo(
        SLOBudget(
            endpoint="/v1/tasks",
            p50_ms=200,
            p95_ms=1000,
            p99_ms=2000,
            error_rate_threshold=0.02,
        )
    )

    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Add response schema validation middleware
    app.add_middleware(
        ResponseSchemaValidationMiddleware,
        enabled=settings.schema_validation_enabled,
        fail_on_error=settings.schema_validation_fail_on_error,
    )

    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware, https_enabled=settings.https_enabled)

    # Add CSRF protection middleware - skip in test mode
    if not test_mode:
        app.add_middleware(
            CSRFMiddleware,
            enabled=(
                settings.csrf_protection_enabled
                if hasattr(settings, "csrf_protection_enabled")
                else True
            ),
            cookie_name="csrf_token",
            header_name="X-CSRF-Token",
            token_expiry_minutes=60,
            secure=settings.https_enabled,  # Set secure flag based on HTTPS configuration
        )

    # Add Compliance Middleware - skip in test mode (requires auth)
    if not test_mode:
        app.add_middleware(ComplianceMiddleware)

    # Add deprecation middleware
    app.add_middleware(DeprecationMiddleware, deprecated_endpoints={})

    # Add rate limiting middleware (Redis client set in lifespan) - skip in test mode
    if not test_mode:
        app.add_middleware(
            RateLimitingMiddleware,
            redis_client=None,  # Will be set in lifespan
            enabled=settings.rate_limit_enabled,
        )

    # Ensure CORS is configured early to handle preflight before other middlewares

    # Add authentication middleware - skip in test mode
    if not test_mode:
        app.add_middleware(AuthenticationMiddleware)

    # Register exception handlers
    register_exception_handlers(app)

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> Dict[str, str]:
        """API root endpoint with basic information."""
        return {
            "name": "APGI System API",
            "version": settings.api_version,
            "description": "REST API for consciousness modeling",
            "docs": "/docs",
            "health": "/v1/health",  # Updated to point to the comprehensive health endpoint
        }

    # Include routers
    app.include_router(auth.router)
    app.include_router(sessions.router)
    app.include_router(state.router)
    app.include_router(tasks.router)
    app.include_router(users.router)
    app.include_router(export.router)
    app.include_router(metrics.router)
    app.include_router(health.router)
    app.include_router(version.router)
    app.include_router(webhooks.router)
    app.include_router(admin.router)
    app.include_router(compliance_routes.router)

    # Configure deprecated endpoints
    version.configure_deprecated_endpoints({})

    # Add OpenTelemetry Instrumentation
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        logger.warning("opentelemetry API not installed, skipping instrumentation")

    logger.info("APGI API application created successfully", version=settings.api_version)
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    default_port = 8000
    default_host = "0.0.0.0"

    # Check if default port is available
    if not is_port_available(default_host, default_port):
        print(f"⚠️  Port {default_port} is already in use. Trying alternative ports...")
        # Try to find an available port
        for port in range(default_port + 1, default_port + 100):
            if is_port_available(default_host, port):
                print(f"✓ Using port {port}")
                default_port = port
                break
        else:
            print(
                f"❌ Could not find an available port between {default_port} and {default_port + 99}"
            )
            sys.exit(1)

    # Configure uvicorn arguments based on HTTPS settings
    uvicorn_kwargs = {
        "host": default_host,
        "port": default_port,
        "reload": settings.reload,
        "log_level": "info",
    }

    if settings.https_enabled:
        if not settings.ssl_certfile or not settings.ssl_keyfile:
            print(
                "❌ HTTPS enabled but SSL_CERTFILE and SSL_KEYFILE environment variables are not set"
            )
            print("   Please set SSL_CERTFILE and SSL_KEYFILE to enable HTTPS")
            print("   Falling back to HTTP...")
        else:
            uvicorn_kwargs["ssl_keyfile"] = settings.ssl_keyfile
            uvicorn_kwargs["ssl_certfile"] = settings.ssl_certfile
            print(
                f"✓ HTTPS enabled with cert: {settings.ssl_certfile}, key: {settings.ssl_keyfile}"
            )

    uvicorn.run("api.main:app", **uvicorn_kwargs)  # type: ignore[arg-type]
