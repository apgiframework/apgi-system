"""
Database Connection Management

SQLAlchemy engine and session configuration with async support.
"""

import logging
import secrets
import string
from contextlib import contextmanager
from typing import Any, AsyncGenerator, Dict, Generator

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from api.config import settings
from api.database.models import Base, User

# Import circuit breaker utilities
from utils.circuit_breaker_utils import CircuitBreakerException, circuit_breaker

logger = logging.getLogger(__name__)


# Bare postgresql:// URLs resolve to whichever default driver the installed
# SQLAlchemy picks (psycopg2 in 2.0, psycopg 3 in newer releases). Pin the
# driver listed in requirements.txt so engine creation is version-independent.
sync_db_url = settings.database_url.replace("postgresql://", "postgresql+psycopg2://", 1)

# Create synchronous SQLAlchemy engine
engine = create_engine(
    sync_db_url,
    echo=settings.db_echo_sql,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
)


# Create async SQLAlchemy engine with asyncpg for PostgreSQL
# Convert postgresql:// to postgresql+asyncpg:// for async driver
async_db_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
async_engine = create_async_engine(
    async_db_url,
    echo=settings.db_echo_sql,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
)


# Create synchronous session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


def generate_secure_password(length: int = 32) -> str:
    """
    Generate a secure random password.

    Args:
        length: Length of the password to generate

    Returns:
        Secure random password string
    """
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?"
    password = "".join(secrets.choice(alphabet) for _ in range(length))
    return password


def generate_secure_username(prefix: str = "user") -> str:
    """
    Generate a secure random username.

    Args:
        prefix: Prefix for the username

    Returns:
        Secure random username string
    """
    random_suffix = secrets.token_hex(8)
    return f"{prefix}_{random_suffix}"


def init_db() -> None:
    """
    Initialize database by creating all tables and default user.

    This should be called during application startup.
    """
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")

        # Create default user if it doesn't exist
        create_default_user()

    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


@circuit_breaker(name="database_init", failure_threshold=2, recovery_timeout=30.0)
def create_default_user() -> None:
    """
    Create a default user for session management with secure random credentials.

    This ensures that the foreign key constraint for sessions
    can be satisfied when creating sessions without explicit user management.
    """
    db = SessionLocal()
    try:
        # Check if any default user already exists (users with username starting with "default_")
        from sqlalchemy import select

        stmt = select(User).where(User.username.like("default_%"))
        result = db.execute(stmt)
        existing_user = result.scalar_one_or_none()

        if existing_user:
            logger.info(f"Default user already exists: {existing_user.username}")
            return

        # Generate secure credentials
        secure_username = generate_secure_username("default")
        secure_password = generate_secure_password()

        # Print credentials to stdout only (never logger) to avoid leak in log files
        import sys

        sys.stdout.write(
            f"\n! SECURITY ALERT: Default admin credentials generated !\n"
            f"ROTATE IMMEDIATELY AFTER FIRST LOGIN\n"
            f"Username: {secure_username}\n"
            f"Password: {secure_password}\n\n"
        )

        # Import here to avoid circular import
        from api.services.auth_manager import AuthManager

        # Create default user with admin privileges for full system access
        default_user = User(
            user_id=secure_username,
            username=secure_username,
            email=f"{secure_username}@apgi-system.local",
            password_hash=AuthManager.hash_password(secure_password),
            roles=["admin"],
        )

        db.add(default_user)
        db.commit()
        logger.info(f"Default user created with secure credentials: {secure_username}")

    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error creating default user: {e}")
        raise CircuitBreakerException(f"Database operation failed: {e}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create default user: {e}")
        raise
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.

    Yields:
        Session: SQLAlchemy database session

    Usage:
        @app.get("/endpoint")
        async def endpoint(db: Session = Depends(get_db)):
            # Use db session
            pass
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:  # type: ignore[misc]
    """
    Dependency function to get async database session.

    Yields:
        AsyncSession: SQLAlchemy async database session

    Usage:
        @app.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_async_db)):
            # Use async db session
            pass
    """
    async with AsyncSessionLocal() as session:
        yield session


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database session.

    Yields:
        Session: SQLAlchemy database session

    Usage:
        with get_db_context() as db:
            # Use db session
            pass
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database operation failed in context: {e}")
        raise CircuitBreakerException(f"Database operation failed: {e}")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@circuit_breaker(name="database_health_check", failure_threshold=3, recovery_timeout=15.0)
def check_database_health() -> bool:
    """
    Check database connectivity and health using ORM-safe queries.

    Returns:
        bool: True if database is healthy, False otherwise
    """
    try:
        db = SessionLocal()
        try:
            # Simple health check using ORM-safe literal
            from sqlalchemy import literal_column, select

            db.execute(select(literal_column("1")))
            logger.debug("Database health check passed")
            return True
        finally:
            db.close()
    except SQLAlchemyError as e:
        logger.error(f"Database health check failed: {e}")
        raise CircuitBreakerException(f"Database health check failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during database health check: {e}")
        raise


@circuit_breaker(name="database_connection_test", failure_threshold=2, recovery_timeout=10.0)
def test_database_connection() -> Dict[str, Any]:
    """
    Test database connection and return connection details.

    Uses SQLAlchemy's inspect() and connection pooling for safer database testing.

    Returns:
        Dict containing connection test results
    """
    try:
        import time

        from sqlalchemy import inspect, literal_column, select

        start_time = time.time()

        db = SessionLocal()
        try:
            # Test basic connectivity using SQLAlchemy's inspect
            inspector = inspect(db.bind)
            version = inspector.dialect.server_version_info

            # Test write operation using ORM-safe temporary table
            db.execute(select(literal_column("1")))

            connection_time = time.time() - start_time

            pool_size = "unknown"
            if hasattr(engine.pool, "size"):
                pool_size = engine.pool.size()

            return {
                "status": "healthy",
                "version": str(version) if version else "unknown",
                "connection_time": connection_time,
                "pool_size": pool_size,
                "checked_at": time.time(),
            }

        finally:
            db.close()

    except SQLAlchemyError as e:
        logger.error(f"Database connection test failed: {e}")
        raise CircuitBreakerException(f"Database connection test failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during database connection test: {e}")
        raise


def close_db() -> None:
    """
    Close database connections.

    This should be called during application shutdown.
    """
    try:
        engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")
