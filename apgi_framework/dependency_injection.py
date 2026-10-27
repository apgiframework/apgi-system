"""Dependency Injection Container for APGI Framework.

This module provides a centralized service container to manage dependencies
and eliminate circular import patterns across the codebase.

Usage:
    from apgi_framework.dependency_injection import ServiceContainer

    # Register services
    ServiceContainer.register('config_manager', ConfigManager())

    # Retrieve services
    config = ServiceContainer.get('config_manager')
"""

import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar

T = TypeVar("T")


class ServiceContainer:
    """Centralized service container for dependency injection.

    Thread-safe singleton pattern for managing application services.
    Eliminates circular dependencies by providing a single point of access
    to all services.
    """

    _instance: Optional["ServiceContainer"] = None
    _lock = threading.Lock()
    _services: Dict[str, Any] = {}
    _factories: Dict[str, Any] = {}
    _singletons: Dict[str, Any] = {}

    def __new__(cls) -> "ServiceContainer":
        """Ensure singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str, service: Any, singleton: bool = True) -> None:
        """Register a service in the container.

        Args:
            name: Service identifier
            service: Service instance or factory function
            singleton: If True, service is cached after first creation

        Raises:
            ValueError: If service name already registered
        """
        container = cls()

        if name in container._services or name in container._factories:
            raise ValueError(f"Service '{name}' already registered")

        if callable(service) and not isinstance(service, type):
            # It's a factory function
            container._factories[name] = service
        else:
            # It's a service instance or class
            container._services[name] = service

    @classmethod
    def get(cls, name: str) -> Any:
        """Retrieve a service from the container.

        Args:
            name: Service identifier

        Returns:
            Service instance

        Raises:
            KeyError: If service not found
        """
        container = cls()

        # Check if already instantiated singleton
        if name in container._singletons:
            return container._singletons[name]

        # Check if registered as instance
        if name in container._services:
            return container._services[name]

        # Check if registered as factory
        if name in container._factories:
            factory = container._factories[name]
            instance = factory()
            container._singletons[name] = instance
            return instance

        raise KeyError(f"Service '{name}' not found in container")

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if service is registered."""
        container = cls()
        return (
            name in container._services
            or name in container._factories
            or name in container._singletons
        )

    @classmethod
    def remove(cls, name: str) -> None:
        """Remove a service from the container."""
        container = cls()
        container._services.pop(name, None)
        container._factories.pop(name, None)
        container._singletons.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered services."""
        container = cls()
        container._services.clear()
        container._factories.clear()
        container._singletons.clear()

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get all registered service names."""
        container = cls()
        return {**container._services, **container._factories, **container._singletons}


class ServiceProvider(ABC):
    """Abstract base class for service providers.

    Implement this to create custom service registration logic.
    """

    @abstractmethod
    def register(self, container: ServiceContainer) -> None:
        """Register services with the container."""
        pass


class CoreServiceProvider(ServiceProvider):
    """Registers core framework services."""

    def register(self, container: ServiceContainer) -> None:
        """Register core services without circular dependencies."""
        # Import here to avoid circular imports at module load time
        from apgi_framework.config.config_manager import ConfigManager

        # Register config manager as singleton factory
        def create_config_manager():
            return ConfigManager()

        container.register("config_manager", create_config_manager)


class DatabaseServiceProvider(ServiceProvider):
    """Registers database-related services."""

    def register(self, container: ServiceContainer) -> None:
        """Register database services."""
        from api.database.connection import AsyncSessionLocal, SessionLocal

        container.register("async_session", AsyncSessionLocal, singleton=False)
        container.register("sync_session", SessionLocal, singleton=False)


class AuthServiceProvider(ServiceProvider):
    """Registers authentication services."""

    def register(self, container: ServiceContainer) -> None:
        """Register authentication services."""
        from api.services.auth_manager import AuthManager

        def create_auth_manager():
            # Get database session from container
            session = container.get("async_session")
            return AuthManager(session)

        container.register("auth_manager", create_auth_manager)


def bootstrap_container() -> ServiceContainer:
    """Bootstrap the service container with all providers.

    Returns:
        Configured ServiceContainer instance
    """
    container = ServiceContainer()

    # Register all service providers
    providers = [
        CoreServiceProvider(),
        DatabaseServiceProvider(),
        AuthServiceProvider(),
    ]

    for provider in providers:
        provider.register(container)

    return container
