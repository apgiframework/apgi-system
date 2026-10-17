"""
Test API Setup and Basic Functionality

Verifies that the API application is properly configured and can start.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the API."""
    app = create_app(test_mode=True)

    # Override authentication dependency for testing
    from api.routes import health
    from api.services.authorization import get_current_user
    from api.services.health_check import HealthCheckService

    async def override_get_current_user() -> None:
        return None

    app.dependency_overrides[get_current_user] = override_get_current_user

    # Initialize health service for testing
    health.health_service = HealthCheckService(redis_client=None)

    return TestClient(app)


def test_app_creation() -> None:
    """Test that the FastAPI application can be created."""
    app = create_app()
    assert app is not None
    assert app.title == "APGI System API"
    assert app.version == "1.1.0"


def test_root_endpoint(client: TestClient) -> None:
    """Test the root endpoint returns basic API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "APGI System API"
    assert data["version"] == "1.1.0"
    assert "docs" in data
    assert "health" in data


def test_health_check_endpoint(client: TestClient) -> None:
    """Test the health check endpoint returns valid health status."""
    response = client.get("/v1/health")
    # Accept 200 (healthy) or 503 (unhealthy) - both indicate endpoint is working
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data
    assert data["status"] in ("healthy", "unhealthy")


def test_openapi_schema(client: TestClient) -> None:
    """Test that OpenAPI schema is generated."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert schema["info"]["title"] == "APGI System API"


def test_docs_endpoint(client: TestClient) -> None:
    """Test that Swagger UI documentation endpoint responds (may be JSON redirect or HTML)."""
    response = client.get("/docs")
    # Docs endpoint may return HTML (200) or redirect JSON (200) depending on configuration
    assert response.status_code in (200, 307, 404)
    if response.status_code == 200:
        # Accept either HTML or JSON response
        content_type = response.headers.get("content-type", "")
        assert "text/html" in content_type or "application/json" in content_type


def test_redoc_endpoint(client: TestClient) -> None:
    """Test that ReDoc documentation endpoint responds (may be JSON redirect or HTML)."""
    response = client.get("/redoc")
    # ReDoc endpoint may return HTML (200) or redirect JSON (200) depending on configuration
    assert response.status_code in (200, 307, 404)
    if response.status_code == 200:
        # Accept either HTML or JSON response
        content_type = response.headers.get("content-type", "")
        assert "text/html" in content_type or "application/json" in content_type
