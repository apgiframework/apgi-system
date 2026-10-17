"""
Version Information Routes

Provides API versioning information including current version,
supported versions, and deprecation notices.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api import __version__

router = APIRouter(prefix="/v1", tags=["Version"])


# Version configuration - use environment variables with defaults
CURRENT_VERSION = os.getenv("API_VERSION", __version__)
API_VERSION = os.getenv("API_VERSION_PREFIX", "v1")
SUPPORTED_VERSIONS = [API_VERSION]
DEPRECATED_VERSIONS: List[str] = []
DEPRECATED_ENDPOINTS: Dict[str, Dict[str, str]] = {}


@router.get("/version")
async def get_version_info() -> JSONResponse:
    """
    Get API version information.

    Returns current version, supported versions, deprecated versions,
    and links to API documentation.

    **Validates: Requirements 6.1, 6.4**
    """
    return JSONResponse(
        status_code=200,
        content={
            "current_version": CURRENT_VERSION,
            "api_version": API_VERSION,
            "supported_versions": SUPPORTED_VERSIONS,
            "deprecated_versions": DEPRECATED_VERSIONS,
            "api_spec_url": "/openapi.json",
            "documentation_url": "/docs",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )


def configure_deprecated_endpoints(deprecated_config: Dict[str, Dict[str, str]]) -> None:
    """
    Configure deprecated endpoints.

    Args:
        deprecated_config: Dictionary mapping endpoint paths to deprecation info
                          Format: {"/v1/endpoint": {"sunset": "2026-01-01", "replacement": "/v2/endpoint"}}
    """
    global DEPRECATED_ENDPOINTS
    DEPRECATED_ENDPOINTS = deprecated_config


def is_endpoint_deprecated(path: str) -> Optional[Dict[str, str]]:
    """
    Check if an endpoint is deprecated.

    Args:
        path: The endpoint path to check

    Returns:
        Deprecation info if deprecated, None otherwise
    """
    return DEPRECATED_ENDPOINTS.get(path)


def get_deprecated_endpoints() -> Dict[str, Dict[str, str]]:
    """
    Get all deprecated endpoints configuration.

    Returns:
        Dictionary of deprecated endpoints and their info
    """
    return DEPRECATED_ENDPOINTS
