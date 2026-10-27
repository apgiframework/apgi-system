"""
OWASP Security Protections Middleware

Enhanced security middleware implementing OWASP Top 10 protections:
- XXE (XML External Entity) Prevention
- SSRF (Server-Side Request Forgery) Protection
- Security Headers
- Input Sanitization
"""

import ipaddress
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# SSRF Protection: Blocked URL patterns
SSRF_BLOCKED_HOSTS: Set[str] = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",  # nosec B104 - blocklist entry, not a bind address
    "::1",
    "[::1]",
}

# Blocked IP ranges (private networks)
SSRF_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),  # IPv6 private
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
]

# XML patterns that indicate potential XXE
XXE_PATTERNS = [
    re.compile(r"<!ENTITY\s+.*SYSTEM\s+['\"]", re.IGNORECASE),
    re.compile(r"<!ENTITY\s+.*PUBLIC\s+['\"]", re.IGNORECASE),
    re.compile(r"<!DOCTYPE\s+.*\[", re.IGNORECASE),
    re.compile(r"\&\w+;"),  # Entity references
]


class SSRFProtector:
    """Protection against Server-Side Request Forgery attacks."""

    @staticmethod
    def is_safe_url(url: str) -> tuple[bool, Optional[str]]:
        """
        Check if a URL is safe from SSRF perspective.

        Args:
            url: URL to check

        Returns:
            Tuple of (is_safe, reason_if_unsafe)
        """
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname

            if not hostname:
                return False, "Invalid URL: no hostname"

            # Check blocked hosts
            if hostname.lower() in SSRF_BLOCKED_HOSTS:
                return False, f"Blocked host: {hostname}"

            # Check if hostname is an IP address
            try:
                ip = ipaddress.ip_address(hostname)
                for network in SSRF_BLOCKED_NETWORKS:
                    if ip in network:
                        return False, f"Blocked IP range: {ip} in {network}"
            except ValueError:
                # Not an IP address, continue with hostname checks
                pass

            # Block common internal DNS patterns
            internal_patterns = [
                r"^localhost\.?",
                r"^.*\.local$",
                r"^.*\.internal$",
                r"^.*\.private$",
            ]
            for pattern in internal_patterns:
                if re.match(pattern, hostname, re.IGNORECASE):
                    return False, f"Internal DNS pattern blocked: {hostname}"

            return True, None

        except Exception as e:
            return False, f"URL parsing error: {e}"

    @staticmethod
    def validate_outbound_request(
        method: str, url: str, headers: Optional[Dict[str, str]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate an outbound HTTP request for SSRF safety.

        Args:
            method: HTTP method
            url: Target URL
            headers: Request headers

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Validate URL
        is_safe, reason = SSRFProtector.is_safe_url(url)
        if not is_safe:
            return False, reason

        # Check for header-based SSRF (e.g., X-Forwarded-For manipulation)
        if headers:
            suspicious_headers = ["x-forwarded-host", "x-http-host-override"]
            for header in suspicious_headers:
                if header in (h.lower() for h in headers.keys()):
                    return False, f"Suspicious header detected: {header}"

        return True, None


class XXEProtector:
    """Protection against XML External Entity attacks."""

    @staticmethod
    def contains_xxe(content: str) -> tuple[bool, Optional[str]]:
        """
        Check if XML content contains XXE patterns.

        Args:
            content: XML content to check

        Returns:
            Tuple of (contains_xxe, pattern_found)
        """
        for pattern in XXE_PATTERNS:
            match = pattern.search(content)
            if match:
                return True, match.group(0)
        return False, None

    @staticmethod
    def safe_xml_parse(content: str) -> Any:
        """
        Parse XML safely with XXE protection.

        Args:
            content: XML content

        Returns:
            Parsed XML object

        Raises:
            ValueError: If XXE patterns detected
        """
        has_xxe, pattern = XXEProtector.contains_xxe(content)
        if has_xxe:
            raise ValueError(f"XXE attack detected: {pattern}")

        # Untrusted XML is only ever parsed with defusedxml (no entity expansion, no external
        # entities); there is deliberately no fallback to the standard-library parser.
        try:
            from defusedxml import ElementTree as ET
        except ImportError as exc:
            raise RuntimeError("defusedxml is required for secure XML parsing") from exc
        return ET.fromstring(content)


class OWASPProtectionMiddleware(BaseHTTPMiddleware):
    """
    Middleware implementing OWASP security protections.

    Provides:
    - XXE prevention for XML content
    - SSRF protection for outbound requests
    - Enhanced security headers
    - Input validation hooks
    """

    def __init__(
        self,
        app: ASGIApp,
        block_ssrf: bool = True,
        block_xxe: bool = True,
        custom_blocked_hosts: Optional[List[str]] = None,
    ):
        """
        Initialize OWASP protection middleware.

        Args:
            app: ASGI application
            block_ssrf: Enable SSRF protection
            block_xxe: Enable XXE protection
            custom_blocked_hosts: Additional blocked hosts for SSRF
        """
        super().__init__(app)
        self.block_ssrf = block_ssrf
        self.block_xxe = block_xxe
        self.ssrf_protector = SSRFProtector()
        self.xxe_protector = XXEProtector()

        # Add custom blocked hosts
        if custom_blocked_hosts:
            SSRF_BLOCKED_HOSTS.update(h.lower() for h in custom_blocked_hosts)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Process request with OWASP protections.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response
        """
        # Check for XXE in XML content
        if self.block_xxe:
            content_type = request.headers.get("Content-Type", "")
            if "xml" in content_type.lower():
                try:
                    body = await request.body()
                    body_text = body.decode("utf-8", errors="ignore")
                    has_xxe, pattern = self.xxe_protector.contains_xxe(body_text)
                    if has_xxe:
                        logger.warning(f"XXE attack blocked: {pattern}")
                        return Response(
                            content='{"error": "XML External Entity attacks are not allowed"}',
                            status_code=400,
                            media_type="application/json",
                        )
                except Exception:
                    pass  # Continue if body can't be read

        # Add SSRF protection info to request state
        request.state.ssrf_protector = self.ssrf_protector
        request.state.xxe_protector = self.xxe_protector

        # Process request
        response = await call_next(request)

        # Add OWASP security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        return response


class SecureHTTPClient:
    """
    HTTP client with built-in SSRF protection.

    Wraps HTTP requests with SSRF validation and safe defaults.
    """

    def __init__(self) -> None:
        self.ssrf_protector = SSRFProtector()
        self._allowed_schemes = {"http", "https"}

    async def request(self, method: str, url: str, **kwargs: Any) -> Any:
        """
        Make an HTTP request with SSRF protection.

        Args:
            method: HTTP method
            url: Target URL
            **kwargs: Additional arguments for httpx/aiohttp

        Returns:
            Response object

        Raises:
            ValueError: If URL fails SSRF checks
        """
        # Validate URL
        is_safe, reason = self.ssrf_protector.is_safe_url(url)
        if not is_safe:
            raise ValueError(f"SSRF protection: {reason}")

        # Check scheme
        parsed = urlparse(url)
        if parsed.scheme not in self._allowed_schemes:
            raise ValueError(f"Disallowed URL scheme: {parsed.scheme}")

        # Make request using httpx
        import httpx

        async with httpx.AsyncClient() as client:
            return await client.request(method, url, **kwargs)

    async def get(self, url: str, **kwargs) -> Any:
        """Make a GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> Any:
        """Make a POST request."""
        return await self.request("POST", url, **kwargs)


# Global secure HTTP client instance
_secure_http_client: Optional[SecureHTTPClient] = None


def get_secure_http_client() -> SecureHTTPClient:
    """Get or create global secure HTTP client."""
    global _secure_http_client
    if _secure_http_client is None:
        _secure_http_client = SecureHTTPClient()
    return _secure_http_client
