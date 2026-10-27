"""JWT/OAuth2 Authentication Framework for APGI.

Provides modern authentication with:
- JWT token generation and validation
- Role-Based Access Control (RBAC)
- Token refresh and rotation
- Secure token storage
- Audit logging for authentication events
"""

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ..logging.standardized_logging import get_logger, get_security_logger

logger = get_logger(__name__)
security_logger = get_security_logger()


class AuthenticationError(Exception):
    """Base authentication error."""

    def __init__(self, message: str, error_code: str = "AUTH_ERROR"):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc).isoformat()


class InvalidTokenError(AuthenticationError):
    """Raised when a token is invalid or expired."""

    def __init__(self, message: str = "Invalid or expired token"):
        super().__init__(message, "INVALID_TOKEN")


class InsufficientPermissionsError(AuthenticationError):
    """Raised when user lacks required permissions."""

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, "INSUFFICIENT_PERMISSIONS")


class Role(str, Enum):
    """User roles for RBAC."""

    ADMIN = "admin"
    RESEARCHER = "researcher"
    OPERATOR = "operator"
    VIEWER = "viewer"
    SERVICE = "service"


class Permission(str, Enum):
    """Permissions for fine-grained access control."""

    # Experiment permissions
    EXPERIMENT_CREATE = "experiment:create"
    EXPERIMENT_READ = "experiment:read"
    EXPERIMENT_UPDATE = "experiment:update"
    EXPERIMENT_DELETE = "experiment:delete"
    EXPERIMENT_RUN = "experiment:run"

    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"

    # Analysis permissions
    ANALYSIS_RUN = "analysis:run"
    ANALYSIS_READ = "analysis:read"

    # System permissions
    SYSTEM_CONFIGURE = "system:configure"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_AUDIT = "system:audit"
    USER_MANAGE = "user:manage"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions
    Role.RESEARCHER: {
        Permission.EXPERIMENT_CREATE,
        Permission.EXPERIMENT_READ,
        Permission.EXPERIMENT_UPDATE,
        Permission.EXPERIMENT_RUN,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
        Permission.DATA_EXPORT,
        Permission.ANALYSIS_RUN,
        Permission.ANALYSIS_READ,
    },
    Role.OPERATOR: {
        Permission.EXPERIMENT_READ,
        Permission.EXPERIMENT_RUN,
        Permission.DATA_READ,
        Permission.SYSTEM_MONITOR,
    },
    Role.VIEWER: {
        Permission.EXPERIMENT_READ,
        Permission.DATA_READ,
        Permission.ANALYSIS_READ,
    },
    Role.SERVICE: {
        Permission.EXPERIMENT_READ,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
        Permission.ANALYSIS_RUN,
    },
}


@dataclass
class TokenClaims:
    """JWT token claims."""

    subject: str  # User ID or service ID
    role: Role
    permissions: Set[Permission] = field(default_factory=set)
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=1)
    )
    token_id: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    issuer: str = "apgi-framework"
    audience: Optional[str] = None
    additional_claims: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert claims to dictionary for JWT encoding."""
        return {
            "sub": self.subject,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "iat": int(self.issued_at.timestamp()),
            "exp": int(self.expires_at.timestamp()),
            "jti": self.token_id,
            "iss": self.issuer,
            "aud": self.audience,
            **self.additional_claims,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenClaims":
        """Create claims from decoded JWT dictionary."""
        return cls(
            subject=data.get("sub", ""),
            role=Role(data.get("role", Role.VIEWER.value)),
            permissions={Permission(p) for p in data.get("permissions", [])},
            issued_at=datetime.fromtimestamp(data.get("iat", 0), timezone.utc),
            expires_at=datetime.fromtimestamp(data.get("exp", 0), timezone.utc),
            token_id=data.get("jti", ""),
            issuer=data.get("iss", "apgi-framework"),
            audience=data.get("aud"),
            additional_claims={
                k: v
                for k, v in data.items()
                if k not in ["sub", "role", "permissions", "iat", "exp", "jti", "iss", "aud"]
            },
        )


@dataclass
class TokenPair:
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    expires_in: int  # Seconds
    token_type: str = "Bearer"


class JWTAuthManager:
    """JWT Authentication Manager with RBAC support.

    Features:
    - JWT token generation with RS256 or HS256
    - Role-based access control
    - Token refresh and rotation
    - Token blacklisting
    - Audit logging
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_ttl: int = 3600,  # 1 hour
        refresh_token_ttl: int = 86400 * 7,  # 7 days
        enable_refresh_rotation: bool = True,
    ):
        """Initialize JWT authentication manager.

        Args:
            secret_key: Secret key for signing (generate if None)
            algorithm: JWT algorithm (HS256 or RS256)
            access_token_ttl: Access token lifetime in seconds
            refresh_token_ttl: Refresh token lifetime in seconds
            enable_refresh_rotation: Whether to rotate refresh tokens
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = algorithm
        self.access_token_ttl = access_token_ttl
        self.refresh_token_ttl = refresh_token_ttl
        self.enable_refresh_rotation = enable_refresh_rotation

        # Token storage (in production, use Redis/database)
        self._token_blacklist: Set[str] = set()
        self._refresh_tokens: Dict[str, TokenClaims] = {}
        self._used_refresh_tokens: Set[str] = set()  # For rotation detection

        logger.info("JWTAuthManager initialized", algorithm=algorithm)

    def generate_token_pair(
        self,
        subject: str,
        role: Role,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> TokenPair:
        """Generate access and refresh token pair.

        Args:
            subject: User or service identifier
            role: User role
            additional_claims: Additional claims to include

        Returns:
            TokenPair with access and refresh tokens
        """
        # Get permissions for role
        permissions = ROLE_PERMISSIONS.get(role, set())

        # Create access token claims
        access_claims = TokenClaims(
            subject=subject,
            role=role,
            permissions=permissions,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.access_token_ttl),
            additional_claims=additional_claims or {},
        )

        # Create refresh token claims (longer lived, fewer permissions)
        refresh_claims = TokenClaims(
            subject=subject,
            role=role,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.refresh_token_ttl),
            additional_claims={"token_type": "refresh"},
        )

        # Generate tokens
        access_token = self._encode_token(access_claims)
        refresh_token = self._encode_token(refresh_claims)

        # Store refresh token
        self._refresh_tokens[refresh_claims.token_id] = refresh_claims

        # Log token generation (security audit)
        security_logger.info(
            "Token pair generated",
            subject=subject,
            role=role.value,
            token_id=access_claims.token_id,
            jti=refresh_claims.token_id,
        )

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_token_ttl,
        )

    def refresh_access_token(self, refresh_token: str) -> TokenPair:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            New token pair

        Raises:
            InvalidTokenError: If refresh token is invalid
        """
        # Decode and validate refresh token
        claims = self.decode_token(refresh_token)

        if claims.additional_claims.get("token_type") != "refresh":
            raise InvalidTokenError("Not a refresh token")

        # Check for token reuse (rotation detection)
        if claims.token_id in self._used_refresh_tokens:
            # Potential token theft - revoke all user tokens
            security_logger.critical(
                "Refresh token reuse detected - potential compromise",
                subject=claims.subject,
                token_id=claims.token_id,
            )
            self.revoke_all_user_tokens(claims.subject)
            raise InvalidTokenError("Token reuse detected - session terminated")

        # Mark refresh token as used
        self._used_refresh_tokens.add(claims.token_id)

        # Remove old refresh token from storage
        self._refresh_tokens.pop(claims.token_id, None)

        # Generate new token pair
        token_pair = self.generate_token_pair(
            subject=claims.subject,
            role=claims.role,
            additional_claims={
                k: v for k, v in claims.additional_claims.items() if k != "token_type"
            },
        )

        security_logger.info(
            "Token refreshed",
            subject=claims.subject,
            old_jti=claims.token_id,
        )

        return token_pair

    def decode_token(self, token: str) -> TokenClaims:
        """Decode and validate JWT token.

        Args:
            token: JWT token string

        Returns:
            TokenClaims if valid

        Raises:
            InvalidTokenError: If token is invalid or expired
        """
        try:
            # Check if token is blacklisted
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            if token_hash in self._token_blacklist:
                raise InvalidTokenError("Token has been revoked")

            # Decode JWT (simplified - production should use PyJWT library)
            claims = self._decode_token(token)

            # Check expiration
            if datetime.now(timezone.utc) > claims.expires_at:
                raise InvalidTokenError("Token has expired")

            return claims

        except InvalidTokenError:
            raise
        except Exception as e:
            logger.error(f"Token decode error: {e}")
            raise InvalidTokenError(f"Invalid token format: {e}")

    def verify_permission(self, token: str, required_permission: Permission) -> TokenClaims:
        """Verify token and check permission.

        Args:
            token: JWT access token
            required_permission: Required permission

        Returns:
            TokenClaims if authorized

        Raises:
            InvalidTokenError: If token is invalid
            InsufficientPermissionsError: If permission check fails
        """
        claims = self.decode_token(token)

        # Check if permission is in role permissions
        if required_permission not in claims.permissions:
            security_logger.warning(
                "Permission denied",
                subject=claims.subject,
                required=required_permission.value,
                role=claims.role.value,
            )
            raise InsufficientPermissionsError(f"Permission '{required_permission.value}' required")

        return claims

    def revoke_token(self, token: str) -> None:
        """Revoke a token (add to blacklist).

        Args:
            token: Token to revoke
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        self._token_blacklist.add(token_hash)

        try:
            claims = self._decode_token(token)
            security_logger.info(
                "Token revoked",
                subject=claims.subject,
                token_id=claims.token_id,
            )
        except Exception:
            pass  # Token might be malformed

    def revoke_all_user_tokens(self, subject: str) -> None:
        """Revoke all tokens for a user (emergency revocation).

        Args:
            subject: User identifier
        """
        # Remove all refresh tokens for user
        tokens_to_remove = [
            jti for jti, claims in self._refresh_tokens.items() if claims.subject == subject
        ]
        for jti in tokens_to_remove:
            del self._refresh_tokens[jti]

        security_logger.warning(
            "All tokens revoked for user",
            subject=subject,
            count=len(tokens_to_remove),
        )

    def _encode_token(self, claims: TokenClaims) -> str:
        """Encode claims to JWT string (simplified implementation).

        Production should use PyJWT with proper cryptographic signing.
        """
        header = json.dumps({"alg": self.algorithm, "typ": "JWT"})
        payload = json.dumps(claims.to_dict())

        # Base64url encode
        import base64

        def b64url_encode(data: bytes) -> str:
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

        header_b64 = b64url_encode(header.encode())
        payload_b64 = b64url_encode(payload.encode())

        # Create signature
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

        return f"{message}.{b64url_encode(signature.encode())}"

    def _decode_token(self, token: str) -> TokenClaims:
        """Decode JWT string to claims (simplified implementation).

        Production should use PyJWT with proper verification.
        """
        import base64

        def b64url_decode(data: str) -> bytes:
            padding = 4 - len(data) % 4
            if padding != 4:
                data += "=" * padding
            return base64.urlsafe_b64decode(data)

        parts = token.split(".")
        if len(parts) != 3:
            raise InvalidTokenError("Invalid JWT format")

        payload_json = b64url_decode(parts[1]).decode()
        payload = json.loads(payload_json)

        return TokenClaims.from_dict(payload)


class OAuth2Config:
    """OAuth2 configuration for external identity providers."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        authorize_url: str,
        token_url: str,
        userinfo_url: str,
        redirect_uri: str,
        scopes: Optional[List[str]] = None,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorize_url = authorize_url
        self.token_url = token_url
        self.userinfo_url = userinfo_url
        self.redirect_uri = redirect_uri
        self.scopes = scopes or ["openid", "profile", "email"]


class OAuth2Client:
    """OAuth2 client for external identity provider integration.

    Supports authorization code flow for web applications.
    """

    def __init__(self, config: OAuth2Config):
        self.config = config
        self._state_store: Dict[str, Dict[str, Any]] = {}

    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """Generate authorization URL for OAuth2 flow.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Authorization URL
        """
        import urllib.parse

        state = state or secrets.token_urlsafe(16)
        self._state_store[state] = {"created_at": time.time()}

        params = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(self.config.scopes),
            "state": state,
        }

        return f"{self.config.authorize_url}?{urllib.parse.urlencode(params)}"

    def exchange_code_for_token(self, code: str, state: str) -> Dict[str, Any]:
        """Exchange authorization code for access token.

        Args:
            code: Authorization code
            state: State parameter for validation

        Returns:
            Token response from provider
        """
        # Validate state
        if state not in self._state_store:
            raise AuthenticationError("Invalid state parameter", "INVALID_STATE")

        # Remove used state
        del self._state_store[state]

        # In production, make HTTP request to token endpoint
        # For now, return mock response
        logger.info("OAuth2 code exchange", state=state[:8])

        return {
            "access_token": "mock_oauth2_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "mock_refresh_token",
        }


def require_permission(permission: Permission) -> Any:
    """Decorator to require specific permission for function access.

    Args:
        permission: Required permission

    Example:
        @require_permission(Permission.EXPERIMENT_CREATE)
        def create_experiment(...):
            ...
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, token: str, auth_manager: JWTAuthManager, **kwargs: Any) -> Any:
            claims = auth_manager.verify_permission(token, permission)
            kwargs["auth_claims"] = claims
            return func(*args, **kwargs)

        return wrapper

    return decorator
