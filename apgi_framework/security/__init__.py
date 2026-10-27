"""Security module for APGI Framework
Provides input sanitization, authentication, and security utilities.
"""

from typing import Any, Dict, List, Optional

from .authentication import (
    AuthenticationError,
    InsufficientPermissionsError,
    InvalidTokenError,
    JWTAuthManager,
    OAuth2Client,
    OAuth2Config,
    Permission,
    Role,
    TokenClaims,
    TokenPair,
    require_permission,
)
from .input_sanitization import (
    InputSanitizer,
    SecureFileHandler,
    SecurityError,
    sanitize_filename,
    sanitize_path,
    sanitize_text_input,
    validate_file_extension,
)
from .secure_pickle import (
    SecurePickleError,
    SecurePickleValidator,
    safe_pickle_dump,
    safe_pickle_load,
    validate_pickle_security,
)


# Mock classes for testing
class AccessController:
    """Mock access controller for testing purposes."""

    def __init__(self) -> None:
        self.access_policies: Dict[str, Any] = {}
        self.user_permissions: Dict[str, List[Any]] = {}
        self.access_logs: List[Dict[str, Any]] = []

    def add_access_policy(self, policy_name: str, policy_rules: Any) -> None:
        """Add an access policy."""
        self.access_policies[policy_name] = policy_rules

    def grant_permission(self, user_id: str, permissions: List[Any]) -> None:
        """Grant permissions to a user."""
        if user_id not in self.user_permissions:
            self.user_permissions[user_id] = []
        self.user_permissions[user_id].extend(permissions)

    def check_access(self, user_id: str, resource: str, action: str) -> bool:
        """Check if a user has access to perform an action on a resource."""
        user_permissions = self.user_permissions.get(user_id, [])
        required_permission = f"{resource}:{action}"

        access_granted = required_permission in user_permissions

        # Log the access attempt
        log_entry = {
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "granted": access_granted,
            "timestamp": "2024-01-01T00:00:00Z",
        }
        self.access_logs.append(log_entry)

        return access_granted

    def get_access_logs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get access logs, optionally filtered by user."""
        if user_id:
            return [log for log in self.access_logs if log["user_id"] == user_id]

        return self.access_logs


__all__ = [
    # Authentication
    "JWTAuthManager",
    "OAuth2Client",
    "OAuth2Config",
    "TokenClaims",
    "TokenPair",
    "Role",
    "Permission",
    "AuthenticationError",
    "InvalidTokenError",
    "InsufficientPermissionsError",
    "require_permission",
    # Input sanitization
    "InputSanitizer",
    "SecureFileHandler",
    "SecurityError",
    "sanitize_filename",
    "sanitize_path",
    "validate_file_extension",
    "sanitize_text_input",
    # Secure pickle
    "SecurePickleValidator",
    "SecurePickleError",
    "safe_pickle_load",
    "safe_pickle_dump",
    "validate_pickle_security",
    # Mock classes for testing
    "AccessController",
]
