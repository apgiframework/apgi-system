"""
Authentication Manager Service

Handles JWT token creation/verification and password hashing for user authentication.
"""

import hashlib
import hmac
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, cast

import bcrypt
import jwt
import redis as redis_sync
import redis.asyncio as redis
from sqlalchemy import and_
from sqlalchemy.orm import Session

from api.config import settings
from api.database.models import RefreshToken, User
from api.exceptions import AuthenticationError, ExpiredTokenError, InvalidTokenError

logger = logging.getLogger(__name__)


class TokenPayload:
    """JWT token payload data."""

    def __init__(
        self,
        user_id: str,
        username: str,
        roles: List[str],
        exp: datetime,
        token_type: str = "access",
        jti: Optional[str] = None,
    ):
        self.user_id = user_id
        self.username = username
        self.roles = roles
        self.exp = exp
        self.token_type = token_type
        self.jti = jti or str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JWT encoding."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "roles": self.roles,
            "exp": int(self.exp.timestamp()),
            "token_type": self.token_type,
            "jti": self.jti,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenPayload":
        """Create from dictionary after JWT decoding."""
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            roles=data["roles"],
            exp=datetime.fromtimestamp(data["exp"]),
            token_type=data.get("token_type", "access"),
            jti=data.get("jti"),
        )


class AuthManager:
    """
    Manages authentication and authorization.

    Responsibilities:
    - JWT token creation and verification
    - Password hashing and verification
    - User authentication
    - Token refresh
    """

    def __init__(self, db: Session):
        """
        Initialize AuthManager.

        Args:
            db: Database session for user lookups
        """
        self.db = db
        self.secret_key = settings.jwt_secret_key
        self.private_key = settings.jwt_private_key
        self.public_key = settings.jwt_public_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.jwt_access_token_expire_minutes
        self.refresh_token_expire_days = settings.jwt_refresh_token_expire_days

        # Async Redis client for token blacklisting (awaited call sites)
        try:
            self.redis_client = redis.Redis.from_url(settings.redis_url)
        except Exception as e:
            logger.warning(f"Failed to connect to Redis for token blacklisting: {e}")
            self.redis_client = None

        # Synchronous Redis client for failed-login tracking, which runs in the
        # synchronous authenticate_user path and cannot await the async client
        try:
            self.sync_redis_client: Optional[redis_sync.Redis] = redis_sync.Redis.from_url(
                settings.redis_url
            )
        except Exception as e:
            logger.warning(f"Failed to create Redis client for login tracking: {e}")
            self.sync_redis_client = None

        # In-memory fallback for failed login tracking when Redis unavailable
        self.failed_login_attempts: Dict[str, Tuple[int, float]] = (
            {}
        )  # user_id -> (count, timestamp)

    # ========================================================================
    # Password Hashing
    # ========================================================================

    def _track_failed_login_attempt(self, user_id: str) -> None:
        """
        Track failed login attempts with Redis or in-memory fallback.

        Args:
            user_id: User identifier
        """
        current_time = time.time()

        if self.sync_redis_client:
            # Redis tracking; fall back to memory when Redis is unreachable
            try:
                failed_key = f"failed_login:{user_id}"
                failed_count = int(self.sync_redis_client.incr(failed_key))  # type: ignore[arg-type]
                self.sync_redis_client.expire(failed_key, 3600)  # Expire in 1 hour
            except redis_sync.RedisError as e:
                logger.warning(f"Redis unavailable for login tracking, using memory: {e}")
            else:
                if failed_count >= 5:
                    # Lock the account - need to get user and update
                    user = self.db.query(User).filter(User.user_id == user_id).first()
                    if user:
                        user.is_active = False  # type: ignore[assignment]
                        self.db.commit()
                        logger.warning(
                            f"Account locked for user {user_id} due to too many failed login attempts"
                        )
                return

        # In-memory fallback
        if user_id not in self.failed_login_attempts:
            self.failed_login_attempts[user_id] = (0, current_time)

        count, _ = self.failed_login_attempts[user_id]
        count += 1
        self.failed_login_attempts[user_id] = (count, current_time)

        # Clean up old entries (older than 1 hour)
        expired = [
            uid for uid, (_, ts) in self.failed_login_attempts.items() if current_time - ts > 3600
        ]
        for uid in expired:
            del self.failed_login_attempts[uid]

        if count >= 5:
            # Lock the account
            user = self.db.query(User).filter(User.user_id == user_id).first()
            if user:
                user.is_active = False  # type: ignore[assignment]
                self.db.commit()
                logger.warning(
                    f"Account locked for user {user_id} due to too many failed login attempts (in-memory fallback)"
                )

    @staticmethod
    def hash_password(password: str) -> str:
        # Truncate password to 72 bytes (bcrypt limit)
        password_bytes = password.encode("utf-8")
        if len(password_bytes) > 72:
            password_bytes = password_bytes[:72]
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode("utf-8")

    @staticmethod
    def create_lookup_hash(token: str) -> str:
        """
        Create a SHA-256 hash for deterministic token lookup.

        Args:
            token: Token string to hash

        Returns:
            SHA-256 hash string
        """
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            plain_password: Plain text password to verify
            hashed_password: Hashed password to compare against

        Returns:
            True if password matches, False otherwise
        """
        # Truncate password to 72 bytes (bcrypt limit) - must match hash_password behavior
        password_bytes = plain_password.encode("utf-8")
        if len(password_bytes) > 72:
            password_bytes = password_bytes[:72]
        hashed_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(password_bytes, hashed_bytes)

    # ========================================================================
    # Token Creation
    # ========================================================================

    def create_access_token(self, user_id: str, username: str, roles: List[str]) -> str:
        """
        Create a JWT access token.

        Args:
            user_id: User identifier
            username: Username
            roles: List of user roles

        Returns:
            Encoded JWT token string
        """
        expires_at = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        payload = TokenPayload(
            user_id=user_id, username=username, roles=roles, exp=expires_at, token_type="access"
        )

        sign_key = self.private_key if self.algorithm.startswith("RS") else self.secret_key
        token = jwt.encode(payload.to_dict(), sign_key, algorithm=self.algorithm)  # type: ignore[arg-type]

        return token

    def create_refresh_token(self, user_id: str, username: str, roles: List[str]) -> str:
        """
        Create a JWT refresh token.

        Args:
            user_id: User identifier
            username: Username
            roles: List of user roles

        Returns:
            Encoded JWT refresh token string
        """
        expires_at = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)

        payload = TokenPayload(
            user_id=user_id, username=username, roles=roles, exp=expires_at, token_type="refresh"
        )

        sign_key = self.private_key if self.algorithm.startswith("RS") else self.secret_key
        token = jwt.encode(payload.to_dict(), sign_key, algorithm=self.algorithm)  # type: ignore[arg-type]

        return token

    # ========================================================================
    # Token Verification
    # ========================================================================

    def verify_token(self, token: str, expected_type: str = "access") -> TokenPayload:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string
            expected_type: Expected token type ("access" or "refresh")

        Returns:
            TokenPayload with decoded token data

        Raises:
            InvalidTokenError: If token is invalid or malformed
            ExpiredTokenError: If token has expired
        """
        try:
            # Decode token
            verify_key = self.public_key if self.algorithm.startswith("RS") else self.secret_key
            assert verify_key is not None, f"JWT verification key ({self.algorithm}) not configured"
            payload_dict = jwt.decode(token, verify_key, algorithms=[self.algorithm])

            # Parse payload
            payload = TokenPayload.from_dict(payload_dict)

            # Verify token type
            if payload.token_type != expected_type:
                raise InvalidTokenError(
                    f"Invalid token type: expected {expected_type}, got {payload.token_type}"
                )

            # Check expiration (jwt.decode already checks this, but we handle it explicitly)
            if datetime.utcnow() > payload.exp:
                raise ExpiredTokenError("Token has expired")

            return payload

        except jwt.ExpiredSignatureError:
            raise ExpiredTokenError("Token has expired")
        except ExpiredTokenError:
            # Re-raise our own ExpiredTokenError
            raise
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Invalid token: {str(e)}")
        except Exception as e:
            raise InvalidTokenError(f"Token verification failed: {str(e)}")

    # ========================================================================
    # User Authentication
    # ========================================================================

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user with username and password.

        Args:
            username: Username
            password: Plain text password

        Returns:
            User object if authentication successful, None otherwise
        """
        # Look up user
        user = self.db.query(User).filter(User.username == username).first()

        # Always perform password verification to prevent timing attacks
        # Use dummy hash for non-existent users that depends on the password to match timing
        if not user:
            # Generate a new salt and hash the password to ensure it takes real time
            import bcrypt

            dummy_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(12)).decode()
            password_hash = dummy_hash
        else:
            password_hash = cast(str, user.password_hash)

        password_valid = self.verify_password(password, str(password_hash))

        if not user or not password_valid:
            # Failed login - track attempts for existing users only
            if user:
                self._track_failed_login_attempt(user.user_id)  # type: ignore[arg-type]
            return None

        # Successful login - reset failed attempts
        if self.sync_redis_client:
            try:
                self.sync_redis_client.delete(f"failed_login:{user.user_id}")
            except redis_sync.RedisError as e:
                logger.warning(f"Redis unavailable while clearing login attempts: {e}")
        self.failed_login_attempts.pop(str(user.user_id), None)

        # Update last login
        try:
            user.last_login = datetime.utcnow()  # type: ignore[assignment]
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update last login for user {username}: {e}")
            # Still return user since login succeeded, just log the error

        return user

    def create_tokens_for_user(self, user: User) -> Dict[str, Any]:
        """
        Create access and refresh tokens for a user.

        Args:
            user: User object

        Returns:
            Dictionary with access_token, refresh_token, token_type, and expires_in
        """
        access_token = self.create_access_token(
            user_id=user.user_id, username=user.username, roles=user.roles  # type: ignore[arg-type]
        )

        refresh_token = self.create_refresh_token(
            user_id=user.user_id, username=user.username, roles=user.roles  # type: ignore[arg-type]
        )

        # Store refresh token in database
        lookup_hash = self.create_lookup_hash(refresh_token)
        token_hash = self.hash_password(refresh_token)
        expires_at = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)

        try:
            db_refresh_token = RefreshToken(
                user_id=user.user_id,
                lookup_hash=lookup_hash,
                token_hash=token_hash,
                expires_at=expires_at,
            )
            self.db.add(db_refresh_token)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create refresh token for user {user.user_id}: {e}")
            raise

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60,
        }

    # ========================================================================
    # Token Refresh
    # ========================================================================

    @staticmethod
    def constant_time_compare(val1: str, val2: str) -> bool:
        """
        Constant-time comparison to prevent timing attacks.

        Args:
            val1: First value to compare
            val2: Second value to compare

        Returns:
            True if values are equal, False otherwise
        """
        return hmac.compare_digest(val1.encode(), val2.encode())

    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Create a new access token and refresh token using an existing refresh token.
        Implements refresh token rotation by revoking the old token and issuing a new one.

        Args:
            refresh_token: Refresh token string

        Returns:
            Dictionary with new access_token, refresh_token, token_type, and expires_in

        Raises:
            InvalidTokenError: If refresh token is invalid
            ExpiredTokenError: If refresh token has expired
            AuthenticationError: If refresh token has been revoked
        """
        # Verify refresh token
        payload = self.verify_token(refresh_token, expected_type="refresh")

        # Create lookup hash to find the token in database
        lookup_hash = self.create_lookup_hash(refresh_token)

        # Look up the specific token by lookup_hash and user_id (not just first non-revoked)
        db_token = (
            self.db.query(RefreshToken)
            .filter(
                and_(
                    RefreshToken.user_id == payload.user_id,
                    RefreshToken.lookup_hash == lookup_hash,
                    RefreshToken.revoked.is_(False),
                )
            )
            .first()
        )

        if not db_token:
            raise AuthenticationError("Invalid or revoked refresh token")

        # Verify the token using bcrypt.checkpw
        if not self.verify_password(refresh_token, db_token.token_hash):  # type: ignore[arg-type]
            raise AuthenticationError("Invalid refresh token")

        # Check expiration in database
        if datetime.utcnow() > db_token.expires_at:
            raise ExpiredTokenError("Refresh token has expired")

        # Revoke the old refresh token (token rotation)
        db_token.revoked = True  # type: ignore[assignment]

        # Create new access token
        access_token = self.create_access_token(
            user_id=payload.user_id, username=payload.username, roles=payload.roles
        )

        # Create new refresh token
        new_refresh_token = self.create_refresh_token(
            user_id=payload.user_id, username=payload.username, roles=payload.roles
        )

        # Store new refresh token in database
        new_token_hash = self.hash_password(new_refresh_token)
        new_lookup_hash = self.create_lookup_hash(new_refresh_token)
        expires_at = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)

        try:
            new_db_token = RefreshToken(
                user_id=payload.user_id,
                lookup_hash=new_lookup_hash,
                token_hash=new_token_hash,
                expires_at=expires_at,
            )
            self.db.add(new_db_token)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create new refresh token for user {payload.user_id}: {e}")
            raise

        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60,
        }

    # ========================================================================
    # Token Revocation
    # ========================================================================

    def revoke_refresh_token(self, refresh_token: str) -> bool:
        """
        Revoke a refresh token (logout).

        Args:
            refresh_token: Refresh token to revoke

        Returns:
            True if token was revoked, False if not found
        """
        try:
            # Verify token to get user_id
            payload = self.verify_token(refresh_token, expected_type="refresh")

            # Create lookup hash to find the token in database
            lookup_hash = self.create_lookup_hash(refresh_token)

            # Find and revoke the specific token by lookup_hash and user_id
            db_token = (
                self.db.query(RefreshToken)
                .filter(
                    and_(
                        RefreshToken.user_id == payload.user_id,
                        RefreshToken.lookup_hash == lookup_hash,
                        RefreshToken.revoked.is_(False),
                    )
                )
                .first()
            )

            try:
                if db_token:
                    # Verify the token using bcrypt.checkpw before revoking
                    if self.verify_password(refresh_token, db_token.token_hash):  # type: ignore[arg-type]
                        db_token.revoked = True  # type: ignore[assignment]
                        self.db.commit()
                        return True
                return False
            except Exception as e:
                self.db.rollback()
                logger.error(f"Failed to revoke refresh token {lookup_hash}: {e}")
                return False

        except (InvalidTokenError, ExpiredTokenError):
            # Token is already invalid, consider it revoked
            return False

    def revoke_all_user_tokens(self, user_id: str) -> int:
        """
        Revoke all refresh tokens for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of tokens revoked
        """
        try:
            tokens = (
                self.db.query(RefreshToken)
                .filter(and_(RefreshToken.user_id == user_id, RefreshToken.revoked.is_(False)))
                .all()
            )
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to retrieve tokens for user {user_id}: {e}")
            raise

        count = 0
        try:
            for token in tokens:
                token.revoked = True  # type: ignore[assignment]
                count += 1

            self.db.commit()
            return count
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to revoke tokens for user {user_id}: {e}")
            raise

    # ========================================================================
    # Token Blacklisting
    # ========================================================================

    async def blacklist_access_token(self, token: str, expires_at: datetime) -> bool:
        """
        Add an access token to the blacklist.

        Args:
            token: The JWT access token to blacklist
            expires_at: When the token expires (to set TTL)

        Returns:
            True if successfully blacklisted, False otherwise
        """
        if not self.redis_client:
            logger.warning("Redis not available, cannot blacklist token")
            return False

        try:
            # Decode token to get JTI without validation (validation happens in verify_token)
            unverified_payload = jwt.decode(token, options={"verify_signature": False})
            jti = unverified_payload.get("jti")

            if not jti:
                logger.warning("Token has no JTI, falling back to hashing token string")
                jti = hashlib.sha256(token.encode("utf-8")).hexdigest()

            # Calculate TTL in seconds from now until expiration
            ttl_seconds = int((expires_at - datetime.utcnow()).total_seconds())
            if ttl_seconds <= 0:
                # Token already expired, no need to blacklist
                return True

            key = f"blacklisted_jti:{jti}"
            await self.redis_client.setex(key, ttl_seconds, "blacklisted")
            return True
        except Exception as e:
            logger.error(f"Failed to blacklist access token: {e}")
            return False

    async def is_token_blacklisted(self, token: str) -> bool:
        """
        Check if an access token is blacklisted.

        Args:
            token: The JWT access token to check

        Returns:
            True if blacklisted, False otherwise
        """
        if not self.redis_client:
            # If Redis not available, assume not blacklisted
            return False

        try:
            unverified_payload = jwt.decode(token, options={"verify_signature": False})
            jti = unverified_payload.get("jti")

            if not jti:
                jti = hashlib.sha256(token.encode("utf-8")).hexdigest()

            key = f"blacklisted_jti:{jti}"
            result = await self.redis_client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to check token blacklist: {e}")
            # On error, allow the token (fail-safe)
            return False
