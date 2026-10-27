"""Keychain-wrapped keys for the encrypted session store (GDPR: encryption at rest).

The SQLCipher key is a random 32-byte *data key*. It is stored on disk only wrapped (AES-256-GCM)
by a *master key* that lives in the operating-system keychain, so a copy of the database
folder alone cannot be opened. Keychain access goes through the optional ``keyring`` package or
any object with ``get_password``/``set_password`` (a test double, or a vault client).
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from .errors import StoreKeyError, StoreUnavailableError
from .store import SessionStore

_SERVICE = "apgi-acquisition"


def _aesgcm() -> Any:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - cryptography ships with the framework
        raise StoreUnavailableError(
            "the 'cryptography' package is required for key wrapping"
        ) from exc
    return AESGCM


def _default_keyring() -> Any:
    try:
        import keyring
    except ImportError as exc:
        raise StoreUnavailableError(
            "no OS keychain available: install 'keyring' (pip install keyring) or pass a keychain object"
        ) from exc
    return keyring


class KeychainKeyProvider:
    """Create, wrap and unwrap the store's data key."""

    def __init__(
        self,
        wrapped_key_path: Path,
        *,
        account: str = "store-master-key",
        service: str = _SERVICE,
        keychain: Optional[Any] = None,
    ) -> None:
        self._path = Path(wrapped_key_path)
        self._account = account
        self._service = service
        self._keychain = keychain

    @property
    def keychain(self) -> Any:
        if self._keychain is None:
            self._keychain = _default_keyring()
        return self._keychain

    def _master_key(self, create: bool) -> Optional[bytes]:
        stored = self.keychain.get_password(self._service, self._account)
        if stored:
            return base64.b64decode(stored)
        if not create:
            return None
        key = os.urandom(32)
        self.keychain.set_password(
            self._service, self._account, base64.b64encode(key).decode("ascii")
        )
        return key

    def get_or_create(self) -> bytes:
        """Return the data key, generating and wrapping a new one on first use."""
        aes = _aesgcm()
        if self._path.exists():
            master = self._master_key(create=False)
            if master is None:
                raise StoreKeyError(
                    "the wrapped key exists but its master key is not in the keychain"
                )
            blob = self._path.read_bytes()
            try:
                return bytes(aes(master).decrypt(blob[:12], blob[12:], self._account.encode()))
            except Exception as exc:
                raise StoreKeyError(
                    "wrapped key could not be unwrapped (wrong master key or tampering)"
                ) from exc
        master = self._master_key(create=True)
        assert master is not None
        data_key = os.urandom(32)
        nonce = os.urandom(12)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_bytes(nonce + aes(master).encrypt(nonce, data_key, self._account.encode()))
        os.chmod(tmp, 0o600)
        os.replace(tmp, self._path)
        return data_key


def open_store_with_keychain(
    db_path: Path, *, keychain: Optional[Any] = None, **store_kwargs: Any
) -> Tuple[SessionStore, KeychainKeyProvider]:
    """Open (or create) an encrypted store whose key is wrapped by the OS keychain."""
    db_path = Path(db_path)
    provider = KeychainKeyProvider(db_path.with_name(db_path.name + ".key"), keychain=keychain)
    return SessionStore(db_path, key=provider.get_or_create(), **store_kwargs), provider
