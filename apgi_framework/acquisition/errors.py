"""Exceptions raised by the acquisition subsystem."""

from __future__ import annotations


class AcquisitionError(Exception):
    """Base class for every acquisition error."""


class DeviceError(AcquisitionError):
    """A device failed or was used in the wrong state."""


class DeviceNotFoundError(AcquisitionError):
    """No driver or device is registered under the requested name."""


class ClockSyncError(AcquisitionError):
    """A clock mapping was requested before enough sync points existed."""


class InvalidTransition(AcquisitionError):
    """A session state-machine event is not allowed in the current state."""


class StoreError(AcquisitionError):
    """The encrypted session store failed."""


class StoreUnavailableError(StoreError):
    """SQLCipher is not installed, so no encrypted store can be opened."""


class StoreKeyError(StoreError):
    """The database could not be decrypted with the supplied key."""


class RecorderError(AcquisitionError):
    """The local recorder failed or found corrupt data."""


class TransportError(AcquisitionError):
    """An upload transport failed; the item may be retried."""


class NetworkUnavailable(TransportError):
    """The network is down; retrying later is expected to work and costs no attempt."""


class PermanentTransportError(TransportError):
    """An upload failed in a way that retrying cannot fix."""


class ExportError(AcquisitionError):
    """A BIDS export could not be produced."""
