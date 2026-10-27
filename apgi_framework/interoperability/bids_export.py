"""BIDS-EEG export (implementation lives in ``acquisition.bids``; this is its public home)."""

from ..acquisition.bids import (
    BidsExportResult,
    export_recording_to_bids,
    export_session_to_bids,
    validate_bids_eeg,
)

__all__ = [
    "BidsExportResult",
    "export_recording_to_bids",
    "export_session_to_bids",
    "validate_bids_eeg",
]
