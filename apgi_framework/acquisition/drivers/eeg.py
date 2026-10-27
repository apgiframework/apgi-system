"""EEG drivers: an LSL-backed device and a simulated EEG amplifier."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .lsl import LslDevice
from .simulated import SimulatedDevice

DEFAULT_EEG_CHANNELS = ("Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2")


class EEGDevice(LslDevice):
    """EEG amplifier published on LSL (stream type ``EEG``)."""

    def __init__(
        self,
        *,
        device_id: str = "eeg-0",
        sampling_rate_hz: float = 250.0,
        channel_names: Sequence[str] = DEFAULT_EEG_CHANNELS,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device_id=device_id,
            stream_type="EEG",
            kind="EEG",
            unit="uV",
            sampling_rate_hz=sampling_rate_hz,
            channel_names=channel_names,
            driver_name="lsl-eeg",
            model=kwargs.pop("model", "LSL-EEG"),
            **kwargs,
        )


class SimulatedEEGDevice(SimulatedDevice):
    """Deterministic EEG source (10 Hz rhythm plus noise), labelled synthetic."""

    def __init__(self, *, channel_names: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        kwargs.setdefault("device_id", "sim-eeg-0")
        super().__init__(
            channel_names=channel_names or DEFAULT_EEG_CHANNELS,
            channel_kind="EEG",
            channel_unit="uV",
            model="APGI-SIM-EEG",
            driver_name="simulated-eeg",
            **kwargs,
        )
