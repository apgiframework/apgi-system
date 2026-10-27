"""PPG drivers: an LSL-backed device and a simulated pulse-oximeter waveform."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .lsl import LslDevice
from .simulated import SimulatedDevice


class PPGDevice(LslDevice):
    """Photoplethysmograph published on LSL (stream type ``PPG``)."""

    def __init__(
        self,
        *,
        device_id: str = "ppg-0",
        sampling_rate_hz: float = 100.0,
        channel_names: Sequence[str] = ("PPG1",),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device_id=device_id,
            stream_type="PPG",
            kind="PPG",
            unit="a.u.",
            sampling_rate_hz=sampling_rate_hz,
            channel_names=channel_names,
            driver_name="lsl-ppg",
            model=kwargs.pop("model", "LSL-PPG"),
            **kwargs,
        )


class SimulatedPPGDevice(SimulatedDevice):
    """Pulse waveform (fundamental plus second harmonic) at ``heart_rate_bpm``."""

    def __init__(
        self,
        *,
        heart_rate_bpm: float = 72.0,
        channel_names: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        self._hz = heart_rate_bpm / 60.0
        kwargs.setdefault("device_id", "sim-ppg-0")
        kwargs.setdefault("sampling_rate_hz", 100.0)
        super().__init__(
            channel_names=channel_names or ("PPG1",),
            channel_kind="PPG",
            channel_unit="a.u.",
            model="APGI-SIM-PPG",
            driver_name="simulated-ppg",
            **kwargs,
        )

    def _waveform(self, t: NDArray[np.float64], ch: int) -> NDArray[np.float64]:
        w = 2 * np.pi * self._hz * t
        return 0.5 * np.sin(w) + 0.15 * np.sin(2 * w + 0.6)

    def _noise_sd(self) -> float:
        return 0.01
