"""ECG drivers: an LSL-backed device and a simulated ECG source."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .lsl import LslDevice
from .simulated import SimulatedDevice


class ECGDevice(LslDevice):
    """ECG amplifier published on LSL (stream type ``ECG``)."""

    def __init__(
        self,
        *,
        device_id: str = "ecg-0",
        sampling_rate_hz: float = 500.0,
        channel_names: Sequence[str] = ("ECG1",),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device_id=device_id,
            stream_type="ECG",
            kind="ECG",
            unit="mV",
            sampling_rate_hz=sampling_rate_hz,
            channel_names=channel_names,
            driver_name="lsl-ecg",
            model=kwargs.pop("model", "LSL-ECG"),
            **kwargs,
        )


class SimulatedECGDevice(SimulatedDevice):
    """Periodic R-peak train at ``heart_rate_bpm`` plus noise, labelled synthetic."""

    def __init__(
        self,
        *,
        heart_rate_bpm: float = 72.0,
        channel_names: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        self._period_s = 60.0 / heart_rate_bpm
        kwargs.setdefault("device_id", "sim-ecg-0")
        kwargs.setdefault("sampling_rate_hz", 500.0)
        super().__init__(
            channel_names=channel_names or ("ECG1",),
            channel_kind="ECG",
            channel_unit="mV",
            model="APGI-SIM-ECG",
            driver_name="simulated-ecg",
            **kwargs,
        )

    def _waveform(self, t: NDArray[np.float64], ch: int) -> NDArray[np.float64]:
        phase = np.mod(t, self._period_s)
        r = 1.2 * np.exp(-(((phase - 0.2) / 0.012) ** 2))  # QRS
        tw = 0.25 * np.exp(-(((phase - 0.45) / 0.05) ** 2))  # T wave
        return r + tw

    def _noise_sd(self) -> float:
        return 0.02
