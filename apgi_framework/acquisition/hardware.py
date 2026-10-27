"""Hardware classes, Track B priorities, surrogate ranking and the stimulation guard.

Eleven hardware classes are registered. Priority 1 classes are acquired live for protocols
P00/P01; priority 2 and 3 are surrogates or covariates; fMRI, TMS/tFUS and iEEG are *import
workflows*: data produced elsewhere is imported, this software never drives the equipment.

Stimulation equipment (TMS/tFUS) is never controlled from here. No driver may be bound to a
non-controllable class if it exposes an output-control surface, and
:func:`require_controllable` raises for every stimulation class, so a manufacturer or
institutional interlock cannot be bypassed through this code.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from .errors import AcquisitionError

INFERENCE_CADENCE_S = 0.1  # the 100 ms inference cadence surrogates must respect


class StimulationControlError(AcquisitionError):
    """Software control of stimulation equipment was requested or attempted."""


class UnknownHardwareClass(AcquisitionError, KeyError):
    """A hardware class key is not registered."""


class Role(str, enum.Enum):
    REALTIME = "realtime"
    REALTIME_SURROGATE = "realtime_surrogate"
    OFFLINE_COVARIATE = "offline_covariate"
    CONTEXT = "context"  # Layer 3 context
    IMPORT_WORKFLOW = "import_workflow"


@dataclass(frozen=True)
class HardwareClass:
    key: str
    name: str
    priority: int  # 1 highest; import workflows use 0 (not ranked for live acquisition)
    role: Role
    protocols: Tuple[str, ...]
    controllable: bool = (
        False  # may this software send control commands to it? (never for stimulation)
    )
    physiological_lag_s: Optional[float] = None  # signal lag relative to the event it reflects
    imported: bool = False

    @property
    def live(self) -> bool:
        return not self.imported


HARDWARE_CLASSES: Tuple[HardwareClass, ...] = (
    HardwareClass("eeg", "EEG", 1, Role.REALTIME, ("P00", "P01")),
    HardwareClass("ecg", "ECG", 1, Role.REALTIME, ("P00", "P01")),
    HardwareClass(
        "eye_tracking",
        "Eye tracking / pupillometry",
        2,
        Role.REALTIME_SURROGATE,
        (),
        physiological_lag_s=0.3,
    ),
    HardwareClass("behavioural", "Behavioural response devices", 1, Role.REALTIME, ("P00", "P01")),
    HardwareClass("ppg", "PPG", 3, Role.REALTIME_SURROGATE, ()),
    HardwareClass("respiration", "Respiration", 3, Role.REALTIME_SURROGATE, ()),
    HardwareClass("eda", "EDA", 3, Role.OFFLINE_COVARIATE, (), physiological_lag_s=2.0),
    HardwareClass("actigraphy", "Actigraphy and wearables", 3, Role.CONTEXT, ()),
    HardwareClass("fmri", "fMRI", 0, Role.IMPORT_WORKFLOW, ("P03",), imported=True),
    HardwareClass("stimulation", "TMS/tFUS", 0, Role.IMPORT_WORKFLOW, ("P05",), imported=True),
    HardwareClass("ieeg", "iEEG", 0, Role.IMPORT_WORKFLOW, ("P06",), imported=True),
)

# Method names that would let software drive or arm stimulation output or defeat an interlock.
FORBIDDEN_CONTROL_SURFACE: FrozenSet[str] = frozenset(
    {
        "fire",
        "trigger_pulse",
        "deliver_pulse",
        "set_intensity",
        "set_amplitude",
        "set_output",
        "enable_output",
        "arm",
        "start_stimulation",
        "stop_stimulation",
        "set_pulse_parameters",
        "bypass_interlock",
        "override_safety",
        "disable_interlock",
        "unlock",
    }
)


class HardwareRegistry:
    """Hardware classes plus the driver names bound to each."""

    def __init__(self, classes: Tuple[HardwareClass, ...] = HARDWARE_CLASSES) -> None:
        self._classes: Dict[str, HardwareClass] = {c.key: c for c in classes}
        if len(self._classes) != len(classes):
            raise ValueError("duplicate hardware class key")
        self._drivers: Dict[str, List[str]] = {c.key: [] for c in classes}

    def __len__(self) -> int:
        return len(self._classes)

    def __contains__(self, key: object) -> bool:
        return key in self._classes

    def get(self, key: str) -> HardwareClass:
        try:
            return self._classes[key]
        except KeyError:
            raise UnknownHardwareClass(
                f"unknown hardware class {key!r}; known: {sorted(self._classes)}"
            ) from None

    def all(self) -> List[HardwareClass]:
        return list(self._classes.values())

    def by_priority(self) -> List[HardwareClass]:
        """Live classes ordered by priority (1 first), then name; import workflows last."""
        return sorted(self._classes.values(), key=lambda c: (c.priority == 0, c.priority, c.key))

    def bind_driver(self, key: str, driver_cls: type, driver_name: str) -> None:
        """Associate a driver with a class, refusing stimulation control surfaces."""
        cls = self.get(key)
        if cls.imported:
            raise StimulationControlError(
                f"{cls.name} is an import workflow; no live driver may be bound to it"
            )
        surface = sorted(
            n for n in FORBIDDEN_CONTROL_SURFACE if callable(getattr(driver_cls, n, None))
        )
        if surface:
            raise StimulationControlError(
                f"driver {driver_name!r} exposes stimulation-control methods {surface}; "
                "equipment controls must not be driven or bypassed from this software"
            )
        self._drivers[key].append(driver_name)

    def drivers(self, key: str) -> List[str]:
        self.get(key)
        return list(self._drivers[key])


def require_controllable(key: str, registry: Optional[HardwareRegistry] = None) -> None:
    """Raise :class:`StimulationControlError` unless software may command the class (never, today)."""
    cls = (registry or default_hardware_registry).get(key)
    if not cls.controllable:
        raise StimulationControlError(
            f"{cls.name} cannot be controlled from this software; use the manufacturer's or "
            "institution's own controls and import the resulting data"
        )


def surrogate_usable_in_realtime(key: str, registry: Optional[HardwareRegistry] = None) -> bool:
    """Whether a class may feed the live (100 ms cadence) inference.

    Skin-conductance responses lag their cause by 1-3 s, far beyond the 100 ms cadence, so EDA
    is an offline covariate only; pupillometry is the real-time surrogate.
    """
    cls = (registry or default_hardware_registry).get(key)
    if cls.role is Role.OFFLINE_COVARIATE:
        return False
    return cls.role in (Role.REALTIME, Role.REALTIME_SURROGATE)


SURROGATE_RANKING: Tuple[str, ...] = ("eye_tracking", "ppg", "respiration", "eda")
"""Surrogate order for interoceptive state: pupillometry (real time) first, EDA (offline) last."""


def default_driver_classes() -> Dict[str, Tuple[str, str]]:
    """Mapping hardware class -> (registry name, import path) for the shipped drivers."""
    return {
        "eeg": ("lsl-eeg", "EEGDevice"),
        "ecg": ("lsl-ecg", "ECGDevice"),
        "ppg": ("lsl-ppg", "PPGDevice"),
    }


def _build_default() -> HardwareRegistry:
    from . import drivers

    reg = HardwareRegistry()
    for key, (name, attr) in default_driver_classes().items():
        reg.bind_driver(key, getattr(drivers, attr), name)
    for key, attr, name in (
        ("eeg", "SimulatedEEGDevice", "simulated-eeg"),
        ("ecg", "SimulatedECGDevice", "simulated-ecg"),
        ("ppg", "SimulatedPPGDevice", "simulated-ppg"),
    ):
        reg.bind_driver(key, getattr(drivers, attr), name)
    return reg


default_hardware_registry = _build_default()
