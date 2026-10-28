"""Stub module for VP15 validation protocol.

This module provides type stubs for VP15 validation protocol
that is being migrated from the old apgi_simulation structure.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np


class VP15Bounds:
    """Bounds for VP15 validation metrics."""

    def __init__(
        self,
        critical_energy: float = 0.05,
        max_allostatic_load: float = 0.90,
        precision_floor: float = 0.1,
        precision_ceil: float = 10.0,
        min_synchrony: float = 0.20,
        energy_restore_rate: float = 0.50,
        load_damping_gain: float = 0.15,
    ):
        """Initialize VP15 bounds.

        Parameters
        ----------
        critical_energy : float, optional
            Critical energy threshold, by default 0.05
        max_allostatic_load : float, optional
            Maximum allostatic load, by default 0.90
        precision_floor : float, optional
            Minimum precision value, by default 0.1
        precision_ceil : float, optional
            Maximum precision value, by default 10.0
        min_synchrony : float, optional
            Minimum synchrony threshold, by default 0.20
        energy_restore_rate : float, optional
            Energy restoration rate, by default 0.50
        load_damping_gain : float, optional
            Load damping gain, by default 0.15
        """
        self.critical_energy = critical_energy
        self.max_allostatic_load = max_allostatic_load
        self.precision_floor = precision_floor
        self.precision_ceil = precision_ceil
        self.min_synchrony = min_synchrony
        self.energy_restore_rate = energy_restore_rate
        self.load_damping_gain = load_damping_gain


class VP15Metrics:
    """Metrics for VP15 validation."""

    def __init__(
        self,
        metabolic_feasibility: float = 0.0,
        allostatic_integrity: float = 0.0,
        information_consistency: float = 0.0,
        temporal_coherence: float = 0.0,
        precision_stability: float = 0.0,
        system_health_score: float = 0.0,
        corrections_applied: int = 0,
    ):
        """Initialize VP15 metrics.

        Parameters
        ----------
        metabolic_feasibility : float, optional
            Metabolic feasibility score, by default 0.0
        allostatic_integrity : float, optional
            Allostatic integrity score, by default 0.0
        information_consistency : float, optional
            Information consistency score, by default 0.0
        temporal_coherence : float, optional
            Temporal coherence score, by default 0.0
        precision_stability : float, optional
            Precision stability score, by default 0.0
        system_health_score : float, optional
            Overall system health score, by default 0.0
        corrections_applied : int, optional
            Number of corrections applied, by default 0
        """
        self.metabolic_feasibility = metabolic_feasibility
        self.allostatic_integrity = allostatic_integrity
        self.information_consistency = information_consistency
        self.temporal_coherence = temporal_coherence
        self.precision_stability = precision_stability
        self.system_health_score = system_health_score
        self.corrections_applied = corrections_applied


class VP15ValidationProtocol:
    """VP15 validation protocol for model validation."""

    _WEIGHTS = {
        "metabolic_feasibility": 0.30,
        "allostatic_integrity": 0.25,
        "information_consistency": 0.20,
        "temporal_coherence": 0.15,
        "precision_stability": 0.10,
    }

    def __init__(self, config: dict):
        """Initialize VP15 validation protocol.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        vp15_config = config.get("validation", {}).get("vp15", {})
        self.bounds = VP15Bounds(
            critical_energy=vp15_config.get("critical_energy", 0.05),
            max_allostatic_load=vp15_config.get("max_allostatic_load", 0.90),
            precision_floor=vp15_config.get("precision_floor", 0.1),
            precision_ceil=vp15_config.get("precision_ceil", 10.0),
            min_synchrony=vp15_config.get("min_synchrony", 0.20),
            energy_restore_rate=vp15_config.get("energy_restore_rate", 0.50),
            load_damping_gain=vp15_config.get("load_damping_gain", 0.15),
        )
        self.violation_history: list[str] = []
        self._total_corrections = 0

    def validate_step(
        self, state: Dict[str, Any], targets: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, VP15Metrics]:
        """Validate a single simulation step.

        Parameters
        ----------
        state : dict
            System state dictionary
        targets : dict, optional
            Target parameter dictionary

        Returns
        -------
        tuple[bool, VP15Metrics]
            (is_valid, metrics)
        """
        corrections = 0

        # Extract state values with defaults
        metabolism = state.get("metabolism", {})
        allostasis = state.get("allostasis", {})
        active_inference = state.get("active_inference", {})
        timeline = state.get("timeline", {})

        # Subsystems may report these as one-element arrays; the bounds below are scalar
        reserves = float(np.ravel(metabolism.get("reserves", 1.0))[0])
        allostatic_load = float(np.ravel(allostasis.get("allostatic_load", 0.0))[0])
        free_energy = active_inference.get("free_energy", 0.0)
        synchrony = timeline.get("synchrony", 1.0)
        precisions = active_inference.get("precisions", [])

        # Constraint 1: Metabolic feasibility
        if reserves < self.bounds.critical_energy:
            reserves = min(1.0, reserves + self.bounds.energy_restore_rate)
            metabolism["reserves"] = reserves
            corrections += 1
            self.violation_history.append("METABOLIC: reserves below critical")
            metabolic_feasibility = 0.5
        else:
            metabolic_feasibility = 1.0

        # Constraint 2: Allostatic integrity
        if allostatic_load > self.bounds.max_allostatic_load:
            load = allostatic_load - self.bounds.load_damping_gain * (
                allostatic_load - self.bounds.max_allostatic_load
            )
            load = float(np.clip(load, 0.0, self.bounds.max_allostatic_load))
            allostasis["allostatic_load"] = load
            corrections += 1
            self.violation_history.append("ALLOSTATIC: load above maximum")
            allostatic_integrity = 0.0
        else:
            allostatic_integrity = max(0.0, 1.0 - allostatic_load / self.bounds.max_allostatic_load)

        # Constraint 3: Information consistency
        if free_energy < 0.0:
            active_inference["free_energy"] = 0.0
            corrections += 1
            self.violation_history.append("INFORMATION: negative free energy")
            information_consistency = 0.5
        else:
            information_consistency = 1.0

        # Constraint 4: Temporal coherence
        if synchrony < self.bounds.min_synchrony:
            timeline["synchrony"] = self.bounds.min_synchrony
            corrections += 1
            self.violation_history.append("TEMPORAL: synchrony below minimum")
            temporal_coherence = 0.0
        else:
            temporal_coherence = synchrony

        # Constraint 5: Precision stability
        if precisions:
            total_values = 0
            in_range: int = 0
            for arr in precisions:
                if isinstance(arr, np.ndarray):
                    total_values += int(len(arr))
                    in_range += int(
                        np.sum(
                            (arr >= self.bounds.precision_floor)
                            & (arr <= self.bounds.precision_ceil)
                        )
                    )
                    # Clip values
                    np.clip(arr, self.bounds.precision_floor, self.bounds.precision_ceil, out=arr)
                    if np.any(
                        (arr < self.bounds.precision_floor) | (arr > self.bounds.precision_ceil)
                    ):
                        corrections += 1
                        self.violation_history.append("PRECISION: values out of range")
            precision_stability = in_range / total_values if total_values > 0 else 1.0
        else:
            precision_stability = 1.0

        # Calculate system health score
        system_health_score = (
            metabolic_feasibility * self._WEIGHTS["metabolic_feasibility"]
            + allostatic_integrity * self._WEIGHTS["allostatic_integrity"]
            + information_consistency * self._WEIGHTS["information_consistency"]
            + temporal_coherence * self._WEIGHTS["temporal_coherence"]
            + precision_stability * self._WEIGHTS["precision_stability"]
        )

        self._total_corrections += corrections

        metrics = VP15Metrics(
            metabolic_feasibility=metabolic_feasibility,
            allostatic_integrity=allostatic_integrity,
            information_consistency=information_consistency,
            temporal_coherence=temporal_coherence,
            precision_stability=precision_stability,
            system_health_score=system_health_score,
            corrections_applied=corrections,
        )

        # A collapsed allostatic integrity is a hard failure, whatever the weighted score says
        is_valid = bool(system_health_score > 0.5 and allostatic_integrity > 0.0)

        return is_valid, metrics

    def get_total_corrections(self) -> int:
        """Get total corrections applied.

        Returns
        -------
        int
            Total corrections
        """
        return self._total_corrections

    def get_violation_report(self) -> str:
        """Get violation report.

        Returns
        -------
        str
            Violation report
        """
        if not self.violation_history:
            return "healthy"

        recent = self.violation_history[-10:]
        lines = ["Recent violations:"]
        for v in recent:
            lines.append(f"– {v}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset the protocol state."""
        self.violation_history.clear()
        self._total_corrections = 0
