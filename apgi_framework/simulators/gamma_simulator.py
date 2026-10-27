"""Gamma-band synchrony simulator for APGI Framework falsification testing.

This module implements the GammaSimulator class that generates realistic gamma-band
synchrony patterns with phase-locking values (PLV) for testing consciousness-related
neural connectivity patterns.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class BrainRegion(Enum):
    """Enumeration of brain regions for gamma synchrony analysis."""

    FRONTAL_LEFT = "frontal_left"
    FRONTAL_RIGHT = "frontal_right"
    PARIETAL_LEFT = "parietal_left"
    PARIETAL_RIGHT = "parietal_right"
    TEMPORAL_LEFT = "temporal_left"
    TEMPORAL_RIGHT = "temporal_right"


@dataclass
class GammaSignature:
    """Data class representing gamma-band synchrony signature."""

    plv_values: Dict[str, float]  # Phase-locking values between region pairs
    duration: float  # Duration of sustained activity in ms
    frequency_range: Tuple[float, float]  # Frequency range in Hz
    peak_frequency: float  # Peak frequency in Hz
    is_conscious_threshold: bool  # Whether PLV > 0.3 threshold
    region_connectivity: Dict[str, List[str]]  # Connectivity patterns


class GammaSimulator:
    """Simulator for gamma-band synchrony with realistic phase-locking characteristics.

    Gamma-band synchrony (30-80 Hz) with PLV >0.3 between frontoparietal and temporal
    regions, sustained >200 ms, indicates conscious processing and global workspace
    activation.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize gamma synchrony simulator.

        Args:
            random_seed: Optional seed for reproducible random number generation
        """
        self.rng = np.random.RandomState(random_seed)
        self.conscious_plv_threshold = 0.3
        self.unconscious_plv_threshold = 0.15
        self.min_conscious_duration = 200.0  # ms
        self.frequency_range = (30.0, 80.0)  # Hz

        # Define key connectivity patterns for consciousness
        self.frontoparietal_pairs = [
            ("frontal_left", "parietal_left"),
            ("frontal_right", "parietal_right"),
            ("frontal_left", "parietal_right"),
            ("frontal_right", "parietal_left"),
        ]

        self.frontotemporal_pairs = [
            ("frontal_left", "temporal_left"),
            ("frontal_right", "temporal_right"),
            ("frontal_left", "temporal_right"),
            ("frontal_right", "temporal_left"),
        ]

        self.parietotemporal_pairs = [
            ("parietal_left", "temporal_left"),
            ("parietal_right", "temporal_right"),
            ("parietal_left", "temporal_right"),
            ("parietal_right", "temporal_left"),
        ]

    def generate_conscious_signature(
        self,
        plv_range: Tuple[float, float] = (0.35, 0.7),
        duration_range: Tuple[float, float] = (220.0, 800.0),
        noise_level: float = 0.05,
    ) -> GammaSignature:
        """Generate gamma synchrony signature indicating conscious processing.

        Args:
            plv_range: Range for PLV generation, default ensures >0.3
            duration_range: Range for duration generation (ms), default ensures >200 ms
            noise_level: Standard deviation of noise to add to PLV values

        Returns:
            GammaSignature with PLV >0.3 threshold
        """
        plv_values = {}

        # Generate high PLV for frontoparietal connections (key for consciousness)
        for region1, region2 in self.frontoparietal_pairs:
            base_plv = self.rng.uniform(plv_range[0], plv_range[1])
            noise = self.rng.normal(0, noise_level)
            final_plv = np.clip(base_plv + noise, 0.0, 1.0)
            # Ensure stays above threshold
            final_plv = max(final_plv, self.conscious_plv_threshold + 0.02)
            plv_values[f"{region1}-{region2}"] = final_plv

        # Generate moderate PLV for frontotemporal connections
        for region1, region2 in self.frontotemporal_pairs:
            base_plv = self.rng.uniform(plv_range[0] * 0.8, plv_range[1] * 0.9)
            noise = self.rng.normal(0, noise_level)
            final_plv = np.clip(base_plv + noise, 0.0, 1.0)
            final_plv = max(final_plv, self.conscious_plv_threshold + 0.01)
            plv_values[f"{region1}-{region2}"] = final_plv

        # Generate lower but still conscious-level PLV for parietotemporal
        for region1, region2 in self.parietotemporal_pairs:
            base_plv = self.rng.uniform(self.conscious_plv_threshold + 0.02, plv_range[1] * 0.7)
            noise = self.rng.normal(0, noise_level)
            final_plv = np.clip(base_plv + noise, 0.0, 1.0)
            final_plv = max(final_plv, self.conscious_plv_threshold + 0.01)
            plv_values[f"{region1}-{region2}"] = final_plv

        # Generate sustained duration
        duration = self.rng.uniform(duration_range[0], duration_range[1])
        duration = max(duration, self.min_conscious_duration + 10)

        # Generate frequency characteristics
        peak_frequency = self.rng.uniform(40.0, 60.0)  # Typical conscious gamma

        return GammaSignature(
            plv_values=plv_values,
            duration=duration,
            frequency_range=self.frequency_range,
            peak_frequency=peak_frequency,
            is_conscious_threshold=self._check_conscious_threshold(plv_values),
            region_connectivity=self._build_connectivity_map(plv_values),
        )

    def generate_unconscious_signature(
        self,
        plv_range: Tuple[float, float] = (0.05, 0.14),
        duration_range: Tuple[float, float] = (50.0, 180.0),
        noise_level: float = 0.03,
    ) -> GammaSignature:
        """Generate gamma synchrony signature indicating unconscious/subthreshold processing.

        Args:
            plv_range: Range for PLV generation, default ensures <0.15
            duration_range: Range for duration generation (ms), default ensures <200 ms
            noise_level: Standard deviation of noise to add to PLV values

        Returns:
            GammaSignature with PLV <0.15 threshold
        """
        plv_values = {}

        # Generate low PLV for all connections
        all_pairs = (
            self.frontoparietal_pairs + self.frontotemporal_pairs + self.parietotemporal_pairs
        )

        for region1, region2 in all_pairs:
            base_plv = self.rng.uniform(plv_range[0], plv_range[1])
            noise = self.rng.normal(0, noise_level)
            final_plv = np.clip(base_plv + noise, 0.0, 1.0)
            # Ensure stays below threshold
            final_plv = min(final_plv, self.unconscious_plv_threshold - 0.01)
            final_plv = max(final_plv, 0.0)
            plv_values[f"{region1}-{region2}"] = final_plv

        # Generate brief duration
        duration = self.rng.uniform(duration_range[0], duration_range[1])
        duration = min(duration, self.min_conscious_duration - 10)

        # Generate frequency characteristics
        peak_frequency = self.rng.uniform(35.0, 45.0)  # Lower gamma for unconscious

        return GammaSignature(
            plv_values=plv_values,
            duration=duration,
            frequency_range=self.frequency_range,
            peak_frequency=peak_frequency,
            is_conscious_threshold=self._check_conscious_threshold(plv_values),
            region_connectivity=self._build_connectivity_map(plv_values),
        )

    def generate_signature(
        self,
        target_plv: float,
        target_duration: float,
        connection_pattern: str = "frontoparietal",
        noise_level: float = 0.04,
    ) -> GammaSignature:
        """Generate gamma synchrony signature with specific target parameters.

        Args:
            target_plv: Target PLV value
            target_duration: Target duration in ms
            connection_pattern: Type of connections ("frontoparietal", "all", "sparse")
            noise_level: Standard deviation of noise to add

        Returns:
            GammaSignature with specified characteristics
        """
        plv_values = {}

        # Select connection pairs based on pattern
        if connection_pattern == "frontoparietal":
            pairs = self.frontoparietal_pairs
        elif connection_pattern == "all":
            pairs = (
                self.frontoparietal_pairs + self.frontotemporal_pairs + self.parietotemporal_pairs
            )
        elif connection_pattern == "sparse":
            pairs = self.frontoparietal_pairs[:2]  # Only strongest connections
        else:
            pairs = self.frontoparietal_pairs

        # Generate PLV values with noise
        for region1, region2 in pairs:
            noise = self.rng.normal(0, noise_level)
            final_plv = np.clip(target_plv + noise, 0.0, 1.0)
            plv_values[f"{region1}-{region2}"] = final_plv

        # Add duration noise
        duration_noise = self.rng.normal(0, noise_level * 20)
        final_duration = max(10.0, target_duration + duration_noise)

        # Generate frequency characteristics
        peak_frequency = self.rng.uniform(35.0, 65.0)

        return GammaSignature(
            plv_values=plv_values,
            duration=final_duration,
            frequency_range=self.frequency_range,
            peak_frequency=peak_frequency,
            is_conscious_threshold=self._check_conscious_threshold(plv_values),
            region_connectivity=self._build_connectivity_map(plv_values),
        )

    def _check_conscious_threshold(self, plv_values: Dict[str, float]) -> bool:
        """Check if PLV values meet conscious threshold criteria."""
        # Check if any frontoparietal connection exceeds threshold
        frontoparietal_plvs = [
            plv
            for conn, plv in plv_values.items()
            if any(
                f"{r1}-{r2}" in conn or f"{r2}-{r1}" in conn for r1, r2 in self.frontoparietal_pairs
            )
        ]

        if not frontoparietal_plvs:
            return False

        return max(frontoparietal_plvs) > self.conscious_plv_threshold

    def _build_connectivity_map(self, plv_values: Dict[str, float]) -> Dict[str, List[str]]:
        """Build connectivity map from PLV values."""
        connectivity: Dict[str, List[str]] = {}

        for connection, plv in plv_values.items():
            if plv > 0.1:  # Only include meaningful connections
                regions = connection.split("-")
                if len(regions) == 2:
                    region1, region2 = regions
                    if region1 not in connectivity:
                        connectivity[region1] = []
                    if region2 not in connectivity:
                        connectivity[region2] = []
                    connectivity[region1].append(region2)
                    connectivity[region2].append(region1)

        return connectivity

    def validate_signature(self, signature: GammaSignature) -> bool:
        """Validate that gamma signature meets basic physiological constraints.

        Args:
            signature: GammaSignature to validate

        Returns:
            True if signature is physiologically plausible
        """
        # Check PLV values are in valid range
        for plv in signature.plv_values.values():
            if not (0.0 <= plv <= 1.0):
                return False

        # Check duration is positive
        if signature.duration <= 0:
            return False

        # Check frequency range is valid
        if (
            signature.frequency_range[0] < 20
            or signature.frequency_range[1] > 100
            or signature.frequency_range[0] >= signature.frequency_range[1]
        ):
            return False

        # Check peak frequency is within range
        if not (
            signature.frequency_range[0] <= signature.peak_frequency <= signature.frequency_range[1]
        ):
            return False

        return True

    def is_conscious_level(self, signature: GammaSignature) -> bool:
        """Determine if gamma signature indicates conscious-level processing.

        Args:
            signature: GammaSignature to evaluate

        Returns:
            True if PLV and duration exceed conscious thresholds
        """
        # Check PLV threshold
        plv_conscious = self._check_conscious_threshold(signature.plv_values)

        # Check duration threshold
        duration_conscious = signature.duration > self.min_conscious_duration

        return plv_conscious and duration_conscious

    def get_max_plv(self, signature: GammaSignature) -> float:
        """Get maximum PLV value from signature."""
        if not signature.plv_values:
            return 0.0
        return max(signature.plv_values.values())

    def get_frontoparietal_plv(self, signature: GammaSignature) -> float:
        """Get average PLV for frontoparietal connections."""
        frontoparietal_plvs = [
            plv
            for conn, plv in signature.plv_values.items()
            if any(
                f"{r1}-{r2}" in conn or f"{r2}-{r1}" in conn for r1, r2 in self.frontoparietal_pairs
            )
        ]

        if not frontoparietal_plvs:
            return 0.0

        return float(np.mean(frontoparietal_plvs))

    def simulate_gamma_synchrony(
        self, plv_range: Tuple[float, float], duration_range: Tuple[float, float]
    ) -> "GammaResult":
        """Legacy method for backward compatibility with falsification tests.

        Args:
            plv_range: Range for PLV generation
            duration_range: Range for duration generation (ms)

        Returns:
            GammaResult with plv and duration attributes for compatibility
        """
        # Determine if this should be conscious or unconscious based on PLV range
        if plv_range[0] > self.conscious_plv_threshold:
            signature = self.generate_conscious_signature(plv_range, duration_range)
        elif plv_range[1] < self.conscious_plv_threshold:
            signature = self.generate_unconscious_signature(plv_range, duration_range)
        else:
            # Mixed range - use target PLV at midpoint
            target_plv = (plv_range[0] + plv_range[1]) / 2
            target_duration = (duration_range[0] + duration_range[1]) / 2
            signature = self.generate_signature(target_plv, target_duration)

        # Return compatibility object
        return GammaResult(signature)


class GammaResult:
    """Compatibility wrapper for gamma simulation results."""

    def __init__(self, signature: GammaSignature):
        self.signature = signature
        self.plv = self.get_max_plv()
        self.duration = signature.duration

    def get_max_plv(self) -> float:
        """Get maximum PLV value from signature."""
        if not self.signature.plv_values:
            return 0.0
        return max(self.signature.plv_values.values())
