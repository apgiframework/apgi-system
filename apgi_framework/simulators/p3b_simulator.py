"""P3b ERP signature simulator for APGI Framework falsification testing.

This module implements the P3bSimulator class that generates realistic P3b ERP signatures
with configurable amplitude and latency parameters for testing consciousness-related
neural activity patterns.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class P3bSignature:
    """Data class representing a P3b ERP signature."""

    amplitude: float  # Amplitude in μV at Pz electrode
    latency: float  # Latency in milliseconds
    electrode: str  # Electrode location (default: Pz)
    noise_level: float  # Added noise level
    is_conscious_threshold: bool  # Whether amplitude > 5 μV threshold


class P3bSimulator:
    """Simulator for P3b ERP signatures with realistic amplitude and latency characteristics.

    The P3b component is a positive ERP deflection typically occurring 250-500 ms after
    stimulus onset, with amplitudes >5 μV at Pz electrode indicating conscious processing.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize P3b simulator.

        Args:
            random_seed: Optional seed for reproducible random number generation
        """
        self.rng = np.random.RandomState(random_seed)
        self.conscious_amplitude_threshold = 5.0  # μV
        self.latency_range = (250.0, 500.0)  # ms
        self.electrode = "Pz"

    def generate_conscious_signature(
        self,
        amplitude_range: Tuple[float, float] = (5.5, 12.0),
        latency_range: Optional[Tuple[float, float]] = None,
        noise_level: float = 0.5,
    ) -> P3bSignature:
        """Generate a P3b signature indicating conscious processing.

        Args:
            amplitude_range: Range for amplitude generation (μV), default ensures >5 μV
            latency_range: Range for latency generation (ms), uses class default if None
            noise_level: Standard deviation of Gaussian noise to add

        Returns:
            P3bSignature with amplitude >5 μV threshold
        """
        if latency_range is None:
            latency_range = self.latency_range

        # Generate base amplitude above conscious threshold
        base_amplitude = self.rng.uniform(amplitude_range[0], amplitude_range[1])

        # Generate latency within specified range
        latency = self.rng.uniform(latency_range[0], latency_range[1])

        # Add realistic noise
        noise = self.rng.normal(0, noise_level)
        final_amplitude = base_amplitude + noise

        # Ensure amplitude stays above threshold even with noise
        final_amplitude = max(final_amplitude, self.conscious_amplitude_threshold + 0.1)

        return P3bSignature(
            amplitude=final_amplitude,
            latency=latency,
            electrode=self.electrode,
            noise_level=noise_level,
            is_conscious_threshold=final_amplitude > self.conscious_amplitude_threshold,
        )

    def generate_unconscious_signature(
        self,
        amplitude_range: Tuple[float, float] = (0.5, 2.0),
        latency_range: Optional[Tuple[float, float]] = None,
        noise_level: float = 0.3,
    ) -> P3bSignature:
        """Generate a P3b signature indicating unconscious/subthreshold processing.

        Args:
            amplitude_range: Range for amplitude generation (μV), default ensures <5 μV
            latency_range: Range for latency generation (ms), uses class default if None
            noise_level: Standard deviation of Gaussian noise to add

        Returns:
            P3bSignature with amplitude <5 μV threshold
        """
        if latency_range is None:
            latency_range = self.latency_range

        # Generate base amplitude below conscious threshold
        base_amplitude = self.rng.uniform(amplitude_range[0], amplitude_range[1])

        # Generate latency within specified range
        latency = self.rng.uniform(latency_range[0], latency_range[1])

        # Add realistic noise
        noise = self.rng.normal(0, noise_level)
        final_amplitude = base_amplitude + noise

        # Ensure amplitude stays below threshold even with noise
        final_amplitude = min(final_amplitude, self.conscious_amplitude_threshold - 0.1)
        final_amplitude = max(final_amplitude, 0.0)  # Ensure non-negative

        return P3bSignature(
            amplitude=final_amplitude,
            latency=latency,
            electrode=self.electrode,
            noise_level=noise_level,
            is_conscious_threshold=final_amplitude > self.conscious_amplitude_threshold,
        )

    def generate_signature(
        self,
        target_amplitude: float,
        target_latency: Optional[float] = None,
        noise_level: float = 0.4,
    ) -> P3bSignature:
        """Generate a P3b signature with specific target parameters.

        Args:
            target_amplitude: Target amplitude in μV
            target_latency: Target latency in ms, random if None
            noise_level: Standard deviation of Gaussian noise to add

        Returns:
            P3bSignature with specified characteristics
        """
        if target_latency is None:
            target_latency = self.rng.uniform(self.latency_range[0], self.latency_range[1])

        # Add noise to target values
        amplitude_noise = self.rng.normal(0, noise_level)
        latency_noise = self.rng.normal(0, noise_level * 10)  # Scale noise for latency

        final_amplitude = max(0.0, target_amplitude + amplitude_noise)
        final_latency = np.clip(
            target_latency + latency_noise, self.latency_range[0], self.latency_range[1]
        )

        return P3bSignature(
            amplitude=final_amplitude,
            latency=final_latency,
            electrode=self.electrode,
            noise_level=noise_level,
            is_conscious_threshold=final_amplitude > self.conscious_amplitude_threshold,
        )

    def validate_signature(self, signature: P3bSignature) -> bool:
        """Validate that a P3b signature meets basic physiological constraints.

        Args:
            signature: P3bSignature to validate

        Returns:
            True if signature is physiologically plausible
        """
        # Check amplitude is non-negative and within reasonable range
        if signature.amplitude < 0 or signature.amplitude > 50.0:
            return False

        # Check latency is within expected range
        if not (self.latency_range[0] <= signature.latency <= self.latency_range[1]):
            return False

        # Check electrode is correct
        if signature.electrode != self.electrode:
            return False

        return True

    def is_conscious_level(self, signature: P3bSignature) -> bool:
        """Determine if P3b signature indicates conscious-level processing.

        Args:
            signature: P3bSignature to evaluate

        Returns:
            True if amplitude exceeds conscious threshold (>5 μV)
        """
        return signature.amplitude > self.conscious_amplitude_threshold

    def simulate_p3b(
        self, amplitude_range: Tuple[float, float], latency_range: Tuple[float, float]
    ) -> P3bSignature:
        """Legacy method for backward compatibility with falsification tests.

        Args:
            amplitude_range: Range for amplitude generation (μV)
            latency_range: Range for latency generation (ms)

        Returns:
            P3bSignature with specified characteristics
        """
        # Determine if this should be conscious or unconscious based on amplitude range
        if amplitude_range[0] > self.conscious_amplitude_threshold:
            return self.generate_conscious_signature(amplitude_range, latency_range)
        elif amplitude_range[1] < self.conscious_amplitude_threshold:
            return self.generate_unconscious_signature(amplitude_range, latency_range)
        else:
            # Mixed range - use target amplitude at midpoint
            target_amplitude = (amplitude_range[0] + amplitude_range[1]) / 2
            target_latency = (latency_range[0] + latency_range[1]) / 2
            return self.generate_signature(target_amplitude, target_latency)
