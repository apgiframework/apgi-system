"""Perturbational Complexity Index (PCI) calculator for APGI Framework falsification testing.

This module implements the PCICalculator class that computes PCI values for testing
consciousness-related complexity measures. PCI > 0.4 indicates conscious states,
while PCI < 0.3 indicates unconscious states.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PCISignature:
    """Data class representing a PCI signature."""

    pci_value: float  # PCI value (0-1 range)
    complexity_components: Dict[str, float]  # Breakdown of complexity measures
    connectivity_strength: float  # Overall connectivity strength
    perturbation_response: float  # Response to perturbation
    is_conscious_threshold: bool  # Whether PCI > 0.4 threshold
    state_classification: str  # "conscious", "unconscious", or "intermediate"


class PCICalculator:
    """Calculator for Perturbational Complexity Index with realistic complexity characteristics.

    PCI measures the complexity of brain responses to perturbations. PCI > 0.4 indicates
    conscious states with rich, differentiated responses, while PCI < 0.3 indicates
    unconscious states with simple, stereotyped responses.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize PCI calculator.

        Args:
            random_seed: Optional seed for reproducible random number generation
        """
        self.rng = np.random.RandomState(random_seed)
        self.conscious_pci_threshold = 0.4
        self.unconscious_pci_threshold = 0.3

        # Default brain network parameters
        self.default_n_regions = 64  # Number of brain regions
        self.default_sampling_rate = 1000  # Hz
        self.default_perturbation_duration = 0.3  # seconds

        # Complexity calculation parameters
        self.lempel_ziv_threshold = 0.5
        self.min_complexity = 0.01
        self.max_complexity = 0.95

    def generate_conscious_signature(
        self, pci_range: Tuple[float, float] = (0.45, 0.8), noise_level: float = 0.02
    ) -> PCISignature:
        """Generate PCI signature indicating conscious processing.

        Args:
            pci_range: Range for PCI generation, default ensures >0.4
            noise_level: Standard deviation of noise to add to PCI

        Returns:
            PCISignature with PCI >0.4 threshold
        """
        # Generate base PCI value
        base_pci = self.rng.uniform(pci_range[0], pci_range[1])
        noise = self.rng.normal(0, noise_level)
        final_pci = np.clip(base_pci + noise, 0.0, 1.0)

        # Ensure stays above threshold
        final_pci = max(final_pci, self.conscious_pci_threshold + 0.02)

        # Generate complexity components for conscious state
        complexity_components = {
            "lempel_ziv_complexity": self.rng.uniform(0.6, 0.9),
            "perturbational_complexity": self.rng.uniform(0.5, 0.8),
            "integration_measure": self.rng.uniform(0.4, 0.7),
            "differentiation_measure": self.rng.uniform(0.5, 0.8),
            "information_integration": self.rng.uniform(0.45, 0.75),
        }

        # High connectivity and perturbation response for conscious states
        connectivity_strength = self.rng.uniform(0.6, 0.9)
        perturbation_response = self.rng.uniform(0.5, 0.8)

        return PCISignature(
            pci_value=final_pci,
            complexity_components=complexity_components,
            connectivity_strength=connectivity_strength,
            perturbation_response=perturbation_response,
            is_conscious_threshold=final_pci > self.conscious_pci_threshold,
            state_classification="conscious",
        )

    def generate_unconscious_signature(
        self, pci_range: Tuple[float, float] = (0.05, 0.25), noise_level: float = 0.015
    ) -> PCISignature:
        """Generate PCI signature indicating unconscious/subthreshold processing.

        Args:
            pci_range: Range for PCI generation, default ensures <0.3
            noise_level: Standard deviation of noise to add to PCI

        Returns:
            PCISignature with PCI <0.3 threshold
        """
        # Generate base PCI value
        base_pci = self.rng.uniform(pci_range[0], pci_range[1])
        noise = self.rng.normal(0, noise_level)
        final_pci = np.clip(base_pci + noise, 0.0, 1.0)

        # Ensure stays below threshold
        final_pci = min(final_pci, self.unconscious_pci_threshold - 0.02)
        final_pci = max(final_pci, 0.01)  # Ensure positive

        # Generate complexity components for unconscious state
        complexity_components = {
            "lempel_ziv_complexity": self.rng.uniform(0.1, 0.3),
            "perturbational_complexity": self.rng.uniform(0.05, 0.25),
            "integration_measure": self.rng.uniform(0.05, 0.2),
            "differentiation_measure": self.rng.uniform(0.1, 0.25),
            "information_integration": self.rng.uniform(0.05, 0.2),
        }

        # Low connectivity and perturbation response for unconscious states
        connectivity_strength = self.rng.uniform(0.1, 0.4)
        perturbation_response = self.rng.uniform(0.05, 0.3)

        return PCISignature(
            pci_value=final_pci,
            complexity_components=complexity_components,
            connectivity_strength=connectivity_strength,
            perturbation_response=perturbation_response,
            is_conscious_threshold=final_pci > self.conscious_pci_threshold,
            state_classification="unconscious",
        )

    def calculate_pci_from_connectivity(
        self,
        connectivity_matrix: np.ndarray,
        perturbation_sites: Optional[List[int]] = None,
        noise_level: float = 0.1,
    ) -> PCISignature:
        """Calculate PCI from a connectivity matrix and perturbation response.

        Args:
            connectivity_matrix: NxN connectivity matrix between brain regions
            perturbation_sites: List of region indices to perturb, random if None
            noise_level: Noise level for realistic simulation

        Returns:
            PCISignature calculated from connectivity data
        """
        # Validate connectivity matrix
        if connectivity_matrix.shape[0] != connectivity_matrix.shape[1]:
            raise ValueError("Connectivity matrix must be square")

        if np.any(np.isnan(connectivity_matrix)):
            raise ValueError("Connectivity matrix contains NaN values")

        if np.any(np.isinf(connectivity_matrix)):
            raise ValueError("Connectivity matrix contains infinite values")

        n_regions = connectivity_matrix.shape[0]

        if perturbation_sites is None:
            # Select random perturbation sites (typically 1-3 regions)
            n_sites = self.rng.randint(1, min(4, n_regions))
            perturbation_sites = self.rng.choice(n_regions, n_sites, replace=False)

        # Simulate perturbation response
        perturbation_response = self._simulate_perturbation_response(
            connectivity_matrix, perturbation_sites, noise_level
        )

        # Calculate complexity measures
        complexity_components = self._calculate_complexity_components(
            connectivity_matrix, perturbation_response
        )

        # Calculate overall PCI
        pci_value = self._compute_pci_from_components(complexity_components)

        # Calculate connectivity strength
        connectivity_strength = self._calculate_connectivity_strength(connectivity_matrix)

        # Determine state classification
        if pci_value > self.conscious_pci_threshold:
            state_classification = "conscious"
        elif pci_value < self.unconscious_pci_threshold:
            state_classification = "unconscious"
        else:
            state_classification = "intermediate"

        return PCISignature(
            pci_value=pci_value,
            complexity_components=complexity_components,
            connectivity_strength=connectivity_strength,
            perturbation_response=np.mean(perturbation_response),
            is_conscious_threshold=pci_value > self.conscious_pci_threshold,
            state_classification=state_classification,
        )

    def generate_signature(self, target_pci: float, noise_level: float = 0.02) -> PCISignature:
        """Generate PCI signature with specific target PCI value.

        Args:
            target_pci: Target PCI value (0-1)
            noise_level: Standard deviation of noise to add

        Returns:
            PCISignature with specified PCI value
        """
        # Add noise to target PCI
        noise = self.rng.normal(0, noise_level)
        final_pci = np.clip(target_pci + noise, 0.01, 0.95)

        # Generate appropriate complexity components based on PCI level
        if final_pci > self.conscious_pci_threshold:
            # High complexity components
            complexity_components = {
                "lempel_ziv_complexity": final_pci * self.rng.uniform(0.8, 1.2),
                "perturbational_complexity": final_pci * self.rng.uniform(0.7, 1.1),
                "integration_measure": final_pci * self.rng.uniform(0.6, 1.0),
                "differentiation_measure": final_pci * self.rng.uniform(0.8, 1.2),
                "information_integration": final_pci * self.rng.uniform(0.7, 1.1),
            }
            connectivity_strength = final_pci * self.rng.uniform(0.8, 1.2)
            perturbation_response = final_pci * self.rng.uniform(0.7, 1.1)
            state_classification = "conscious"
        else:
            # Low complexity components
            complexity_components = {
                "lempel_ziv_complexity": final_pci * self.rng.uniform(0.5, 1.5),
                "perturbational_complexity": final_pci * self.rng.uniform(0.6, 1.4),
                "integration_measure": final_pci * self.rng.uniform(0.4, 1.2),
                "differentiation_measure": final_pci * self.rng.uniform(0.5, 1.3),
                "information_integration": final_pci * self.rng.uniform(0.4, 1.2),
            }
            connectivity_strength = final_pci * self.rng.uniform(0.6, 1.4)
            perturbation_response = final_pci * self.rng.uniform(0.5, 1.3)
            state_classification = (
                "unconscious" if final_pci < self.unconscious_pci_threshold else "intermediate"
            )

        # Clip all components to valid ranges
        for key in complexity_components:
            complexity_components[key] = np.clip(complexity_components[key], 0.01, 0.95)

        connectivity_strength = np.clip(connectivity_strength, 0.01, 0.95)
        perturbation_response = np.clip(perturbation_response, 0.01, 0.95)

        return PCISignature(
            pci_value=final_pci,
            complexity_components=complexity_components,
            connectivity_strength=connectivity_strength,
            perturbation_response=perturbation_response,
            is_conscious_threshold=final_pci > self.conscious_pci_threshold,
            state_classification=state_classification,
        )

    def _simulate_perturbation_response(
        self,
        connectivity_matrix: np.ndarray,
        perturbation_sites: List[int],
        noise_level: float,
    ) -> np.ndarray:
        """Simulate brain response to perturbation."""
        n_regions = connectivity_matrix.shape[0]
        n_timepoints = int(self.default_perturbation_duration * self.default_sampling_rate)

        # Initialize response matrix
        response = np.zeros((n_regions, n_timepoints))

        # Apply initial perturbation
        for site in perturbation_sites:
            # Create perturbation signal (brief pulse)
            pulse_duration = int(0.01 * self.default_sampling_rate)  # 10ms pulse
            response[site, :pulse_duration] = 1.0

        # Simulate propagation through network
        for t in range(1, n_timepoints):
            # Calculate network influence with extreme value handling
            try:
                network_input = np.dot(connectivity_matrix, response[:, t - 1])

                # Handle extreme values
                if np.any(np.isinf(network_input)) or np.any(np.isnan(network_input)):
                    network_input = np.zeros_like(network_input)
                else:
                    # Clip to reasonable range
                    network_input = np.clip(network_input, -1000, 1000)

                # Apply decay and network dynamics
                decay_factor = 0.95
                response[:, t] = response[:, t - 1] * decay_factor + network_input * 0.1

                # Handle extreme values in response
                if np.any(np.isinf(response[:, t])) or np.any(np.isnan(response[:, t])):
                    response[:, t] = response[:, t - 1] * decay_factor
                else:
                    # Clip to reasonable range
                    response[:, t] = np.clip(response[:, t], -100, 100)

                # Add noise
                noise = self.rng.normal(0, noise_level, n_regions)
                response[:, t] += noise

            except Exception:
                # Fallback: just apply decay
                response[:, t] = response[:, t - 1] * 0.95

        return response

    def _calculate_complexity_components(
        self, connectivity_matrix: np.ndarray, perturbation_response: np.ndarray
    ) -> Dict[str, float]:
        """Calculate various complexity measures."""
        # Validate inputs
        if perturbation_response.size == 0:
            raise ValueError("Response array cannot be empty")

        if connectivity_matrix.shape[0] != perturbation_response.shape[0]:
            raise ValueError("Response and connectivity dimensions must match")

        components = {}

        # Lempel-Ziv complexity of response patterns
        components["lempel_ziv_complexity"] = self._lempel_ziv_complexity(perturbation_response)

        # Perturbational complexity (diversity of responses)
        components["perturbational_complexity"] = self._perturbational_complexity(
            perturbation_response
        )

        # Integration measure (global connectivity)
        components["integration_measure"] = self._integration_measure(connectivity_matrix)

        # Differentiation measure (local specialization)
        components["differentiation_measure"] = self._differentiation_measure(perturbation_response)

        # Information integration (Phi-like measure)
        components["information_integration"] = self._information_integration(
            connectivity_matrix, perturbation_response
        )

        return components

    def _lempel_ziv_complexity(self, response: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity of response patterns."""
        # Simplified LZ complexity calculation
        # Binarize response based on threshold
        binary_response = (response > self.lempel_ziv_threshold).astype(int)

        # Flatten to 1D sequence
        sequence = binary_response.flatten()

        # Calculate LZ complexity (simplified version)
        n = len(sequence)
        if n == 0:
            return 0.0

        complexity = 0
        i = 0
        while i < n:
            # Find longest match in previous subsequence
            max_match = 0
            for j in range(i):
                match_length = 0
                k = 0
                while i + k < n and j + k < i and sequence[i + k] == sequence[j + k]:
                    match_length += 1
                    k += 1
                max_match = max(max_match, match_length)

            # Move to next unmatched position
            i += max(1, max_match)
            complexity += 1

        # Normalize by theoretical maximum
        normalized_complexity = complexity / (n / 2)  # Rough normalization
        return float(np.clip(normalized_complexity, self.min_complexity, self.max_complexity))

    def _perturbational_complexity(self, response: np.ndarray) -> float:
        """Calculate complexity of perturbation response patterns."""
        # Calculate variance across regions and time
        spatial_variance = np.var(response, axis=0)  # Variance across regions
        temporal_variance = np.var(response, axis=1)  # Variance across time

        # Combine spatial and temporal complexity
        spatial_complexity = np.mean(spatial_variance)
        temporal_complexity = np.mean(temporal_variance)

        # Normalize and combine
        complexity = (spatial_complexity + temporal_complexity) / 2
        return float(np.clip(complexity, self.min_complexity, self.max_complexity))

    def _integration_measure(self, connectivity_matrix: np.ndarray) -> float:
        """Calculate integration measure from connectivity."""
        # Calculate global efficiency as integration measure
        try:
            # Compute shortest path lengths
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # inv_matrix is calculated but not used in current implementation
                # inv_matrix = np.linalg.pinv(
                #     connectivity_matrix + np.eye(connectivity_matrix.shape[0]) * 0.01
                # )

            # Global efficiency approximation
            n = connectivity_matrix.shape[0]
            efficiency_sum = 0
            count = 0

            for i in range(n):
                for j in range(i + 1, n):
                    if connectivity_matrix[i, j] > 0:
                        efficiency_sum += connectivity_matrix[i, j]
                        count += 1

            if count > 0:
                integration = efficiency_sum / count
            else:
                integration = 0.0

        except (ValueError, IndexError, ZeroDivisionError) as e:
            # Fallback: use mean connectivity
            logger.warning(
                f"Integration calculation failed: {e}. Using mean connectivity fallback."
            )
            integration = float(np.mean(connectivity_matrix))

        return float(np.clip(integration, self.min_complexity, self.max_complexity))

    def _differentiation_measure(self, response: np.ndarray) -> float:
        """Calculate differentiation measure from response patterns."""
        # Calculate how different each region's response is from others
        n_regions = response.shape[0]
        if n_regions < 2:
            return 0.0

        differentiation_sum = 0
        count = 0

        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                # Calculate correlation between regions
                corr = np.corrcoef(response[i, :], response[j, :])[0, 1]
                if not np.isnan(corr):
                    # Differentiation is inverse of correlation
                    differentiation_sum += 1 - abs(corr)
                    count += 1

        if count > 0:
            differentiation = differentiation_sum / count
        else:
            differentiation = 0.5

        return float(np.clip(differentiation, self.min_complexity, self.max_complexity))

    def _information_integration(
        self, connectivity_matrix: np.ndarray, response: np.ndarray
    ) -> float:
        """Calculate information integration measure."""
        # Simplified Phi-like measure
        # Calculate mutual information between different parts of the network
        n_regions = connectivity_matrix.shape[0]

        if n_regions < 4:
            return float(np.mean(np.abs(connectivity_matrix)))

        # Split network into two parts
        mid = n_regions // 2
        part1_response = response[:mid, :]
        part2_response = response[mid:, :]

        # Calculate cross-correlation between parts
        part1_mean = np.mean(part1_response, axis=0)
        part2_mean = np.mean(part2_response, axis=0)

        if len(part1_mean) > 1 and len(part2_mean) > 1:
            # Handle extreme values that might cause NaN
            if np.any(np.isinf(part1_mean)) or np.any(np.isinf(part2_mean)):
                cross_corr = 0.0
            elif np.any(np.isnan(part1_mean)) or np.any(np.isnan(part2_mean)):
                cross_corr = 0.0
            else:
                try:
                    cross_corr = np.corrcoef(part1_mean, part2_mean)[0, 1]
                    if np.isnan(cross_corr) or np.isinf(cross_corr):
                        cross_corr = 0.0
                except (ValueError, IndexError, RuntimeWarning) as e:
                    logger.warning(
                        f"Cross-correlation calculation failed: {e}. Using fallback value."
                    )
                    cross_corr = 0.0
        else:
            cross_corr = 0.0

        # Convert correlation to integration measure
        integration = abs(cross_corr)
        return float(np.clip(integration, self.min_complexity, self.max_complexity))

    def _compute_pci_from_components(self, components: Dict[str, float]) -> float:
        """Compute overall PCI from complexity components."""
        if not components:
            raise ValueError("No complexity components provided")

        # Weighted average of components
        weights = {
            "lempel_ziv_complexity": 0.25,
            "perturbational_complexity": 0.25,
            "integration_measure": 0.2,
            "differentiation_measure": 0.15,
            "information_integration": 0.15,
        }

        pci = 0.0
        total_weight = 0.0

        for component, value in components.items():
            if component in weights:
                pci += weights[component] * value
                total_weight += weights[component]

        if total_weight > 0:
            pci /= total_weight

        return float(np.clip(pci, self.min_complexity, self.max_complexity))

    def _calculate_connectivity_strength(self, connectivity_matrix: np.ndarray) -> float:
        """Calculate overall connectivity strength."""
        return float(np.mean(np.abs(connectivity_matrix)))

    def validate_signature(self, signature: PCISignature) -> bool:
        """Validate that PCI signature meets basic constraints.

        Args:
            signature: PCISignature to validate

        Returns:
            True if signature is valid
        """
        # Check PCI value is in valid range
        if not (0.0 <= signature.pci_value <= 1.0):
            return False

        # Check complexity components are in valid range
        for value in signature.complexity_components.values():
            if not (0.0 <= value <= 1.0):
                return False

        # Check connectivity strength is in valid range
        if not (0.0 <= signature.connectivity_strength <= 1.0):
            return False

        # Check perturbation response is in valid range
        if not (0.0 <= signature.perturbation_response <= 1.0):
            return False

        # Check state classification consistency
        if signature.is_conscious_threshold != (signature.pci_value > self.conscious_pci_threshold):
            return False

        return True

    def is_conscious_level(self, signature: PCISignature) -> bool:
        """Determine if PCI signature indicates conscious-level processing.

        Args:
            signature: PCISignature to evaluate

        Returns:
            True if PCI exceeds conscious threshold (>0.4)
        """
        return signature.pci_value > self.conscious_pci_threshold

    def calculate_pci(self, connectivity_matrix: np.ndarray) -> float:
        """Legacy method for backward compatibility with falsification tests.

        Args:
            connectivity_matrix: NxN connectivity matrix between brain regions

        Returns:
            PCI value (float) for compatibility
        """
        signature = self.calculate_pci_from_connectivity(connectivity_matrix)
        return signature.pci_value
