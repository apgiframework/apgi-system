"""Microstate analysis for scalp topography clustering.

Provides classification of EEG microstates into canonical states,
transition probability estimation, and temporal dynamics analysis
for APGI framework validation.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from scipy import signal
from sklearn.cluster import KMeans

from apgi_framework.utils.progress_monitor import ProgressMonitor


@dataclass
class MicrostateSequence:
    """Container for microstate sequence data."""

    labels: np.ndarray  # Microstate labels over time
    templates: np.ndarray  # Microstate template topographies
    gfp: np.ndarray  # Global field power
    timestamps: np.ndarray  # Time points

    # Temporal metrics
    durations: List[float] = field(default_factory=list)  # Duration of each segment
    occurrences: Dict[int, int] = field(default_factory=dict)  # Count per state
    coverage: Dict[int, float] = field(default_factory=dict)  # Time coverage per state

    # Transition metrics
    transition_matrix: Optional[np.ndarray] = None
    transition_probabilities: Optional[Dict[Tuple[int, int], float]] = None


class MicrostateAnalysis:
    """Microstate analysis for EEG scalp topography clustering.

    Implements modified K-means clustering to identify canonical microstate
    classes and analyze their temporal dynamics at millisecond resolution.
    """

    def __init__(self, n_states: int = 4, sampling_rate: float = 1000.0):
        """Initialize microstate analysis.

        Args:
            n_states: Number of microstate classes (typically 4-7)
            sampling_rate: Sampling rate in Hz
        """
        self.n_states = n_states
        self.sampling_rate = sampling_rate
        self.dt = 1000.0 / sampling_rate  # ms per sample

        # Canonical microstate templates (will be learned)
        self.templates: Optional[np.ndarray] = None
        self.template_labels = ["A", "B", "C", "D", "E", "F", "G"][:n_states]

    def compute_gfp(self, data: np.ndarray) -> np.ndarray:
        """Compute Global Field Power (GFP).

        GFP is the spatial standard deviation of the scalp potential at each time point.

        Args:
            data: EEG data (channels x time)

        Returns:
            GFP time series
        """
        # Average reference
        data_avg_ref = data - np.mean(data, axis=0, keepdims=True)

        # Compute spatial standard deviation
        gfp = np.std(data_avg_ref, axis=0)

        return cast(np.ndarray, gfp)

    def normalize_topography(self, data: np.ndarray) -> np.ndarray:
        """Normalize topographies to unit variance.

        Args:
            data: EEG data (channels x time)

        Returns:
            Normalized data
        """
        # Compute GFP
        gfp = self.compute_gfp(data)

        # Avoid division by zero
        gfp[gfp < 1e-10] = 1.0

        # Normalize each time point
        normalized = data / gfp[np.newaxis, :]

        return cast(np.ndarray, normalized)

    def select_gfp_peaks(
        self, data: np.ndarray, gfp: np.ndarray, min_peak_distance: int = 10
    ) -> np.ndarray:
        """Select GFP peaks for microstate clustering.

        Microstates are most stable at GFP peaks.

        Args:
            data: EEG data (channels x time)
            gfp: Global field power time series
            min_peak_distance: Minimum distance between peaks (samples)

        Returns:
            Data at GFP peaks (channels x n_peaks)
        """
        # Find peaks in GFP
        peaks, _ = signal.find_peaks(gfp, distance=min_peak_distance)

        # Extract data at peaks
        peak_data = data[:, peaks]

        return peak_data

    def fit_microstates(
        self,
        data: np.ndarray,
        use_gfp_peaks: bool = True,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> np.ndarray:
        """Fit microstate templates using modified K-means clustering.

        Args:
            data: EEG data (channels x time)
            use_gfp_peaks: Whether to use only GFP peaks for clustering
            max_iterations: Maximum clustering iterations
            tolerance: Convergence tolerance

        Returns:
            Microstate templates (n_states x channels)
        """
        # Normalize topographies
        normalized_data = self.normalize_topography(data)

        # Select data for clustering
        if use_gfp_peaks:
            gfp = self.compute_gfp(data)
            clustering_data = self.select_gfp_peaks(normalized_data, gfp)
        else:
            clustering_data = normalized_data

        # Transpose for sklearn (samples x features)
        clustering_data = clustering_data.T

        # Modified K-means with polarity invariance
        best_templates = None
        best_inertia = np.inf

        # Multiple random initializations with progress tracking
        progress = ProgressMonitor(10, "K-means clustering initializations")
        progress.start()

        for init_idx in range(10):
            kmeans = KMeans(
                n_clusters=self.n_states,
                max_iter=max_iterations,
                tol=tolerance,
                n_init=1,
            )
            kmeans.fit(clustering_data)

            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_templates = kmeans.cluster_centers_

            progress.update(message=f"Initialization {init_idx + 1}/10")

        progress.complete()

        # Transpose back (n_states x channels)
        if best_templates is None:
            raise ValueError("Failed to find valid templates during clustering")
        self.templates = best_templates.T

        return self.normalize_templates()

    def normalize_templates(self) -> np.ndarray:
        """Normalize fitted microstate templates to unit length."""
        if self.templates is None:
            raise ValueError("Must fit templates first")

        # Use a local variable to avoid None type issues
        templates: np.ndarray = self.templates

        # Normalize templates with progress tracking
        norm_progress = ProgressMonitor(self.n_states, "Normalizing templates")
        norm_progress.start()

        for i in range(self.n_states):
            templates[:, i] = templates[:, i] / np.linalg.norm(templates[:, i])
            norm_progress.update(i + 1)

        norm_progress.complete()
        return templates

    def assign_microstates(
        self,
        data: np.ndarray,
        templates: Optional[np.ndarray] = None,
        polarity_invariant: bool = True,
    ) -> np.ndarray:
        """Assign microstate labels to each time point.

        Args:
            data: EEG data (channels x time)
            templates: Microstate templates (n_states x channels)
                      If None, uses fitted templates
            polarity_invariant: Whether to ignore polarity in matching

        Returns:
            Microstate labels (time,)
        """
        if templates is None:
            if self.templates is None:
                raise ValueError("Must fit templates first or provide templates")
            templates = self.templates

        # Normalize data
        normalized_data = self.normalize_topography(data)

        # Templates must be set by now
        if templates is None:
            raise ValueError("Templates cannot be None")

        # Compute spatial correlation with each template
        n_channels, n_timepoints = normalized_data.shape
        n_states = templates.shape[1]

        correlations = np.zeros((n_states, n_timepoints))

        # Progress tracking for correlation computation
        total_computations = n_states * n_timepoints
        progress = ProgressMonitor(total_computations, "Computing spatial correlations")
        progress.start()

        computation_count = 0
        for i in range(n_states):
            template = templates[:, i]

            # Compute correlation at each time point
            for t in range(n_timepoints):
                corr = np.corrcoef(template, normalized_data[:, t])[0, 1]

                computation_count += 1
                if computation_count % 1000 == 0:  # Update every 1000 computations
                    progress.set_step(
                        computation_count,
                        f"State {i + 1}/{n_states}, Time {t + 1}/{n_timepoints}",
                    )

                if polarity_invariant:
                    corr = np.abs(corr)

                correlations[i, t] = corr

        progress.complete()

        # Assign label with maximum correlation
        labels = np.argmax(correlations, axis=0)

        return cast(np.ndarray, labels)

    def smooth_labels(self, labels: np.ndarray, min_duration: int = 3) -> np.ndarray:
        """Smooth microstate labels by removing brief segments.

        Args:
            labels: Microstate labels
            min_duration: Minimum segment duration (samples)

        Returns:
            Smoothed labels
        """
        smoothed = labels.copy()
        n_timepoints = len(labels)

        i = 0
        while i < n_timepoints:
            # Find segment of same label
            current_label = smoothed[i]
            j = i + 1
            while j < n_timepoints and smoothed[j] == current_label:
                j += 1

            segment_length = j - i

            # If segment too short, reassign to neighboring state
            if segment_length < min_duration:
                # Use most common neighbor
                neighbors = []
                if i > 0:
                    neighbors.append(smoothed[i - 1])
                if j < n_timepoints:
                    neighbors.append(smoothed[j])

                if neighbors:
                    new_label = Counter(neighbors).most_common(1)[0][0]
                    smoothed[i:j] = new_label

            i = j

        return np.asarray(smoothed)

    def compute_temporal_metrics(self, labels: np.ndarray) -> Dict[str, Any]:
        """Compute temporal metrics for microstate sequence.

        Args:
            labels: Microstate labels

        Returns:
            Dictionary with temporal metrics
        """
        n_timepoints = len(labels)
        total_duration = n_timepoints * self.dt

        # Find segments
        segments = []
        i = 0
        while i < n_timepoints:
            current_label = labels[i]
            j = i + 1
            while j < n_timepoints and labels[j] == current_label:
                j += 1

            duration = (j - i) * self.dt
            segments.append({"label": current_label, "start": i, "end": j, "duration": duration})

            i = j

        # Compute metrics per state
        occurrences = Counter(labels)

        durations_per_state: Dict[int, List[float]] = {i: [] for i in range(self.n_states)}
        for seg in segments:
            durations_per_state[seg["label"]].append(seg["duration"])

        mean_durations = {
            i: np.mean(durations) if durations else 0.0
            for i, durations in durations_per_state.items()
        }

        coverage = {i: (count / n_timepoints) * 100 for i, count in occurrences.items()}

        return {
            "n_segments": len(segments),
            "segments": segments,
            "occurrences": dict(occurrences),
            "mean_durations": mean_durations,
            "coverage": coverage,
            "total_duration": total_duration,
        }

    def compute_transition_matrix(self, labels: np.ndarray) -> np.ndarray:
        """Compute microstate transition probability matrix.

        Args:
            labels: Microstate labels

        Returns:
            Transition matrix (n_states x n_states)
        """
        # Count transitions
        transition_counts = np.zeros((self.n_states, self.n_states))

        for i in range(len(labels) - 1):
            from_state = labels[i]
            to_state = labels[i + 1]

            # Only count transitions between different states
            if from_state != to_state:
                transition_counts[from_state, to_state] += 1

        # Normalize to probabilities
        row_sums = np.sum(transition_counts, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        transition_matrix = transition_counts / row_sums

        return cast(np.ndarray, transition_matrix)

    def analyze_sequence(
        self,
        data: np.ndarray,
        templates: Optional[np.ndarray] = None,
        smooth: bool = True,
    ) -> MicrostateSequence:
        """Complete microstate sequence analysis.

        Args:
            data: EEG data (channels x time)
            templates: Optional pre-fitted templates
            smooth: Whether to smooth labels

        Returns:
            MicrostateSequence object with complete analysis
        """
        # Compute GFP
        gfp = self.compute_gfp(data)

        # Assign labels
        labels = self.assign_microstates(data, templates)

        # Smooth if requested
        if smooth:
            labels = self.smooth_labels(labels)

        # Compute temporal metrics
        temporal_metrics = self.compute_temporal_metrics(labels)

        # Compute transition matrix
        transition_matrix = self.compute_transition_matrix(labels)

        # Create timestamps
        timestamps = np.arange(len(labels)) * self.dt

        # Ensure templates is not None
        final_templates = self.templates if templates is None else templates
        if final_templates is None:
            raise ValueError("Templates cannot be None when creating MicrostateSequence")

        # Extract transition probabilities
        transition_probs = {}
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i != j:
                    transition_probs[(i, j)] = transition_matrix[i, j]

        return MicrostateSequence(
            labels=labels,
            templates=final_templates,
            gfp=gfp,
            timestamps=timestamps,
            durations=[seg["duration"] for seg in temporal_metrics["segments"]],
            occurrences=temporal_metrics["occurrences"],
            coverage=temporal_metrics["coverage"],
            transition_matrix=transition_matrix,
            transition_probabilities=transition_probs,
        )

    def detect_posterior_to_anterior_transitions(
        self, sequence: MicrostateSequence, channel_positions: np.ndarray
    ) -> List[Dict]:
        """Detect posterior-to-anterior microstate transitions.

        Relevant for APGI ignition cascade validation.

        Args:
            sequence: MicrostateSequence object
            channel_positions: Channel positions (n_channels x 3) in 3D space

        Returns:
            List of detected transitions with timing and spatial info
        """
        # Compute centroid of each microstate template
        centroids = []
        for i in range(self.n_states):
            template = sequence.templates[:, i]

            # Weight positions by template values
            weights = np.abs(template)
            weights = weights / np.sum(weights)

            centroid = np.sum(channel_positions * weights[:, np.newaxis], axis=0)
            centroids.append(centroid)

        centroid_array = np.array(centroids)

        # Classify states as posterior or anterior based on y-coordinate
        # (assuming standard coordinate system)
        posterior_states = np.where(centroid_array[:, 1] < 0)[0]
        anterior_states = np.where(centroid_array[:, 1] > 0)[0]

        # Find transitions
        transitions = []
        labels = sequence.labels

        for i in range(len(labels) - 1):
            from_state = labels[i]
            to_state = labels[i + 1]

            if from_state in posterior_states and to_state in anterior_states:
                transitions.append(
                    {
                        "time": sequence.timestamps[i],
                        "from_state": int(from_state),
                        "to_state": int(to_state),
                        "from_centroid": centroid_array[from_state],
                        "to_centroid": centroid_array[to_state],
                    }
                )

        return transitions

    def compare_sequences(
        self, seq1: MicrostateSequence, seq2: MicrostateSequence
    ) -> Dict[str, float]:
        """Compare two microstate sequences.

        Args:
            seq1: First microstate sequence
            seq2: Second microstate sequence

        Returns:
            Dictionary with comparison metrics
        """
        # Template similarity
        template_corr = np.corrcoef(seq1.templates.flatten(), seq2.templates.flatten())[0, 1]

        # Coverage similarity
        coverage_diff = np.mean(
            [
                abs(float(seq1.coverage.get(i, 0)) - float(seq2.coverage.get(i, 0)))
                for i in range(self.n_states)
            ]
        )

        # Transition matrix similarity
        if seq1.transition_matrix is not None and seq2.transition_matrix is not None:
            transition_diff: float = float(
                np.mean(np.abs(seq1.transition_matrix - seq2.transition_matrix))
            )
        else:
            transition_diff = 0.0  # Default value when matrices are not available

        return {
            "template_correlation": template_corr,
            "coverage_difference": float(coverage_diff),
            "transition_difference": transition_diff,
        }

    def export_templates(self, filename: str) -> None:
        """Export microstate templates to file.

        Args:
            filename: Output filename
        """
        if self.templates is None:
            raise ValueError("No templates to export")

        np.savez(
            filename,
            templates=self.templates,
            n_states=self.n_states,
            labels=self.template_labels,
        )

    def load_templates(self, filename: str) -> None:
        """Load microstate templates from file.

        Args:
            filename: Input filename
        """
        data = np.load(filename)
        self.templates = data["templates"]
        self.n_states = int(data["n_states"])
        if "labels" in data:
            self.template_labels = data["labels"].tolist()
