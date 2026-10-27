"""
Gamma-band synchrony analysis for long-range coherence.

Provides cross-frequency coupling, frontal-posterior coherence,
and phase-amplitude coupling detection for APGI framework validation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal


@dataclass
class CoherenceMetrics:
    """Container for coherence analysis results."""

    # Coherence measures
    coherence: np.ndarray  # Coherence spectrum
    frequencies: np.ndarray  # Frequency bins

    # Gamma-band specific
    gamma_coherence: float  # Mean coherence in gamma band
    gamma_peak_freq: float  # Peak frequency in gamma band
    gamma_peak_coherence: float  # Peak coherence value

    # Phase metrics
    phase_lag: Optional[np.ndarray] = None
    phase_locking_value: Optional[float] = None

    # Cross-frequency coupling
    pac_strength: Optional[float] = None  # Phase-amplitude coupling
    cfc_modulation_index: Optional[float] = None

    # Metadata
    channel_pair: Optional[Tuple[str, str]] = None
    time_window: Optional[Tuple[float, float]] = None


@dataclass
class NetworkConnectivity:
    """Container for network-level connectivity metrics."""

    coherence_matrix: np.ndarray  # Channel x channel coherence
    channel_names: List[str]

    # Network metrics
    mean_connectivity: float
    frontal_posterior_coherence: float
    interhemispheric_coherence: float

    # Graph metrics
    clustering_coefficient: Optional[float] = None
    path_length: Optional[float] = None
    small_worldness: Optional[float] = None


class GammaSynchronyAnalysis:
    """
    Gamma-band synchrony analysis for APGI framework.

    Implements cross-frequency coupling, long-range coherence,
    and phase-amplitude coupling detection for ignition validation.
    """

    def __init__(self, sampling_rate: float = 1000.0):
        """
        Initialize gamma synchrony analysis.

        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2.0

        # Gamma band definition
        self.gamma_low = 30.0  # Hz
        self.gamma_high = 80.0  # Hz

    def extract_gamma_band(
        self,
        data: np.ndarray,
        low: Optional[float] = None,
        high: Optional[float] = None,
    ) -> np.ndarray:
        """
        Extract gamma-band signal using bandpass filter.

        Args:
            data: EEG data (channels x time)
            low: Low frequency cutoff (default: 30 Hz)
            high: High frequency cutoff (default: 80 Hz)

        Returns:
            Gamma-band filtered data
        """
        if low is None:
            low = self.gamma_low
        if high is None:
            high = self.gamma_high

        # Design bandpass filter
        sos = signal.butter(
            4, [low / self.nyquist, high / self.nyquist], btype="band", output="sos"
        )

        # Apply filter
        gamma_data = signal.sosfiltfilt(sos, data, axis=-1)

        return np.asarray(gamma_data)

    def compute_hilbert_transform(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute analytic signal using Hilbert transform.

        Args:
            data: Signal data

        Returns:
            Tuple of (amplitude envelope, instantaneous phase)
        """
        # Compute analytic signal
        analytic = signal.hilbert(data, axis=-1)

        # Extract amplitude and phase
        amplitude = np.abs(analytic)
        phase = np.angle(analytic)

        return amplitude, phase

    def compute_coherence(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude-squared coherence between two signals.

        Args:
            signal1: First signal
            signal2: Second signal
            nperseg: Length of each segment for FFT
            noverlap: Number of overlapping samples

        Returns:
            Tuple of (frequencies, coherence)
        """
        if noverlap is None:
            noverlap = nperseg // 2

        # Compute coherence
        freqs, coherence = signal.coherence(
            signal1, signal2, fs=self.sampling_rate, nperseg=nperseg, noverlap=noverlap
        )

        return freqs, coherence

    def compute_gamma_coherence(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        channel_pair: Optional[Tuple[str, str]] = None,
    ) -> CoherenceMetrics:
        """
        Compute gamma-band coherence between two signals.

        Args:
            signal1: First signal
            signal2: Second signal
            channel_pair: Optional channel names

        Returns:
            CoherenceMetrics object
        """
        # Compute full coherence spectrum
        freqs, coherence = self.compute_coherence(signal1, signal2)

        # Extract gamma band
        gamma_mask = (freqs >= self.gamma_low) & (freqs <= self.gamma_high)
        gamma_freqs = freqs[gamma_mask]
        gamma_coh = coherence[gamma_mask]

        # Compute gamma metrics
        gamma_coherence = np.mean(gamma_coh)
        peak_idx = np.argmax(gamma_coh)
        gamma_peak_freq = gamma_freqs[peak_idx]
        gamma_peak_coherence = gamma_coh[peak_idx]

        # Compute phase lag
        phase_lag = self._compute_phase_lag(signal1, signal2, freqs)

        # Compute phase locking value
        plv = self._compute_phase_locking_value(signal1, signal2)

        return CoherenceMetrics(
            coherence=coherence,
            frequencies=freqs,
            gamma_coherence=gamma_coherence,
            gamma_peak_freq=gamma_peak_freq,
            gamma_peak_coherence=gamma_peak_coherence,
            phase_lag=phase_lag,
            phase_locking_value=plv,
            channel_pair=channel_pair,
        )

    def _compute_phase_lag(
        self, signal1: np.ndarray, signal2: np.ndarray, freqs: np.ndarray
    ) -> np.ndarray:
        """
        Compute phase lag between signals.

        Args:
            signal1: First signal
            signal2: Second signal
            freqs: Frequency bins

        Returns:
            Phase lag at each frequency
        """
        # Compute cross-spectral density
        freqs_csd, csd = signal.csd(signal1, signal2, fs=self.sampling_rate, nperseg=256)

        # Phase lag is angle of cross-spectrum
        phase_lag = np.angle(csd)

        return np.asarray(phase_lag)

    def _compute_phase_locking_value(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """
        Compute phase locking value (PLV) between signals.

        Args:
            signal1: First signal
            signal2: Second signal

        Returns:
            Phase locking value (0-1)
        """
        # Extract gamma band
        gamma1 = self.extract_gamma_band(signal1[np.newaxis, :])[0]
        gamma2 = self.extract_gamma_band(signal2[np.newaxis, :])[0]

        # Compute phases
        _, phase1 = self.compute_hilbert_transform(gamma1)
        _, phase2 = self.compute_hilbert_transform(gamma2)

        # Phase difference
        phase_diff = phase1 - phase2

        # PLV is magnitude of mean complex phase difference
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))

        return float(plv)

    def compute_phase_amplitude_coupling(
        self,
        low_freq_signal: np.ndarray,
        high_freq_signal: np.ndarray,
        phase_band: Tuple[float, float] = (4, 8),
        amp_band: Tuple[float, float] = (30, 80),
    ) -> float:
        """
        Compute phase-amplitude coupling (PAC).

        Measures coupling between phase of low-frequency oscillation
        and amplitude of high-frequency oscillation.

        Args:
            low_freq_signal: Signal for phase extraction
            high_freq_signal: Signal for amplitude extraction
            phase_band: Frequency band for phase (Hz)
            amp_band: Frequency band for amplitude (Hz)

        Returns:
            PAC strength (modulation index)
        """
        # Filter signals
        sos_phase = signal.butter(
            4,
            [phase_band[0] / self.nyquist, phase_band[1] / self.nyquist],
            btype="band",
            output="sos",
        )
        phase_signal = signal.sosfiltfilt(sos_phase, low_freq_signal)

        sos_amp = signal.butter(
            4,
            [amp_band[0] / self.nyquist, amp_band[1] / self.nyquist],
            btype="band",
            output="sos",
        )
        amp_signal = signal.sosfiltfilt(sos_amp, high_freq_signal)

        # Extract phase and amplitude
        _, phase = self.compute_hilbert_transform(phase_signal)
        amplitude, _ = self.compute_hilbert_transform(amp_signal)

        # Compute modulation index using Kullback-Leibler divergence
        n_bins = 18  # 20-degree bins
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)

        # Compute mean amplitude in each phase bin
        mean_amp_per_bin = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
            if np.any(mask):
                mean_amp_per_bin[i] = np.mean(amplitude[mask])

        # Normalize to probability distribution
        p = mean_amp_per_bin / np.sum(mean_amp_per_bin)

        # Uniform distribution
        q = np.ones(n_bins) / n_bins

        # KL divergence (modulation index)
        # Avoid log(0)
        p[p == 0] = 1e-10
        mi = np.sum(p * np.log(p / q)) / np.log(n_bins)

        return float(mi)

    def compute_cross_frequency_coupling(
        self,
        data: np.ndarray,
        phase_freqs: List[float] = [4, 8, 13],
        amp_freqs: List[float] = [30, 50, 70],
    ) -> np.ndarray:
        """
        Compute cross-frequency coupling matrix.

        Args:
            data: EEG signal
            phase_freqs: Center frequencies for phase extraction
            amp_freqs: Center frequencies for amplitude extraction

        Returns:
            CFC matrix (phase_freqs x amp_freqs)
        """
        cfc_matrix = np.zeros((len(phase_freqs), len(amp_freqs)))

        for i, phase_freq in enumerate(phase_freqs):
            # Phase band: ±2 Hz around center
            phase_band = (phase_freq - 2, phase_freq + 2)

            for j, amp_freq in enumerate(amp_freqs):
                # Amplitude band: ±10 Hz around center
                amp_band = (amp_freq - 10, amp_freq + 10)

                # Compute PAC
                pac = self.compute_phase_amplitude_coupling(data, data, phase_band, amp_band)
                cfc_matrix[i, j] = pac

        return cfc_matrix

    def compute_frontal_posterior_coherence(
        self,
        data: np.ndarray,
        frontal_channels: List[int],
        posterior_channels: List[int],
    ) -> float:
        """
        Compute mean gamma coherence between frontal and posterior regions.

        Args:
            data: EEG data (channels x time)
            frontal_channels: Indices of frontal channels
            posterior_channels: Indices of posterior channels

        Returns:
            Mean frontal-posterior gamma coherence
        """
        coherences = []

        for f_ch in frontal_channels:
            for p_ch in posterior_channels:
                metrics = self.compute_gamma_coherence(data[f_ch], data[p_ch])
                coherences.append(metrics.gamma_coherence)

        return float(np.mean(coherences))

    def compute_connectivity_matrix(
        self, data: np.ndarray, channel_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Compute full connectivity matrix across all channels.

        Args:
            data: EEG data (channels x time)
            channel_names: Optional channel names

        Returns:
            Connectivity matrix (channels x channels)
        """
        n_channels = data.shape[0]
        connectivity = np.zeros((n_channels, n_channels))

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                metrics = self.compute_gamma_coherence(data[i], data[j])
                connectivity[i, j] = metrics.gamma_coherence
                connectivity[j, i] = metrics.gamma_coherence

        # Diagonal is 1 (self-coherence)
        np.fill_diagonal(connectivity, 1.0)

        return connectivity

    def analyze_network_connectivity(
        self,
        data: np.ndarray,
        channel_names: List[str],
        frontal_channels: List[int],
        posterior_channels: List[int],
        left_channels: Optional[List[int]] = None,
        right_channels: Optional[List[int]] = None,
    ) -> NetworkConnectivity:
        """
        Comprehensive network connectivity analysis.

        Args:
            data: EEG data (channels x time)
            channel_names: Channel names
            frontal_channels: Frontal channel indices
            posterior_channels: Posterior channel indices
            left_channels: Left hemisphere channel indices
            right_channels: Right hemisphere channel indices

        Returns:
            NetworkConnectivity object
        """
        # Compute connectivity matrix
        connectivity = self.compute_connectivity_matrix(data, channel_names)

        # Mean connectivity
        n_channels = len(channel_names)
        mean_connectivity = np.sum(connectivity) / (n_channels * (n_channels - 1))

        # Frontal-posterior coherence
        fp_coherence = self.compute_frontal_posterior_coherence(
            data, frontal_channels, posterior_channels
        )

        # Interhemispheric coherence
        if left_channels and right_channels:
            inter_coherences = []
            for l_ch in left_channels:
                for r_ch in right_channels:
                    inter_coherences.append(connectivity[l_ch, r_ch])
            interhemispheric = np.mean(inter_coherences)
        else:
            interhemispheric = 0.0

        # Graph metrics
        clustering = self._compute_clustering_coefficient(connectivity)
        path_length = self._compute_path_length(connectivity)

        # Small-worldness
        if clustering > 0 and path_length > 0:
            # Compare to random network (simplified)
            random_clustering = mean_connectivity
            random_path_length = np.log(n_channels) / np.log(mean_connectivity * n_channels)
            small_worldness = (clustering / random_clustering) / (path_length / random_path_length)
        else:
            small_worldness = None

        return NetworkConnectivity(
            coherence_matrix=connectivity,
            channel_names=channel_names,
            mean_connectivity=mean_connectivity,
            frontal_posterior_coherence=fp_coherence,
            interhemispheric_coherence=interhemispheric,
            clustering_coefficient=clustering,
            path_length=path_length,
            small_worldness=small_worldness,
        )

    def _compute_clustering_coefficient(
        self, connectivity: np.ndarray, threshold: float = 0.5
    ) -> float:
        """
        Compute clustering coefficient of connectivity network.

        Args:
            connectivity: Connectivity matrix
            threshold: Threshold for binarizing connections

        Returns:
            Mean clustering coefficient
        """
        # Binarize
        binary_net = (connectivity > threshold).astype(int)
        np.fill_diagonal(binary_net, 0)

        n_nodes = binary_net.shape[0]
        clustering_coeffs = []

        for i in range(n_nodes):
            # Find neighbors
            neighbors = np.where(binary_net[i] > 0)[0]
            k = len(neighbors)

            if k < 2:
                continue

            # Count connections among neighbors
            neighbor_connections = 0
            for j in range(len(neighbors)):
                for neighbor_idx in range(j + 1, len(neighbors)):
                    if binary_net[neighbors[j], neighbors[neighbor_idx]] > 0:
                        neighbor_connections += 1

            # Clustering coefficient for node i
            cc = (2 * neighbor_connections) / (k * (k - 1))
            clustering_coeffs.append(cc)

        return float(np.mean(clustering_coeffs)) if clustering_coeffs else 0.0

    def _compute_path_length(self, connectivity: np.ndarray, threshold: float = 0.5) -> float:
        """
        Compute characteristic path length of connectivity network.

        Args:
            connectivity: Connectivity matrix
            threshold: Threshold for binarizing connections

        Returns:
            Mean shortest path length
        """
        # Binarize
        binary_net = (connectivity > threshold).astype(int)
        np.fill_diagonal(binary_net, 0)

        n_nodes = binary_net.shape[0]

        # Compute shortest paths using Floyd-Warshall
        dist = np.where(binary_net > 0, 1, np.inf)
        np.fill_diagonal(dist, 0)

        for k in range(n_nodes):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if dist[i, j] > dist[i, k] + dist[k, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        # Mean path length (excluding infinite distances)
        finite_dists = dist[np.isfinite(dist) & (dist > 0)]

        return float(np.mean(finite_dists)) if len(finite_dists) > 0 else 0.0

    def detect_gamma_bursts(
        self, data: np.ndarray, threshold: float = 2.0, min_duration: float = 50.0
    ) -> List[Dict]:
        """
        Detect gamma-band burst events.

        Args:
            data: EEG signal
            threshold: Threshold in standard deviations above mean
            min_duration: Minimum burst duration in ms

        Returns:
            List of detected bursts with timing and amplitude info
        """
        # Extract gamma band
        gamma = self.extract_gamma_band(data[np.newaxis, :])[0]

        # Compute amplitude envelope
        amplitude, _ = self.compute_hilbert_transform(gamma)

        # Threshold
        mean_amp = np.mean(amplitude)
        std_amp = np.std(amplitude)
        threshold_value = mean_amp + threshold * std_amp

        # Detect bursts
        above_threshold = amplitude > threshold_value

        # Find burst segments
        bursts = []
        in_burst = False
        burst_start = 0

        dt = 1000.0 / self.sampling_rate  # ms per sample

        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_burst:
                # Burst onset
                in_burst = True
                burst_start = i
            elif not above_threshold[i] and in_burst:
                # Burst offset
                in_burst = False
                burst_duration = (i - burst_start) * dt

                if burst_duration >= min_duration:
                    bursts.append(
                        {
                            "start_time": burst_start * dt,
                            "end_time": i * dt,
                            "duration": burst_duration,
                            "peak_amplitude": np.max(amplitude[burst_start:i]),
                            "mean_amplitude": np.mean(amplitude[burst_start:i]),
                        }
                    )

        return bursts
