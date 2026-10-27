"""Metabolic Calibration Module for APGI Framework.

This module provides ground-truth calibration of metabolic cost coefficients (c_1, c_2)
using high-resolution metabolic imaging datasets:
- Two-photon metabolic imaging (iATPSnFR2, ATeam sensors): 100-300 ms temporal resolution
- Phosphorus Magnetic Resonance Spectroscopy (31P-MRS/31P-fMRS): Bulk ATP turnover measurements

The c_1 coefficient represents dynamic/signaling costs (ATP per ignition event),
driven by ion pumping and glutamate recycling during neural activity bursts.

The c_2 coefficient represents static/maintenance costs (ATP per ms per neuron),
covering resting potential maintenance, protein synthesis, and baseline metabolism.

References:
-----------
- Marvin et al. (2024): iATPSnFR2 sensor - Nature Communications/PNAS
- Díaz-García et al. (2017): ATeam/Peredox sensors - Cell Metabolism
- Chen et al. (2020): 31P-fMRS at 3T/7T/9.4T
- Attwell & Laughlin (2001): Energy budget for signaling in grey matter
- Lennie (2003): Cost of cortical computation

Usage:
------
    # Load Two-photon dataset for c_1 calibration
    loader = TwoPhotonDatasetLoader.from_zenodo("10.5281/zenodo.xxxxx")
    traces = loader.load_traces()
    c1 = loader.compute_c1_ground_truth(traces)

    # Load P-MRS dataset for c_2 calibration
    pmrs_loader = PMRSDatasetLoader.from_parrec("ck_flux_data.parrec")
    c2 = pmrs_loader.compute_c2_ground_truth()

    # Create calibrated metabolic layer
    calibrator = MetabolicCalibrator(c1=c1, c2=c2)
    vml = CalibratedVirtualMetabolicLayer(calibrator=calibrator)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.integrate import trapezoid

# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class CalibratedCostCoefficients:
    """Ground-truth cost coefficients from high-resolution metabolic imaging.

    These coefficients partition ATP consumption into:
    - c_1 (dynamic): Activity-driven costs from ion pumping, glutamate recycling
    - c_2 (static): Baseline maintenance costs (resting potential, protein synthesis)

    Attributes:
        c_1_dynamic: ATP molecules per ignition event per neuron
        c_2_static: ATP molecules per millisecond per neuron
        c_1_uncertainty: Standard error of c_1 estimate from fit
        c_2_uncertainty: Standard error of c_2 estimate from fit
        calibration_source: Origin of calibration data
        temporal_resolution_ms: Actual temporal resolution of source data
        reference_temperature_c: Temperature at which measurements taken
        dataset_doi: Digital Object Identifier of source dataset

    Biological Ranges (for validation):
        - c_1: ~10^7 to 10^8 ATP per AP (glutamate recycling + ion pumping)
        - c_2: ~10^8 to 10^10 ATP/s per neuron (baseline metabolism)
        - Ratio c_1/c_2 per ms: ~0.01 to 0.1 (matches 5-20 Hz firing rates)
    """

    c_1_dynamic: float  # ATP molecules per ignition event per neuron
    c_2_static: float  # ATP molecules per ms per neuron

    # Uncertainty quantification
    c_1_uncertainty: float = 0.0
    c_2_uncertainty: float = 0.0

    # Provenance
    calibration_source: Literal["two_photon", "p_mrs", "combined", "literature"] = "literature"
    temporal_resolution_ms: float = 0.0
    reference_temperature_c: float = 37.0
    dataset_doi: str = ""

    def __post_init__(self) -> None:
        """Validate coefficient values against biological plausibility."""
        if self.c_1_dynamic < 0 or self.c_2_static < 0:
            raise ValueError("Cost coefficients must be non-negative")

    @property
    def total_baseline_per_second(self) -> float:
        """Total baseline ATP consumption per neuron per second."""
        return self.c_2_static * 1000.0  # Convert ms to seconds

    @property
    def cost_per_action_potential(self) -> float:
        """ATP cost per action potential (ignition-normalized)."""
        return self.c_1_dynamic

    def compute_expected_atp(
        self,
        num_ignitions: float,
        duration_ms: float,
        num_neurons: int,
    ) -> Tuple[float, float, float]:
        """Compute expected ATP consumption from c_1 and c_2.

        Returns:
            (total_atp, dynamic_component, static_component)
        """
        dynamic_atp = self.c_1_dynamic * num_ignitions * num_neurons
        static_atp = self.c_2_static * duration_ms * num_neurons
        return dynamic_atp + static_atp, dynamic_atp, static_atp


@dataclass
class TwoPhotonTrace:
    """Single trace from Two-photon metabolic imaging (e.g., iATPSnFR2, ATeam).

    Attributes:
        fluorescence: Raw fluorescence signal (ΔF/F0 or absolute)
        time_ms: Time vector in milliseconds
        cell_id: Identifier for the imaged cell/synapse
        stimulation_times_ms: Times of electrical/optical stimulation
        sampling_rate_hz: Actual sampling rate of the recording
        temperature_c: Temperature during recording
        sensor_type: Type of fluorescent sensor used
    """

    fluorescence: npt.NDArray[np.float64]
    time_ms: npt.NDArray[np.float64]
    cell_id: str = ""
    stimulation_times_ms: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    sampling_rate_hz: float = 10.0  # Default: 10 Hz = 100 ms resolution
    temperature_c: float = 37.0
    sensor_type: Literal["iATPSnFR2", "ATeam", "Peredox", "unknown"] = "unknown"

    def __post_init__(self) -> None:
        """Validate trace dimensions."""
        if len(self.fluorescence) != len(self.time_ms):
            raise ValueError("Fluorescence and time vectors must have same length")

    @property
    def temporal_resolution_ms(self) -> float:
        """Actual temporal resolution of the recording."""
        if len(self.time_ms) < 2:
            return 1000.0 / self.sampling_rate_hz
        return float(np.mean(np.diff(self.time_ms)))

    def get_baseline_fluorescence(self, window_ms: float = 1000.0) -> float:
        """Compute baseline fluorescence from pre-stimulation period."""
        if len(self.stimulation_times_ms) == 0 or self.stimulation_times_ms[0] == 0:
            # Use first window_ms if no stimulation or stimulation at 0
            end_idx = int(window_ms / self.temporal_resolution_ms)
            return float(np.mean(self.fluorescence[:end_idx]))

        # Use period before first stimulation
        pre_stim_mask = self.time_ms < self.stimulation_times_ms[0]
        if np.any(pre_stim_mask):
            return float(np.mean(self.fluorescence[pre_stim_mask]))
        return float(np.mean(self.fluorescence))

    def detect_bursts(
        self,
        threshold_std: float = 3.0,
        min_duration_ms: float = 50.0,
    ) -> List[Dict[str, Any]]:
        """Detect bursts (ignition events) from fluorescence signal.

        Uses threshold crossing on fluorescence signal to identify
        periods of elevated ATP consumption.

        Returns:
            List of burst dictionaries with start_ms, end_ms, peak_f, integrated_atp
        """
        baseline = self.get_baseline_fluorescence()
        threshold = baseline + threshold_std * np.std(self.fluorescence)

        # Find threshold crossings
        above_threshold = self.fluorescence > threshold

        if not np.any(above_threshold):
            return []

        # Find contiguous regions
        changes = np.diff(above_threshold.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        # Handle edge cases
        if above_threshold[0]:
            starts = np.insert(starts, 0, 0)
        if above_threshold[-1]:
            ends = np.append(ends, len(self.fluorescence))

        bursts = []
        for start_idx, end_idx in zip(starts, ends):
            duration_ms = (end_idx - start_idx) * self.temporal_resolution_ms

            if duration_ms < min_duration_ms:
                continue

            burst_slice = self.fluorescence[start_idx:end_idx]

            bursts.append(
                {
                    "start_ms": float(self.time_ms[start_idx]),
                    "end_ms": float(self.time_ms[end_idx - 1]),
                    "duration_ms": duration_ms,
                    "peak_f": float(np.max(burst_slice)),
                    "mean_f": float(np.mean(burst_slice)),
                    "integrated_signal": float(
                        trapezoid(burst_slice - baseline, self.time_ms[start_idx:end_idx])
                    ),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                }
            )

        return bursts


@dataclass
class PMRSSpectrum:
    """31P-MRS spectrum with ATP/PCr/PCi peaks.

    Attributes:
        chemical_shift_ppm: Chemical shift axis in ppm
        intensity: Signal intensity (arbitrary units)
        field_strength_t: Magnetic field strength in Tesla
        sequence_type: MRS sequence used
        tr_ms: Repetition time in milliseconds
        te_ms: Echo time in milliseconds
        temperature_c: Sample temperature
    """

    chemical_shift_ppm: npt.NDArray[np.float64]
    intensity: npt.NDArray[np.float64]
    field_strength_t: float = 7.0
    sequence_type: Literal["fid", "steam", "press", "csi", "unknown"] = "unknown"
    tr_ms: float = 1000.0
    te_ms: float = 0.0
    temperature_c: float = 37.0

    # Peak assignments (typical chemical shifts at 7T)
    ATP_PEAKS_PPM = {
        "gamma": -2.8,
        "alpha": -7.5,
        "beta": -16.0,
    }
    PCr_PEAK_PPM = -3.0
    PCi_PEAK_PPM = 3.0

    def find_peaks(
        self,
        peak_ppm: float,
        window_ppm: float = 0.5,
    ) -> Tuple[float, float, int]:
        """Find peak position, amplitude, and index.

        Args:
            peak_ppm: Expected chemical shift of the peak
            window_ppm: Search window around expected position

        Returns:
            (peak_ppm_found, peak_amplitude, peak_index)
        """
        mask = np.abs(self.chemical_shift_ppm - peak_ppm) < window_ppm
        if not np.any(mask):
            return peak_ppm, 0.0, 0

        local_intensity = self.intensity[mask]
        local_ppm = self.chemical_shift_ppm[mask]

        peak_idx = np.argmax(local_intensity)
        return (
            float(local_ppm[peak_idx]),
            float(local_intensity[peak_idx]),
            int(np.where(mask)[0][peak_idx]),
        )

    def compute_atp_pcr_ratio(self) -> float:
        """Compute ATP/PCr ratio for metabolic state assessment."""
        _, atp_amp, _ = self.find_peaks(self.ATP_PEAKS_PPM["beta"])
        _, pcr_amp, _ = self.find_peaks(self.PCr_PEAK_PPM)

        if pcr_amp == 0:
            return 0.0
        return atp_amp / pcr_amp


# =============================================================================
# Dataset Loaders
# =============================================================================


class DatasetLoader(Protocol):
    """Protocol for metabolic dataset loaders."""

    def load(self, source: Union[str, Path]) -> Any:
        """Load dataset from source."""
        ...

    def validate(self, data: Any) -> bool:
        """Validate loaded data integrity."""
        ...


class TwoPhotonDatasetLoader:
    """Load and parse Two-photon metabolic imaging datasets.

    Supports datasets from:
    - Looger Lab (Janelia): iATPSnFR2, iATPSnFR1
    - Yellen Lab (Harvard): ATeam, Peredox sensors
    - Díaz-García et al. (Harvard): Combined NADH/ATP measurements

    Expected formats:
    - Zenodo/Dryad repositories (zip/tar.gz with CSV/TIFF/HDF5)
    - CSV files with columns: time_ms, fluorescence, cell_id, stimulation
    - HDF5 files with /traces, /metadata, /stimulation groups
    - TIFF stacks (raw imaging data)

    Processing pipeline:
    1. Load fluorescence traces (ΔF/F0 or absolute)
    2. Detect action potential clusters (bursts)
    3. Fit: ATP(t) = c_1 × N_AP(t) + c_2 × t + noise
    4. Return c_1 in ATP molecules per AP

    Example:
        loader = TwoPhotonDatasetLoader()
        traces = loader.load_from_zenodo("10.5281/zenodo.xxxxx")
        c1 = loader.compute_c1_ground_truth(traces, sensor_calibration=15000.0)
    """

    # Sensor calibration factors (ATP molecules per unit fluorescence)
    # These convert ΔF/F0 to absolute ATP molecule counts
    SENSOR_CALIBRATION = {
        "iATPSnFR2": 15000.0,  # ~15K ATP per 1% ΔF/F0 (estimated from Kd ~4 mM)
        "iATPSnFR1": 18000.0,  # Slightly higher Kd
        "ATeam": 20000.0,  # FRET-based, higher dynamic range
        "Peredox": 12000.0,  # NADH sensor, indirect ATP proxy
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir or Path.home() / ".apgi" / "datasets" / "two_photon"  # fmt: skip
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_traces: List[TwoPhotonTrace] = []

    @classmethod
    def from_zenodo(cls, doi: str, cache_dir: Optional[Path] = None) -> TwoPhotonDatasetLoader:
        """Create loader from Zenodo DOI.

        Args:
            doi: Zenodo DOI (e.g., "10.5281/zenodo.12345")
            cache_dir: Optional custom cache directory

        Returns:
            Configured loader with downloaded dataset
        """
        loader = cls(cache_dir=cache_dir)
        loader.download_zenodo(doi)
        return loader

    def download_zenodo(self, doi: str) -> Path:
        """Download dataset from Zenodo.

        Args:
            doi: Zenodo DOI

        Returns:
            Path to downloaded/extracted dataset
        """
        import urllib.request
        import zipfile

        # Parse DOI to get record ID
        record_id = doi.split(".")[-1] if "." in doi else doi.split("/")[-1]

        download_url = f"https://zenodo.org/record/{record_id}/files/dataset.zip"
        zip_path = self.cache_dir / f"zenodo_{record_id}.zip"

        if not zip_path.exists():
            try:
                urllib.request.urlretrieve(
                    download_url, zip_path
                )  # nosec B310 - fixed https://zenodo.org host
            except Exception as e:
                warnings.warn(f"Failed to download from Zenodo: {e}")
                # Return empty path for offline mode
                return self.cache_dir

        # Extract
        extract_dir = self.cache_dir / f"zenodo_{record_id}"
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        return extract_dir

    def load_traces(
        self,
        source: Union[str, Path, List[Path]],
        file_pattern: str = "*.csv",
        sensor_type: Literal["iATPSnFR2", "ATeam", "Peredox", "unknown"] = "unknown",
    ) -> List[TwoPhotonTrace]:
        """Load fluorescence traces from CSV/HDF5 files.

        Args:
            source: Directory path, file path, or list of file paths
            file_pattern: Glob pattern to match trace files
            sensor_type: Type of sensor used in recording

        Returns:
            List of TwoPhotonTrace objects
        """
        self._loaded_traces = []

        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if source_path.is_dir():
                files = list(source_path.glob(file_pattern))
            else:
                files = [source_path]
        else:
            files = [Path(f) for f in source]

        for file_path in files:
            try:
                trace = self._load_single_file(file_path, sensor_type)
                self._loaded_traces.append(trace)
            except Exception as e:
                warnings.warn(f"Failed to load {file_path}: {e}")

        return self._loaded_traces

    def _load_single_file(
        self,
        file_path: Path,
        sensor_type: Literal["iATPSnFR2", "ATeam", "Peredox", "unknown"],
    ) -> TwoPhotonTrace:
        """Load a single trace file."""
        import pandas as pd

        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(file_path)

            # Standardize column names
            col_map = {
                "time": "time_ms",
                "time_s": "time_ms",
                "fluorescence": "fluorescence",
                "f": "fluorescence",
                "delta_f_f0": "fluorescence",
                "dff": "fluorescence",
            }

            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            time_ms_arr: npt.NDArray[np.float64] = np.asarray(df["time_ms"], dtype=np.float64)
            if np.max(time_ms_arr) < 100:  # Likely in seconds
                time_ms_arr = time_ms_arr * 1000.0

            fluorescence_arr: npt.NDArray[np.float64] = np.asarray(
                df["fluorescence"], dtype=np.float64
            )

            # Extract stimulation times if available
            stim_times_arr: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
            if "stimulation" in df.columns or "stim" in df.columns:
                stim_col = "stimulation" if "stimulation" in df.columns else "stim"
                stim_mask: npt.NDArray[np.bool_] = np.asarray(df[stim_col] > 0, dtype=np.bool_)
                stim_times_arr = time_ms_arr[stim_mask]

            # Compute sampling rate
            if len(time_ms_arr) > 1:
                dt_ms = float(np.mean(np.diff(time_ms_arr)))
                sampling_rate_hz = 1000.0 / dt_ms
            else:
                sampling_rate_hz = 10.0

            return TwoPhotonTrace(
                fluorescence=fluorescence_arr,
                time_ms=time_ms_arr,
                cell_id=file_path.stem,
                stimulation_times_ms=stim_times_arr,
                sampling_rate_hz=sampling_rate_hz,
                sensor_type=sensor_type,
            )

        elif suffix in [".h5", ".hdf5"]:
            import h5py

            with h5py.File(file_path, "r") as f:
                # Try standard structure
                if "traces" in f:
                    traces = f["traces"]
                    fluorescence = traces["fluorescence"][:]
                    time_ms = traces["time_ms"][:] * 1000.0  # Convert to ms

                    stim_times = np.array([])
                    if "stimulation" in traces:
                        stim_times = time_ms[traces["stimulation"][:] > 0]

                    # Get sensor type from metadata if available
                    if "metadata" in f and "sensor_type" in f["metadata"]:
                        sensor_type = f["metadata"]["sensor_type"][()].decode()

                    return TwoPhotonTrace(
                        fluorescence=fluorescence.astype(np.float64),
                        time_ms=time_ms.astype(np.float64),
                        cell_id=file_path.stem,
                        stimulation_times_ms=stim_times.astype(np.float64),
                        sensor_type=sensor_type,
                    )
                else:
                    raise ValueError(f"Unknown HDF5 structure in {file_path}")

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def compute_c1_ground_truth(
        self,
        traces: Optional[List[TwoPhotonTrace]] = None,
        sensor_calibration: Optional[float] = None,
        ap_rate_hz: float = 10.0,  # Assumed AP rate during burst
        fit_method: Literal["linear", "integral", "regression"] = "integral",
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Compute c_1 coefficient from Two-photon traces.

        Model: ΔATP(t) = c_1 × N_AP(t) + noise

        Where:
        - N_AP(t) is the number of action potentials up to time t
        - c_1 is ATP molecules consumed per action potential

        Args:
            traces: List of traces (uses loaded traces if None)
            sensor_calibration: ATP molecules per unit fluorescence (auto-detect if None)
            ap_rate_hz: Assumed action potential rate during stimulation
            fit_method: Method to estimate c_1

        Returns:
            (c_1_mean, c_1_std, metadata_dict)
        """
        if traces is None:
            traces = self._loaded_traces

        if not traces:
            raise ValueError("No traces loaded. Call load_traces() first.")

        c1_estimates = []
        burst_statistics = []

        for trace in traces:
            # Auto-detect sensor calibration
            if sensor_calibration is None:
                calibration = self.SENSOR_CALIBRATION.get(trace.sensor_type, 15000.0)
            else:
                calibration = sensor_calibration

            # Detect bursts
            bursts = trace.detect_bursts()

            for burst in bursts:
                # Convert integrated fluorescence to ATP molecules
                # integrated_signal is in (ΔF/F0) × ms units
                atp_consumed = burst["integrated_signal"] * calibration

                # Estimate number of APs during burst
                num_aps = burst["duration_ms"] / 1000.0 * ap_rate_hz

                if num_aps > 0:
                    c1_estimate = atp_consumed / num_aps
                    c1_estimates.append(c1_estimate)

                    burst_statistics.append(
                        {
                            "cell_id": trace.cell_id,
                            "duration_ms": burst["duration_ms"],
                            "atp_consumed": atp_consumed,
                            "num_aps": num_aps,
                            "c1_estimate": c1_estimate,
                        }
                    )

        if not c1_estimates:
            warnings.warn("No bursts detected in traces. Returning default c_1.")
            return 1.5e7, 0.0, {"method": "default", "num_bursts": 0}

        c1_array = np.array(c1_estimates)
        c1_mean = float(np.median(c1_array))  # Median for robustness
        c1_std = float(np.std(c1_array))

        metadata = {
            "method": fit_method,
            "num_bursts": len(c1_estimates),
            "num_cells": len(set(b["cell_id"] for b in burst_statistics)),
            "c1_mean": c1_mean,
            "c1_std": c1_std,
            "c1_min": float(np.min(c1_array)),
            "c1_max": float(np.max(c1_array)),
            "sensor_calibration_used": calibration,
            "ap_rate_assumed_hz": ap_rate_hz,
            "burst_statistics": burst_statistics,
        }

        return c1_mean, c1_std, metadata


class PMRSDatasetLoader:
    """Load and parse 31P-Magnetic Resonance Spectroscopy datasets.

    Supports datasets from:
    - Chen et al. (2020): 3T functional MRS
    - University of Nottingham: 7T 31P-fMRS
    - University of Minnesota: 9.4T ultra-high field

    Expected formats:
    - Philips PAR/REC files (.par/.rec)
    - Siemens TWIX files (.dat)
    - NIfTI-MRS (.nii.gz with MRS extension)
    - LCModel output files (.csv/.txt)
    - CSV with pre-computed flux values

    Processing:
    1. Load 31P spectra with ATP/PCr/PCi peaks
    2. Compute Creatine Kinase forward flux (PCr → ATP)
    3. Calculate baseline ATP synthesis rate
    4. Map to c_2 = baseline_ATP_rate / num_voxels / temporal_resolution

    Example:
        loader = PMRSDatasetLoader()
        spectra = loader.load_parrec("subject_001.par")
        c2 = loader.compute_c2_ground_truth(spectra, voxel_volume_ml=8.0)
    """

    # Physical constants
    ATP_MOLECULES_PER_UMOL = 6.022e17  # Avogadro's number / 1e6 (μmol to molecules)
    T1_ATP_S = 3.0  # T1 relaxation time for ATP at 7T (seconds)

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the P-MRS loader.

        Args:
            cache_dir: Directory to cache loaded datasets
        """
        self.cache_dir = cache_dir or Path.home() / ".apgi" / "datasets" / "pmrs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_spectra: List[PMRSSpectrum] = []

    def load_parrec(
        self,
        par_path: Union[str, Path],
        rec_path: Optional[Union[str, Path]] = None,
    ) -> List[PMRSSpectrum]:
        """Load Philips PAR/REC format 31P-MRS data.

        Args:
            par_path: Path to .par header file
            rec_path: Path to .rec data file (auto-detected if None)

        Returns:
            List of PMRSSpectrum objects (one per time point/dynamic)
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel required for PAR/REC loading. Install: pip install nibabel")

        par_path = Path(par_path)

        if rec_path is None:
            rec_path = par_path.with_suffix(".rec")
        else:
            rec_path = Path(rec_path)

        # Load with nibabel
        img = nib.parrec.load(par_path)
        data = img.get_fdata()

        # Extract header info
        hdr = img.header
        field_strength = getattr(hdr, "magnetic_field_strength", 7.0)

        # PAR/REC stores MRS as (freq, x, y, z, dynamics) or similar
        # Reshape to extract spectra
        if data.ndim >= 4:
            # Multi-voxel, multi-dynamic data
            n_freq = data.shape[0]
            n_dynamics = data.shape[-1] if data.ndim > 4 else 1

            # Create ppm axis (simplified - would need proper calibration)
            # Typical 31P spectral width: -25 to 10 ppm
            ppm = np.linspace(-25, 10, n_freq)

            spectra = []
            for dyn in range(n_dynamics):
                if data.ndim > 4:
                    # Average across spatial dimensions for single voxel
                    spectrum_data = np.mean(data[..., dyn], axis=(1, 2, 3))
                else:
                    spectrum_data = data[..., dyn]

                spectra.append(
                    PMRSSpectrum(
                        chemical_shift_ppm=ppm,
                        intensity=spectrum_data.astype(np.float64),
                        field_strength_t=float(field_strength),
                        sequence_type="fid",
                    )
                )
        else:
            # Single spectrum
            n_freq = data.shape[0]
            ppm = np.linspace(-25, 10, n_freq)
            spectra = [
                PMRSSpectrum(
                    chemical_shift_ppm=ppm,
                    intensity=data.astype(np.float64),
                    field_strength_t=float(field_strength),
                    sequence_type="fid",
                )
            ]

        self._loaded_spectra = spectra
        return spectra

    def load_csv_flux(
        self,
        csv_path: Union[str, Path],
        ck_flux_column: str = "CK_flux_umol_g_min",
        atp_synthase_column: str = "ATP_synthase_umol_g_min",
    ) -> Dict[str, npt.NDArray[np.float64]]:
        """Load pre-computed flux values from CSV.

        Many studies provide LCModel-derived flux values in CSV format.

        Args:
            csv_path: Path to CSV file
            ck_flux_column: Column name for CK forward flux
            atp_synthase_column: Column name for ATP synthase flux

        Returns:
            Dictionary with flux arrays
        """
        import pandas as pd

        df = pd.read_csv(csv_path)

        ck_flux_vals: npt.NDArray[np.float64] = np.asarray(df[ck_flux_column], dtype=np.float64)
        atp_synth_vals: npt.NDArray[np.float64] = np.asarray(
            df.get(atp_synthase_column, pd.Series([0.0] * len(df))), dtype=np.float64
        )
        time_min_vals: npt.NDArray[np.float64] = np.asarray(
            df.get("time_min", pd.Series(np.arange(len(df)))), dtype=np.float64
        )
        return {
            "ck_flux_umol_g_min": ck_flux_vals,
            "atp_synthase_umol_g_min": atp_synth_vals,
            "time_min": time_min_vals,
        }

    def compute_c2_ground_truth(
        self,
        spectra_or_flux: Union[List[PMRSSpectrum], Dict[str, npt.NDArray[np.float64]], None] = None,
        voxel_volume_ml: float = 8.0,
        brain_density_g_ml: float = 1.04,
        resting_neuron_fraction: float = 0.8,
        method: Literal["flux_baseline", "spectral_ratio", "literature"] = "flux_baseline",
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Compute c_2 coefficient from 31P-MRS data.

        Model: ATP_baseline(t) = c_2 × t + noise

        Where c_2 is ATP molecules per millisecond per neuron at rest.

        Args:
            spectra_or_flux: Loaded spectra or flux dictionary (uses loaded if None)
            voxel_volume_ml: Voxel volume in milliliters
            brain_density_g_ml: Brain tissue density
            resting_neuron_fraction: Fraction of neurons metabolically active at rest
            method: Method to estimate c_2

        Returns:
            (c_2_mean, c_2_std, metadata_dict)
        """
        if spectra_or_flux is None:
            spectra_or_flux = self._loaded_spectra

        if method == "literature":
            # Use literature values from Attwell & Laughlin
            # Baseline: ~10^9 ATP/s per neuron
            c2_lit = 1e9 / 1000.0  # Convert to per ms
            return (
                c2_lit,
                0.2 * c2_lit,
                {
                    "method": "literature",
                    "source": "Attwell & Laughlin 2001",
                    "value_per_s": 1e9,
                },
            )

        if isinstance(spectra_or_flux, dict) and "ck_flux_umol_g_min" in spectra_or_flux:
            # Use pre-computed flux values
            flux_data = spectra_or_flux

            # CK flux in μmol/g/min → molecules/g/ms
            ck_flux_umol_g_min = flux_data["ck_flux_umol_g_min"]

            # Convert to ATP molecules per ms
            # μmol/g/min × (molecules/μmol) / (60×1000 ms/min) = molecules/g/ms
            molecules_per_g_ms = (
                np.mean(ck_flux_umol_g_min) * self.ATP_MOLECULES_PER_UMOL / (60.0 * 1000.0)
            )

            # Convert to per voxel
            voxel_mass_g = voxel_volume_ml * brain_density_g_ml
            atp_per_voxel_ms = molecules_per_g_ms * voxel_mass_g

            # Estimate neurons per voxel
            # ~100,000 neurons per mm³ = 100,000 neurons per μL
            # voxel_volume_ml = voxel_volume_ml × 1000 μL/mL
            neurons_per_voxel = 100_000 * voxel_volume_ml * 1000 * resting_neuron_fraction

            c2_mean = atp_per_voxel_ms / neurons_per_voxel
            c2_std = c2_mean * 0.3  # 30% uncertainty estimate

            metadata = {
                "method": "flux_baseline",
                "ck_flux_umol_g_min_mean": float(np.mean(ck_flux_umol_g_min)),
                "ck_flux_umol_g_min_std": float(np.std(ck_flux_umol_g_min)),
                "voxel_volume_ml": voxel_volume_ml,
                "neurons_per_voxel": neurons_per_voxel,
                "atp_per_voxel_ms": atp_per_voxel_ms,
                "c2_mean": c2_mean,
                "c2_std": c2_std,
            }

            return float(c2_mean), float(c2_std), metadata

        elif isinstance(spectra_or_flux, list) and len(spectra_or_flux) > 0:
            # Use spectral ratios (simplified - would need proper quantification)
            # This is a placeholder for full spectral fitting

            # Get ATP/PCr ratios
            ratios = [s.compute_atp_pcr_ratio() for s in spectra_or_flux]
            mean_ratio = np.mean(ratios)

            # Estimate ATP synthesis rate from ratio
            # Higher ratio → higher ATP turnover → higher c_2
            # This is a rough approximation
            base_c2 = 1e6  # Base value per ms per neuron
            c2_mean = base_c2 * (1 + mean_ratio)
            c2_std = c2_mean * 0.4  # Higher uncertainty for spectral method

            metadata = {
                "method": "spectral_ratio",
                "atp_pcr_ratio_mean": float(mean_ratio),
                "atp_pcr_ratio_std": float(np.std(ratios)),
                "num_spectra": len(spectra_or_flux),
            }

            return float(c2_mean), float(c2_std), metadata

        else:
            raise ValueError("No valid data provided for c_2 computation")


# =============================================================================
# Calibration Engine
# =============================================================================


class MetabolicCalibrator:
    """Main calibration engine for fitting c_1 and c_2 from ground-truth data.

    Combines Two-photon (c_1) and P-MRS (c_2) datasets to produce calibrated
    cost coefficients with uncertainty quantification.

    Usage:
        calibrator = MetabolicCalibrator()

        # Load datasets
        calibrator.load_two_photon_dataset("zenodo://marvin_2024")
        calibrator.load_pmrs_dataset("path/to/7t_fmrs.par")

        # Fit coefficients
        coeffs = calibrator.fit_coefficients()

        # Use in VirtualMetabolicLayer
        vml = CalibratedVirtualMetabolicLayer(calibrator=calibrator)
    """

    def __init__(self) -> None:
        """Initialize the calibrator."""
        self.two_photon_loader = TwoPhotonDatasetLoader()
        self.pmrs_loader = PMRSDatasetLoader()

        self._c1_estimate: Optional[Tuple[float, float]] = None
        self._c2_estimate: Optional[Tuple[float, float]] = None
        self._fit_metadata: Dict[str, Any] = {}

    def load_two_photon_dataset(
        self,
        source: Union[str, Path],
        sensor_type: Literal["iATPSnFR2", "ATeam", "Peredox", "unknown"] = "unknown",
    ) -> List[TwoPhotonTrace]:
        """Load Two-photon imaging dataset.

        Args:
            source: Path to dataset or Zenodo DOI (e.g., "zenodo://10.5281/zenodo.xxxxx")
            sensor_type: Type of fluorescent sensor

        Returns:
            List of loaded traces
        """
        # Handle Zenodo DOI
        if isinstance(source, str) and source.startswith("zenodo://"):
            doi = source.replace("zenodo://", "")
            self.two_photon_loader = TwoPhotonDatasetLoader.from_zenodo(doi)
            doi_id = doi.split(".")[-1]
            extract_dir = self.two_photon_loader.cache_dir / f"zenodo_{doi_id}"
            traces = self.two_photon_loader.load_traces(extract_dir, sensor_type=sensor_type)
        else:
            traces = self.two_photon_loader.load_traces(source, sensor_type=sensor_type)

        return traces

    def load_pmrs_dataset(
        self,
        source: Union[str, Path],
        format_type: Literal["parrec", "csv", "twix"] = "parrec",
    ) -> Union[List[PMRSSpectrum], Dict[str, npt.NDArray[np.float64]]]:
        """Load 31P-MRS dataset.

        Args:
            source: Path to dataset file
            format_type: Format of the data

        Returns:
            Loaded spectra or flux data
        """
        if format_type == "parrec":
            return self.pmrs_loader.load_parrec(source)
        elif format_type == "csv":
            return self.pmrs_loader.load_csv_flux(source)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def fit_coefficients(
        self,
        c1_method: Literal["integral", "regression", "literature"] = "integral",
        c2_method: Literal["flux_baseline", "spectral_ratio", "literature"] = "flux_baseline",
        combine_uncertainty: bool = True,
    ) -> CalibratedCostCoefficients:
        """Fit c_1 and c_2 from loaded datasets.

        Args:
            c1_method: Method for c_1 estimation
            c2_method: Method for c_2 estimation
            combine_uncertainty: Whether to combine uncertainties if both methods available

        Returns:
            CalibratedCostCoefficients with fitted values
        """
        # Fit c_1 from Two-photon data
        if c1_method == "literature":
            c1_mean, c1_std = 1.5e7, 3e6
            c1_metadata = {"method": "literature", "source": "Attwell & Laughlin 2001"}
        else:
            c1_mean, c1_std, c1_metadata = self.two_photon_loader.compute_c1_ground_truth(
                fit_method=c1_method
            )

        self._c1_estimate = (c1_mean, c1_std)

        # Fit c_2 from P-MRS data
        if c2_method == "literature":
            c2_mean, c2_std, c2_metadata = (
                1e6,  # 1e9 ATP/s per neuron → 1e6 per ms
                2e5,
                {"method": "literature", "source": "Attwell & Laughlin 2001"},
            )
        else:
            c2_mean, c2_std, c2_metadata = self.pmrs_loader.compute_c2_ground_truth(
                method=c2_method
            )

        self._c2_estimate = (c2_mean, c2_std)

        # Store metadata
        self._fit_metadata = {
            "c1": c1_metadata,
            "c2": c2_metadata,
            "timestamp": str(np.datetime64("now")),
        }

        # Determine calibration source
        if c1_method != "literature" and c2_method != "literature":
            source: Literal["two_photon", "p_mrs", "combined", "literature"] = "combined"
        elif c1_method != "literature":
            source = "two_photon"
        elif c2_method != "literature":
            source = "p_mrs"
        else:
            source = "literature"

        return CalibratedCostCoefficients(
            c_1_dynamic=c1_mean,
            c_2_static=c2_mean,
            c_1_uncertainty=c1_std,
            c_2_uncertainty=c2_std,
            calibration_source=source,
            temporal_resolution_ms=self._get_temporal_resolution(),
            dataset_doi=self._get_dataset_doi(),
        )

    def _get_temporal_resolution(self) -> float:
        """Get temporal resolution from loaded data."""
        if self.two_photon_loader._loaded_traces:
            return self.two_photon_loader._loaded_traces[0].temporal_resolution_ms
        return 100.0  # Default 100 ms

    def _get_dataset_doi(self) -> str:
        """Get DOI from loaded datasets."""
        # Would track actual DOI from loaded data
        return ""

    def validate_coefficients(
        self,
        coefficients: CalibratedCostCoefficients,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Validate coefficients against biological plausibility ranges.

        Args:
            coefficients: Coefficients to validate
            verbose: Print validation results

        Returns:
            Validation report dictionary
        """
        # Biological plausibility ranges (from literature)
        C1_RANGE = (1e7, 1e8)  # ATP per AP
        C2_RANGE = (1e5, 1e7)  # ATP per ms per neuron

        c1_valid = C1_RANGE[0] <= coefficients.c_1_dynamic <= C1_RANGE[1]
        c2_valid = C2_RANGE[0] <= coefficients.c_2_static <= C2_RANGE[1]

        # Check ratio (should be consistent with firing rates ~5-20 Hz)
        ratio = coefficients.c_1_dynamic / (coefficients.c_2_static * 1000)
        ratio_valid = 0.005 <= ratio <= 0.5  # 5-50 AP/s equivalent

        warnings_list: List[str] = []
        report: Dict[str, Any] = {
            "c1_valid": c1_valid,
            "c2_valid": c2_valid,
            "ratio_valid": ratio_valid,
            "c1_value": coefficients.c_1_dynamic,
            "c2_value": coefficients.c_2_static,
            "ratio": ratio,
            "c1_range": C1_RANGE,
            "c2_range": C2_RANGE,
            "warnings": warnings_list,
        }

        if not c1_valid:
            report["warnings"].append(
                f"c_1 ({coefficients.c_1_dynamic:.2e}) outside typical range {C1_RANGE}"
            )
        if not c2_valid:
            report["warnings"].append(
                f"c_2 ({coefficients.c_2_static:.2e}) outside typical range {C2_RANGE}"
            )
        if not ratio_valid:
            report["warnings"].append(
                f"c_1/c_2 ratio ({ratio:.4f}) inconsistent with typical firing rates"
            )

        if verbose and report["warnings"]:
            print("Validation Warnings:")
            for w in report["warnings"]:
                print(f"  - {w}")

        return report


# =============================================================================
# Validation Tools
# =============================================================================


class CostCoefficientValidator:
    """Validate calibrated cost coefficients against established frameworks.

    References:
    - Attwell & Laughlin (2001): 19.1 ATP per spike
    - Lennie (2003): Energy budget proportions
    - Howarth et al. (2012): Updated estimates
    """

    # Literature values for comparison
    LITERATURE_VALUES = {
        "attwell_laughlin_2001": {
            "atp_per_spike": 1.91e7,  # 19.1 million per spike (rounded)
            "atp_per_s_resting": 1e9,  # 1 billion per neuron per second
        },
        "lennie_2003": {
            "atp_per_spike": 1.5e7,
            "atp_per_s_resting": 9e8,
        },
        "howarth_2012": {
            "atp_per_spike": 2.2e7,
            "atp_per_s_resting": 1.1e9,
        },
    }

    def __init__(self):
        """Initialize validator."""
        pass

    def compare_to_literature(
        self,
        coefficients: CalibratedCostCoefficients,
    ) -> Dict[str, Any]:
        """Compare coefficients to published literature values.

        Returns:
            Comparison report with deviations from each reference
        """
        results = {}

        for ref_name, ref_values in self.LITERATURE_VALUES.items():
            c1_deviation = (coefficients.c_1_dynamic - ref_values["atp_per_spike"]) / ref_values[
                "atp_per_spike"
            ]

            c2_per_s = coefficients.c_2_static * 1000.0
            c2_deviation = (c2_per_s - ref_values["atp_per_s_resting"]) / ref_values[
                "atp_per_s_resting"
            ]

            results[ref_name] = {
                "c1_deviation": float(c1_deviation),
                "c2_deviation": float(c2_deviation),
                "c1_matches": abs(c1_deviation) < 0.5,  # Within 50%
                "c2_matches": abs(c2_deviation) < 0.5,
            }

        return results

    def compute_confidence_score(
        self,
        coefficients: CalibratedCostCoefficients,
    ) -> float:
        """Compute overall confidence score (0-1) based on multiple factors.

        Factors:
        - Uncertainty magnitude (lower = better)
        - Literature agreement
        - Data source quality (Two-photon > P-MRS > Literature)
        """
        score = 1.0

        # Penalty for high uncertainty
        if coefficients.c_1_uncertainty > coefficients.c_1_dynamic * 0.5:
            score -= 0.3
        if coefficients.c_2_uncertainty > coefficients.c_2_static * 0.5:
            score -= 0.3

        # Bonus for empirical data
        if coefficients.calibration_source == "combined":
            score += 0.1
        elif coefficients.calibration_source in ["two_photon", "p_mrs"]:
            score += 0.05

        return max(0.0, min(1.0, score))


# =============================================================================
# Convenience Functions
# =============================================================================


def calibrate_from_datasets(
    two_photon_path: Optional[Union[str, Path]] = None,
    pmrs_path: Optional[Union[str, Path]] = None,
    two_photon_sensor: Literal["iATPSnFR2", "ATeam", "Peredox", "unknown"] = "unknown",
    pmrs_format: Literal["parrec", "csv", "twix"] = "csv",
    validate: bool = True,
) -> CalibratedCostCoefficients:
    """Convenience function to calibrate from datasets.

    Args:
        two_photon_path: Path to Two-photon dataset (or None to use literature)
        pmrs_path: Path to P-MRS dataset (or None to use literature)
        two_photon_sensor: Sensor type for Two-photon data
        pmrs_format: Format of P-MRS data
        validate: Run validation after fitting

    Returns:
        Calibrated cost coefficients
    """
    calibrator = MetabolicCalibrator()

    # Load datasets if provided
    if two_photon_path is not None:
        calibrator.load_two_photon_dataset(two_photon_path, two_photon_sensor)
        c1_method: Literal["integral", "regression", "literature"] = "integral"
    else:
        c1_method = "literature"

    if pmrs_path is not None:
        calibrator.load_pmrs_dataset(pmrs_path, pmrs_format)
        c2_method: Literal["flux_baseline", "spectral_ratio", "literature"] = "flux_baseline"
    else:
        c2_method = "literature"

    # Fit coefficients
    coeffs = calibrator.fit_coefficients(c1_method=c1_method, c2_method=c2_method)

    # Validate
    if validate:
        validator = CostCoefficientValidator()
        _ = calibrator.validate_coefficients(coeffs, verbose=True)
        _ = validator.compare_to_literature(coeffs)

        print("\nCalibration Complete:")
        print(f"  c_1 (dynamic): {coeffs.c_1_dynamic:.2e} ± {coeffs.c_1_uncertainty:.2e} ATP/AP")
        print(
            f"  c_2 (static):  {coeffs.c_2_static:.2e} ± {coeffs.c_2_uncertainty:.2e} ATP/ms/neuron"
        )
        print(f"  Source: {coeffs.calibration_source}")
        print(f"  Confidence: {validator.compute_confidence_score(coeffs):.2f}")

    return coeffs


def get_default_coefficients() -> CalibratedCostCoefficients:
    """Get default coefficients from literature values.

    These are conservative estimates based on Attwell & Laughlin (2001):
    - c_1: ~1.5×10^7 ATP per action potential
    - c_2: ~1×10^6 ATP per ms per neuron (equivalent to 10^9 ATP/s)
    """
    return CalibratedCostCoefficients(
        c_1_dynamic=1.5e7,
        c_2_static=1e6,
        c_1_uncertainty=3e6,  # 20% uncertainty
        c_2_uncertainty=2e5,
        calibration_source="literature",
        temporal_resolution_ms=100.0,
        dataset_doi="attwell_laughlin_2001",
    )


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Data structures
    "CalibratedCostCoefficients",
    "TwoPhotonTrace",
    "PMRSSpectrum",
    # Loaders
    "TwoPhotonDatasetLoader",
    "PMRSDatasetLoader",
    # Calibration
    "MetabolicCalibrator",
    "CostCoefficientValidator",
    # Convenience functions
    "calibrate_from_datasets",
    "get_default_coefficients",
]
