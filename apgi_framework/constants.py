"""Constants for APGI System.

Contains named constants for magic numbers used throughout the codebase
to improve maintainability and readability.
"""

import numpy as np

# Time constants
MILLISECONDS_TO_SECONDS = 0.001
SECONDS_TO_MILLISECONDS = 1000.0
DEFAULT_TIMESTEP_MS = 1.0
DEFAULT_TIMESTEP_SEC = DEFAULT_TIMESTEP_MS * MILLISECONDS_TO_SECONDS

# Neural network dimensions
DEFAULT_NEURAL_DIMS = {"SENSORY": 256, "HIDDEN_1": 128, "HIDDEN_2": 64, "IDENTITY": 64}

# Oscillation frequencies (Hz)
OSCILLATION_FREQS = {
    "DELTA": (1.0, 4.0),
    "THETA": (4.0, 8.0),
    "ALPHA": (8.0, 13.0),
    "BETA": (13.0, 30.0),
    "GAMMA": (30.0, 100.0),
}

# Default oscillation frequencies
DEFAULT_FREQS = {"BETA": 20.0, "GAMMA": 40.0}  # Hz  # Hz

# Ignition timing (milliseconds)
IGNITION_TIMING = {
    "AMPLIFICATION_DURATION": 300.0,
    "MAINTENANCE_DURATION": 1000.0,
    "FADE_DURATION": 200.0,
    "PRE_IGNITION_DURATION": 500.0,
    "POST_IGNITION_DURATION": 1000.0,
}

# Thresholds and gains
THRESHOLDS = {
    "DEFAULT_PREDICTION_STRENGTH": 0.8,
    "DEFAULT_ERROR_GAIN": 1.0,
    "DEFAULT_RECURRENT_STRENGTH": 0.8,
    "DEFAULT_AMPLIFICATION_GAIN": 2.0,
    "COHERENCE_THRESHOLD": 0.8,
}

# Gains and modulations
GAINS = {"GAIN_RANGE_MIN": 0.5, "GAIN_RANGE_MAX": 2.0, "BETA_MODULATION_STRENGTH": 0.1}

# Memory and capacity
MEMORY_LIMITS = {
    "DEFAULT_EPISODIC_CAPACITY": 1000,
    "DEFAULT_MAX_EVENTS": 1000,
    "DEFAULT_PROJECTION_CACHE_SIZE": 100,
}

# Metabolic costs
METABOLIC_COSTS = {
    "DEFAULT_IGNITION_COST": 0.1,
    "DEFAULT_TASK_OVERHEAD": 0.05,
    "DEFAULT_BASELINE_RATE": 0.01,
    "DEFAULT_RECOVERY_RATE": 0.02,
}

# Learning rates
LEARNING_RATES = {"DEFAULT_CONSOLIDATION_RATE": 0.01, "DEFAULT_LEARNING_RATE": 0.01}

# Precision ranges
PRECISION_RANGE = {"MIN": 0.1, "MAX": 10.0}

# Validation ranges
VALIDATION_RANGES = {"DT_RANGE": (1e-6, 1.0), "POSITIVE_RANGE": (0.0, float("inf"))}

# Noise scales
NOISE_SCALES = {"DEFAULT_SENSORY_NOISE": 0.2, "DEFAULT_IDENTITY_INIT_SCALE": 0.1}

# Common array operations
TWO_PI = 2.0 * np.pi
