"""Neural data acquisition and processing pipeline for APGI framework.

This module provides interfaces for EEG/MEG data acquisition, real-time processing,
neural signature extraction including ERPs, microstates, and gamma synchrony,
pupillometry, and multi-modal physiological monitoring.
"""

from .cardiac_processor import (
    CardiacProcessor,
    CardiacQualityAssessor,
    CardiacQualityMetrics,
    HEPExtractor,
    HRVAnalyzer,
    HRVMetrics,
    RPeakAlgorithm,
)
from .eeg_interface import (
    ArtifactDetector,
    ChannelInfo,
    ChannelType,
    EEGConfig,
    EEGInterface,
)

# Signal processing components
from .eeg_processor import (
    BaselineCorrector,
    EEGProcessor,
    EpochExtractor,
    ERPExtractor,
    ERPFeatures,
    FASTERArtifactDetector,
    FilterType,
    ProcessedEEG,
)
from .erp_analysis import ERPAnalysis, ERPComponents, P3bMetrics
from .gamma_synchrony import (
    CoherenceMetrics,
    GammaSynchronyAnalysis,
    NetworkConnectivity,
)
from .microstate_analysis import MicrostateAnalysis, MicrostateSequence
from .physiological_monitoring import (
    HeartRateMonitor,
    PhysiologicalConfig,
    PhysiologicalMonitoring,
    PhysiologicalSample,
    RespirationMonitor,
    RespirationPhase,
    SignalType,
    SkinConductanceMonitor,
)
from .pupillometry_interface import (
    BlinkDetectionMethod,
    EyeType,
    PupillometryConfig,
    PupillometryInterface,
    PupilSample,
)
from .pupillometry_processor import (
    PupilFeatureExtractor,
    PupilFeatures,
    PupillometryProcessor,
    PupilMetricsCalculator,
    PupilQualityMetrics,
    PupilQualityScorer,
)
from .quality_control import (
    AdaptiveProtocolManager,
    AlertSeverity,
)
from .quality_control import ArtifactDetector as UnifiedArtifactDetector
from .quality_control import (
    MultiModalQualityMetrics,
    OperatorNotificationSystem,
    QualityAlert,
    QualityLevel,
    SignalQualityMonitor,
)

__all__ = [
    # Data acquisition interfaces
    "EEGInterface",
    "EEGConfig",
    "ArtifactDetector",
    "ChannelInfo",
    "ChannelType",
    "PupillometryInterface",
    "PupillometryConfig",
    "PupilSample",
    "EyeType",
    "BlinkDetectionMethod",
    "PhysiologicalMonitoring",
    "PhysiologicalConfig",
    "PhysiologicalSample",
    "SignalType",
    "RespirationPhase",
    "HeartRateMonitor",
    "SkinConductanceMonitor",
    "RespirationMonitor",
    # Neural analysis
    "ERPAnalysis",
    "ERPComponents",
    "P3bMetrics",
    "MicrostateAnalysis",
    "MicrostateSequence",
    "GammaSynchronyAnalysis",
    "CoherenceMetrics",
    "NetworkConnectivity",
    # Signal processing
    "EEGProcessor",
    "FASTERArtifactDetector",
    "EpochExtractor",
    "BaselineCorrector",
    "ERPExtractor",
    "ProcessedEEG",
    "ERPFeatures",
    "FilterType",
    "PupillometryProcessor",
    "PupilFeatureExtractor",
    "PupilMetricsCalculator",
    "PupilQualityScorer",
    "PupilFeatures",
    "PupilQualityMetrics",
    "CardiacProcessor",
    "HRVAnalyzer",
    "HEPExtractor",
    "CardiacQualityAssessor",
    "HRVMetrics",
    "CardiacQualityMetrics",
    "RPeakAlgorithm",
    # Quality control
    "SignalQualityMonitor",
    "UnifiedArtifactDetector",
    "AdaptiveProtocolManager",
    "OperatorNotificationSystem",
    "MultiModalQualityMetrics",
    "QualityAlert",
    "QualityLevel",
    "AlertSeverity",
]
