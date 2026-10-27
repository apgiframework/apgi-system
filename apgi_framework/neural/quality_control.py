"""Real-time quality control system for multi-modal data acquisition.

Integrates EEG, pupillometry, and cardiac quality monitoring with adaptive
protocol management and operator notifications for APGI experiments.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np

from ..logging.standardized_logging import get_logger


class QualityLevel(Enum):
    """Data quality levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityAlert:
    """Quality control alert."""

    timestamp: float
    severity: AlertSeverity
    modality: str  # eeg, pupil, cardiac, or multi-modal
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    recovery_suggestion: str


@dataclass
class MultiModalQualityMetrics:
    """Integrated quality metrics across all modalities."""

    timestamp: float

    # EEG metrics
    eeg_quality: float  # 0-1
    eeg_artifact_rate: float
    eeg_bad_channels: int

    # Pupillometry metrics
    pupil_quality: float  # 0-1
    pupil_data_loss: float
    pupil_tracking_confidence: float

    # Cardiac metrics
    cardiac_quality: float  # 0-1
    cardiac_sqi: float
    cardiac_ectopic_rate: float

    # Overall metrics
    overall_quality: float  # 0-1
    quality_level: QualityLevel
    active_alerts: int

    # Recommendations
    continue_recording: bool
    suggested_actions: List[str] = field(default_factory=list)


class SignalQualityMonitor:
    """Real-time signal quality monitor integrating all modalities.

    Continuously assesses EEG, pupillometry, and cardiac signal quality
    with unified quality scoring and alerting.
    """

    def __init__(self, update_interval: float = 1.0):
        """Initialize signal quality monitor.

        Args:
            update_interval: Quality update interval in seconds
        """
        self.update_interval = update_interval

        # Quality history
        self.quality_history: deque = deque(maxlen=1000)

        # Alert history
        self.alert_history: deque = deque(maxlen=100)

        # Thresholds
        self.thresholds = {
            "eeg_quality_min": 0.6,
            "eeg_artifact_rate_max": 0.15,
            "eeg_bad_channels_max": 5,
            "pupil_quality_min": 0.6,
            "pupil_data_loss_max": 0.20,
            "pupil_tracking_confidence_min": 0.7,
            "cardiac_quality_min": 0.6,
            "cardiac_sqi_min": 0.7,
            "cardiac_ectopic_rate_max": 0.05,
            "overall_quality_min": 0.5,
        }

        # Monitoring state
        self.is_monitoring = False
        self.last_update_time: Optional[float] = None

    def update_quality_metrics(
        self,
        eeg_metrics: Dict[str, Any],
        pupil_metrics: Dict[str, Any],
        cardiac_metrics: Dict[str, Any],
    ) -> MultiModalQualityMetrics:
        """Update and integrate quality metrics from all modalities.

        Args:
            eeg_metrics: EEG quality metrics dictionary
            pupil_metrics: Pupillometry quality metrics dictionary
            cardiac_metrics: Cardiac quality metrics dictionary

        Returns:
            MultiModalQualityMetrics object
        """
        current_time = time.time()

        # Extract EEG metrics
        eeg_quality = eeg_metrics.get("overall_quality", 0.0)
        eeg_artifact_rate = eeg_metrics.get("artifact_rate", 0.0)
        eeg_bad_channels = eeg_metrics.get("bad_channels_count", 0)

        # Extract pupil metrics
        pupil_quality = pupil_metrics.get("overall_quality", 0.0)
        pupil_data_loss = pupil_metrics.get("data_loss_percentage", 0.0) / 100
        pupil_tracking_confidence = pupil_metrics.get("tracking_confidence", 0.0)

        # Extract cardiac metrics
        cardiac_quality = cardiac_metrics.get("overall_quality", 0.0)
        cardiac_sqi = cardiac_metrics.get("signal_quality_index", 0.0)
        cardiac_ectopic_rate = cardiac_metrics.get("ectopic_beat_percentage", 0.0) / 100

        # Compute overall quality (weighted average)
        overall_quality = eeg_quality * 0.4 + pupil_quality * 0.3 + cardiac_quality * 0.3

        # Determine quality level
        if overall_quality >= 0.8:
            quality_level = QualityLevel.EXCELLENT
        elif overall_quality >= 0.6:
            quality_level = QualityLevel.GOOD
        elif overall_quality >= 0.4:
            quality_level = QualityLevel.ACCEPTABLE
        elif overall_quality >= 0.2:
            quality_level = QualityLevel.POOR
        else:
            quality_level = QualityLevel.CRITICAL

        # Check for quality issues and generate alerts
        alerts = self._check_quality_thresholds(
            eeg_quality,
            eeg_artifact_rate,
            eeg_bad_channels,
            pupil_quality,
            pupil_data_loss,
            pupil_tracking_confidence,
            cardiac_quality,
            cardiac_sqi,
            cardiac_ectopic_rate,
            overall_quality,
        )

        # Generate suggested actions
        suggested_actions = self._generate_suggestions(
            eeg_quality,
            eeg_artifact_rate,
            eeg_bad_channels,
            pupil_quality,
            pupil_data_loss,
            cardiac_quality,
            cardiac_sqi,
        )

        # Determine if recording should continue
        continue_recording = overall_quality >= self.thresholds["overall_quality_min"]

        # Create metrics object
        metrics = MultiModalQualityMetrics(
            timestamp=current_time,
            eeg_quality=eeg_quality,
            eeg_artifact_rate=eeg_artifact_rate,
            eeg_bad_channels=eeg_bad_channels,
            pupil_quality=pupil_quality,
            pupil_data_loss=pupil_data_loss,
            pupil_tracking_confidence=pupil_tracking_confidence,
            cardiac_quality=cardiac_quality,
            cardiac_sqi=cardiac_sqi,
            cardiac_ectopic_rate=cardiac_ectopic_rate,
            overall_quality=overall_quality,
            quality_level=quality_level,
            active_alerts=len(alerts),
            continue_recording=continue_recording,
            suggested_actions=suggested_actions,
        )

        # Store in history
        self.quality_history.append(metrics)
        self.last_update_time = current_time

        return metrics

    def _check_quality_thresholds(
        self,
        eeg_quality: float,
        eeg_artifact_rate: float,
        eeg_bad_channels: int,
        pupil_quality: float,
        pupil_data_loss: float,
        pupil_tracking_confidence: float,
        cardiac_quality: float,
        cardiac_sqi: float,
        cardiac_ectopic_rate: float,
        overall_quality: float,
    ) -> List[QualityAlert]:
        """Check quality metrics against thresholds and generate alerts."""
        alerts = []
        current_time = time.time()

        # EEG alerts
        if eeg_quality < self.thresholds["eeg_quality_min"]:
            alerts.append(
                QualityAlert(
                    timestamp=current_time,
                    severity=(AlertSeverity.WARNING if eeg_quality > 0.4 else AlertSeverity.ERROR),
                    modality="eeg",
                    message=f"EEG quality below threshold: {eeg_quality:.2f}",
                    metric_name="eeg_quality",
                    metric_value=eeg_quality,
                    threshold=self.thresholds["eeg_quality_min"],
                    recovery_suggestion="Check electrode impedances and participant movement",
                )
            )

        if eeg_artifact_rate > self.thresholds["eeg_artifact_rate_max"]:
            alerts.append(
                QualityAlert(
                    timestamp=current_time,
                    severity=AlertSeverity.WARNING,
                    modality="eeg",
                    message=f"High EEG artifact rate: {eeg_artifact_rate:.1%}",
                    metric_name="eeg_artifact_rate",
                    metric_value=eeg_artifact_rate,
                    threshold=self.thresholds["eeg_artifact_rate_max"],
                    recovery_suggestion="Ask participant to relax and minimize movement",
                )
            )

        if eeg_bad_channels > self.thresholds["eeg_bad_channels_max"]:
            alerts.append(
                QualityAlert(
                    timestamp=current_time,
                    severity=AlertSeverity.ERROR,
                    modality="eeg",
                    message=f"Too many bad EEG channels: {eeg_bad_channels}",
                    metric_name="eeg_bad_channels",
                    metric_value=float(eeg_bad_channels),
                    threshold=float(self.thresholds["eeg_bad_channels_max"]),
                    recovery_suggestion="Check and re-apply electrodes with high impedance",
                )
            )

        # Pupillometry alerts
        if pupil_quality < self.thresholds["pupil_quality_min"]:
            alerts.append(
                QualityAlert(
                    timestamp=current_time,
                    severity=AlertSeverity.WARNING,
                    modality="pupil",
                    message=f"Pupillometry quality below threshold: {pupil_quality:.2f}",
                    metric_name="pupil_quality",
                    metric_value=pupil_quality,
                    threshold=self.thresholds["pupil_quality_min"],
                    recovery_suggestion="Recalibrate eye tracker and check lighting conditions",
                )
            )

        if pupil_data_loss > self.thresholds["pupil_data_loss_max"]:
            alerts.append(
                QualityAlert(
                    timestamp=current_time,
                    severity=AlertSeverity.ERROR,
                    modality="pupil",
                    message=f"High pupil data loss: {pupil_data_loss:.1%}",
                    metric_name="pupil_data_loss",
                    metric_value=pupil_data_loss,
                    threshold=self.thresholds["pupil_data_loss_max"],
                    recovery_suggestion="Adjust eye tracker position and participant head position",
                )
            )

        # Cardiac alerts
        if cardiac_quality < self.thresholds["cardiac_quality_min"]:
            alerts.append(
                QualityAlert(
                    timestamp=current_time,
                    severity=AlertSeverity.WARNING,
                    modality="cardiac",
                    message=f"Cardiac signal quality below threshold: {cardiac_quality:.2f}",
                    metric_name="cardiac_quality",
                    metric_value=cardiac_quality,
                    threshold=self.thresholds["cardiac_quality_min"],
                    recovery_suggestion="Check ECG electrode placement and contact",
                )
            )

        if cardiac_ectopic_rate > self.thresholds["cardiac_ectopic_rate_max"]:
            alerts.append(
                QualityAlert(
                    timestamp=current_time,
                    severity=AlertSeverity.INFO,
                    modality="cardiac",
                    message=f"Elevated ectopic beat rate: {cardiac_ectopic_rate:.1%}",
                    metric_name="cardiac_ectopic_rate",
                    metric_value=cardiac_ectopic_rate,
                    threshold=self.thresholds["cardiac_ectopic_rate_max"],
                    recovery_suggestion="Monitor participant stress level and allow rest if needed",
                )
            )

        # Overall quality alert
        if overall_quality < self.thresholds["overall_quality_min"]:
            alerts.append(
                QualityAlert(
                    timestamp=current_time,
                    severity=AlertSeverity.CRITICAL,
                    modality="multi-modal",
                    message=f"Overall data quality critically low: {overall_quality:.2f}",
                    metric_name="overall_quality",
                    metric_value=overall_quality,
                    threshold=self.thresholds["overall_quality_min"],
                    recovery_suggestion="Consider pausing recording to address quality issues",
                )
            )

        # Store alerts
        for alert in alerts:
            self.alert_history.append(alert)

        return alerts

    def _generate_suggestions(
        self,
        eeg_quality: float,
        eeg_artifact_rate: float,
        eeg_bad_channels: int,
        pupil_quality: float,
        pupil_data_loss: float,
        cardiac_quality: float,
        cardiac_sqi: float,
    ) -> List[str]:
        """Generate actionable suggestions based on quality metrics."""
        suggestions = []

        # EEG suggestions
        if eeg_quality < 0.6:
            suggestions.append("Improve EEG signal quality by checking electrode impedances")
        if eeg_artifact_rate > 0.15:
            suggestions.append("Reduce EEG artifacts by minimizing participant movement")
        if eeg_bad_channels > 3:
            suggestions.append("Re-apply or replace problematic EEG electrodes")

        # Pupillometry suggestions
        if pupil_quality < 0.6:
            suggestions.append("Recalibrate eye tracker for better pupil tracking")
        if pupil_data_loss > 0.15:
            suggestions.append("Adjust participant head position or eye tracker placement")

        # Cardiac suggestions
        if cardiac_quality < 0.6:
            suggestions.append("Check ECG electrode placement and skin contact")
        if cardiac_sqi < 0.7:
            suggestions.append("Improve cardiac signal quality by reducing movement artifacts")

        # General suggestions
        if not suggestions:
            suggestions.append("Data quality is good - continue recording")

        return suggestions

    def get_quality_summary(self, window_seconds: float = 60.0) -> Dict[str, Any]:
        """Get quality summary over recent time window.

        Args:
            window_seconds: Time window for summary (seconds)

        Returns:
            Dictionary with quality summary statistics
        """
        if not self.quality_history:
            return {"status": "no_data"}

        # Filter recent metrics
        current_time = time.time()
        recent_metrics = [
            m for m in self.quality_history if current_time - m.timestamp <= window_seconds
        ]

        if not recent_metrics:
            return {"status": "no_recent_data"}

        # Compute summary statistics
        overall_qualities = [m.overall_quality for m in recent_metrics]
        eeg_qualities = [m.eeg_quality for m in recent_metrics]
        pupil_qualities = [m.pupil_quality for m in recent_metrics]
        cardiac_qualities = [m.cardiac_quality for m in recent_metrics]

        return {
            "status": "ok",
            "window_seconds": window_seconds,
            "n_samples": len(recent_metrics),
            "overall_quality_mean": np.mean(overall_qualities),
            "overall_quality_std": np.std(overall_qualities),
            "overall_quality_min": np.min(overall_qualities),
            "eeg_quality_mean": np.mean(eeg_qualities),
            "pupil_quality_mean": np.mean(pupil_qualities),
            "cardiac_quality_mean": np.mean(cardiac_qualities),
            "current_quality_level": recent_metrics[-1].quality_level.value,
            "active_alerts": recent_metrics[-1].active_alerts,
        }


class ArtifactDetector:
    """Unified artifact detector coordinating across all modalities.

    Detects artifacts in EEG, pupillometry, and cardiac signals
    with cross-modal validation.
    """

    def __init__(self) -> None:
        """Initialize unified artifact detector."""
        self.artifact_history: deque = deque(maxlen=1000)

    def detect_multi_modal_artifacts(
        self,
        eeg_artifacts: Optional[np.ndarray],
        pupil_artifacts: Optional[np.ndarray],
        cardiac_artifacts: Optional[np.ndarray],
        timestamps: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Detect artifacts across all modalities with cross-validation.

        Args:
            eeg_artifacts: EEG artifact mask (samples)
            pupil_artifacts: Pupil artifact mask (samples)
            cardiac_artifacts: Cardiac artifact mask (samples)
            timestamps: Timestamp array

        Returns:
            Dictionary with artifact masks and combined mask
        """
        result = {
            "timestamps": timestamps,
            "eeg_artifacts": (
                eeg_artifacts
                if eeg_artifacts is not None
                else np.zeros(len(timestamps), dtype=bool)
            ),
            "pupil_artifacts": (
                pupil_artifacts
                if pupil_artifacts is not None
                else np.zeros(len(timestamps), dtype=bool)
            ),
            "cardiac_artifacts": (
                cardiac_artifacts
                if cardiac_artifacts is not None
                else np.zeros(len(timestamps), dtype=bool)
            ),
        }

        # Combine artifacts (logical OR)
        combined = np.zeros(len(timestamps), dtype=bool)
        if eeg_artifacts is not None:
            combined |= eeg_artifacts
        if pupil_artifacts is not None:
            combined |= pupil_artifacts
        if cardiac_artifacts is not None:
            combined |= cardiac_artifacts

        result["combined_artifacts"] = combined

        # Detect correlated artifacts (artifacts present in multiple modalities)
        artifact_count = np.zeros(len(timestamps), dtype=int)

        if eeg_artifacts is not None:
            artifact_count += eeg_artifacts.astype(int)
        if pupil_artifacts is not None:
            artifact_count += pupil_artifacts.astype(int)
        if cardiac_artifacts is not None:
            artifact_count += cardiac_artifacts.astype(int)

        correlated_mask = artifact_count >= 2  # Artifacts in 2+ modalities
        result["correlated_artifacts"] = correlated_mask

        # Store artifact statistics
        self.artifact_history.append(
            {
                "timestamp": time.time(),
                "total_artifacts": np.sum(combined),
                "correlated_artifacts": np.sum(correlated_mask),
                "artifact_rate": np.sum(combined) / len(combined),
            }
        )

        return result


class AdaptiveProtocolManager:
    """Adaptive protocol manager for quality-based adjustments.

    Automatically adjusts experimental protocols based on real-time
    quality metrics to maintain data integrity.
    """

    def __init__(self) -> None:
        """Initialize adaptive protocol manager."""
        self.adjustment_history: List[Dict[str, Any]] = []
        self.current_adjustments: Dict[str, Any] = {}

        # Setup logging
        self.logger = get_logger(__name__)

    def evaluate_protocol_adjustments(
        self, quality_metrics: MultiModalQualityMetrics
    ) -> Dict[str, Any]:
        """Evaluate if protocol adjustments are needed based on quality.

        Args:
            quality_metrics: Current quality metrics

        Returns:
            Dictionary with recommended adjustments
        """
        adjustments: Dict[str, Any] = {
            "timestamp": time.time(),
            "quality_level": quality_metrics.quality_level.value,
            "adjustments_needed": False,
            "recommendations": cast(List[Dict[str, Any]], []),
        }

        # Check if adjustments are needed
        if quality_metrics.quality_level in [QualityLevel.POOR, QualityLevel.CRITICAL]:
            adjustments["adjustments_needed"] = True

            # EEG-specific adjustments
            if quality_metrics.eeg_quality < 0.5:
                adjustments["recommendations"].append(
                    {
                        "modality": "eeg",
                        "action": "increase_trial_duration",
                        "reason": "Low EEG quality - extend trials to collect more clean data",
                        "parameter": "trial_duration",
                        "adjustment": "+20%",
                    }
                )

            # Pupillometry-specific adjustments
            if quality_metrics.pupil_data_loss > 0.25:
                adjustments["recommendations"].append(
                    {
                        "modality": "pupil",
                        "action": "increase_inter_trial_interval",
                        "reason": "High pupil data loss - allow more time for blink recovery",
                        "parameter": "inter_trial_interval",
                        "adjustment": "+500ms",
                    }
                )

            # Cardiac-specific adjustments
            if quality_metrics.cardiac_quality < 0.5:
                adjustments["recommendations"].append(
                    {
                        "modality": "cardiac",
                        "action": "add_rest_periods",
                        "reason": "Low cardiac signal quality - add rest periods",
                        "parameter": "rest_period_frequency",
                        "adjustment": "every_10_trials",
                    }
                )

            # Overall adjustments
            if quality_metrics.overall_quality < 0.3:
                adjustments["recommendations"].append(
                    {
                        "modality": "all",
                        "action": "pause_recording",
                        "reason": "Critical quality issues - pause to address problems",
                        "parameter": "recording_state",
                        "adjustment": "pause",
                    }
                )

        # Store adjustment
        self.adjustment_history.append(adjustments)
        self.current_adjustments = adjustments

        return adjustments

    def apply_adjustment(self, adjustment: Dict[str, Any]) -> bool:
        """Apply a protocol adjustment.

        Args:
            adjustment: Adjustment dictionary

        Returns:
            Success status
        """
        # This would interface with the actual experimental control system
        # For now, just log the adjustment
        self.logger.info(
            f"Applying adjustment: {adjustment['action']} for {adjustment['modality']}"
        )
        return True


class OperatorNotificationSystem:
    """Real-time operator notification system.

    Provides alerts, recovery suggestions, and status updates
    to the experiment operator.
    """

    def __init__(self) -> None:
        """Initialize operator notification system."""
        self.notification_callbacks: List[Callable] = []
        self.notification_history: deque = deque(maxlen=100)

        # Setup logging
        self.logger = get_logger(__name__)

    def register_callback(self, callback: Callable) -> None:
        """Register callback for notifications.

        Args:
            callback: Function to call with notification data
        """
        self.notification_callbacks.append(callback)

    def send_alert(self, alert: QualityAlert) -> None:
        """Send quality alert to operator.

        Args:
            alert: QualityAlert object
        """
        notification = {
            "type": "alert",
            "timestamp": alert.timestamp,
            "severity": alert.severity.value,
            "modality": alert.modality,
            "message": alert.message,
            "recovery_suggestion": alert.recovery_suggestion,
        }

        # Store notification
        self.notification_history.append(notification)

        # Send to callbacks
        for callback in self.notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                self.logger.error(f"Notification callback error: {e}")

    def send_quality_update(self, quality_metrics: MultiModalQualityMetrics) -> None:
        """Send quality update to operator.

        Args:
            quality_metrics: Current quality metrics
        """
        notification = {
            "type": "quality_update",
            "timestamp": quality_metrics.timestamp,
            "overall_quality": quality_metrics.overall_quality,
            "quality_level": quality_metrics.quality_level.value,
            "eeg_quality": quality_metrics.eeg_quality,
            "pupil_quality": quality_metrics.pupil_quality,
            "cardiac_quality": quality_metrics.cardiac_quality,
            "active_alerts": quality_metrics.active_alerts,
            "suggested_actions": quality_metrics.suggested_actions,
        }

        # Store notification
        self.notification_history.append(notification)

        # Send to callbacks
        for callback in self.notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                self.logger.error(f"Notification callback error: {e}")

    def send_status_message(self, message: str, severity: str = "info") -> None:
        """Send general status message to operator.

        Args:
            message: Status message
            severity: Message severity (info, warning, error)
        """
        notification = {
            "type": "status",
            "timestamp": time.time(),
            "severity": severity,
            "message": message,
        }

        # Store notification
        self.notification_history.append(notification)

        # Send to callbacks
        for callback in self.notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                self.logger.error(f"Notification callback error: {e}")

    def get_recent_notifications(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent notifications.

        Args:
            n: Number of recent notifications to retrieve

        Returns:
            List of notification dictionaries
        """
        return list(self.notification_history)[-n:]
