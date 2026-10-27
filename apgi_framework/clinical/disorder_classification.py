"""Disorder classification module for anxiety disorders.

Implements GAD, panic disorder, and social anxiety differentiation using
neural signatures and machine learning classification with cross-validation.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class DisorderType(Enum):
    """Types of anxiety disorders."""

    CONTROL = "control"
    GAD = "generalized_anxiety_disorder"
    PANIC = "panic_disorder"
    SOCIAL_ANXIETY = "social_anxiety_disorder"
    DEPRESSION = "major_depressive_disorder"
    PTSD = "post_traumatic_stress_disorder"


@dataclass
class NeuralSignatureProfile:
    """Neural signature profile for disorder classification."""

    # P3b metrics
    p3b_amplitude_extero: float = 0.0  # µV
    p3b_amplitude_intero: float = 0.0  # µV
    p3b_latency_extero: float = 350.0  # ms
    p3b_latency_intero: float = 350.0  # ms

    # Gamma synchrony
    gamma_power_frontal: float = 0.0
    gamma_power_posterior: float = 0.0
    gamma_coherence: float = 0.0

    # Microstate metrics
    microstate_duration: float = 0.0  # ms
    microstate_transitions: float = 0.0  # transitions/sec

    # Pupillometry
    pupil_dilation_intero: float = 0.0  # mm
    pupil_latency: float = 0.0  # ms

    # APGI parameters
    threshold: float = 3.5
    intero_precision: float = 1.5
    extero_precision: float = 2.0
    somatic_gain: float = 1.2

    # Behavioral metrics
    detection_threshold_intero: float = 0.5
    detection_threshold_extero: float = 0.5
    reaction_time_mean: float = 500.0  # ms
    reaction_time_variability: float = 100.0  # ms

    def to_feature_vector(self) -> np.ndarray:
        """Convert profile to feature vector for ML."""
        return np.array(
            [
                self.p3b_amplitude_extero,
                self.p3b_amplitude_intero,
                self.p3b_latency_extero,
                self.p3b_latency_intero,
                self.gamma_power_frontal,
                self.gamma_power_posterior,
                self.gamma_coherence,
                self.microstate_duration,
                self.microstate_transitions,
                self.pupil_dilation_intero,
                self.pupil_latency,
                self.threshold,
                self.intero_precision,
                self.extero_precision,
                self.somatic_gain,
                self.detection_threshold_intero,
                self.detection_threshold_extero,
                self.reaction_time_mean,
                self.reaction_time_variability,
            ]
        )

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for interpretation."""
        return [
            "p3b_amplitude_extero",
            "p3b_amplitude_intero",
            "p3b_latency_extero",
            "p3b_latency_intero",
            "gamma_power_frontal",
            "gamma_power_posterior",
            "gamma_coherence",
            "microstate_duration",
            "microstate_transitions",
            "pupil_dilation_intero",
            "pupil_latency",
            "threshold",
            "intero_precision",
            "extero_precision",
            "somatic_gain",
            "detection_threshold_intero",
            "detection_threshold_extero",
            "reaction_time_mean",
            "reaction_time_variability",
        ]


@dataclass
class ClassificationResult:
    """Result of disorder classification."""

    predicted_disorder: DisorderType
    confidence: float  # 0-1
    probabilities: Dict[DisorderType, float] = field(default_factory=dict)

    # Cross-validation metrics
    cv_accuracy: Optional[float] = None
    cv_std: Optional[float] = None

    # Feature importance
    feature_importance: Optional[Dict[str, float]] = None

    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None

    # Classification report
    classification_report: Optional[Dict[str, float]] = None

    # Neural signature profile
    neural_profile: Optional[NeuralSignatureProfile] = None


class DisorderClassification:
    """Disorder classification system for anxiety disorders.

    Implements GAD, panic disorder, and social anxiety differentiation
    using neural signatures and machine learning with cross-validation.
    """

    def __init__(self, classifier_type: str = "random_forest", random_state: int = 42) -> None:
        """Initialize disorder classifier.

        Args:
            classifier_type: Type of classifier ('random_forest', 'gradient_boosting', 'svm')
            random_state: Random state for reproducibility
        """
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.classifier = self._create_classifier()
        self.is_trained = False
        self.feature_names = NeuralSignatureProfile.feature_names()

    def _create_classifier(self) -> Any:
        """Create classifier based on type."""
        if self.classifier_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state,
                class_weight="balanced",
            )
        elif self.classifier_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
            )
        elif self.classifier_type == "svm":
            return SVC(
                kernel="rbf",
                probability=True,
                random_state=self.random_state,
                class_weight="balanced",
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def extract_neural_signature(
        self,
        p3b_data: Dict[str, float],
        gamma_data: Dict[str, float],
        microstate_data: Dict[str, float],
        pupil_data: Dict[str, float],
        apgi_params: Dict[str, float],
        behavioral_data: Dict[str, float],
    ) -> NeuralSignatureProfile:
        """Extract neural signature profile from experimental data.

        Args:
            p3b_data: P3b metrics (amplitude_extero, amplitude_intero, latency_extero, latency_intero)
            gamma_data: Gamma synchrony metrics (power_frontal, power_posterior, coherence)
            microstate_data: Microstate metrics (duration, transitions)
            pupil_data: Pupillometry metrics (dilation_intero, latency)
            apgi_params: APGI parameters (threshold, intero_precision, extero_precision, somatic_gain)
            behavioral_data: Behavioral metrics (detection_threshold_intero, detection_threshold_extero, rt_mean, rt_variability)

        Returns:
            NeuralSignatureProfile object

        Raises:
            ValueError: If input data is invalid or missing required fields
            TypeError: If input data types are incorrect
        """
        # Input validation
        self._validate_neural_signature_inputs(
            p3b_data,
            gamma_data,
            microstate_data,
            pupil_data,
            apgi_params,
            behavioral_data,
        )

        return NeuralSignatureProfile(
            p3b_amplitude_extero=p3b_data.get("amplitude_extero", 0.0),
            p3b_amplitude_intero=p3b_data.get("amplitude_intero", 0.0),
            p3b_latency_extero=p3b_data.get("latency_extero", 350.0),
            p3b_latency_intero=p3b_data.get("latency_intero", 350.0),
            gamma_power_frontal=gamma_data.get("power_frontal", 0.0),
            gamma_power_posterior=gamma_data.get("power_posterior", 0.0),
            gamma_coherence=gamma_data.get("coherence", 0.0),
            microstate_duration=microstate_data.get("duration", 0.0),
            microstate_transitions=microstate_data.get("transitions", 0.0),
            pupil_dilation_intero=pupil_data.get("dilation_intero", 0.0),
            pupil_latency=pupil_data.get("latency", 0.0),
            threshold=apgi_params.get("threshold", 3.5),
            intero_precision=apgi_params.get("intero_precision", 1.5),
            extero_precision=apgi_params.get("extero_precision", 2.0),
            somatic_gain=apgi_params.get("somatic_gain", 1.2),
            detection_threshold_intero=behavioral_data.get("detection_threshold_intero", 0.5),
            detection_threshold_extero=behavioral_data.get("detection_threshold_extero", 0.5),
            reaction_time_mean=behavioral_data.get("rt_mean", 500.0),
            reaction_time_variability=behavioral_data.get("rt_variability", 100.0),
        )

    def _validate_neural_signature_inputs(
        self,
        p3b_data: Dict[str, float],
        gamma_data: Dict[str, float],
        microstate_data: Dict[str, float],
        pupil_data: Dict[str, float],
        apgi_params: Dict[str, float],
        behavioral_data: Dict[str, float],
    ) -> None:
        """Validate inputs for neural signature extraction.

        Args:
            All input data dictionaries

        Raises:
            ValueError: If any input is invalid
            TypeError: If any input has wrong type
        """
        # Type validation
        for name, data in [
            ("p3b_data", p3b_data),
            ("gamma_data", gamma_data),
            ("microstate_data", microstate_data),
            ("pupil_data", pupil_data),
            ("apgi_params", apgi_params),
            ("behavioral_data", behavioral_data),
        ]:
            if not isinstance(data, dict):
                raise TypeError(f"{name} must be a dictionary, got {type(data).__name__}")

            if len(data) == 0:
                raise ValueError(f"{name} cannot be empty")

        # P3b data validation
        self._validate_p3b_data(p3b_data)

        # Gamma data validation
        self._validate_gamma_data(gamma_data)

        # Microstate data validation
        self._validate_microstate_data(microstate_data)

        # Pupil data validation
        self._validate_pupil_data(pupil_data)

        # APGI parameters validation
        self._validate_apgi_params(apgi_params)

        # Behavioral data validation
        self._validate_behavioral_data(behavioral_data)

    def _validate_p3b_data(self, p3b_data: Dict[str, float]) -> None:
        """Validate P3b data."""
        amplitude_keys = ["amplitude_extero", "amplitude_intero"]
        latency_keys = ["latency_extero", "latency_intero"]

        for key in amplitude_keys:
            if key in p3b_data:
                value = p3b_data[key]
                if not isinstance(value, (int, float)):
                    raise TypeError(f"P3b {key} must be numeric, got {type(value).__name__}")
                if not (0 <= value <= 50):  # P3b amplitude typically 0-50 μV
                    raise ValueError(f"P3b {key} must be between 0 and 50 μV, got {value}")

        for key in latency_keys:
            if key in p3b_data:
                value = p3b_data[key]
                if not isinstance(value, (int, float)):
                    raise TypeError(f"P3b {key} must be numeric, got {type(value).__name__}")
                if not (200 <= value <= 800):  # P3b latency typically 200-800 ms
                    raise ValueError(f"P3b {key} must be between 200 and 800 ms, got {value}")

    def _validate_gamma_data(self, gamma_data: Dict[str, float]) -> None:
        """Validate gamma data."""
        power_keys = ["power_frontal", "power_posterior"]

        for key in power_keys:
            if key in gamma_data:
                value = gamma_data[key]
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Gamma {key} must be numeric, got {type(value).__name__}")
                if not (0 <= value <= 100):  # Power normalized 0-100
                    raise ValueError(f"Gamma {key} must be between 0 and 100, got {value}")

        if "coherence" in gamma_data:
            value = gamma_data["coherence"]
            if not isinstance(value, (int, float)):
                raise TypeError(f"Gamma coherence must be numeric, got {type(value).__name__}")
            if not (0 <= value <= 1):  # Coherence 0-1
                raise ValueError(f"Gamma coherence must be between 0 and 1, got {value}")

    def _validate_microstate_data(self, microstate_data: Dict[str, float]) -> None:
        """Validate microstate data."""
        if "duration" in microstate_data:
            value = microstate_data["duration"]
            if not isinstance(value, (int, float)):
                raise TypeError(f"Microstate duration must be numeric, got {type(value).__name__}")
            if not (10 <= value <= 200):  # Duration 10-200 ms
                raise ValueError(f"Microstate duration must be between 10 and 200 ms, got {value}")

        if "transitions" in microstate_data:
            value = microstate_data["transitions"]
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Microstate transitions must be numeric, got {type(value).__name__}"
                )
            if not (0 <= value <= 50):  # Transitions per second
                raise ValueError(f"Microstate transitions must be between 0 and 50, got {value}")

    def _validate_pupil_data(self, pupil_data: Dict[str, float]) -> None:
        """Validate pupil data."""
        if "dilation_intero" in pupil_data:
            value = pupil_data["dilation_intero"]
            if not isinstance(value, (int, float)):
                raise TypeError(f"Pupil dilation must be numeric, got {type(value).__name__}")
            if not (-5 <= value <= 5):  # Dilation in mm, -5 to 5
                raise ValueError(f"Pupil dilation must be between -5 and 5 mm, got {value}")

        if "latency" in pupil_data:
            value = pupil_data["latency"]
            if not isinstance(value, (int, float)):
                raise TypeError(f"Pupil latency must be numeric, got {type(value).__name__}")
            if not (200 <= value <= 2000):  # Latency 200-2000 ms
                raise ValueError(f"Pupil latency must be between 200 and 2000 ms, got {value}")

    def _validate_apgi_params(self, apgi_params: Dict[str, float]) -> None:
        """Validate APGI parameters."""
        if "threshold" in apgi_params:
            value = apgi_params["threshold"]
            if not isinstance(value, (int, float)):
                raise TypeError(f"APGI threshold must be numeric, got {type(value).__name__}")
            if not (0.1 <= value <= 10):  # Threshold 0.1-10
                raise ValueError(f"APGI threshold must be between 0.1 and 10, got {value}")

        precision_keys = ["intero_precision", "extero_precision"]
        for key in precision_keys:
            if key in apgi_params:
                value = apgi_params[key]
                if not isinstance(value, (int, float)):
                    raise TypeError(f"APGI {key} must be numeric, got {type(value).__name__}")
                if not (0.1 <= value <= 10):  # Precision 0.1-10
                    raise ValueError(f"APGI {key} must be between 0.1 and 10, got {value}")

        if "somatic_gain" in apgi_params:
            value = apgi_params["somatic_gain"]
            if not isinstance(value, (int, float)):
                raise TypeError(f"APGI somatic_gain must be numeric, got {type(value).__name__}")
            if not (0.0 <= value <= 5.0):  # Gain 0-5
                raise ValueError(f"APGI somatic_gain must be between 0.0 and 5.0, got {value}")

    def _validate_behavioral_data(self, behavioral_data: Dict[str, float]) -> None:
        """Validate behavioral data."""
        threshold_keys = ["detection_threshold_intero", "detection_threshold_extero"]
        for key in threshold_keys:
            if key in behavioral_data:
                value = behavioral_data[key]
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Behavioral {key} must be numeric, got {type(value).__name__}")
                if not (0.0 <= value <= 1.0):  # Threshold 0-1
                    raise ValueError(f"Behavioral {key} must be between 0.0 and 1.0, got {value}")

        if "rt_mean" in behavioral_data:
            value = behavioral_data["rt_mean"]
            if not isinstance(value, (int, float)):
                raise TypeError(f"Behavioral rt_mean must be numeric, got {type(value).__name__}")
            if not (100 <= value <= 3000):  # RT 100-3000 ms
                raise ValueError(f"Behavioral rt_mean must be between 100 and 3000 ms, got {value}")

        if "rt_variability" in behavioral_data:
            value = behavioral_data["rt_variability"]
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Behavioral rt_variability must be numeric, got {type(value).__name__}"
                )
            if not (0 <= value <= 1000):  # Variability 0-1000 ms
                raise ValueError(
                    f"Behavioral rt_variability must be between 0 and 1000 ms, got {value}"
                )

    def train(
        self,
        profiles: List[NeuralSignatureProfile],
        labels: List[DisorderType],
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Train classifier with cross-validation.

        Args:
            profiles: List of neural signature profiles
            labels: List of disorder labels
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with training metrics
        """
        # Convert profiles to feature matrix
        X = np.array([p.to_feature_vector() for p in profiles])
        y = np.array([label.value for label in labels])

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Dynamic CV fold adjustment based on sample size
        n_samples = len(profiles)
        if n_samples < cv_folds:
            adjusted_folds = max(2, n_samples)
            logger.warning(
                f"Insufficient samples ({n_samples}) for {cv_folds}-fold CV. "
                f"Adjusting to {adjusted_folds}-fold CV."
            )
            cv_folds = adjusted_folds

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.classifier, X_scaled, y, cv=cv, scoring="accuracy")

        # Train on full dataset
        self.classifier.fit(X_scaled, y)
        self.is_trained = True

        # Get feature importance if available
        feature_importance = {}
        if hasattr(self.classifier, "feature_importances_"):
            importances = self.classifier.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))

        return {
            "cv_mean_accuracy": float(np.mean(cv_scores)),
            "cv_std_accuracy": float(np.std(cv_scores)),
            "cv_scores": cv_scores.tolist(),
            "n_samples": len(profiles),
            "n_features": X.shape[1],
            "feature_importance": (feature_importance if feature_importance is not None else {}),
        }

    def classify(
        self,
        profile: Optional[NeuralSignatureProfile] = None,
        neural_profile: Optional[NeuralSignatureProfile] = None,
    ) -> ClassificationResult:
        """Classify disorder from neural signature profile.

        Args:
            profile: Neural signature profile (primary parameter)
            neural_profile: Alternative parameter name for backward compatibility

        Returns:
            ClassificationResult with prediction and confidence
        """
        # Use whichever parameter was provided
        effective_profile = profile if profile is not None else neural_profile
        if effective_profile is None:
            raise ValueError("Either profile or neural_profile must be provided")

        if not self.is_trained:
            raise ValueError("Classifier must be trained before classification")

        # Convert to feature vector
        X = effective_profile.to_feature_vector().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Predict
        prediction = self.classifier.predict(X_scaled)[0]
        probabilities = self.classifier.predict_proba(X_scaled)[0]

        # Get class labels
        classes = self.classifier.classes_

        # Create probability dictionary
        prob_dict = {}
        for cls, prob in zip(classes, probabilities):
            disorder_type = DisorderType(cls)
            prob_dict[disorder_type] = float(prob)

        # Get predicted disorder and confidence
        predicted_disorder = DisorderType(prediction)
        confidence = float(np.max(probabilities))

        # Get feature importance if available
        feature_importance = None
        if hasattr(self.classifier, "feature_importances_"):
            importances = self.classifier.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))

        return ClassificationResult(
            predicted_disorder=predicted_disorder,
            confidence=confidence,
            probabilities=prob_dict,
            feature_importance=feature_importance,
            neural_profile=effective_profile,
        )

    def evaluate(
        self, profiles: List[NeuralSignatureProfile], true_labels: List[DisorderType]
    ) -> ClassificationResult:
        """Evaluate classifier on test set.

        Args:
            profiles: List of neural signature profiles
            true_labels: List of true disorder labels

        Returns:
            ClassificationResult with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before evaluation")

        # Convert to feature matrix
        X = np.array([p.to_feature_vector() for p in profiles])
        y_true = np.array([label.value for label in true_labels])

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        y_pred = self.classifier.predict(X_scaled)

        # Compute metrics
        accuracy = np.mean(y_pred == y_true)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, target_names=self.classifier.classes_)

        # Get feature importance
        feature_importance = None
        if hasattr(self.classifier, "feature_importances_"):
            importances = self.classifier.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))

        # Determine most common predicted disorder
        from collections import Counter

        most_common_pred = Counter(y_pred).most_common(1)[0][0]
        predicted_disorder = DisorderType(most_common_pred)

        return ClassificationResult(
            predicted_disorder=predicted_disorder,
            confidence=float(accuracy),
            cv_accuracy=float(accuracy),
            feature_importance=feature_importance,
            confusion_matrix=conf_matrix,
            classification_report=class_report,
        )

    def get_disorder_specific_signatures(self) -> Dict[DisorderType, Dict[str, Any]]:
        """Get disorder-specific neural signature patterns.

        Returns:
            Dictionary mapping disorders to their characteristic signatures
        """
        signatures = {
            DisorderType.GAD: {
                "threshold": 2.5,  # Lower threshold
                "intero_precision": 2.5,  # Elevated interoceptive precision
                "p3b_amplitude_intero": 8.0,  # Enhanced interoceptive P3b
                "gamma_coherence": 0.6,  # Elevated gamma synchrony
                "description": "Chronic low threshold with elevated interoceptive precision",
            },
            DisorderType.PANIC: {
                "threshold": 2.0,  # Very low threshold
                "intero_precision": 3.0,  # Highly elevated interoceptive precision
                "p3b_amplitude_intero": 10.0,  # Very high interoceptive P3b
                "somatic_gain": 2.0,  # Strong somatic bias
                "description": "Catastrophic interoceptive ignition with extreme somatic bias",
            },
            DisorderType.SOCIAL_ANXIETY: {
                "threshold": 2.8,  # Moderately low threshold
                "intero_precision": 2.0,  # Moderately elevated interoceptive precision
                "p3b_amplitude_extero": 7.0,  # Enhanced social stimulus P3b
                "somatic_gain": 1.8,  # Elevated somatic bias for social contexts
                "description": "Context-specific threshold lowering for social stimuli",
            },
            DisorderType.DEPRESSION: {
                "threshold": 4.5,  # Elevated threshold
                "intero_precision": 0.8,  # Reduced interoceptive precision
                "p3b_amplitude_extero": 3.0,  # Blunted P3b
                "somatic_gain": 0.5,  # Reduced somatic bias
                "description": "Elevated threshold with blunted interoceptive processing",
            },
            DisorderType.PTSD: {
                "threshold": 2.0,  # Low threshold for trauma cues
                "intero_precision": 2.8,  # Elevated interoceptive precision
                "p3b_amplitude_intero": 9.0,  # Enhanced P3b to trauma-related interoception
                "somatic_gain": 2.5,  # Very strong trauma-specific somatic markers
                "description": "Trauma-specific threshold lowering with strong somatic markers",
            },
            DisorderType.CONTROL: {
                "threshold": 3.5,  # Normal threshold
                "intero_precision": 1.5,  # Normal interoceptive precision
                "p3b_amplitude_intero": 5.0,  # Normal P3b
                "somatic_gain": 1.2,  # Normal somatic bias
                "description": "Balanced ignition dynamics with adaptive threshold",
            },
        }

        return signatures

    def compare_to_normative(self, profile: NeuralSignatureProfile) -> Dict[str, float]:
        """Compare profile to normative control data.

        Args:
            profile: Neural signature profile

        Returns:
            Dictionary with z-scores for each feature
        """
        # Get control signature
        control_sig = self.get_disorder_specific_signatures()[DisorderType.CONTROL]

        # Compute z-scores (simplified - would use actual normative data in practice)
        z_scores = {
            "threshold": (profile.threshold - float(control_sig["threshold"])) / 0.5,
            "intero_precision": (profile.intero_precision - float(control_sig["intero_precision"]))
            / 0.3,
            "p3b_amplitude_intero": (
                profile.p3b_amplitude_intero - float(control_sig["p3b_amplitude_intero"])
            )
            / 1.5,
            "somatic_gain": (profile.somatic_gain - float(control_sig["somatic_gain"])) / 0.2,
        }

        return z_scores
