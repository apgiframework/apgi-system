"""ML Integration Module for APGI Framework
===========================================

This module provides a unified interface for accessing machine learning classification
tools in the APGI Framework, including consciousness state classification and
clinical disorder classification.

Features:
- Unified API for all ML classification tools
- Automatic model selection and hyperparameter tuning
- Cross-validation and performance evaluation
- Feature extraction and preprocessing
- Model persistence and loading
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. ML features will be limited.")

from ..analysis.ml_classification import (
    BiomarkerClassifierEnsemble,
    ConsciousnessClassifier,
    create_biomarker_features,
    feature_selection,
    hyperparameter_tuning,
)
from ..clinical.disorder_classification import (
    DisorderClassification,
    DisorderType,
    NeuralSignatureProfile,
)
from ..exceptions import APGIFrameworkError
from ..logging.standardized_logging import get_logger


class MLIntegrationError(APGIFrameworkError):
    """ML integration specific errors."""


class UnifiedMLClassifier:
    """Unified interface for all ML classification capabilities in APGI Framework.

    Provides access to consciousness state classification, disorder classification,
    and biomarker analysis through a single API.
    """

    def __init__(self, enable_advanced: bool = False):
        """Initialize unified ML classifier.

        Args:
            enable_advanced: Whether to enable advanced features (deep learning, etc.)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for ML classification. "
                "Install with: pip install scikit-learn"
            )

        self.enable_advanced = enable_advanced
        self.logger = get_logger(__name__)

        # Initialize classifiers
        self.consciousness_classifier: ConsciousnessClassifier | None = None
        self.disorder_classifier: DisorderClassification | None = None
        self.ensemble_classifier: BiomarkerClassifierEnsemble | None = None

        # Model storage
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

    def train_consciousness_classifier(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        algorithm: str = "random_forest",
        feature_names: Optional[List[str]] = None,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Train consciousness state classifier.

        Args:
            data: Feature matrix
            labels: Target labels (0=unconscious, 1=conscious)
            algorithm: Classification algorithm
            feature_names: Names of features
            test_size: Test set proportion
            cv_folds: Cross-validation folds

        Returns:
            Dictionary with training results and model performance
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                data, labels, test_size=test_size, random_state=42, stratify=labels
            )

            # Train classifier
            self.consciousness_classifier = ConsciousnessClassifier(
                algorithm=algorithm, random_state=42
            )
            self.consciousness_classifier.fit(X_train, y_train, feature_names)

            # Evaluate
            train_results = self.consciousness_classifier.evaluate(X_train, y_train, cv_folds=1)
            test_results = self.consciousness_classifier.evaluate(X_test, y_test, cv_folds=1)

            # Make predictions
            y_pred = self.consciousness_classifier.predict(X_test)
            y_proba = self.consciousness_classifier.predict_proba(X_test)

            results = {
                "algorithm": algorithm,
                "train_performance": {
                    "accuracy": train_results.accuracy,
                    "f1_score": train_results.f1_score,
                    "cross_val_scores": train_results.cross_val_scores,
                },
                "test_performance": {
                    "accuracy": test_results.accuracy,
                    "f1_score": test_results.f1_score,
                    "precision": test_results.precision,
                    "recall": test_results.recall,
                },
                "predictions": {
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "y_proba": y_proba[:, 1],
                },
                "feature_importance": test_results.feature_importance,
                "confusion_matrix": test_results.confusion_matrix,
            }

            self.logger.info(f"Trained {algorithm} consciousness classifier")
            return results

        except Exception as e:
            self.logger.error(f"Failed to train consciousness classifier: {e}")
            raise MLIntegrationError(f"Training failed: {e}")

    def train_disorder_classifier(
        self,
        profiles: List[NeuralSignatureProfile],
        disorders: List[DisorderType],
        algorithm: str = "random_forest",
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Train clinical disorder classifier.

        Args:
            profiles: List of neural signature profiles
            disorders: List of disorder types
            algorithm: Classification algorithm
            test_size: Test set proportion
            cv_folds: Cross-validation folds

        Returns:
            Dictionary with training results
        """
        try:
            # Initialize disorder classifier
            self.disorder_classifier = DisorderClassification(classifier_type=algorithm)

            # Train classifier
            results: Dict[str, Any] = self.disorder_classifier.train(profiles, disorders, cv_folds)

            self.logger.info(f"Trained {algorithm} disorder classifier")
            return results

        except Exception as e:
            self.logger.error(f"Failed to train disorder classifier: {e}")
            raise MLIntegrationError(f"Training failed: {e}")

    def train_ensemble_classifier(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        algorithms: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Train ensemble classifier combining multiple algorithms.

        Args:
            data: Feature matrix
            labels: Target labels
            algorithms: List of algorithms to include
            feature_names: Names of features
            test_size: Test set proportion

        Returns:
            Dictionary with ensemble results
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                data, labels, test_size=test_size, random_state=42, stratify=labels
            )

            # Train ensemble
            self.ensemble_classifier = BiomarkerClassifierEnsemble(algorithms)
            self.ensemble_classifier.fit(X_train, y_train, feature_names)

            # Evaluate ensemble
            ensemble_results = self.ensemble_classifier.evaluate_ensemble(X_test, y_test)

            # Get individual classifier results
            individual_results = {}
            for name, result in ensemble_results.items():
                if name != "ensemble":
                    individual_results[name] = {
                        "accuracy": result.accuracy,
                        "f1_score": result.f1_score,
                        "precision": result.precision,
                        "recall": result.recall,
                    }

            results = {
                "ensemble_performance": {
                    "accuracy": ensemble_results["ensemble"].accuracy,
                    "f1_score": ensemble_results["ensemble"].f1_score,
                    "precision": ensemble_results["ensemble"].precision,
                    "recall": ensemble_results["ensemble"].recall,
                    "confusion_matrix": ensemble_results["ensemble"].confusion_matrix,
                },
                "individual_performance": individual_results,
                "algorithms_used": algorithms,
            }

            self.logger.info("Trained ensemble classifier")
            return results

        except Exception as e:
            self.logger.error(f"Failed to train ensemble classifier: {e}")
            raise MLIntegrationError(f"Training failed: {e}")

    def predict_consciousness(
        self, data: Union[np.ndarray, pd.DataFrame], return_probabilities: bool = True
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Predict consciousness states.

        Args:
            data: Feature matrix
            return_probabilities: Whether to return probabilities

        Returns:
            Predictions or dictionary with predictions and probabilities
        """
        if self.consciousness_classifier is None:
            raise ValueError(
                "Consciousness classifier not trained. Call train_consciousness_classifier() first."
            )

        predictions = self.consciousness_classifier.predict(data)

        if return_probabilities:
            probabilities = self.consciousness_classifier.predict_proba(data)
            return {"predictions": predictions, "probabilities": probabilities[:, 1]}
        else:
            return predictions

    def predict_disorder(self, profile: NeuralSignatureProfile) -> Dict[str, Any]:
        """Predict disorder from neural signature profile.

        Args:
            profile: Neural signature profile

        Returns:
            Dictionary with prediction results
        """
        if self.disorder_classifier is None:
            raise ValueError(
                "Disorder classifier not trained. Call train_disorder_classifier() first."
            )

        result = self.disorder_classifier.classify(profile=profile)
        return {
            "predicted_disorder": (
                result.predicted_disorder.value
                if hasattr(result.predicted_disorder, "value")
                else str(result.predicted_disorder)
            ),
            "confidence": result.confidence,
            "probabilities": result.probabilities,
            "feature_importance": getattr(result, "feature_importance", None),
        }

    def extract_biomarker_features(
        self,
        eeg_data: np.ndarray,
        sfreq: float,
        channel_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Extract biomarker features from EEG data.

        Args:
            eeg_data: EEG data
            sfreq: Sampling frequency
            channel_names: Channel names

        Returns:
            Feature matrix
        """
        return create_biomarker_features(eeg_data, sfreq, channel_names)

    def optimize_hyperparameters(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        algorithm: str = "random_forest",
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for a classifier.

        Args:
            data: Feature matrix
            labels: Target labels
            algorithm: Algorithm to optimize
            param_grid: Parameter grid for optimization
            cv_folds: Cross-validation folds

        Returns:
            Optimization results
        """
        if param_grid is None:
            # Default parameter grids
            param_grids: Dict[str, Dict[str, List[Any]]] = {
                "random_forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                },
                "svm": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "linear"],
                },
                "gradient_boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                },
            }
            param_grid = param_grids.get(algorithm, param_grids["random_forest"])

        # Get classifier class
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.svm import SVC

        classifier_classes = {
            "random_forest": RandomForestClassifier,
            "svm": SVC,
            "gradient_boosting": GradientBoostingClassifier,
        }

        classifier_class = classifier_classes.get(algorithm, RandomForestClassifier)

        return hyperparameter_tuning(
            classifier_class,
            data,
            labels,
            cast(Dict[str, List[Any]], param_grid),
            cv_folds,
        )

    def select_features(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        k: int = 10,
        method: str = "f_classif",
    ) -> Dict[str, Any]:
        """Select most important features.

        Args:
            data: Feature matrix
            labels: Target labels
            k: Number of features to select
            method: Feature selection method

        Returns:
            Dictionary with selected features and importance scores
        """
        X_selected, selected_names = feature_selection(data, labels, k, method)

        # Calculate feature importance using a simple classifier
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(data, labels)

        if hasattr(rf, "feature_importances_"):
            importance_dict = dict(zip(selected_names, rf.feature_importances_))
        else:
            importance_dict = {}

        return {
            "selected_features": X_selected,
            "feature_names": selected_names,
            "importance_scores": importance_dict,
        }

    def save_model(self, model_type: str, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Save trained model to file.

        Args:
            model_type: Type of model to save
            filepath: Path to save file

        Returns:
            Path to saved model
        """
        import pickle

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.model_dir / f"{model_type}_model_{timestamp}.pkl"

        filepath = Path(filepath)

        # Select model to save
        model_map = {
            "consciousness": self.consciousness_classifier,
            "disorder": self.disorder_classifier,
            "ensemble": self.ensemble_classifier,
        }

        model = model_map.get(model_type)
        if model is None:
            raise ValueError(f"No trained model of type '{model_type}' available")

        with open(filepath, "wb") as f:
            pickle.dump(model, f)

        self.logger.info(f"Saved {model_type} model to {filepath}")
        return filepath

    def load_model(self, model_type: str, filepath: Union[str, Path]) -> None:
        """Load trained model from file.

        Args:
            model_type: Type of model to load
            filepath: Path to model file
        """
        from ..security.secure_pickle import safe_pickle_load

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Define expected model types for validation
        expected_types = {
            ConsciousnessClassifier,
            DisorderClassification,
            BiomarkerClassifierEnsemble,
        }

        model = safe_pickle_load(filepath, expected_types=expected_types)

        # Assign to appropriate classifier
        if model_type == "consciousness":
            self.consciousness_classifier = model
        elif model_type == "disorder":
            self.disorder_classifier = model
        elif model_type == "ensemble":
            self.ensemble_classifier = model
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.logger.info(f"Loaded {model_type} model from {filepath}")

    def get_available_algorithms(self) -> List[str]:
        """Get list of available classification algorithms.

        Returns:
            List of algorithm names
        """
        return [
            "random_forest",
            "svm",
            "gradient_boosting",
            "logistic_regression",
            "mlp",
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models.

        Returns:
            Dictionary with model information
        """
        info = {
            "consciousness_classifier": self.consciousness_classifier is not None,
            "disorder_classifier": self.disorder_classifier is not None,
            "ensemble_classifier": self.ensemble_classifier is not None,
            "available_algorithms": self.get_available_algorithms(),
            "model_directory": str(self.model_dir),
        }

        if self.consciousness_classifier:
            info["consciousness_algorithm"] = self.consciousness_classifier.algorithm

        if self.ensemble_classifier:
            info["ensemble_algorithms"] = self.ensemble_classifier.algorithms

        return info


# Convenience functions for quick access
def create_consciousness_classifier(
    algorithm: str = "random_forest",
) -> ConsciousnessClassifier:
    """Create a consciousness classifier with default settings."""
    return ConsciousnessClassifier(algorithm=algorithm)


def create_disorder_classifier(
    algorithm: str = "random_forest",
) -> DisorderClassification:
    """Create a disorder classifier with default settings."""
    return DisorderClassification(classifier_type=algorithm)


def create_ensemble_classifier(
    algorithms: Optional[List[str]] = None,
) -> BiomarkerClassifierEnsemble:
    """Create an ensemble classifier with default settings."""
    return BiomarkerClassifierEnsemble(algorithms=algorithms)
