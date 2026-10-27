#!/usr/bin/env python3
"""Machine Learning Classification Tools for APGI Framework
=========================================================

This module provides machine learning tools for classifying consciousness states,
biomarker analysis, and predictive modeling in neuroscience experiments.

Features:
- Consciousness state classification
- Biomarker-based prediction models
- Cross-validation and model evaluation
- Feature importance analysis
- Automated model selection
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from apgi_framework.logging.standardized_logging import get_logger

logger = get_logger(__name__)


try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. ML classification features will be limited.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning features will be limited.")


@dataclass
class ClassificationResult:
    """Container for classification results."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    feature_importance: Optional[Dict[str, float]] = None
    cross_val_scores: Optional[List[float]] = None
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None


class ConsciousnessClassifier:
    """Machine learning classifier for consciousness states.

    Supports multiple algorithms and provides comprehensive evaluation
    metrics for classifying conscious vs unconscious states from biomarkers.
    """

    def __init__(self, algorithm: str = "random_forest", random_state: int = 42):
        """Initialize the consciousness classifier.

        Parameters
        ----------
        algorithm : str
            Classification algorithm ('random_forest', 'svm', 'gradient_boosting',
                                    'logistic_regression', 'mlp', 'deep_nn')
        random_state : int
            Random state for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for ConsciousnessClassifier")

        self.algorithm = algorithm
        self.random_state = random_state
        self.model: Union[
            RandomForestClassifier,
            SVC,
            GradientBoostingClassifier,
            LogisticRegression,
            MLPClassifier,
            "DeepConsciousnessNet",
            None,
        ] = None
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None

        self._initialize_model()  # type: ignore[no-untyped-call]

    def _initialize_model(self) -> None:
        """Initialize the ML model based on selected algorithm."""
        if self.algorithm == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )
        elif self.algorithm == "svm":
            self.model = SVC(kernel="rbf", probability=True, random_state=self.random_state)
        elif self.algorithm == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state
            )
        elif self.algorithm == "logistic_regression":
            self.model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif self.algorithm == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=self.random_state,
                max_iter=1000,
            )
        elif self.algorithm == "deep_nn":
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch required for deep neural network")
            # Will be initialized when fitting
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
    ) -> "ConsciousnessClassifier":
        """Fit the classifier to training data.

        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels (0=unconscious, 1=conscious)
        feature_names : list, optional
            Names of features for interpretability

        Returns
        -------
        ConsciousnessClassifier
            Fitted classifier
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values
        else:
            X = np.array(X)

        if isinstance(y, pd.Series):
            y = np.asarray(y.values)
        else:
            y = np.asarray(y)

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        if self.algorithm == "deep_nn":
            self.model = DeepConsciousnessNet(input_size=X.shape[1])
            self._train_deep_model(X_scaled, np.asarray(y))
        else:
            if self.model is not None and hasattr(self.model, "fit"):
                self.model.fit(X_scaled, np.asarray(y))  # type: ignore[operator]

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict consciousness states.

        Parameters
        ----------
        X : array-like
            Feature matrix

        Returns
        -------
        np.ndarray
            Predicted labels (0=unconscious, 1=conscious)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.array(X)

        X_scaled = self.scaler.transform(X)

        if self.algorithm == "deep_nn":
            return self._predict_deep_model(X_scaled)
        else:
            if self.model is not None and hasattr(self.model, "predict"):
                return np.asarray(self.model.predict(X_scaled))  # type: ignore
            else:
                raise ValueError("Model not available for prediction")

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict consciousness probabilities.

        Parameters
        ----------
        X : array-like
            Feature matrix

        Returns
        -------
        np.ndarray
            Predicted probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.array(X)

        X_scaled = self.scaler.transform(X)

        if self.algorithm == "deep_nn":
            return self._predict_proba_deep_model(X_scaled)
        elif self.model is not None and hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X_scaled))  # type: ignore
        else:
            # For models without predict_proba, use decision function
            decision = self.model.decision_function(X_scaled)  # type: ignore[operator]
            prob_positive = 1 / (1 + np.exp(-np.asarray(decision)))
            return np.column_stack([1 - prob_positive, prob_positive])

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        cv_folds: int = 5,
    ) -> ClassificationResult:
        """Evaluate classifier performance.

        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            True labels
        cv_folds : int
            Number of cross-validation folds

        Returns
        -------
        ClassificationResult
            Comprehensive evaluation results
        """
        if isinstance(X, pd.DataFrame):
            X = np.asarray(X.values)
        if isinstance(y, pd.Series):
            y = np.asarray(y.values)

        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        # Calculate metrics
        report = classification_report(y, predictions, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y, predictions)

        # Cross-validation
        cv_scores = None
        if cv_folds > 1:
            pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", self.model)])
            cv_scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                scoring="accuracy",
            ).tolist()

        # Feature importance
        feature_importance = None
        if self.model is not None and hasattr(self.model, "feature_importances_"):
            feature_importance = dict(
                zip(
                    self.feature_names or [],
                    list(np.asarray(self.model.feature_importances_)),
                )
            )
        elif self.model is not None and hasattr(self.model, "coef_"):
            # For linear models
            importance = np.abs(np.asarray(self.model.coef_[0]))  # type: ignore
            feature_importance = dict(zip(self.feature_names or [], importance))

        return ClassificationResult(
            model_name=f"{self.algorithm}_classifier",
            accuracy=report["accuracy"],
            precision=report["weighted avg"]["precision"],
            recall=report["weighted avg"]["recall"],
            f1_score=report["weighted avg"]["f1-score"],
            confusion_matrix=conf_matrix,
            feature_importance=feature_importance,
            cross_val_scores=cv_scores,
            predictions=predictions,
            probabilities=probabilities[:, 1],  # Probability of conscious state
        )

    def _train_deep_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train deep neural network model."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for deep learning")

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Training loop
        if self.model is None:
            raise ValueError("Model not initialized")
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(50):  # Simple training
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def _predict_deep_model(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with deep model."""
        if self.model is None:
            raise ValueError("Model not initialized")
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()

    def _predict_proba_deep_model(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with deep model."""
        if self.model is None:
            raise ValueError("Model not initialized")
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()


if PYTORCH_AVAILABLE:

    class DeepConsciousnessNet(nn.Module):
        """Deep neural network for consciousness classification."""

        def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32]):
            """Initialize deep network.

            Parameters
            ----------
            input_size : int
                Number of input features
            """
            super(DeepConsciousnessNet, self).__init__()

            layers = []
            prev_size = input_size

            for hidden_size in hidden_sizes:
                layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(0.3)])
                prev_size = hidden_size

            layers.extend([nn.Linear(prev_size, 2), nn.Softmax(dim=1)])  # Binary classification

            self.network = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            """Forward pass."""
            return self.network(x)

else:
    # Create a dummy class when PyTorch is not available
    class DeepConsciousnessNetDummy:
        """Dummy class when PyTorch is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch required for DeepConsciousnessNet")

    # Alias for compatibility - use type: ignore to bypass assignment check
    DeepConsciousnessNet: type[Any] = DeepConsciousnessNetDummy  # type: ignore


class BiomarkerClassifierEnsemble:
    """Ensemble classifier combining multiple algorithms for robust consciousness detection."""

    def __init__(self, algorithms: Optional[List[str]] = None, random_state: int = 42):
        """Initialize ensemble classifier.

        Parameters
        ----------
        algorithms : list, optional
            List of algorithms to include in ensemble
        random_state : int
            Random state for reproducibility
        """
        if algorithms is None:
            algorithms = ["random_forest", "gradient_boosting", "svm", "mlp"]

        self.algorithms = algorithms
        self.random_state = random_state
        self.classifiers: Dict[str, ConsciousnessClassifier] = {}
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
    ) -> "BiomarkerClassifierEnsemble":
        """Fit ensemble classifiers.

        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        feature_names : list, optional
            Feature names

        Returns
        -------
        BiomarkerClassifierEnsemble
            Fitted ensemble
        """
        self.feature_names = feature_names

        for algorithm in self.algorithms:
            classifier = ConsciousnessClassifier(algorithm, self.random_state)
            classifier.fit(X, y, feature_names)
            self.classifiers[algorithm] = classifier

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict using ensemble voting.

        Parameters
        ----------
        X : array-like
            Feature matrix

        Returns
        -------
        np.ndarray
            Ensemble predictions
        """
        if not self.classifiers:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get predictions from all classifiers
        predictions_list: List[np.ndarray] = []
        for classifier in self.classifiers.values():
            pred = classifier.predict(X)
            predictions_list.append(pred)

        predictions_array = np.array(predictions_list)

        # Majority voting
        ensemble_predictions = []
        for i in range(predictions_array.shape[1]):
            votes = predictions_array[:, i]
            # Use mode (most frequent prediction)
            unique, counts = np.unique(votes, return_counts=True)
            ensemble_predictions.append(unique[np.argmax(counts)])

        return np.array(ensemble_predictions)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities using ensemble averaging.

        Parameters
        ----------
        X : array-like
            Feature matrix

        Returns
        -------
        np.ndarray
            Average predicted probabilities
        """
        if not self.classifiers:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get probabilities from all classifiers
        probabilities_list = []
        for classifier in self.classifiers.values():
            probabilities_list.append(classifier.predict_proba(X))

        probabilities_array = np.array(probabilities_list)

        # Average probabilities
        return np.asarray(np.mean(probabilities_array, axis=0))

    def evaluate_ensemble(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, ClassificationResult]:
        """Evaluate all classifiers in the ensemble.

        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            True labels

        Returns
        -------
        dict
            Evaluation results for each classifier
        """
        results = {}

        for name, classifier in self.classifiers.items():
            result = classifier.evaluate(X, y)
            results[name] = result

        # Add ensemble result
        ensemble_predictions = self.predict(X)
        ensemble_probabilities = self.predict_proba(X)

        report = classification_report(y, ensemble_predictions, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y, ensemble_predictions)

        results["ensemble"] = ClassificationResult(
            model_name="ensemble_classifier",
            accuracy=report["accuracy"],
            precision=report["weighted avg"]["precision"],
            recall=report["weighted avg"]["recall"],
            f1_score=report["weighted avg"]["f1-score"],
            confusion_matrix=conf_matrix,
            predictions=ensemble_predictions,
            probabilities=ensemble_probabilities[:, 1],
        )

        return results


def create_biomarker_features(
    eeg_data: np.ndarray, sfreq: float, channel_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Extract biomarker features from EEG data for classification.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data, shape (n_channels, n_times) or (n_times,)
    sfreq : float
        Sampling frequency
    channel_names : list, optional
        Names of EEG channels

    Returns
    -------
    pd.DataFrame
        Feature matrix with biomarker features
    """
    if eeg_data.ndim == 1:
        eeg_data = eeg_data[np.newaxis, :]

    if channel_names is None:
        channel_names = [f"Ch{i + 1}" for i in range(eeg_data.shape[0])]

    features = {}

    # Basic statistical features
    for i, ch_name in enumerate(channel_names):
        channel_data = eeg_data[i]

        features[f"{ch_name}_mean"] = np.mean(channel_data)
        features[f"{ch_name}_std"] = np.std(channel_data)
        features[f"{ch_name}_skew"] = pd.Series(channel_data).skew()
        features[f"{ch_name}_kurtosis"] = pd.Series(channel_data).kurtosis()
        features[f"{ch_name}_rms"] = np.sqrt(np.mean(channel_data**2))

    # Frequency domain features (simplified)
    from scipy import signal

    freqs, psd = signal.welch(eeg_data, fs=sfreq, nperseg=min(1024, eeg_data.shape[1]))

    # Band power ratios
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 100),
    }

    for band_name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.mean(psd[:, mask], axis=1)

        for i, ch_name in enumerate(channel_names):
            features[f"{ch_name}_{band_name}_power"] = band_power[i]

        # Alpha/beta ratio (consciousness marker)
        if band_name == "alpha":
            alpha_power = band_power
        elif band_name == "beta":
            beta_power = band_power

    # Add alpha/beta ratio
    for i, ch_name in enumerate(channel_names):
        features[f"{ch_name}_alpha_beta_ratio"] = alpha_power[i] / (beta_power[i] + 1e-10)

    # Connectivity features (simplified - correlation between channels)
    if eeg_data.shape[0] > 1:
        corr_matrix = np.corrcoef(eeg_data)
        # Use upper triangle correlations as features
        upper_corr = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        for i, corr in enumerate(upper_corr):
            features[f"ch_corr_{i}"] = corr

        features["mean_correlation"] = np.mean(upper_corr)

    return pd.DataFrame([features])


def hyperparameter_tuning(
    classifier_class: type,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    param_grid: Dict[str, List[Any]],
    cv: int = 5,
    scoring: str = "accuracy",
) -> Dict[str, Any]:
    """Perform hyperparameter tuning for a classifier.

    Parameters
    ----------
    classifier_class : type
        Classifier class to tune
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    param_grid : dict
        Parameter grid for tuning
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric

    Returns
    -------
    dict
        Best parameters and score
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for hyperparameter tuning")

    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", classifier_class())])

    # Update parameter names for pipeline
    pipeline_params = {f"classifier__{k}": v for k, v in param_grid.items()}

    grid_search = GridSearchCV(pipeline, pipeline_params, cv=cv, scoring=scoring, n_jobs=-1)

    grid_search.fit(X, y)

    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
    }


def feature_selection(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    k: int = 10,
    method: str = "f_classif",
) -> Tuple[np.ndarray, List[str]]:
    """Select most important features for classification.

    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    k : int
        Number of features to select
    method : str
        Feature selection method

    Returns
    -------
    tuple
        Selected features and feature names
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for feature selection")

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_array = X.values
    else:
        X_array = np.array(X)
        feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

    if method == "f_classif":
        selector = SelectKBest(score_func=f_classif, k=min(k, X_array.shape[1]))
    else:
        raise ValueError(f"Unknown feature selection method: {method}")

    X_selected = selector.fit_transform(X_array, y)

    # Get selected feature names
    selected_mask = selector.get_support()
    selected_names = [name for name, selected in zip(feature_names, selected_mask) if selected]

    return X_selected, selected_names


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample biomarker data
    n_samples = 200
    n_features = 20

    # Create synthetic features
    X = np.random.randn(n_samples, n_features)
    feature_names = [f"biomarker_{i}" for i in range(n_features)]

    # Create synthetic labels (0=unconscious, 1=conscious)
    # Make some features more predictive
    X[:, 0] += 2  # Strong predictor for conscious
    X[:, 1] -= 1.5  # Strong predictor for unconscious

    # Generate labels based on features
    linear_combo = 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.randn(n_samples) * 0.5
    y = (linear_combo > 0).astype(int)

    # Train and evaluate classifier
    classifier = ConsciousnessClassifier(algorithm="random_forest")
    classifier.fit(X, y, feature_names)

    results = classifier.evaluate(X, y)

    logger.info("Classification Results:")
    logger.info(f"Accuracy: {results.accuracy:.3f}")
    logger.info(f"F1 Score: {results.f1_score:.3f}")
    logger.info(f"Cross-validation scores: {results.cross_val_scores}")

    if results.feature_importance:
        logger.info("\nTop 5 Important Features:")
        sorted_features = sorted(
            results.feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        for name, importance in sorted_features[:5]:
            logger.info(f"{name}: {importance:.3f}")

    # Ensemble classification
    logger.info("\n" + "=" * 50)
    logger.info("Ensemble Classification:")

    ensemble = BiomarkerClassifierEnsemble()
    ensemble.fit(X, y, feature_names)

    ensemble_results = ensemble.evaluate_ensemble(X, y)
    logger.info(f"Ensemble Accuracy: {ensemble_results['ensemble'].accuracy:.3f}")
    logger.info(f"Ensemble F1 Score: {ensemble_results['ensemble'].f1_score:.3f}")
