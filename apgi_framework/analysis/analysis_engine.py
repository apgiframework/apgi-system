"""APGI Analysis Engine.

Provides comprehensive analysis capabilities for experimental data including:
- Statistical analysis
- Visualization generation
- Results processing and reporting
"""

import concurrent.futures
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional seaborn import
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None  # type: ignore

from ..exceptions import AnalysisError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    analysis_id: str
    timestamp: datetime
    analysis_type: str

    # Statistical results
    statistics: Dict[str, float]
    p_values: Dict[str, float]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

    # Visualization data
    plots: Dict[str, str]  # plot_id -> file_path
    figure_data: Dict[str, Any]  # raw figure data for interactive plots

    # Metadata
    parameters: Dict[str, Any]
    data_summary: Dict[str, Any]
    notes: List[str]


class AnalysisEngine:
    """Comprehensive analysis engine for APGI experimental data.

    Features:
    - Statistical analysis (t-tests, ANOVA, correlation, regression)
    - Effect size calculations
    - Visualization generation
    - Results processing and export
    """

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the analysis engine."""
        self.output_dir = Path(output_dir) if output_dir else Path("apgi_outputs/analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize plotting style
        try:
            plt.style.use("seaborn-v0_8")
        except Exception:
            try:
                plt.style.use("seaborn")
            except Exception:
                plt.style.use("ggplot")
        if HAS_SEABORN and sns is not None:
            sns.set_palette("husl")

        # Analysis registry
        self.analysis_functions = {
            "descriptive": self._descriptive_analysis,
            "comparative": self._comparative_analysis,
            "correlation": self._correlation_analysis,
            "regression": self._regression_analysis,
            "time_series": self._time_series_analysis,
            "bayesian": self._bayesian_analysis,
        }

        logger.info(f"AnalysisEngine initialized with output directory: {self.output_dir}")

    def analyze(
        self,
        data: pd.DataFrame,
        analysis_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        generate_plots: bool = True,
    ) -> AnalysisResult:
        """Perform comprehensive analysis on experimental data.

        Args:
            data: Experimental data as DataFrame
            analysis_type: Type of analysis to perform
            parameters: Analysis-specific parameters
            generate_plots: Whether to generate visualizations

        Returns:
            AnalysisResult with all findings

        Raises:
            AnalysisError: If analysis fails
            ValidationError: If inputs are invalid
        """
        if analysis_type not in self.analysis_functions:
            raise ValidationError(f"Unknown analysis type: {analysis_type}")

        if data.empty:
            raise ValidationError("Data cannot be empty")

        analysis_id = f"{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        try:
            logger.info(f"Starting {analysis_type} analysis: {analysis_id}")

            # Validate data
            self._validate_data(data, analysis_type)

            # Perform analysis
            analysis_func = self.analysis_functions[analysis_type]
            stats, p_values, effect_sizes, conf_intervals = analysis_func(data, parameters or {})

            # Generate visualizations
            plots: Dict[str, str] = {}
            figure_data: Dict[str, Any] = {}
            if generate_plots:
                plots, figure_data = self._generate_visualizations(
                    data, analysis_type, stats, analysis_id
                )

            # Create result
            result = AnalysisResult(
                analysis_id=analysis_id,
                timestamp=start_time,
                analysis_type=analysis_type,
                statistics=stats,
                p_values=p_values,
                effect_sizes=effect_sizes,
                confidence_intervals=conf_intervals,
                plots=plots,
                figure_data=figure_data,
                parameters=parameters or {},
                data_summary=self._summarize_data(data),
                notes=[],
            )

            # Save results
            self._save_results(result)

            logger.info(f"Analysis completed: {analysis_id}")
            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise AnalysisError(f"Analysis {analysis_type} failed: {e}")

    def _validate_data(self, data: pd.DataFrame, analysis_type: str) -> None:
        """Validate input data for specific analysis type."""
        if analysis_type == "comparative":
            # Check for grouping variable
            if not any(col.endswith("_group") for col in data.columns):
                raise ValidationError("Comparative analysis requires grouping variables")

        elif analysis_type == "correlation":
            # Check for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                raise ValidationError("Correlation analysis requires at least 2 numeric columns")

        elif analysis_type == "time_series":
            # Check for time column
            time_cols = [col for col in data.columns if "time" in col.lower() or col == "timestamp"]
            if not time_cols:
                raise ValidationError("Time series analysis requires time/timestamp column")

    def _descriptive_analysis(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        """Perform descriptive statistical analysis."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        stats = {}
        p_values = {}
        effect_sizes = {}
        conf_intervals = {}

        for col in numeric_cols:
            col_data = data[col].dropna()

            # Basic statistics
            stats[col] = {
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "count": len(col_data),
                "missing": data[col].isna().sum(),
            }

            # Confidence intervals
            ci = self._calculate_confidence_interval(np.asarray(col_data.values))
            conf_intervals[col] = ci

            # Normality test
            from scipy import stats as scipy_stats

            _, p_normal = scipy_stats.shapiro(col_data[:5000])  # Limit sample size
            p_values[f"{col}_normality"] = p_normal

            # Effect size (skewness, kurtosis)
            effect_sizes[f"{col}_skewness"] = scipy_stats.skew(col_data)
            effect_sizes[f"{col}_kurtosis"] = scipy_stats.kurtosis(col_data)

        return stats, p_values, effect_sizes, conf_intervals

    def _comparative_analysis(self, data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, float],
        Dict[str, Tuple[float, float]],
    ]:
        """Perform comparative analysis with parallel processing."""
        from scipy import stats as scipy_stats

        # Identify grouping variables
        group_cols = [col for col in data.columns if col.endswith("_group")]
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        stats: Dict[str, Any] = {}
        p_values: Dict[str, float] = {}
        effect_sizes: Dict[str, float] = {}
        conf_intervals: Dict[str, Tuple[float, float]] = {}

        if not group_cols:
            return stats, p_values, effect_sizes, conf_intervals

        # Function to analyze a single grouping variable
        def analyze_grouping_variable(
            group_col: str,
        ) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
            group_stats = {}
            group_p_values = {}
            group_effect_sizes = {}

            groups = data[group_col].unique()

            for num_col in numeric_cols:
                if num_col == group_col:
                    continue

                # Group data
                group_data = [data[data[group_col] == g][num_col].dropna() for g in groups]
                group_data = [g for g in group_data if len(g) > 0]

                if len(group_data) < 2:
                    continue

                # Perform appropriate test
                if len(group_data) == 2:
                    # T-test
                    t_stat, p_val = scipy_stats.ttest_ind(*group_data)
                    test_name = f"{num_col}_ttest_{group_col}"

                    # Cohen's d
                    pooled_std = np.sqrt(
                        sum((len(g) - 1) * g.var() for g in group_data)
                        / sum(len(g) - 1 for g in group_data)
                    )
                    cohens_d = (group_data[0].mean() - group_data[1].mean()) / pooled_std
                    group_effect_sizes[f"{test_name}_cohens_d"] = cohens_d
                else:
                    # ANOVA
                    f_stat, p_val = scipy_stats.f_oneway(*group_data)
                    test_name = f"{num_col}_anova_{group_col}"

                    # Eta squared
                    total_var = np.concatenate(group_data).var()
                    between_var = sum(
                        len(g) * (g.mean() - np.concatenate(group_data).mean()) ** 2
                        for g in group_data
                    ) / sum(len(g) for g in group_data)
                    eta_squared = between_var / total_var if total_var != 0 else 0.0
                    group_effect_sizes[f"{test_name}_eta_squared"] = eta_squared

                group_stats[test_name] = {
                    "test_statistic": t_stat if len(group_data) == 2 else f_stat,
                    "group_means": [g.mean() for g in group_data],
                    "group_stds": [g.std() for g in group_data],
                    "group_sizes": [len(g) for g in group_data],
                }

                group_p_values[test_name] = p_val

            return group_col, group_stats, group_p_values, group_effect_sizes

        # Process each grouping variable sequentially to avoid GIL issues with DataFrames
        for group_col in group_cols:
            try:
                (
                    group_col,
                    group_stats,
                    group_p_values,
                    group_effect_sizes,
                ) = analyze_grouping_variable(group_col)

                # Merge results
                stats.update(group_stats)
                p_values.update(group_p_values)
                effect_sizes.update(group_effect_sizes)

            except Exception as exc:
                logger.warning(f"Comparative analysis for {group_col} failed: {exc}")

        logger.info(
            f"Completed parallel comparative analysis for {len(group_cols)} grouping variables"
        )
        return stats, p_values, effect_sizes, conf_intervals

    def _correlation_analysis(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        """Perform correlation analysis with parallel processing."""
        from scipy import stats as scipy_stats

        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        numeric_cols = numeric_data.columns

        stats: Dict[str, Any] = {}
        p_values: Dict[str, Any] = {}
        effect_sizes: Dict[str, Any] = {}
        conf_intervals: Dict[str, Any] = {}

        # Generate pairs for parallel processing
        column_pairs = [
            (col1, col2) for i, col1 in enumerate(numeric_cols) for col2 in numeric_cols[i + 1 :]
        ]

        if not column_pairs:
            return stats, p_values, effect_sizes, conf_intervals

        # Function to compute correlation for a single pair
        def compute_correlation(pair: Tuple[str, str]) -> Dict[str, Any]:
            col1, col2 = pair
            x = numeric_data[col1]
            y = numeric_data[col2]

            # Pearson correlation
            r, p_val = scipy_stats.pearsonr(x, y)
            corr_name = f"{col1}_vs_{col2}_pearson"

            # Confidence interval for correlation
            ci = self._correlation_confidence_interval(r, len(x))

            return {
                "name": corr_name,
                "stats": {"correlation": r, "n": len(x)},
                "p_value": p_val,
                "effect_size": abs(r),
                "ci": ci,
            }

        # Use parallel processing for correlation computation
        max_workers = min(len(column_pairs), 8)  # Limit to 8 concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all correlation tasks
            future_to_pair = {
                executor.submit(compute_correlation, pair): pair for pair in column_pairs
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_pair):
                try:
                    result = future.result()
                    corr_name = result["name"]

                    stats[corr_name] = result["stats"]
                    p_values[corr_name] = result["p_value"]
                    effect_sizes[corr_name] = result["effect_size"]
                    conf_intervals[corr_name] = result["ci"]

                except Exception as exc:
                    pair = future_to_pair[future]
                    logger.warning(f"Correlation computation for {pair} failed: {exc}")

        logger.info(f"Completed parallel correlation analysis for {len(column_pairs)} pairs")
        return stats, p_values, effect_sizes, conf_intervals

    def _regression_analysis(self, data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, float],
        Dict[str, Tuple[float, float]],
    ]:
        """Perform regression analysis."""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score

        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        numeric_cols = numeric_data.columns

        stats: Dict[str, Any] = {}
        p_values: Dict[str, float] = {}
        effect_sizes: Dict[str, float] = {}
        conf_intervals: Dict[str, Tuple[float, float]] = {}

        # Simple linear regression for each pair
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols[i + 1 :], i + 1):
                X = np.asarray(numeric_data[col1].values).reshape(-1, 1)
                y = np.asarray(numeric_data[col2].values)

                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

                reg_name = f"{col2}_on_{col1}_regression"

                stats[reg_name] = {
                    "slope": model.coef_[0],
                    "intercept": model.intercept_,
                    "r_squared": r2_score(y, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y, y_pred)),
                    "n": len(X),
                }

                # Effect size: R-squared
                effect_sizes[reg_name] = r2_score(y, y_pred)

        return stats, p_values, effect_sizes, conf_intervals

    def _time_series_analysis(self, data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, float],
        Dict[str, Tuple[float, float]],
    ]:
        """Perform time series analysis."""
        numeric_data = data.select_dtypes(include=[np.number]).dropna()

        stats: Dict[str, Any] = {}
        p_values: Dict[str, float] = {}
        effect_sizes: Dict[str, float] = {}
        conf_intervals: Dict[str, Tuple[float, float]] = {}

        for col in numeric_data.columns:
            series = numeric_data[col]

            # Autocorrelation
            autocorr = [series.autocorr(lag=i) for i in range(1, min(11, len(series) // 2))]

            stats[f"{col}_autocorr"] = {
                "autocorrelations": autocorr,
                "trend": np.polyfit(range(len(series)), series, 1)[0],  # Linear trend
            }

            # Stationarity test (simplified)
            diff_series = series.diff().dropna()
            stationarity_ratio = diff_series.var() / series.var() if series.var() > 0 else 0
            effect_sizes[f"{col}_stationarity"] = stationarity_ratio

        return stats, p_values, effect_sizes, conf_intervals

    def _bayesian_analysis(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        """Perform basic Bayesian analysis."""
        numeric_data = data.select_dtypes(include=[np.number]).dropna()

        stats = {}
        p_values: Dict[str, float] = {}
        effect_sizes: Dict[str, float] = {}
        conf_intervals: Dict[str, Tuple[float, float]] = {}

        for col in numeric_data.columns:
            series = numeric_data[col]

            # Bayesian credible intervals (using normal approximation)
            mean = series.mean()
            std_err = series.std() / np.sqrt(len(series))

            # 95% credible interval
            ci_lower = mean - 1.96 * std_err
            ci_upper = mean + 1.96 * std_err

            stats[f"{col}_bayesian"] = {
                "posterior_mean": mean,
                "posterior_std": series.std(),
                "credible_interval_95": (ci_lower, ci_upper),
            }

            conf_intervals[f"{col}_bayesian"] = (ci_lower, ci_upper)

        return stats, p_values, effect_sizes, conf_intervals

    def _generate_visualizations(
        self,
        data: pd.DataFrame,
        analysis_type: str,
        stats: Dict[str, Any],
        analysis_id: str,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Generate visualizations for the analysis."""
        plots = {}
        figure_data = {}

        # Create plots directory
        plots_dir = self.output_dir / "plots" / analysis_id
        plots_dir.mkdir(parents=True, exist_ok=True)

        numeric_data = data.select_dtypes(include=[np.number])

        # Distribution plots
        if analysis_type in ["descriptive", "comparative"]:
            for col in numeric_data.columns[:6]:  # Limit to 6 columns
                fig, ax = plt.subplots(figsize=(8, 6))

                # Histogram with KDE
                sns.histplot(data[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")

                plot_path = plots_dir / f"{col}_distribution.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                plots[f"{col}_distribution"] = str(plot_path)

        # Correlation heatmap
        if analysis_type in ["correlation", "descriptive"] and len(numeric_data.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))

            corr_matrix = numeric_data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Correlation Matrix")

            plot_path = plots_dir / "correlation_heatmap.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            plots["correlation_heatmap"] = str(plot_path)
            figure_data["correlation_matrix"] = corr_matrix.to_dict()

        # Box plots for comparative analysis
        if analysis_type == "comparative":
            group_cols = [col for col in data.columns if col.endswith("_group")]

            for group_col in group_cols[:3]:  # Limit to 3 grouping variables
                for num_col in numeric_data.columns[:4]:  # Limit to 4 numeric variables
                    fig, ax = plt.subplots(figsize=(10, 6))

                    sns.boxplot(data=data, x=group_col, y=num_col, ax=ax)
                    ax.set_title(f"{num_col} by {group_col}")
                    ax.tick_params(axis="x", rotation=45)

                    plot_path = plots_dir / f"{num_col}_by_{group_col}_boxplot.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close()

                    plots[f"{num_col}_by_{group_col}_boxplot"] = str(plot_path)

        # Time series plots
        if analysis_type == "time_series":
            for col in numeric_data.columns[:4]:  # Limit to 4 columns
                fig, ax = plt.subplots(figsize=(12, 6))

                ax.plot(data.index, numeric_data[col])
                ax.set_title(f"Time Series: {col}")
                ax.set_xlabel("Time/Index")
                ax.set_ylabel(col)

                plot_path = plots_dir / f"{col}_timeseries.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                plots[f"{col}_timeseries"] = str(plot_path)

        return plots, figure_data

    def _calculate_confidence_interval(
        self, data: np.ndarray, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for mean."""
        from scipy import stats as scipy_stats

        n = len(data)
        mean = np.mean(data)
        std_err = scipy_stats.sem(data)

        t_critical = scipy_stats.t.ppf((1 + confidence) / 2, n - 1)

        ci_lower = mean - t_critical * std_err
        ci_upper = mean + t_critical * std_err

        return (ci_lower, ci_upper)

    def _correlation_confidence_interval(
        self, r: float, n: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for correlation coefficient."""
        from scipy import stats as scipy_stats

        # Fisher's z-transform
        z = np.arctanh(r)
        std_err = 1 / np.sqrt(n - 3)

        z_critical = scipy_stats.norm.ppf((1 + confidence) / 2)

        z_lower = z - z_critical * std_err
        z_upper = z + z_critical * std_err

        # Transform back
        r_lower = np.tanh(z_lower)
        r_upper = np.tanh(z_upper)

        return (r_lower, r_upper)

    def _summarize_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create summary of the dataset."""
        return {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "missing_values": data.isna().sum().to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "numeric_columns": list(data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(data.select_dtypes(include=["object", "category"]).columns),
        }

    def _save_results(self, result: AnalysisResult) -> None:
        """Save analysis results to files."""
        results_dir = self.output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = results_dir / f"{result.analysis_id}_summary.json"

        import json

        summary = {
            "analysis_id": result.analysis_id,
            "timestamp": result.timestamp.isoformat(),
            "analysis_type": result.analysis_type,
            "statistics": result.statistics,
            "p_values": result.p_values,
            "effect_sizes": result.effect_sizes,
            "confidence_intervals": result.confidence_intervals,
            "plots": result.plots,
            "parameters": result.parameters,
            "data_summary": result.data_summary,
            "notes": result.notes,
        }

        # Convert numpy types for JSON serialization
        def convert_numpy(obj: Any) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif hasattr(obj, "name") and "dtype" in str(type(obj)):
                return str(obj.name)
            elif isinstance(obj, (np.dtype, type)):
                return str(obj)
            return obj

        summary = convert_numpy(summary)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save detailed data
        data_path = results_dir / f"{result.analysis_id}_data.csv"

        # Create a summary DataFrame
        summary_data = []
        for key, value in result.statistics.items():
            summary_data.append({"metric": key, "value": str(value), "type": "statistic"})

        for key, value in result.p_values.items():
            summary_data.append({"metric": key, "value": str(value), "type": "p_value"})

        for key, value in result.effect_sizes.items():
            summary_data.append({"metric": key, "value": str(value), "type": "effect_size"})

        if summary_data:
            pd.DataFrame(summary_data).to_csv(data_path, index=False)

        logger.info(f"Results saved to {results_dir}")

    def get_analysis_summary(self, analysis_id: str) -> Dict[str, Any]:
        """Load and return analysis summary."""
        results_dir = self.output_dir / "results"
        summary_path = results_dir / f"{analysis_id}_summary.json"

        if not summary_path.exists():
            raise FileNotFoundError(f"Analysis {analysis_id} not found")

        import json

        with open(summary_path, "r") as f:
            return json.load(f)  # type: ignore

    def list_analyses(self) -> List[str]:
        """List all available analysis IDs."""
        results_dir = self.output_dir / "results"
        if not results_dir.exists():
            return []

        analysis_files = list(results_dir.glob("*_summary.json"))
        return [f.stem.replace("_summary", "") for f in analysis_files]
