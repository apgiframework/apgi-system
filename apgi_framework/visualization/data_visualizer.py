"""APGI Data Visualizer.

Provides comprehensive visualization capabilities for experimental data including:
- Interactive dashboards
- Real-time plotting
- Advanced statistical visualizations
- Export capabilities
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import plotly.offline as pyo  # type: ignore[import-untyped]
import seaborn as sns
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ..config.constants import VisualizationConstants
from ..exceptions import VisualizationError

logger = logging.getLogger(__name__)


class DataVisualizer:
    """Comprehensive data visualization system for APGI experiments.

    Features:
    - Interactive Plotly visualizations
    - Static matplotlib plots
    - Real-time dashboards
    - Statistical visualizations
    - Export capabilities
    """

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the data visualizer."""
        self.output_dir = Path(output_dir) if output_dir else Path("apgi_outputs/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize plotting styles
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # Color palettes
        self.color_palettes = {
            "default": px.colors.qualitative.Set1,
            "neural": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "physiological": ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"],
            "behavioral": ["#34495e", "#16a085", "#27ae60", "#2980b9", "#8e44ad"],
        }

        logger.info(f"DataVisualizer initialized with output directory: {self.output_dir}")

    def create_interactive_dashboard(
        self,
        data: pd.DataFrame,
        title: str = "APGI Analysis Dashboard",
        save_html: bool = True,
    ) -> go.Figure:
        """Create an interactive dashboard with multiple visualizations.

        Args:
            data: Experimental data
            title: Dashboard title
            save_html: Whether to save as HTML file

        Returns:
            Plotly figure object
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Data Overview",
                    "Correlations",
                    "Distributions",
                    "Time Series",
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "heatmap"}],
                    [{"type": "histogram"}, {"type": "scatter"}],
                ],
            )

            numeric_data = data.select_dtypes(include=[np.number])

            # 1. Data Overview (scatter plot of first two numeric columns)
            if len(numeric_data.columns) >= 2:
                col1, col2 = numeric_data.columns[:2]
                fig.add_trace(
                    go.Scatter(
                        x=numeric_data[col1],
                        y=numeric_data[col2],
                        mode="markers",
                        name=f"{col1} vs {col2}",
                        marker=dict(
                            size=VisualizationConstants.DEFAULT_MARKER_SIZE,
                            color=numeric_data[col1],
                            colorscale="Viridis",
                        ),
                    ),
                    row=1,
                    col=1,
                )

            # 2. Correlation heatmap
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr()
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale="RdBu",
                        zmid=0,
                        name="Correlation",
                    ),
                    row=1,
                    col=2,
                )

            # 3. Distribution histogram
            if len(numeric_data.columns) > 0:
                col = numeric_data.columns[0]
                fig.add_trace(
                    go.Histogram(
                        x=numeric_data[col].dropna(),
                        nbinsx=VisualizationConstants.DEFAULT_HISTOGRAM_BINS,
                        name=f"{col} Distribution",
                        marker_color="lightblue",
                    ),
                    row=2,
                    col=1,
                )

            # 4. Time series (if index is datetime or has time column)
            time_cols = [col for col in data.columns if "time" in col.lower()]
            if time_cols and len(numeric_data.columns) > 0:
                time_col = time_cols[0]
                value_col = numeric_data.columns[0]

                fig.add_trace(
                    go.Scatter(
                        x=data[time_col],
                        y=data[value_col],
                        mode="lines+markers",
                        name=f"{value_col} over Time",
                        line=dict(color="red"),
                    ),
                    row=2,
                    col=2,
                )
            elif len(numeric_data.columns) > 0:
                # Use index as time
                col = numeric_data.columns[0]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(numeric_data))),
                        y=numeric_data[col],
                        mode="lines+markers",
                        name=f"{col} Series",
                        line=dict(color="red"),
                    ),
                    row=2,
                    col=2,
                )

            # Update layout
            fig.update_layout(
                title=title,
                height=VisualizationConstants.DEFAULT_PLOT_HEIGHT,
                showlegend=True,
                template=VisualizationConstants.DEFAULT_TEMPLATE,
            )

            # Save HTML if requested
            if save_html:
                html_path = (
                    self.output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                )
                pyo.plot(fig, filename=str(html_path), auto_open=False)
                logger.info(f"Dashboard saved to {html_path}")

            return fig

        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            raise VisualizationError(f"Failed to create dashboard: {e}")

    def plot_neural_signatures(
        self, signatures: Dict[str, Any], save_path: Optional[str] = None
    ) -> go.Figure:
        """Create specialized plots for neural signature data.

        Args:
            signatures: Dictionary containing neural signature data
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        try:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "P3b Amplitude",
                    "Gamma PLV",
                    "BOLD Activation",
                    "PCI Values",
                ),
                specs=[
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "box"}],
                ],
            )

            # P3b Amplitude plot
            if "p3b_amplitude" in signatures:
                p3b_data = signatures["p3b_amplitude"]
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(p3b_data))),
                        y=p3b_data,
                        name="P3b Amplitude (μV)",
                        marker_color="blue",
                    ),
                    row=1,
                    col=1,
                )

            # Gamma PLV plot
            if "gamma_plv" in signatures:
                gamma_data = signatures["gamma_plv"]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(gamma_data))),
                        y=gamma_data,
                        mode="lines+markers",
                        name="Gamma PLV",
                        line=dict(color="green", width=2),
                    ),
                    row=1,
                    col=2,
                )

            # BOLD Activation
            if "bold_activations" in signatures:
                bold_data = signatures["bold_activations"]
                if isinstance(bold_data, dict):
                    regions = list(bold_data.keys())
                    activations = list(bold_data.values())
                    fig.add_trace(
                        go.Bar(
                            x=regions,
                            y=activations,
                            name="BOLD Z-scores",
                            marker_color="red",
                        ),
                        row=2,
                        col=1,
                    )

            # PCI Values
            if "pci_values" in signatures:
                pci_data = signatures["pci_values"]
                fig.add_trace(
                    go.Box(y=pci_data, name="PCI Distribution", marker_color="purple"),
                    row=2,
                    col=2,
                )

            # Update layout
            fig.update_layout(
                title="Neural Signatures Analysis",
                height=800,
                showlegend=True,
                template="plotly_white",
            )

            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Neural signatures plot saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Neural signatures plot failed: {e}")
            raise VisualizationError(f"Failed to plot neural signatures: {e}")

    def plot_falsification_results(
        self, results: Dict[str, Any], save_path: Optional[str] = None
    ) -> go.Figure:
        """Visualize falsification test results.

        Args:
            results: Falsification test results
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        try:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Test Outcomes",
                    "P-values",
                    "Effect Sizes",
                    "Confidence Intervals",
                ),
                specs=[
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "errorbar"}],
                ],
            )

            test_names = list(results.keys())

            # Test outcomes (passed/failed)
            outcomes = [1 if results[test].get("passed", False) else 0 for test in test_names]
            colors = ["green" if outcome == 1 else "red" for outcome in outcomes]

            fig.add_trace(
                go.Bar(
                    x=test_names,
                    y=outcomes,
                    name="Test Outcome",
                    marker_color=colors,
                    text=["PASS" if outcome == 1 else "FAIL" for outcome in outcomes],
                    textposition="auto",
                ),
                row=1,
                col=1,
            )

            # P-values
            p_values = [results[test].get("p_value", 1.0) for test in test_names]
            fig.add_trace(
                go.Scatter(
                    x=test_names,
                    y=p_values,
                    mode="markers",
                    name="P-values",
                    marker=dict(size=10, color=p_values, colorscale="Viridis"),
                    text=[f"p={p:.3f}" for p in p_values],
                    textposition="top center",
                ),
                row=1,
                col=2,
            )

            # Effect sizes
            effect_sizes = [results[test].get("effect_size", 0.0) for test in test_names]
            fig.add_trace(
                go.Bar(
                    x=test_names,
                    y=effect_sizes,
                    name="Effect Sizes",
                    marker_color="orange",
                ),
                row=2,
                col=1,
            )

            # Confidence intervals
            for i, test in enumerate(test_names):
                ci = results[test].get("confidence_interval", (0, 0))
                if isinstance(ci, (list, tuple)) and len(ci) == 2:
                    fig.add_trace(
                        go.Scatter(
                            x=[test, test],
                            y=[ci[0], ci[1]],
                            mode="lines",
                            name=f"CI {test}",
                            line=dict(width=5),
                            showlegend=False,
                        ),
                        row=2,
                        col=2,
                    )

            # Add significance threshold line for p-values
            fig.add_hline(
                y=0.05,
                line_dash="dash",
                line_color="red",
                annotation_text="α = 0.05",
                row=1,
                col=2,
            )

            # Update layout
            fig.update_layout(
                title="Falsification Test Results",
                height=800,
                showlegend=True,
                template="plotly_white",
                xaxis_tickangle=-45,
            )

            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Falsification results plot saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Falsification results plot failed: {e}")
            raise VisualizationError(f"Failed to plot falsification results: {e}")

    def create_statistical_summary_plot(
        self, stats_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> go.Figure:
        """Create comprehensive statistical summary visualization.

        Args:
            stats_data: Statistical analysis results
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        try:
            fig = make_subplots(
                rows=3,
                cols=2,
                subplot_titles=(
                    "Means Comparison",
                    "Standard Deviations",
                    "Sample Sizes",
                    "Confidence Intervals",
                    "Effect Sizes",
                    "P-value Distribution",
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "errorbar"}],
                    [{"type": "bar"}, {"type": "histogram"}],
                ],
            )

            # Extract data from stats
            variables = list(stats_data.get("statistics", {}).keys())

            if variables:
                # Means
                means = [stats_data["statistics"][var].get("mean", 0) for var in variables]
                fig.add_trace(
                    go.Bar(x=variables, y=means, name="Means", marker_color="blue"),
                    row=1,
                    col=1,
                )

                # Standard deviations
                stds = [stats_data["statistics"][var].get("std", 0) for var in variables]
                fig.add_trace(
                    go.Bar(x=variables, y=stds, name="Std Dev", marker_color="green"),
                    row=1,
                    col=2,
                )

                # Sample sizes
                counts = [stats_data["statistics"][var].get("count", 0) for var in variables]
                fig.add_trace(
                    go.Bar(x=variables, y=counts, name="N", marker_color="orange"),
                    row=2,
                    col=1,
                )

                # Confidence intervals
                for i, var in enumerate(variables):
                    ci = stats_data.get("confidence_intervals", {}).get(var, (0, 0))
                    if isinstance(ci, (list, tuple)) and len(ci) == 2:
                        fig.add_trace(
                            go.Scatter(
                                x=[var, var],
                                y=[ci[0], ci[1]],
                                mode="lines",
                                name=f"CI {var}",
                                line=dict(width=8),
                                showlegend=False,
                            ),
                            row=2,
                            col=2,
                        )

                # Effect sizes
                effect_vars = list(stats_data.get("effect_sizes", {}).keys())
                if effect_vars:
                    effect_values = list(stats_data["effect_sizes"].values())
                    fig.add_trace(
                        go.Bar(
                            x=effect_vars,
                            y=effect_values,
                            name="Effect Sizes",
                            marker_color="purple",
                        ),
                        row=3,
                        col=1,
                    )

                # P-value distribution
                p_values = list(stats_data.get("p_values", {}).values())
                if p_values:
                    fig.add_trace(
                        go.Histogram(x=p_values, name="P-values", nbinsx=20, marker_color="red"),
                        row=3,
                        col=2,
                    )

            # Update layout
            fig.update_layout(
                title="Statistical Summary",
                height=1000,
                showlegend=True,
                template="plotly_white",
                xaxis_tickangle=-45,
            )

            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Statistical summary plot saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Statistical summary plot failed: {e}")
            raise VisualizationError(f"Failed to create statistical summary: {e}")

    def plot_time_series_analysis(
        self, time_series_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> go.Figure:
        """Create time series analysis visualizations.

        Args:
            time_series_data: Time series data and analysis results
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        try:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Original Series",
                    "Trend Analysis",
                    "Autocorrelation",
                    "Decomposition",
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "scatter"}],
                ],
            )

            # Original time series
            for var_name, series_data in time_series_data.items():
                if isinstance(series_data, (list, np.ndarray)) and len(series_data) > 0:
                    x_values = list(range(len(series_data)))

                    # Original series
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=series_data,
                            mode="lines",
                            name=f"{var_name} Original",
                            line=dict(width=2),
                        ),
                        row=1,
                        col=1,
                    )

                    # Trend (simple moving average)
                    if len(series_data) > 10:
                        window = min(20, len(series_data) // 4)
                        trend = pd.Series(series_data).rolling(window=window).mean()
                        fig.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=trend,
                                mode="lines",
                                name=f"{var_name} Trend",
                                line=dict(width=3, dash="dash"),
                            ),
                            row=1,
                            col=2,
                        )

                    # Autocorrelation
                    autocorr = [
                        pd.Series(series_data).autocorr(lag=i)
                        for i in range(1, min(21, len(series_data) // 2))
                    ]
                    fig.add_trace(
                        go.Bar(
                            x=list(range(1, len(autocorr) + 1)),
                            y=autocorr,
                            name=f"{var_name} Autocorr",
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )

                    # Simple decomposition (trend + residual)
                    if len(series_data) > 10:
                        trend_line = np.polyfit(x_values, series_data, 1)
                        trend_values = np.polyval(trend_line, x_values)
                        residual = series_data - trend_values

                        fig.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=residual,
                                mode="lines",
                                name=f"{var_name} Residual",
                                line=dict(width=1),
                                showlegend=False,
                            ),
                            row=2,
                            col=2,
                        )

                    break  # Only plot first variable for clarity

            # Update layout
            fig.update_layout(
                title="Time Series Analysis",
                height=800,
                showlegend=True,
                template="plotly_white",
            )

            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Time series analysis plot saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Time series plot failed: {e}")
            raise VisualizationError(f"Failed to create time series plot: {e}")

    def export_plots(
        self,
        figures: List[go.Figure],
        format: str = "html",
        filename_prefix: str = "apgi_plot",
    ) -> List[str]:
        """Export multiple plots to specified format.

        Args:
            figures: List of Plotly figures
            format: Export format ('html', 'png', 'pdf', 'svg')
            filename_prefix: Prefix for output files

        Returns:
            List of output file paths
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_paths = []

            for i, fig in enumerate(figures):
                if format == "html":
                    filename = f"{filename_prefix}_{i + 1}_{timestamp}.html"
                    filepath = self.output_dir / filename
                    fig.write_html(str(filepath))

                elif format == "png":
                    filename = f"{filename_prefix}_{i + 1}_{timestamp}.png"
                    filepath = self.output_dir / filename
                    fig.write_image(str(filepath), width=1200, height=800)

                elif format == "pdf":
                    filename = f"{filename_prefix}_{i + 1}_{timestamp}.pdf"
                    filepath = self.output_dir / filename
                    fig.write_image(str(filepath), format="pdf")

                elif format == "svg":
                    filename = f"{filename_prefix}_{i + 1}_{timestamp}.svg"
                    filepath = self.output_dir / filename
                    fig.write_image(str(filepath), format="svg")

                output_paths.append(str(filepath))
                logger.info(f"Plot exported to {filepath}")

            return output_paths

        except Exception as e:
            logger.error(f"Plot export failed: {e}")
            raise VisualizationError(f"Failed to export plots: {e}")

    def create_comparison_plot(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        labels: Tuple[str, str] = ("Group 1", "Group 2"),
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Create comparison plots between two datasets.

        Args:
            data1: First dataset
            data2: Second dataset
            labels: Labels for the two datasets
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        try:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Distribution Comparison",
                    "Box Plot Comparison",
                    "Scatter Comparison",
                    "Summary Statistics",
                ),
                specs=[
                    [{"type": "histogram"}, {"type": "box"}],
                    [{"type": "scatter"}, {"type": "bar"}],
                ],
            )

            # Get common numeric columns
            numeric_cols1 = data1.select_dtypes(include=[np.number]).columns
            numeric_cols2 = data2.select_dtypes(include=[np.number]).columns
            common_cols = list(set(numeric_cols1) & set(numeric_cols2))

            if common_cols:
                col = common_cols[0]  # Use first common column

                # Distribution comparison
                fig.add_trace(
                    go.Histogram(
                        x=data1[col].dropna(),
                        name=f"{labels[0]}",
                        opacity=0.7,
                        marker_color="blue",
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Histogram(
                        x=data2[col].dropna(),
                        name=f"{labels[1]}",
                        opacity=0.7,
                        marker_color="red",
                    ),
                    row=1,
                    col=1,
                )

                # Box plot comparison
                combined_data = pd.concat(
                    [
                        data1[col].dropna().to_frame().assign(group=labels[0]),
                        data2[col].dropna().to_frame().assign(group=labels[1]),
                    ]
                )

                for label, group_data in combined_data.groupby("group"):
                    fig.add_trace(
                        go.Box(
                            y=group_data[col],
                            name=label,
                            marker_color="blue" if label == labels[0] else "red",
                        ),
                        row=1,
                        col=2,
                    )

                # Scatter comparison (if both have at least 2 numeric columns)
                if len(common_cols) >= 2:
                    col1, col2 = common_cols[:2]

                    fig.add_trace(
                        go.Scatter(
                            x=data1[col1],
                            y=data1[col2],
                            mode="markers",
                            name=labels[0],
                            marker=dict(color="blue", size=8),
                        ),
                        row=2,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=data2[col1],
                            y=data2[col2],
                            mode="markers",
                            name=labels[1],
                            marker=dict(color="red", size=8),
                        ),
                        row=2,
                        col=1,
                    )

                # Summary statistics comparison
                stats1 = [data1[col].mean(), data1[col].std(), data1[col].median()]
                stats2 = [data2[col].mean(), data2[col].std(), data2[col].median()]
                stat_labels = ["Mean", "Std", "Median"]

                fig.add_trace(
                    go.Bar(x=stat_labels, y=stats1, name=labels[0], marker_color="blue"),
                    row=2,
                    col=2,
                )

                fig.add_trace(
                    go.Bar(x=stat_labels, y=stats2, name=labels[1], marker_color="red"),
                    row=2,
                    col=2,
                )

            # Update layout
            fig.update_layout(
                title=f"Comparison: {labels[0]} vs {labels[1]}",
                height=800,
                showlegend=True,
                template="plotly_white",
                barmode="group",
            )

            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Comparison plot saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Comparison plot failed: {e}")
            raise VisualizationError(f"Failed to create comparison plot: {e}")
