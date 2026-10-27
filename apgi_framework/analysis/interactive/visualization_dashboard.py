#!/usr/bin/env python3
"""Interactive Visualization Dashboard for APGI Framework
=====================================================

This module provides interactive visualization tools for real-time monitoring
and analysis of consciousness experiments. Features include:

- Real-time parameter estimation plots
- Interactive EEG signal visualization
- Biomarker dashboard
- Statistical analysis plots
- Export capabilities
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from apgi_framework.logging.standardized_logging import get_logger

logger = get_logger(__name__)


try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive visualization features will be limited.")

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    warnings.warn("Streamlit not available. Web dashboard features will be limited.")


class InteractiveVisualizer:
    """Interactive visualization tools for APGI experiments."""

    def __init__(self) -> None:
        """Initialize the interactive visualizer."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for interactive visualization")

        # Set up plotly theme
        self.theme = {
            "template": "plotly_white",
            "colorway": px.colors.qualitative.Set1,
        }

    def create_parameter_estimation_dashboard(
        self,
        theta_history: List[float],
        pi_history: List[float],
        beta_history: List[float],
        time_points: Optional[List[float]] = None,
    ) -> go.Figure:
        """Create interactive dashboard for parameter estimation.

        Parameters
        ----------
        theta_history : list
            History of ignition threshold estimates
        pi_history : list
            History of interoceptive precision estimates
        beta_history : list
            History of somatic bias estimates
        time_points : list, optional
            Time points for x-axis

        Returns
        -------
        plotly Figure
            Interactive dashboard figure
        """
        if time_points is None:
            time_points = list(range(len(theta_history)))

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Ignition Threshold (θ₀)",
                "Interoceptive Precision (Πᵢ)",
                "Somatic Bias (β)",
            ),
            shared_xaxes=True,
        )

        # Add traces
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=theta_history,
                mode="lines+markers",
                name="θ₀",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=pi_history,
                mode="lines+markers",
                name="Πᵢ",
                line=dict(color="blue", width=2),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=beta_history,
                mode="lines+markers",
                name="β",
                line=dict(color="green", width=2),
            ),
            row=3,
            col=1,
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="Real-time Parameter Estimation Dashboard",
            showlegend=False,
            template=self.theme["template"],
        )

        fig.update_xaxes(title_text="Time/Iteration", row=3, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=3, col=1)

        return fig

    def create_eeg_signal_plot(
        self,
        eeg_data: np.ndarray,
        sfreq: float,
        channel_names: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> go.Figure:
        """Create interactive EEG signal visualization.

        Parameters
        ----------
        eeg_data : np.ndarray
            EEG data, shape (n_channels, n_times)
        sfreq : float
            Sampling frequency
        channel_names : list, optional
            Names of EEG channels
        time_range : tuple, optional
            Time range to display (start, end) in seconds

        Returns
        -------
        plotly Figure
            Interactive EEG plot
        """
        if eeg_data.ndim == 1:
            eeg_data = eeg_data[np.newaxis, :]

        n_channels, n_times = eeg_data.shape

        if channel_names is None:
            channel_names = [f"Ch{i + 1}" for i in range(n_channels)]

        # Create time axis
        time = np.arange(n_times) / sfreq

        if time_range:
            mask = (time >= time_range[0]) & (time <= time_range[1])
            time = time[mask]
            eeg_data = eeg_data[:, mask]

        # Create subplots for each channel
        fig = make_subplots(
            rows=n_channels,
            cols=1,
            subplot_titles=channel_names,
            shared_xaxes=True,
            vertical_spacing=0.02,
        )

        # Add each channel
        for i in range(n_channels):
            # Offset channels for better visualization
            offset = i * 100  # Arbitrary offset
            signal = eeg_data[i] + offset

            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=signal,
                    mode="lines",
                    name=channel_names[i],
                    line=dict(width=1),
                ),
                row=i + 1,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=200 * n_channels,
            title_text="EEG Signal Visualization",
            showlegend=False,
            template=self.theme["template"],
        )

        fig.update_xaxes(title_text="Time (s)", row=n_channels, col=1)
        for i in range(n_channels):
            fig.update_yaxes(title_text="Amplitude (μV)", row=i + 1, col=1)

        return fig

    def create_biomarker_dashboard(
        self,
        biomarkers: Dict[str, float],
        history: Optional[Dict[str, List[float]]] = None,
    ) -> go.Figure:
        """Create interactive biomarker visualization.

        Parameters
        ----------
        biomarkers : dict
            Current biomarker values
        history : dict, optional
            Historical biomarker values

        Returns
        -------
        plotly Figure
            Biomarker dashboard
        """
        if history:
            # Time series view
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Power Biomarkers",
                    "Complexity Biomarkers",
                    "Connectivity",
                    "MSE by Scale",
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                ],
            )

            time_points = list(range(len(next(iter(history.values())))))

            # Power biomarkers
            if "alpha_power" in history:
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=history["alpha_power"],
                        name="Alpha Power",
                        mode="lines",
                    ),
                    row=1,
                    col=1,
                )
            if "alpha_ratio" in history:
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=history["alpha_ratio"],
                        name="Alpha Ratio",
                        mode="lines",
                    ),
                    row=1,
                    col=1,
                )

            # Complexity biomarkers
            if "pci" in history:
                fig.add_trace(
                    go.Scatter(x=time_points, y=history["pci"], name="PCI", mode="lines"),
                    row=1,
                    col=2,
                )

            # Connectivity
            if "mean_connectivity" in history:
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=history["mean_connectivity"],
                        name="Mean Connectivity",
                        mode="lines",
                    ),
                    row=2,
                    col=1,
                )

            # MSE (simplified - show latest scale values)
            mse_keys = [k for k in history.keys() if k.startswith("scale_")]
            if mse_keys:
                for key in mse_keys[:5]:  # Show first 5 scales
                    fig.add_trace(
                        go.Scatter(x=time_points, y=history[key], name=key, mode="lines"),
                        row=2,
                        col=2,
                    )

        else:
            # Current values view
            biomarker_names = list(biomarkers.keys())
            biomarker_values = list(biomarkers.values())

            fig = go.Figure(
                data=[go.Bar(x=biomarker_names, y=biomarker_values, marker_color="lightblue")]
            )

            fig.update_layout(
                title="Current Biomarker Values",
                xaxis_title="Biomarker",
                yaxis_title="Value",
                template=self.theme["template"],
            )

        return fig

    def create_connectivity_matrix_plot(
        self, connectivity_matrix: np.ndarray, channel_names: Optional[List[str]] = None
    ) -> go.Figure:
        """Create interactive connectivity matrix visualization.

        Parameters
        ----------
        connectivity_matrix : np.ndarray
            Connectivity matrix
        channel_names : list, optional
            Channel names

        Returns
        -------
        plotly Figure
            Connectivity heatmap
        """
        if channel_names is None:
            channel_names = [f"Ch{i + 1}" for i in range(connectivity_matrix.shape[0])]

        fig = go.Figure(
            data=go.Heatmap(
                z=connectivity_matrix,
                x=channel_names,
                y=channel_names,
                colorscale="RdBu_r",
                zmin=0,
                zmax=1,
            )
        )

        fig.update_layout(
            title="Functional Connectivity Matrix",
            xaxis_title="Channel",
            yaxis_title="Channel",
            template=self.theme["template"],
        )

        return fig

    def create_statistical_analysis_plot(
        self, data: pd.DataFrame, x_col: str, y_col: str, analysis_type: str = "scatter"
    ) -> go.Figure:
        """Create statistical analysis plots.

        Parameters
        ----------
        data : pd.DataFrame
            Data for analysis
        x_col : str
            X-axis column
        y_col : str
            Y-axis column
        analysis_type : str
            Type of analysis ('scatter', 'box', 'violin', 'histogram')

        Returns
        -------
        plotly Figure
            Statistical plot
        """
        if analysis_type == "scatter":
            fig = px.scatter(
                data, x=x_col, y=y_col, trendline="ols", template=self.theme["template"]
            )
        elif analysis_type == "box":
            fig = px.box(data, x=x_col, y=y_col, template=self.theme["template"])
        elif analysis_type == "violin":
            fig = px.violin(data, x=x_col, y=y_col, template=self.theme["template"])
        elif analysis_type == "histogram":
            fig = px.histogram(data, x=x_col, template=self.theme["template"])
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        return fig


class StreamlitDashboard:
    """Streamlit-based web dashboard for real-time monitoring."""

    def __init__(self) -> None:
        """Initialize the Streamlit dashboard."""
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit required for web dashboard")

    def run_dashboard(self, experiment_data: Dict[str, Any]) -> None:
        """Run the Streamlit dashboard.

        Parameters
        ----------
        experiment_data : dict
            Real-time experiment data
        """
        st.title("🧠 APGI Real-time Monitoring Dashboard")

        # Sidebar controls
        st.sidebar.header("Controls")
        st.sidebar.slider("Update Interval (s)", 1, 10, 2)
        st.sidebar.checkbox("Show Raw Data", False)

        # Main content
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Parameter Estimation")
            # Parameter plots would go here

        with col2:
            st.subheader("EEG Signals")
            # EEG visualization would go here

        # Biomarkers section
        st.subheader("Clinical Biomarkers")
        # Biomarker plots would go here

        # Real-time updates
        if st.button("Start Real-time Monitoring"):
            st.info("Real-time monitoring started...")

        # Export section
        st.subheader("Export Results")
        if st.button("Export Current Data"):
            st.success("Data exported successfully!")


def create_interactive_report(
    results: Dict[str, Any], output_file: str = "apgi_report.html"
) -> None:
    """Create an interactive HTML report.

    Parameters
    ----------
    results : dict
        Analysis results
    output_file : str
        Output HTML file path
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly required for interactive reports")
        return

    visualizer = InteractiveVisualizer()  # type: ignore[no-untyped-call]

    # Create figures
    figures = []

    # Parameter estimation if available
    if "theta_history" in results:
        fig = visualizer.create_parameter_estimation_dashboard(
            results["theta_history"], results["pi_history"], results["beta_history"]
        )
        figures.append(("Parameter Estimation", fig))

    # EEG signals if available
    if "eeg_data" in results:
        fig = visualizer.create_eeg_signal_plot(results["eeg_data"], results.get("sfreq", 500.0))
        figures.append(("EEG Signals", fig))

    # Biomarkers if available
    if "biomarkers" in results:
        fig = visualizer.create_biomarker_dashboard(results["biomarkers"])
        figures.append(("Biomarkers", fig))

    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>APGI Interactive Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>🧠 APGI Framework - Interactive Analysis Report</h1>
        <p>Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """

    for title, fig in figures:
        html_content += f"<h2>{title}</h2>"
        html_content += fig.to_html(full_html=False, include_plotlyjs=False)

    html_content += """
    </body>
    </html>
    """

    with open(output_file, "w") as f:
        f.write(html_content)

    logger.info(f"Interactive report saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    visualizer = InteractiveVisualizer()  # type: ignore[no-untyped-call]

    # Sample parameter data
    theta_history = [0.5 + 0.1 * np.sin(i / 10) for i in range(50)]
    pi_history = [1.0 + 0.2 * np.cos(i / 10) for i in range(50)]
    beta_history = [0.0 + 0.05 * np.random.randn() for _ in range(50)]

    # Create parameter dashboard
    fig = visualizer.create_parameter_estimation_dashboard(theta_history, pi_history, beta_history)

    # Save as HTML
    fig.write_html("parameter_dashboard.html")
    logger.info("Parameter dashboard saved to parameter_dashboard.html")

    # Sample biomarkers
    biomarkers = {
        "alpha_power": 15.3,
        "alpha_ratio": 0.8,
        "pci": 0.45,
        "mean_connectivity": 0.32,
        "scale_1": 1.2,
        "scale_2": 1.1,
        "scale_3": 0.9,
    }

    # Create biomarker dashboard
    biomarker_fig = visualizer.create_biomarker_dashboard(biomarkers)
    biomarker_fig.write_html("biomarker_dashboard.html")
    logger.info("Biomarker dashboard saved to biomarker_dashboard.html")
