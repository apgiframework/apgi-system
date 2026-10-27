"""APGI Results Processor.

Provides comprehensive results processing capabilities including:
- Data aggregation and summarization
- Report generation
- Export functionality
- Integration with analysis and visualization
"""

import json
import logging
import pickle
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd

from ..analysis.analysis_engine import AnalysisResult
from ..exceptions import ProcessingError, ValidationError
from ..security.secure_pickle import safe_pickle_dump, safe_pickle_load

logger = logging.getLogger(__name__)


@dataclass
class ProcessedResult:
    """Container for processed experimental results."""

    result_id: str
    timestamp: datetime

    experiment_type: str

    # Raw data
    raw_data: Dict[str, Any]

    # Processed data
    summary_statistics: Dict[str, float]
    aggregated_results: Dict[str, Any]

    # Analysis results
    analysis_results: Optional[Dict[str, AnalysisResult]] = None

    # Visualization data
    visualization_data: Optional[Dict[str, Any]] = None

    # Metadata
    parameters: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


class ResultsProcessor:
    """Comprehensive results processing system for APGI experiments.

    Features:
    - Data aggregation and summarization
    - Quality assessment
    - Report generation
    - Export capabilities
    - Batch processing
    """

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the results processor."""
        self.output_dir = Path(output_dir) if output_dir else Path("apgi_outputs/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "exports").mkdir(exist_ok=True)
        (self.output_dir / "archives").mkdir(exist_ok=True)

        # Processing configuration
        self.quality_thresholds = {
            "completeness": 0.8,  # 80% of data must be present
            "outlier_ratio": 0.1,  # Max 10% outliers
            "min_sample_size": 10,  # Minimum samples for analysis
            "max_missing_rate": 0.2,  # Max 20% missing values
        }

        logger.info(f"ResultsProcessor initialized with output directory: {self.output_dir}")

    def process_experiment_results(
        self,
        raw_results: Dict[str, Any],
        experiment_type: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ProcessedResult:
        """Process raw experimental results into structured format.

        Args:
            raw_results: Raw experimental data and results
            experiment_type: Type of experiment
            parameters: Experimental parameters

        Returns:
            ProcessedResult with structured data
        """
        try:
            result_id = f"{experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            logger.info(f"Processing results for {experiment_type}: {result_id}")

            # Validate input data
            self._validate_raw_results(raw_results)

            # Extract and process data
            summary_stats = self._calculate_summary_statistics(raw_results)
            aggregated_results = self._aggregate_results(raw_results)
            quality_metrics = self._assess_data_quality(raw_results)

            # Create processed result
            processed = ProcessedResult(
                result_id=result_id,
                timestamp=datetime.now(),
                experiment_type=experiment_type,
                raw_data=raw_results,
                summary_statistics=summary_stats,
                aggregated_results=aggregated_results,
                parameters=parameters or {},
                quality_metrics=quality_metrics,
            )

            # Save processed result
            self._save_processed_result(processed)

            logger.info(f"Results processed successfully: {result_id}")
            return processed

        except Exception as e:
            logger.error(f"Results processing failed: {e}")
            raise ProcessingError(f"Failed to process results: {e}")

    def batch_process_results(
        self,
        results_list: List[Dict[str, Any]],
        experiment_type: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[ProcessedResult]:
        """Process multiple experimental results in batch.

        Args:
            results_list: List of raw experimental results
            experiment_type: Type of experiments
            parameters: Experimental parameters

        Returns:
            List of ProcessedResult objects
        """
        try:
            processed_results = []

            logger.info(f"Starting batch processing of {len(results_list)} results")

            for i, raw_results in enumerate(results_list):
                try:
                    processed = self.process_experiment_results(
                        raw_results, experiment_type, parameters
                    )
                    processed_results.append(processed)

                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(results_list)} results")

                except Exception as e:
                    logger.error(
                        f"Failed to process result {i} (type: {type(raw_results).__name__}): {e}"
                    )
                    logger.debug(f"Result data that failed: {raw_results}")
                    continue

            # Create batch summary
            self._create_batch_summary(processed_results, experiment_type)

            logger.info(
                f"Batch processing completed: {len(processed_results)}/{len(results_list)} results processed"
            )
            return processed_results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise ProcessingError(f"Failed to batch process results: {e}")

    def consolidate_results(
        self,
        processed_results: List[ProcessedResult],
        consolidation_method: str = "aggregate",
    ) -> ProcessedResult:
        """Consolidate multiple processed results into a single summary.

        Args:
            processed_results: List of processed results to consolidate
            consolidation_method: Method for consolidation ('aggregate', 'meta_analysis', 'summary')

        Returns:
            Consolidated ProcessedResult
        """
        try:
            if not processed_results:
                raise ValidationError("No results to consolidate")

            result_id = f"consolidated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            logger.info(
                f"Consolidating {len(processed_results)} results using {consolidation_method}"
            )

            # Extract experiment types
            experiment_types = list(set(r.experiment_type for r in processed_results))
            primary_type = experiment_types[0] if len(experiment_types) == 1 else "mixed"

            # Consolidate based on method
            if consolidation_method == "aggregate":
                consolidated_data = self._aggregate_consolidation(processed_results)
            elif consolidation_method == "meta_analysis":
                consolidated_data = self._meta_analysis_consolidation(processed_results)
            else:  # summary
                consolidated_data = self._summary_consolidation(processed_results)

            # Create consolidated result
            consolidated = ProcessedResult(
                result_id=result_id,
                timestamp=datetime.now(),
                experiment_type=f"consolidated_{primary_type}",
                raw_data={"consolidated_from": [r.result_id for r in processed_results]},
                summary_statistics=consolidated_data["summary_statistics"],
                aggregated_results=consolidated_data["aggregated_results"],
                parameters={"consolidation_method": consolidation_method},
                quality_metrics=consolidated_data["quality_metrics"],
                notes=[f"Consolidated from {len(processed_results)} results"],
            )

            # Save consolidated result
            self._save_processed_result(consolidated)

            logger.info(f"Results consolidated: {result_id}")
            return consolidated

        except Exception as e:
            logger.error(f"Results consolidation failed: {e}")
            raise ProcessingError(f"Failed to consolidate results: {e}")

    def generate_report(
        self,
        processed_result: ProcessedResult,
        report_format: str = "html",
        include_visualizations: bool = True,
    ) -> str:
        """Generate comprehensive report from processed results.

        Args:
            processed_result: Processed result to report on
            report_format: Format of report ('html', 'pdf', 'json')
            include_visualizations: Whether to include visualizations

        Returns:
            Path to generated report
        """
        try:
            report_filename = f"{processed_result.result_id}_report.{report_format}"
            report_path = self.output_dir / "reports" / report_filename

            logger.info(f"Generating {report_format} report for {processed_result.result_id}")

            report_content: Union[str, Dict[str, Any]]

            if report_format == "html":
                report_content = self._generate_html_report(
                    processed_result, include_visualizations
                )
            elif report_format == "json":
                report_content = self._generate_json_report(processed_result)
            elif report_format == "pdf":
                report_content = self._generate_pdf_report(processed_result, include_visualizations)
            else:
                raise ValidationError(f"Unsupported report format: {report_format}")

            # Write report
            with open(report_path, "w", encoding="utf-8") as f:
                if report_format == "json":
                    json.dump(report_content, f, indent=2, default=str)
                else:
                    f.write(str(report_content))

            logger.info(f"Report generated: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise ProcessingError(f"Failed to generate report: {e}")

    def export_results(
        self,
        processed_results: Union[ProcessedResult, List[ProcessedResult]],
        export_format: str = "csv",
        include_metadata: bool = True,
    ) -> List[str]:
        """Export processed results to various formats.

        Args:
            processed_results: Single result or list of results to export
            export_format: Export format ('csv', 'excel', 'json', 'pickle')
            include_metadata: Whether to include metadata

        Returns:
            List of export file paths
        """
        try:
            if isinstance(processed_results, ProcessedResult):
                processed_results = [processed_results]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_paths = []

            logger.info(f"Exporting {len(processed_results)} results to {export_format}")

            if export_format == "csv":
                export_paths = self._export_to_csv(processed_results, timestamp, include_metadata)
            elif export_format == "excel":
                export_paths = self._export_to_excel(processed_results, timestamp, include_metadata)
            elif export_format == "json":
                export_paths = self._export_to_json(processed_results, timestamp, include_metadata)
            elif export_format == "pickle":
                export_paths = self._export_to_pickle(
                    processed_results, timestamp, include_metadata
                )
            else:
                raise ValidationError(f"Unsupported export format: {export_format}")

            logger.info(f"Results exported to {len(export_paths)} files")
            return export_paths

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise ProcessingError(f"Failed to export results: {e}")

    def _validate_raw_results(self, raw_results: Dict[str, Any]) -> None:
        """Validate raw experimental results."""
        if not isinstance(raw_results, dict):
            raise ValidationError("Raw results must be a dictionary")

        if not raw_results:
            raise ValidationError("Raw results cannot be empty")

        # Check for required fields based on experiment type
        required_fields = ["data", "metadata"]
        for req_field in required_fields:
            if req_field not in raw_results:
                logger.warning(f"Missing required field: {req_field}")

    def _calculate_summary_statistics(self, raw_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate summary statistics for the results."""
        stats = {}

        # Extract numeric data
        data = raw_results.get("data", {})

        if isinstance(data, dict):
            # Process dictionary data
            for key, value in data.items():
                if isinstance(value, (list, np.ndarray)):
                    numeric_array = np.array(value)
                    if numeric_array.size > 0:
                        stats[f"{key}_mean"] = float(np.mean(numeric_array))
                        stats[f"{key}_std"] = float(np.std(numeric_array))
                        stats[f"{key}_min"] = float(np.min(numeric_array))
                        stats[f"{key}_max"] = float(np.max(numeric_array))
                        stats[f"{key}_count"] = int(len(numeric_array))
                elif isinstance(value, (int, float)):
                    stats[key] = float(value)

        elif isinstance(data, (list, np.ndarray)):
            # Process array data
            numeric_array = np.array(data)
            if numeric_array.size > 0:
                stats["mean"] = float(np.mean(numeric_array))
                stats["std"] = float(np.std(numeric_array))
                stats["min"] = float(np.min(numeric_array))
                stats["max"] = float(np.max(numeric_array))
                stats["count"] = int(len(numeric_array))

        # Add metadata statistics
        metadata = raw_results.get("metadata", {})
        if "duration" in metadata:
            stats["duration_seconds"] = float(metadata["duration"])

        if "participant_count" in metadata:
            stats["participant_count"] = int(metadata["participant_count"])

        if "trial_count" in metadata:
            stats["trial_count"] = int(metadata["trial_count"])

        return stats

    def _aggregate_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results by categories and groups."""
        aggregated: Dict[str, Any] = {}

        data = raw_results.get("data", {})

        # Group by participant if available
        if "participants" in data:
            participant_data = data["participants"]
            aggregated["by_participant"] = {}

            for participant_id, participant_results in participant_data.items():
                if isinstance(participant_results, dict):
                    aggregated["by_participant"][participant_id] = {
                        "summary": self._calculate_summary_statistics(
                            {"data": participant_results}
                        ),
                        "trials": len(participant_results.get("trials", [])),
                        "success_rate": participant_results.get("success_rate", 0.0),
                    }

        # Group by trial if available
        if "trials" in data:
            trial_data = data["trials"]
            if isinstance(trial_data, list):
                aggregated["by_trial"] = {
                    "total_trials": len(trial_data),
                    "successful_trials": sum(
                        1 for trial in trial_data if trial.get("success", False)
                    ),
                    "average_duration": np.mean([trial.get("duration", 0) for trial in trial_data]),
                }

        # Group by condition if available
        if "conditions" in data:
            condition_data = data["conditions"]
            aggregated["by_condition"] = {}

            for condition_id, condition_results in condition_data.items():
                aggregated["by_condition"][condition_id] = {
                    "summary": self._calculate_summary_statistics({"data": condition_results}),
                    "participant_count": len(set(condition_results.get("participants", []))),
                }

        return aggregated

    def _assess_data_quality(self, raw_results: Dict[str, Any]) -> Dict[str, float]:
        """Assess quality of the experimental data."""
        quality_metrics = {}

        data = raw_results.get("data", {})

        # Completeness: ratio of non-missing values
        total_values = 0
        missing_values = 0

        def count_values(obj: Any) -> None:
            nonlocal total_values, missing_values
            if isinstance(obj, dict):
                for val in obj.values():
                    count_values(val)
            elif isinstance(obj, (list, np.ndarray)):
                for val in obj:
                    count_values(val)
            elif isinstance(obj, (int, float)):
                total_values += 1
                if pd.isna(obj) or obj is None:
                    missing_values += 1

        count_values(data)

        if total_values > 0:
            completeness = 1.0 - (missing_values / total_values)
            quality_metrics["completeness"] = completeness
        else:
            quality_metrics["completeness"] = 0.0

        # Sample size assessment
        sample_sizes = []
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (list, np.ndarray)):
                    sample_sizes.append(len(value))

        if sample_sizes:
            quality_metrics["min_sample_size"] = min(sample_sizes)
            quality_metrics["max_sample_size"] = max(sample_sizes)
            quality_metrics["avg_sample_size"] = float(np.mean(sample_sizes))

        # Outlier detection (simplified)
        outlier_count = 0
        total_numeric = 0

        def detect_outliers(obj: Any) -> None:
            nonlocal outlier_count, total_numeric
            if isinstance(obj, dict):
                for value in obj.values():
                    detect_outliers(value)
            elif isinstance(obj, (list, np.ndarray)):
                numeric_array = np.array(obj)
                if numeric_array.size > 0 and np.issubdtype(numeric_array.dtype, np.number):
                    q1, q3 = np.percentile(numeric_array, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = (numeric_array < lower_bound) | (numeric_array > upper_bound)
                    outlier_count += np.sum(outliers)
                    total_numeric += len(numeric_array)

        detect_outliers(data)

        if total_numeric > 0:
            outlier_ratio = outlier_count / total_numeric
            quality_metrics["outlier_ratio"] = outlier_ratio

        # Overall quality score
        quality_score = 0.0
        weights = {"completeness": 0.4, "sample_size": 0.3, "outlier_ratio": 0.3}

        if "completeness" in quality_metrics:
            quality_score += weights["completeness"] * quality_metrics["completeness"]

        if "avg_sample_size" in quality_metrics:
            size_score = min(
                1.0,
                quality_metrics["avg_sample_size"] / self.quality_thresholds["min_sample_size"],
            )
            quality_score += weights["sample_size"] * size_score

        if "outlier_ratio" in quality_metrics:
            outlier_score = max(
                0.0,
                1.0 - (quality_metrics["outlier_ratio"] / self.quality_thresholds["outlier_ratio"]),
            )
            quality_score += weights["outlier_ratio"] * outlier_score

        quality_metrics["overall_quality_score"] = quality_score

        return quality_metrics

    def _save_processed_result(self, processed_result: ProcessedResult) -> None:
        """Save processed result to file."""
        # Save as JSON
        json_path = self.output_dir / f"{processed_result.result_id}.json"

        # Convert to dict and handle numpy types
        result_dict = asdict(processed_result)

        def convert_numpy(obj: Any) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            return obj

        result_dict = convert_numpy(result_dict)

        with open(json_path, "w") as json_file:
            json.dump(result_dict, json_file, indent=2)

        # Save as pickle for faster loading
        pickle_path = self.output_dir / f"{processed_result.result_id}.pkl"
        safe_pickle_dump(processed_result, pickle_path)

        logger.debug(f"Processed result saved: {processed_result.result_id}")

    def _aggregate_consolidation(self, processed_results: List[ProcessedResult]) -> Dict[str, Any]:
        """Aggregate multiple results using simple aggregation."""
        all_stats: Dict[str, List[float]] = defaultdict(list)
        all_aggregated: Dict[str, List[Any]] = defaultdict(list)
        all_quality: Dict[str, List[float]] = defaultdict(list)

        for result in processed_results:
            # Aggregate summary statistics
            for key, value in result.summary_statistics.items():
                if key not in all_stats:
                    all_stats[key] = []
                all_stats[key].append(value)

            # Aggregate results
            for key, value in result.aggregated_results.items():
                if key not in all_aggregated:
                    all_aggregated[key] = []
                all_aggregated[key].append(value)

            # Aggregate quality metrics
            for key, value in result.quality_metrics.items():
                if key not in all_quality:
                    all_quality[key] = []
                all_quality[key].append(value)

        # Compute aggregate statistics
        consolidated_stats = {}
        for key, values in all_stats.items():
            if values:  # Only process non-empty lists
                consolidated_stats[f"{key}_mean"] = np.mean(values)
                consolidated_stats[f"{key}_std"] = np.std(values)
                consolidated_stats[f"{key}_min"] = np.min(values)
                consolidated_stats[f"{key}_max"] = np.max(values)

        return {
            "summary_statistics": consolidated_stats,
            "aggregated_results": all_aggregated,
            "quality_metrics": {k: np.mean(v) for k, v in all_quality.items() if v},
        }

    def _meta_analysis_consolidation(
        self, processed_results: List[ProcessedResult]
    ) -> Dict[str, Any]:
        """Consolidate results using meta-analysis approach."""
        # For now, implement as weighted average based on quality scores
        weights = []
        for result in processed_results:
            quality_score = result.quality_metrics.get("overall_quality_score", 0.5)
            weights.append(quality_score)

        weights_array: np.ndarray = np.array(weights) / np.sum(weights)  # Normalize

        all_stats = defaultdict(list)

        for result, weight in zip(processed_results, weights_array):
            for key, value in result.summary_statistics.items():
                all_stats[key].append(value * weight)

        consolidated_stats = {k: np.sum(v) for k, v in all_stats.items()}

        return {
            "summary_statistics": consolidated_stats,
            "aggregated_results": {},
            "quality_metrics": {"meta_analysis_weighted": True},
        }

    def _summary_consolidation(self, processed_results: List[ProcessedResult]) -> Dict[str, Any]:
        """Create summary consolidation."""
        # Simple summary statistics
        summary_stats = {
            "total_results": len(processed_results),
            "experiment_types": list(set(r.experiment_type for r in processed_results)),
            "date_range": {
                "start": min(r.timestamp for r in processed_results).isoformat(),
                "end": max(r.timestamp for r in processed_results).isoformat(),
            },
        }

        return {
            "summary_statistics": summary_stats,
            "aggregated_results": {},
            "quality_metrics": {"summary_type": "basic"},
        }

    def _generate_html_report(
        self, processed_result: ProcessedResult, include_visualizations: bool
    ) -> str:
        """Generate HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>APGI Experiment Report: {result_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .stats-table {{ border-collapse: collapse; width: 100%; }}
                .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .stats-table th {{ background-color: #f2f2f2; }}
                .quality-good {{ color: green; }}
                .quality-warning {{ color: orange; }}
                .quality-poor {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>APGI Experiment Report</h1>
                <h2>{experiment_type}</h2>
                <p>Result ID: {result_id}</p>
                <p>Generated: {timestamp}</p>
            </div>
            
            <div class="section">
                <h3>Summary Statistics</h3>
                <table class="stats-table">
                    {stats_rows}
                </table>
            </div>
            
            <div class="section">
                <h3>Data Quality Assessment</h3>
                <table class="stats-table">
                    {quality_rows}
                </table>
            </div>
            
            <div class="section">
                <h3>Parameters</h3>
                <pre>{parameters}</pre>
            </div>
            
            {visualization_section}
            
            <div class="section">
                <h3>Notes</h3>
                <ul>
                    {notes_list}
                </ul>
            </div>
        </body>
        </html>
        """

        # Generate statistics rows
        stats_rows = ""
        for key, value in processed_result.summary_statistics.items():
            stats_rows += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"

        # Generate quality rows
        quality_rows = ""
        for key, value in processed_result.quality_metrics.items():
            if "quality" in key.lower():
                css_class = (
                    "quality-good"
                    if value > 0.8
                    else "quality-warning" if value > 0.6 else "quality-poor"
                )
                quality_rows += f"<tr><td>{key}</td><td class='{css_class}'>{value:.4f}</td></tr>"

        # Generate notes list
        notes_list = ""
        for note in processed_result.notes:
            notes_list += f"<li>{note}</li>"

        # Visualization section (placeholder)
        visualization_section = ""
        if include_visualizations and processed_result.visualization_data:
            visualization_section = "<div class='section'><h3>Visualizations</h3><p>Visualizations would be embedded here</p></div>"

        return html_template.format(
            result_id=processed_result.result_id,
            experiment_type=processed_result.experiment_type,
            timestamp=processed_result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            stats_rows=stats_rows,
            quality_rows=quality_rows,
            parameters=json.dumps(processed_result.parameters, indent=2),
            visualization_section=visualization_section,
            notes_list=notes_list,
        )

    def _generate_json_report(self, processed_result: ProcessedResult) -> Dict[str, Any]:
        """Generate JSON report."""
        return asdict(processed_result)

    def _generate_pdf_report(
        self, processed_result: ProcessedResult, include_visualizations: bool
    ) -> str:
        """Generate PDF report (placeholder)."""
        # For now, return HTML as placeholder
        # In a real implementation, you would use a library like reportlab or weasyprint
        return self._generate_html_report(processed_result, include_visualizations)

    def _export_to_csv(
        self,
        processed_results: List[ProcessedResult],
        timestamp: str,
        include_metadata: bool,
    ) -> List[str]:
        """Export results to CSV format."""
        export_paths = []

        for result in processed_results:
            # Export summary statistics
            stats_df = pd.DataFrame([result.summary_statistics])
            stats_path = self.output_dir / "exports" / f"{result.result_id}_stats_{timestamp}.csv"
            stats_df.to_csv(stats_path, index=False)
            export_paths.append(str(stats_path))

            # Export quality metrics
            quality_df = pd.DataFrame([result.quality_metrics])
            quality_path = (
                self.output_dir / "exports" / f"{result.result_id}_quality_{timestamp}.csv"
            )
            quality_df.to_csv(quality_path, index=False)
            export_paths.append(str(quality_path))

            if include_metadata:
                # Export metadata
                metadata_df = pd.DataFrame([result.parameters])
                metadata_path = (
                    self.output_dir / "exports" / f"{result.result_id}_metadata_{timestamp}.csv"
                )
                metadata_df.to_csv(metadata_path, index=False)
                export_paths.append(str(metadata_path))

        return export_paths

    def _export_to_excel(
        self,
        processed_results: List[ProcessedResult],
        timestamp: str,
        include_metadata: bool,
    ) -> List[str]:
        """Export results to Excel format."""
        export_paths = []

        for result in processed_results:
            excel_path = self.output_dir / "exports" / f"{result.result_id}_{timestamp}.xlsx"

            with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
                # Summary statistics
                pd.DataFrame([result.summary_statistics]).to_excel(
                    writer, sheet_name="Summary Statistics", index=False
                )

                # Quality metrics
                pd.DataFrame([result.quality_metrics]).to_excel(
                    writer, sheet_name="Quality Metrics", index=False
                )

                if include_metadata:
                    # Parameters
                    pd.DataFrame([result.parameters]).to_excel(
                        writer, sheet_name="Parameters", index=False
                    )

            export_paths.append(str(excel_path))

        return export_paths

    def _export_to_json(
        self,
        processed_results: List[ProcessedResult],
        timestamp: str,
        include_metadata: bool,
    ) -> List[str]:
        """Export results to JSON format."""
        export_paths = []

        for result in processed_results:
            json_path = self.output_dir / "exports" / f"{result.result_id}_{timestamp}.json"

            export_data = {
                "result_id": result.result_id,
                "timestamp": result.timestamp.isoformat(),
                "experiment_type": result.experiment_type,
                "summary_statistics": result.summary_statistics,
                "quality_metrics": result.quality_metrics,
            }

            if include_metadata:
                export_data["parameters"] = result.parameters
                export_data["notes"] = result.notes

            with open(json_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            export_paths.append(str(json_path))

        return export_paths

    def _export_to_pickle(
        self,
        processed_results: List[ProcessedResult],
        timestamp: str,
        include_metadata: bool,
    ) -> List[str]:
        """Export results to pickle format."""
        export_paths = []

        for result in processed_results:
            pickle_path = self.output_dir / "exports" / f"{result.result_id}_{timestamp}.pkl"

            safe_pickle_dump(result, pickle_path)

            export_paths.append(str(pickle_path))

        return export_paths

    def _create_batch_summary(
        self, processed_results: List[ProcessedResult], experiment_type: str
    ) -> None:
        """Create summary of batch processing results."""
        summary = {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "experiment_type": experiment_type,
            "total_processed": len(processed_results),
            "successful_processed": len(processed_results),
            "processing_date": datetime.now().isoformat(),
            "result_ids": [r.result_id for r in processed_results],
        }

        # Calculate aggregate statistics
        if processed_results:
            all_quality_scores = [
                r.quality_metrics.get("overall_quality_score", 0) for r in processed_results
            ]
            summary["average_quality_score"] = np.mean(all_quality_scores)
            summary["quality_score_range"] = [
                np.min(all_quality_scores),
                np.max(all_quality_scores),
            ]

        # Save batch summary
        summary_path = self.output_dir / f"batch_summary_{summary['batch_id']}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Batch summary created: {summary_path}")

    def load_processed_result(self, result_id: str) -> ProcessedResult:
        """Load a processed result from file."""
        pickle_path = self.output_dir / f"{result_id}.pkl"

        if not pickle_path.exists():
            raise FileNotFoundError(f"Processed result {result_id} not found")

        return cast(ProcessedResult, safe_pickle_load(pickle_path))

    def list_processed_results(self, experiment_type: Optional[str] = None) -> List[str]:
        """List all processed results, optionally filtered by experiment type."""
        result_files = list(self.output_dir.glob("*.pkl"))
        result_ids = [f.stem for f in result_files]

        if experiment_type:
            filtered_ids = []
            for result_id in result_ids:
                try:
                    result = self.load_processed_result(result_id)
                    if result.experiment_type == experiment_type:
                        filtered_ids.append(result_id)
                except (
                    FileNotFoundError,
                    PermissionError,
                    pickle.UnpicklingError,
                ) as e:
                    logger.warning(f"Failed to load result {result_id}: {e}")
                    continue
            return filtered_ids

        return result_ids
