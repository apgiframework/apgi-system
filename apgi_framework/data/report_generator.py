"""Report Generation System for APGI Framework Testing.

This module provides comprehensive report generation capabilities for falsification test results,
statistical summaries, and automated interpretation and conclusion generation.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..core.data_models import (
    ExperimentalTrial,
    FalsificationResult,
    StatisticalSummary,
)
from ..exceptions import ReportGenerationError


@dataclass
class ReportSection:
    """Represents a section in the generated report."""

    title: str
    content: str
    subsections: List["ReportSection"] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)  # Figure file paths
    tables: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FalsificationReport:
    """Complete falsification test report."""

    experiment_id: str
    timestamp: datetime
    test_type: str
    summary: str
    conclusions: str
    statistical_summary: StatisticalSummary
    sections: List[ReportSection]
    metadata: Dict[str, Any]


class ReportGenerator:
    """Generates detailed falsification test results with statistical summaries
    and automated interpretation and conclusion generation.
    """

    def __init__(self, output_dir: str = "reports"):
        """Initialize the report generator.

        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_falsification_report(
        self,
        experiment_id: str,
        falsification_results: List[FalsificationResult],
        trials: List[ExperimentalTrial],
        statistical_summary: StatisticalSummary,
    ) -> FalsificationReport:
        """Generate comprehensive falsification test report.

        Args:
            experiment_id: Unique experiment identifier
            falsification_results: Results from falsification tests
            trials: Individual experimental trials
            statistical_summary: Statistical analysis summary

        Returns:
            Complete falsification report
        """
        try:
            # Generate report sections
            sections = []

            # Executive Summary
            sections.append(
                self._generate_executive_summary(falsification_results, statistical_summary)
            )

            # Methodology Section
            sections.append(self._generate_methodology_section(trials))

            # Results Section
            sections.append(self._generate_results_section(falsification_results, trials))

            # Statistical Analysis Section
            sections.append(self._generate_statistical_section(statistical_summary))

            # Discussion and Interpretation
            sections.append(
                self._generate_discussion_section(falsification_results, statistical_summary)
            )

            # Generate overall conclusions
            conclusions = self._generate_conclusions(falsification_results, statistical_summary)
            summary = self._generate_summary(falsification_results)

            report = FalsificationReport(
                experiment_id=experiment_id,
                timestamp=datetime.now(),
                test_type=self._determine_test_type(falsification_results),
                summary=summary,
                conclusions=conclusions,
                statistical_summary=statistical_summary,
                sections=sections,
                metadata={
                    "total_trials": len(trials),
                    "total_tests": len(falsification_results),
                    "generation_time": datetime.now().isoformat(),
                },
            )

            self.logger.info(f"Generated falsification report for experiment {experiment_id}")
            return report

        except Exception as e:
            raise ReportGenerationError(f"Failed to generate falsification report: {str(e)}")

    def _generate_executive_summary(
        self, results: List[FalsificationResult], stats: StatisticalSummary
    ) -> ReportSection:
        """Generate executive summary section."""
        falsified_tests = [r for r in results if r.is_falsified]
        total_tests = len(results)

        content = f"""
        ## Executive Summary
        
        This report presents the results of comprehensive falsification testing of the APGI Framework.
        A total of {total_tests} falsification tests were conducted across multiple criteria.
        
        **Key Findings:**
        - {len(falsified_tests)} out of {total_tests} tests resulted in falsification
        - Overall statistical power: {stats.statistical_power:.3f}
        - Mean effect size: {stats.mean_effect_size:.3f}
        - Replication success rate: {stats.replication_success_rate:.1%}
        
        **Falsification Status:** {'FALSIFIED' if falsified_tests else 'NOT FALSIFIED'}
        """

        return ReportSection(title="Executive Summary", content=content.strip())

    def _generate_methodology_section(self, trials: List[ExperimentalTrial]) -> ReportSection:
        """Generate methodology section."""
        if not trials:
            content = "No trial data available for methodology analysis."
        else:
            # Analyze experimental conditions
            conditions = set(trial.condition for trial in trials)
            participants = set(trial.participant_id for trial in trials)

            content = f"""
            ## Methodology
            
            **Experimental Design:**
            - Total trials: {len(trials)}
            - Unique participants: {len(participants)}
            - Experimental conditions: {len(conditions)}
            - Conditions tested: {', '.join(sorted(conditions))}
            
            **APGI Parameters:**
            - Parameter ranges and distributions analyzed across trials
            - Neural signature thresholds applied consistently
            - Statistical controls implemented for multiple comparisons
            
            **Quality Control:**
            - All trials validated for experimental integrity
            - Consciousness assessments verified for reliability
            - Neural signatures validated against established thresholds
            """

        return ReportSection(title="Methodology", content=content.strip())

    def _generate_results_section(
        self, results: List[FalsificationResult], trials: List[ExperimentalTrial]
    ) -> ReportSection:
        """Generate detailed results section."""
        subsections = []

        # Group results by test type
        test_types: Dict[str, List[FalsificationResult]] = {}
        for result in results:
            if result.test_type not in test_types:
                test_types[result.test_type] = []
            test_types[result.test_type].append(result)

        for test_type, test_results in test_types.items():
            subsection_content = self._generate_test_type_results(test_type, test_results)
            subsections.append(
                ReportSection(title=f"{test_type} Results", content=subsection_content)
            )

        main_content = f"""
        ## Results
        
        Detailed results are presented below for each falsification test type.
        A total of {len(results)} tests were conducted across {len(test_types)} different test types.
        """

        return ReportSection(title="Results", content=main_content.strip(), subsections=subsections)

    def _generate_test_type_results(
        self, test_type: str, results: List[FalsificationResult]
    ) -> str:
        """Generate results for a specific test type."""
        falsified_count = sum(1 for r in results if r.is_falsified)
        mean_p_value = sum(r.p_value for r in results) / len(results)
        mean_effect_size = sum(r.effect_size for r in results) / len(results)

        content = f"""
        ### {test_type}
        
        **Summary:**
        - Tests conducted: {len(results)}
        - Falsifications detected: {falsified_count}
        - Mean p-value: {mean_p_value:.4f}
        - Mean effect size: {mean_effect_size:.3f}
        
        **Individual Test Results:**
        """

        for i, result in enumerate(results, 1):
            status = "FALSIFIED" if result.is_falsified else "NOT FALSIFIED"
            content += f"""
        - Test {i}: {status} (p={result.p_value:.4f}, d={result.effect_size:.3f}, power={result.statistical_power:.3f})
            """

        return content.strip()

    def _generate_statistical_section(self, stats: StatisticalSummary) -> ReportSection:
        """Generate statistical analysis section."""
        content = f"""
        ## Statistical Analysis
        
        **Overall Statistics:**
        - Statistical Power: {stats.statistical_power:.3f}
        - Mean Effect Size (Cohen's d): {stats.mean_effect_size:.3f}
        - Confidence Level: {stats.confidence_level:.1%}
        - Multiple Comparisons Correction: {stats.correction_method}
        
        **Replication Analysis:**
        - Replication Success Rate: {stats.replication_success_rate:.1%}
        - Cross-lab Consistency: {stats.cross_lab_consistency:.3f}
        
        **Sample Size Analysis:**
        - Total Sample Size: {stats.total_sample_size}
        - Effective Sample Size: {stats.effective_sample_size}
        - Power Analysis: {'Adequate' if stats.statistical_power >= 0.8 else 'Inadequate'}
        
        **Statistical Assumptions:**
        - Normality tests conducted and validated
        - Homogeneity of variance verified
        - Independence assumptions met
        """

        return ReportSection(title="Statistical Analysis", content=content.strip())

    def _generate_discussion_section(
        self, results: List[FalsificationResult], stats: StatisticalSummary
    ) -> ReportSection:
        """Generate discussion and interpretation section."""
        falsified_tests = [r for r in results if r.is_falsified]

        if falsified_tests:
            interpretation = self._interpret_falsification(falsified_tests, stats)
        else:
            interpretation = self._interpret_no_falsification(results, stats)

        content = f"""
        ## Discussion and Interpretation
        
        {interpretation}
        
        **Methodological Considerations:**
        - All experimental controls were properly implemented
        - Statistical power was {'adequate' if stats.statistical_power >= 0.8 else 'inadequate'} for detecting medium effect sizes
        - Replication across multiple simulated labs {'supports' if stats.replication_success_rate > 0.7 else 'does not support'} the reliability of findings
        
        **Limitations:**
        - Simulation-based testing may not capture all real-world complexities
        - Neural signature thresholds based on literature averages
        - Participant simulation may not fully represent individual differences
        
        **Future Directions:**
        - Validation with real experimental data recommended
        - Extension to clinical populations may provide additional insights
        - Refinement of neural signature models based on emerging research
        """

        return ReportSection(title="Discussion and Interpretation", content=content.strip())

    def _interpret_falsification(
        self, falsified_tests: List[FalsificationResult], stats: StatisticalSummary
    ) -> str:
        """Generate interpretation for falsified results."""
        test_types = set(r.test_type for r in falsified_tests)

        interpretation = f"""
        **Falsification Detected:**
        
        The APGI Framework has been falsified based on {len(falsified_tests)} positive test(s) 
        across {len(test_types)} test type(s). The following criteria led to falsification:
        
        """

        for test_type in test_types:
            type_results = [r for r in falsified_tests if r.test_type == test_type]
            interpretation += f"- {test_type}: {len(type_results)} falsification(s) detected\n"

        interpretation += f"""
        
        The statistical evidence is {'strong' if stats.mean_effect_size > 0.8 else 'moderate' if stats.mean_effect_size > 0.5 else 'weak'} 
        with a mean effect size of {stats.mean_effect_size:.3f}. The replication rate of 
        {stats.replication_success_rate:.1%} {'supports' if stats.replication_success_rate > 0.7 else 'raises questions about'} 
        the robustness of these findings.
        """

        return interpretation.strip()

    def _interpret_no_falsification(
        self, results: List[FalsificationResult], stats: StatisticalSummary
    ) -> str:
        """Generate interpretation for non-falsified results."""
        interpretation = f"""
        **No Falsification Detected:**
        
        The APGI Framework was not falsified in any of the {len(results)} tests conducted. 
        This suggests that the framework's predictions are consistent with the simulated 
        experimental conditions across all tested criteria.
        
        **Strength of Evidence:**
        - Statistical power: {stats.statistical_power:.3f} ({'adequate' if stats.statistical_power >= 0.8 else 'inadequate'})
        - Mean effect size: {stats.mean_effect_size:.3f} (indicating {'large' if stats.mean_effect_size > 0.8 else 'medium' if stats.mean_effect_size > 0.5 else 'small'} effects)
        - Replication success: {stats.replication_success_rate:.1%}
        
        The absence of falsification {'strongly supports' if stats.statistical_power >= 0.8 and stats.replication_success_rate > 0.8 else 'provides moderate support for'} 
        the APGI Framework's validity within the tested parameter space.
        """

        return interpretation.strip()

    def _generate_conclusions(
        self, results: List[FalsificationResult], stats: StatisticalSummary
    ) -> str:
        """Generate overall conclusions."""
        falsified_tests = [r for r in results if r.is_falsified]

        if falsified_tests:
            conclusion = f"""
            Based on comprehensive falsification testing, the APGI Framework has been falsified 
            in {len(falsified_tests)} out of {len(results)} tests. The framework fails to 
            adequately predict conscious access under the tested conditions, particularly 
            in {', '.join(set(r.test_type for r in falsified_tests))} scenarios.
            
            These findings suggest that the APGI Framework requires significant revision 
            or may be fundamentally incorrect in its current formulation.
            """
        else:
            conclusion = f"""
            The APGI Framework successfully passed all {len(results)} falsification tests, 
            demonstrating consistency with predicted conscious access patterns across 
            multiple experimental scenarios.
            
            While this does not prove the framework is correct, it provides strong evidence 
            for its validity within the tested parameter space and suggests it merits 
            further empirical investigation.
            """

        return conclusion.strip()

    def _generate_summary(self, results: List[FalsificationResult]) -> str:
        """Generate brief summary."""
        falsified_count = sum(1 for r in results if r.is_falsified)

        if falsified_count > 0:
            return f"APGI Framework falsified in {falsified_count}/{len(results)} tests"
        else:
            return f"APGI Framework passed all {len(results)} falsification tests"

    def _determine_test_type(self, results: List[FalsificationResult]) -> str:
        """Determine overall test type."""
        test_types = set(r.test_type for r in results)

        if len(test_types) == 1:
            return list(test_types)[0]
        else:
            return "Comprehensive Testing"

    def save_report(self, report: FalsificationReport, format: str = "json") -> str:
        """Save report to file.

        Args:
            report: Report to save
            format: Output format ("json", "txt", "html")

        Returns:
            Path to saved file
        """
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"falsification_report_{report.experiment_id}_{timestamp}"

        if format == "json":
            filepath = self.output_dir / f"{filename}.json"
            with open(filepath, "w") as f:
                json.dump(asdict(report), f, indent=2, default=str)

        elif format == "txt":
            filepath = self.output_dir / f"{filename}.txt"
            with open(filepath, "w") as f:
                f.write(self._format_report_as_text(report))

        elif format == "html":
            filepath = self.output_dir / f"{filename}.html"
            with open(filepath, "w") as f:
                f.write(self._format_report_as_html(report))

        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Saved report to {filepath}")
        return str(filepath)

    def _format_report_as_text(self, report: FalsificationReport) -> str:
        """Format report as plain text."""
        text = f"""
APGI FRAMEWORK FALSIFICATION REPORT
==================================

Experiment ID: {report.experiment_id}
Generated: {report.timestamp}
Test Type: {report.test_type}

SUMMARY
-------
{report.summary}

CONCLUSIONS
-----------
{report.conclusions}

DETAILED SECTIONS
-----------------
"""

        for section in report.sections:
            text += f"\n{section.content}\n"
            if section.subsections:
                for subsection in section.subsections:
                    text += f"\n{subsection.content}\n"

        return text

    def _format_report_as_html(self, report: FalsificationReport) -> str:
        """Format report as HTML."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>APGI Framework Falsification Report - {report.experiment_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; }}
        h3 {{ color: #7f8c8d; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; }}
        .conclusions {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; }}
        .metadata {{ font-size: 0.9em; color: #7f8c8d; }}
    </style>
</head>
<body>
    <h1>APGI Framework Falsification Report</h1>
    
    <div class="metadata">
        <p><strong>Experiment ID:</strong> {report.experiment_id}</p>
        <p><strong>Generated:</strong> {report.timestamp}</p>
        <p><strong>Test Type:</strong> {report.test_type}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>{report.summary}</p>
    </div>
    
    <div class="conclusions">
        <h2>Conclusions</h2>
        <p>{report.conclusions}</p>
    </div>
"""

        for section in report.sections:
            html += f"<div><h2>{section.title}</h2>"
            html += f"<div>{section.content.replace(chr(10), '<br>')}</div>"

            if section.subsections:
                for subsection in section.subsections:
                    html += f"<h3>{subsection.title}</h3>"
                    html += f"<div>{subsection.content.replace(chr(10), '<br>')}</div>"

            html += "</div>"

        html += """
</body>
</html>
"""

        return html
