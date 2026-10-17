"""
Statistical report generator for APGI falsification testing.

This module provides comprehensive statistical report generation including
detailed statistical summaries, falsification probability assessments,
and publication-ready statistical reporting for APGI framework validation studies.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..logging.standardized_logging import get_logger
from .effect_size_calculator import EffectSizeCalculator, EffectSizeResult
from .replication_tracker import (
    ExperimentResult,
    ReplicationSummary,
    ReplicationTracker,
)
from .sample_size_validator import PowerReport, SampleSizeValidator
from .statistical_tester import StatisticalResult, StatisticalTester


class ReportFormat(Enum):
    """Available report formats."""

    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    PDF = "pdf"
    LATEX = "latex"


class FalsificationConclusion(Enum):
    """Possible falsification conclusions."""

    FALSIFIED = "falsified"
    NOT_FALSIFIED = "not_falsified"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class FalsificationAssessment:
    """Assessment of falsification probability."""

    criterion_type: str
    evidence_strength: float  # 0-1 scale
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    power: float
    conclusion: FalsificationConclusion
    confidence_level: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]


@dataclass
class StatisticalSummary:
    """Comprehensive statistical summary."""

    study_id: str
    test_results: List[StatisticalResult]
    effect_sizes: List[EffectSizeResult]
    power_analysis: PowerReport
    replication_summary: Optional[ReplicationSummary]
    falsification_assessments: List[FalsificationAssessment]
    overall_conclusion: str
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PublicationReport:
    """Publication-ready statistical report."""

    title: str
    abstract: str
    methods_section: str
    results_section: str
    discussion_section: str
    tables: List[Dict[str, Any]]
    figures: List[Dict[str, Any]]
    references: List[str]
    supplementary_materials: Dict[str, Any]


class StatisticalReportGenerator:
    """
    Comprehensive statistical report generator for APGI framework studies.

    Generates detailed statistical summaries, falsification probability
    assessments, and publication-ready reports for APGI framework
    validation studies.
    """

    def __init__(
        self,
        significance_threshold: float = 0.05,
        effect_size_threshold: float = 0.2,
        power_threshold: float = 0.8,
    ):
        """
        Initialize the report generator.

        Args:
            significance_threshold: P-value threshold for significance
            effect_size_threshold: Minimum meaningful effect size
            power_threshold: Minimum acceptable statistical power
        """
        self.significance_threshold = significance_threshold
        self.effect_size_threshold = effect_size_threshold
        self.power_threshold = power_threshold

        # Initialize logger
        self.logger = get_logger(__name__)

        # Initialize component analyzers
        self.statistical_tester = StatisticalTester(alpha=significance_threshold)
        self.effect_calculator = EffectSizeCalculator()
        self.sample_validator = SampleSizeValidator(
            alpha=significance_threshold, power_threshold=power_threshold
        )

    def generate_falsification_assessment(
        self,
        criterion_type: str,
        statistical_results: List[StatisticalResult],
        effect_sizes: List[EffectSizeResult],
        sample_size: int,
        theoretical_predictions: Dict[str, Any],
    ) -> FalsificationAssessment:
        """
        Generate comprehensive falsification assessment for a specific criterion.

        Args:
            criterion_type: Type of falsification criterion
            statistical_results: List of statistical test results
            effect_sizes: List of effect size calculations
            sample_size: Total sample size
            theoretical_predictions: Expected theoretical outcomes

        Returns:
            FalsificationAssessment with detailed analysis
        """
        # Extract key statistics
        p_values = [result.p_value for result in statistical_results]
        effect_size_values = [es.value for es in effect_sizes]

        # Primary statistics
        primary_p_value = min(p_values) if p_values else 1.0
        primary_effect_size = max(effect_size_values, key=abs) if effect_size_values else 0.0

        # Calculate confidence interval for primary effect
        if effect_sizes:
            primary_ci = effect_sizes[0].confidence_interval
        else:
            primary_ci = (0.0, 0.0)

        # Estimate statistical power
        estimated_power = self._estimate_power(primary_effect_size, sample_size)

        # Assess evidence strength
        evidence_strength = self._calculate_evidence_strength(
            p_values, effect_size_values, sample_size, criterion_type
        )

        # Determine conclusion
        conclusion = self._determine_falsification_conclusion(
            criterion_type,
            primary_p_value,
            primary_effect_size,
            evidence_strength,
            theoretical_predictions,
        )

        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(
            evidence_strength, estimated_power, sample_size
        )

        # Gather supporting and contradicting evidence
        supporting_evidence, contradicting_evidence = self._analyze_evidence(
            statistical_results, effect_sizes, theoretical_predictions, criterion_type
        )

        return FalsificationAssessment(
            criterion_type=criterion_type,
            evidence_strength=evidence_strength,
            p_value=primary_p_value,
            effect_size=primary_effect_size,
            confidence_interval=primary_ci,
            sample_size=sample_size,
            power=estimated_power,
            conclusion=conclusion,
            confidence_level=confidence_level,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
        )

    def generate_comprehensive_summary(
        self,
        study_id: str,
        test_results: List[StatisticalResult],
        effect_sizes: List[EffectSizeResult],
        power_analysis: PowerReport,
        replication_data: Optional[List[ExperimentResult]] = None,
        theoretical_framework: Optional[Dict[str, Any]] = None,
    ) -> StatisticalSummary:
        """
        Generate comprehensive statistical summary for the study.

        Args:
            study_id: Unique identifier for the study
            test_results: List of statistical test results
            effect_sizes: List of effect size calculations
            power_analysis: Power analysis report
            replication_data: Optional replication experiment data
            theoretical_framework: Optional theoretical predictions

        Returns:
            StatisticalSummary with complete analysis
        """
        # Process replication data if available
        replication_summary = None
        if replication_data:
            replication_tracker = ReplicationTracker()
            replication_tracker.add_multiple_results(replication_data)

            # Assume original effect size from first effect size result
            original_effect = effect_sizes[0].value if effect_sizes else 0.3
            replication_summary = replication_tracker.evaluate_replication_success(original_effect)

        # Generate falsification assessments for each criterion
        falsification_assessments = []

        # APGI Framework falsification criteria
        criteria_types = [
            "primary_falsification",
            "consciousness_without_ignition",
            "threshold_insensitivity",
            "soma_bias_absence",
        ]

        for criterion in criteria_types:
            # Filter relevant results for this criterion
            relevant_results = [r for r in test_results if criterion in r.test_type]
            relevant_effects = [e for e in effect_sizes if criterion in e.effect_size_type]

            if relevant_results or relevant_effects:
                assessment = self.generate_falsification_assessment(
                    criterion_type=criterion,
                    statistical_results=relevant_results,
                    effect_sizes=relevant_effects,
                    sample_size=int(power_analysis.power_summary.get("total_tests", 100)),
                    theoretical_predictions=theoretical_framework or {},
                )
                falsification_assessments.append(assessment)

        # Generate overall conclusion
        overall_conclusion = self._generate_overall_conclusion(
            falsification_assessments, replication_summary
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            test_results, effect_sizes, power_analysis, falsification_assessments
        )

        return StatisticalSummary(
            study_id=study_id,
            test_results=test_results,
            effect_sizes=effect_sizes,
            power_analysis=power_analysis,
            replication_summary=replication_summary,
            falsification_assessments=falsification_assessments,
            overall_conclusion=overall_conclusion,
            recommendations=recommendations,
        )

    def generate_publication_report(
        self,
        summary: StatisticalSummary,
        title: str,
        authors: List[str],
        format_type: ReportFormat = ReportFormat.MARKDOWN,
    ) -> PublicationReport:
        """
        Generate publication-ready report.

        Args:
            summary: Statistical summary to convert
            title: Publication title
            authors: List of author names
            format_type: Desired output format

        Returns:
            PublicationReport with formatted sections
        """
        # Generate abstract
        abstract = self._generate_abstract(summary)

        # Generate methods section
        methods_section = self._generate_methods_section(summary)

        # Generate results section
        results_section = self._generate_results_section(summary)

        # Generate discussion section
        discussion_section = self._generate_discussion_section(summary)

        # Generate tables
        tables = self._generate_tables(summary)

        # Generate figure descriptions
        figures = self._generate_figure_descriptions(summary)

        # Generate references
        references = self._generate_references()

        # Generate supplementary materials
        supplementary_materials = self._generate_supplementary_materials(summary)

        return PublicationReport(
            title=title,
            abstract=abstract,
            methods_section=methods_section,
            results_section=results_section,
            discussion_section=discussion_section,
            tables=tables,
            figures=figures,
            references=references,
            supplementary_materials=supplementary_materials,
        )

    def export_report(
        self,
        report: Union[StatisticalSummary, PublicationReport],
        filename: str,
        format_type: ReportFormat = ReportFormat.JSON,
    ) -> str:
        """
        Export report to specified format.

        Args:
            report: Report to export
            filename: Output filename
            format_type: Export format

        Returns:
            Path to exported file
        """
        if format_type == ReportFormat.JSON:
            return self._export_json(report, filename)
        elif format_type == ReportFormat.MARKDOWN:
            return self._export_markdown(report, filename)
        elif format_type == ReportFormat.HTML:
            return self._export_html(report, filename)
        else:
            raise ValueError(f"Export format {format_type} not yet implemented")

    def _calculate_evidence_strength(
        self,
        p_values: List[float],
        effect_sizes: List[float],
        sample_size: int,
        criterion_type: str,
    ) -> float:
        """Calculate overall evidence strength (0-1 scale)."""
        if not p_values and not effect_sizes:
            return 0.0

        # Statistical significance component
        min_p = min(p_values) if p_values else 1.0
        significance_strength = max(0, 1 - (min_p / self.significance_threshold))

        # Effect size component
        max_effect = max([abs(es) for es in effect_sizes]) if effect_sizes else 0.0
        effect_strength = min(1.0, max_effect / self.effect_size_threshold)

        # Sample size component
        sample_strength = min(1.0, sample_size / 100)  # Normalize to reasonable range

        # Criterion-specific weighting
        if criterion_type == "primary_falsification":
            weights = [0.4, 0.4, 0.2]  # High weight on significance and effect
        elif criterion_type == "soma_bias_absence":
            weights = [0.3, 0.5, 0.2]  # High weight on effect size
        else:
            weights = [0.33, 0.33, 0.34]  # Equal weighting

        # Weighted combination
        evidence_strength = (
            weights[0] * significance_strength
            + weights[1] * effect_strength
            + weights[2] * sample_strength
        )

        return min(1.0, evidence_strength)

    def _determine_falsification_conclusion(
        self,
        criterion_type: str,
        p_value: float,
        effect_size: float,
        evidence_strength: float,
        theoretical_predictions: Dict[str, Any],
    ) -> FalsificationConclusion:
        """Determine falsification conclusion based on evidence."""
        # Get theoretical predictions for this criterion

        # Strong evidence thresholds
        strong_significance = p_value < 0.001
        strong_effect = abs(effect_size) > 0.5
        strong_evidence = evidence_strength > 0.8

        # Criterion-specific logic
        if criterion_type == "primary_falsification":
            # Primary falsification: need strong evidence of signatures without consciousness
            if strong_significance and strong_effect and strong_evidence:
                return FalsificationConclusion.FALSIFIED
            elif p_value < self.significance_threshold and evidence_strength > 0.5:
                return FalsificationConclusion.INCONCLUSIVE
            else:
                return FalsificationConclusion.NOT_FALSIFIED

        elif criterion_type == "consciousness_without_ignition":
            # Need evidence of consciousness without neural signatures
            if strong_significance and evidence_strength > 0.7:
                return FalsificationConclusion.FALSIFIED
            elif p_value < self.significance_threshold:
                return FalsificationConclusion.INCONCLUSIVE
            else:
                return FalsificationConclusion.NOT_FALSIFIED

        elif criterion_type == "soma_bias_absence":
            # Need evidence that β ≈ 1.0 (no interoceptive bias). Under protocol
            # specification 1.1.0 nonsignificance alone does not establish
            # equivalence or falsification: that needs a registered interval
            # (ROPE/TOST) test, which this point estimate cannot provide.
            if abs(effect_size - 1.0) < 0.1 and p_value > 0.05:
                return FalsificationConclusion.INCONCLUSIVE
            elif evidence_strength > 0.6:
                return FalsificationConclusion.INCONCLUSIVE
            else:
                return FalsificationConclusion.NOT_FALSIFIED

        else:
            # General criterion
            if strong_significance and strong_evidence:
                return FalsificationConclusion.FALSIFIED
            elif evidence_strength > 0.5:
                return FalsificationConclusion.INCONCLUSIVE
            else:
                return FalsificationConclusion.INSUFFICIENT_EVIDENCE

    def _calculate_confidence_level(
        self, evidence_strength: float, power: float, sample_size: int
    ) -> float:
        """Calculate confidence level in the conclusion."""
        # Base confidence from evidence strength
        base_confidence = evidence_strength

        # Power adjustment
        power_adjustment = min(1.0, power / self.power_threshold)

        # Sample size adjustment
        sample_adjustment = min(1.0, sample_size / 100)

        # Combined confidence
        confidence = base_confidence * 0.5 + power_adjustment * 0.3 + sample_adjustment * 0.2

        return min(1.0, confidence)

    def _analyze_evidence(
        self,
        statistical_results: List[StatisticalResult],
        effect_sizes: List[EffectSizeResult],
        theoretical_predictions: Dict[str, Any],
        criterion_type: str,
    ) -> Tuple[List[str], List[str]]:
        """Analyze supporting and contradicting evidence."""
        supporting = []
        contradicting = []

        # Analyze statistical results
        for result in statistical_results:
            if result.p_value < self.significance_threshold:
                supporting.append(
                    f"Significant {result.test_type} result (p = {result.p_value:.4f})"
                )
            else:
                contradicting.append(
                    f"Non-significant {result.test_type} result (p = {result.p_value:.4f})"
                )

        # Analyze effect sizes
        for es in effect_sizes:
            if abs(es.value) > self.effect_size_threshold:
                supporting.append(f"Meaningful {es.effect_size_type} effect size ({es.value:.3f})")
            else:
                contradicting.append(f"Small {es.effect_size_type} effect size ({es.value:.3f})")

        return supporting, contradicting

    def _estimate_power(self, effect_size: float, sample_size: int) -> float:
        """Estimate statistical power for given effect size and sample size."""
        try:
            power_result = self.sample_validator.power_analyzer.t_test_power(
                effect_size=abs(effect_size),
                sample_size=sample_size,
                test_type="two_sample",
            )
            return power_result.power
        except (ValueError, RuntimeError, AttributeError) as e:
            # Fallback approximation
            self.logger.warning(f"Power calculation failed: {e}. Using approximation.")
            return float(min(1.0, (abs(effect_size) * np.sqrt(sample_size)) / 2.8))

    def _generate_overall_conclusion(
        self,
        assessments: List[FalsificationAssessment],
        replication_summary: Optional[ReplicationSummary],
    ) -> str:
        """Generate overall study conclusion."""
        if not assessments:
            return "Insufficient data for falsification assessment"

        # Count conclusions by type
        falsified_count = sum(
            1 for a in assessments if a.conclusion == FalsificationConclusion.FALSIFIED
        )
        total_count = len(assessments)

        # Replication consideration
        replication_text = ""
        if replication_summary:
            if replication_summary.success_rate > 0.7:
                replication_text = " with good replication success"
            elif replication_summary.success_rate > 0.4:
                replication_text = " with moderate replication success"
            else:
                replication_text = " with poor replication success"

        # Generate conclusion
        if falsified_count == 0:
            return (
                f"APGI Framework not falsified by any of {total_count} criteria{replication_text}"
            )
        elif falsified_count == total_count:
            return f"APGI Framework falsified by all {total_count} criteria{replication_text}"
        else:
            return f"APGI Framework partially falsified ({falsified_count}/{total_count} criteria){replication_text}"

    def _generate_recommendations(
        self,
        test_results: List[StatisticalResult],
        effect_sizes: List[EffectSizeResult],
        power_analysis: PowerReport,
        assessments: List[FalsificationAssessment],
    ) -> List[str]:
        """Generate study recommendations."""
        recommendations = []

        # Power-based recommendations
        if not power_analysis.overall_adequacy:
            recommendations.append("Increase sample sizes to achieve adequate statistical power")

        # Effect size recommendations
        small_effects = [es for es in effect_sizes if abs(es.value) < 0.2]
        if len(small_effects) > len(effect_sizes) / 2:
            recommendations.append("Consider whether small effect sizes are practically meaningful")

        # Falsification-specific recommendations
        inconclusive_count = sum(
            1 for a in assessments if a.conclusion == FalsificationConclusion.INCONCLUSIVE
        )
        if inconclusive_count > 0:
            recommendations.append(
                "Conduct additional studies to resolve inconclusive falsification criteria"
            )

        # Replication recommendations
        if any("replication" in r.test_type for r in test_results):
            recommendations.append("Conduct independent replications to validate findings")

        return recommendations

    def _generate_abstract(self, summary: StatisticalSummary) -> str:
        """Generate publication abstract."""
        return f"""
        Background: This study tested the falsification criteria of the Allostatic Precision-Gated Ignition (APGI) Framework.
        
        Methods: We conducted {len(summary.test_results)} statistical tests across {len(summary.falsification_assessments)} falsification criteria with a total sample of {summary.power_analysis.power_summary.get('total_tests', 'N')} participants.
        
        Results: {summary.overall_conclusion}
        
        Conclusions: The findings provide {'strong' if 'falsified by all' in summary.overall_conclusion else 'mixed' if 'partially' in summary.overall_conclusion else 'limited'} evidence regarding the APGI Framework's validity.
        """

    def _generate_methods_section(self, summary: StatisticalSummary) -> str:
        """Generate methods section."""
        return f"""
        ## Statistical Analysis
        
        Statistical analyses were conducted using a comprehensive falsification testing framework. 
        We employed {len(set(r.test_type for r in summary.test_results))} different statistical tests 
        with a significance threshold of α = {self.significance_threshold}.
        
        Power analysis indicated {'adequate' if summary.power_analysis.overall_adequacy else 'inadequate'} 
        statistical power across test configurations.
        
        Effect sizes were calculated using Cohen's d and eta-squared measures with 95% confidence intervals.
        """

    def _generate_results_section(self, summary: StatisticalSummary) -> str:
        """Generate results section."""
        results = "## Results\n\n"

        for assessment in summary.falsification_assessments:
            results += f"### {assessment.criterion_type.replace('_', ' ').title()}\n\n"
            results += f"- Evidence strength: {assessment.evidence_strength:.3f}\n"
            results += f"- Primary p-value: {assessment.p_value:.4f}\n"
            results += f"- Effect size: {assessment.effect_size:.3f}\n"
            results += f"- Conclusion: {assessment.conclusion.value}\n\n"

        return results

    def _generate_discussion_section(self, summary: StatisticalSummary) -> str:
        """Generate discussion section."""
        return f"""
        ## Discussion
        
        {summary.overall_conclusion}
        
        The statistical evidence {'strongly supports' if 'falsified by all' in summary.overall_conclusion else 'provides mixed support for' if 'partially' in summary.overall_conclusion else 'does not support'} 
        falsification of the APGI Framework.
        
        ### Limitations and Future Directions
        
        {chr(10).join(f"- {rec}" for rec in summary.recommendations)}
        """

    def _generate_tables(self, summary: StatisticalSummary) -> List[Dict[str, Any]]:
        """Generate statistical tables."""
        tables = []

        # Main results table
        results_data = []
        for assessment in summary.falsification_assessments:
            results_data.append(
                {
                    "Criterion": assessment.criterion_type.replace("_", " ").title(),
                    "P-value": f"{assessment.p_value:.4f}",
                    "Effect Size": f"{assessment.effect_size:.3f}",
                    "95% CI": f"[{assessment.confidence_interval[0]:.3f}, {assessment.confidence_interval[1]:.3f}]",
                    "Conclusion": assessment.conclusion.value.replace("_", " ").title(),
                }
            )

        tables.append(
            {
                "title": "Falsification Criteria Results",
                "data": results_data,
                "caption": "Statistical results for each APGI Framework falsification criterion",
            }
        )

        return tables

    def _generate_figure_descriptions(self, summary: StatisticalSummary) -> List[Dict[str, Any]]:
        """Generate figure descriptions."""
        return [
            {
                "title": "Effect Size Forest Plot",
                "description": "Forest plot showing effect sizes and confidence intervals for each falsification criterion",
                "type": "forest_plot",
            },
            {
                "title": "Power Analysis Summary",
                "description": "Statistical power across different test configurations",
                "type": "power_plot",
            },
        ]

    def _generate_references(self) -> List[str]:
        """Generate reference list."""
        return [
            "Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.). Erlbaum.",
            "Lakens, D. (2013). Calculating and reporting effect sizes to facilitate cumulative science. Frontiers in Psychology, 4, 863.",
            "Open Science Collaboration. (2015). Estimating the reproducibility of psychological science. Science, 349(6251).",
        ]

    def _generate_supplementary_materials(self, summary: StatisticalSummary) -> Dict[str, Any]:
        """Generate supplementary materials."""
        return {
            "raw_data": "Available upon request",
            "analysis_code": "Statistical analysis code available at [repository URL]",
            "power_calculations": asdict(summary.power_analysis),
            "detailed_results": [asdict(result) for result in summary.test_results],
        }

    def _export_json(
        self, report: Union[StatisticalSummary, PublicationReport], filename: str
    ) -> str:
        """Export report as JSON."""
        output_path = f"{filename}.json"

        # Convert dataclass to dictionary
        if isinstance(report, (StatisticalSummary, PublicationReport)):
            report_dict = asdict(report)

            # Handle datetime serialization
            def json_serializer(obj: Any) -> str:
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            with open(output_path, "w") as f:
                json.dump(report_dict, f, indent=2, default=json_serializer)

        return output_path

    def _export_markdown(
        self, report: Union[StatisticalSummary, PublicationReport], filename: str
    ) -> str:
        """Export report as Markdown."""
        output_path = f"{filename}.md"

        if isinstance(report, PublicationReport):
            content = f"""# {report.title}

{report.abstract}

{report.methods_section}

{report.results_section}

{report.discussion_section}

## Tables

{chr(10).join(f"### {table['title']}{chr(10)}{table['caption']}" for table in report.tables)}

## Figures

{chr(10).join(f"### {fig['title']}{chr(10)}{fig['description']}" for fig in report.figures)}
"""
        else:
            # StatisticalSummary
            content = f"""# Statistical Summary: {report.study_id}

## Overall Conclusion
{report.overall_conclusion}

## Falsification Assessments

{chr(10).join(f"### {assessment.criterion_type}{chr(10)}- Conclusion: {assessment.conclusion.value}{chr(10)}- Evidence strength: {assessment.evidence_strength:.3f}" for assessment in report.falsification_assessments)}

## Recommendations

{chr(10).join(f"- {rec}" for rec in report.recommendations)}
"""

        with open(output_path, "w") as f:
            f.write(content)

        return output_path

    def _export_html(
        self, report: Union[StatisticalSummary, PublicationReport], filename: str
    ) -> str:
        """Export report as HTML."""
        output_path = f"{filename}.html"

        # Convert markdown to HTML (simplified)
        markdown_path = self._export_markdown(report, filename + "_temp")

        with open(markdown_path, "r") as f:
            markdown_content = f.read()

        # Simple markdown to HTML conversion
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>APGI Framework Statistical Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
<pre>{markdown_content}</pre>
</body>
</html>"""

        with open(output_path, "w") as f:
            f.write(html_content)

        # Clean up temporary file
        import os

        os.remove(markdown_path)

        return output_path
