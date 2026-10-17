"""
PDF Report Generator for APGI Framework

Provides comprehensive PDF report generation capabilities using ReportLab
for experimental results, statistical analyses, and visualizations.
"""

import datetime
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

try:
    from reportlab.lib import colors  # type: ignore
    from reportlab.lib.enums import TA_CENTER  # type: ignore
    from reportlab.lib.pagesizes import A4  # type: ignore
    from reportlab.lib.styles import ParagraphStyle  # type: ignore
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch  # type: ignore
    from reportlab.platypus import KeepTogether  # type: ignore
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


@dataclass
class ReportSection:
    """Represents a section in the PDF report."""

    title: str
    content: List[Any]
    page_break: bool = False


@dataclass
class ReportConfig:
    """Configuration for PDF report generation."""

    title: str = "APGI Framework Report"
    author: str = "APGI Framework"
    subject: str = "Experimental Results"
    creator: str = "APGI PDF Generator"
    page_size: tuple = A4
    margins: Dict[str, float] = field(
        default_factory=lambda: {
            "top": 1 * inch,
            "bottom": 1 * inch,
            "left": 1 * inch,
            "right": 1 * inch,
        }
    )


class PDFReportGenerator:
    """Generates comprehensive PDF reports for APGI Framework results."""

    def __init__(self, config: Optional[ReportConfig] = None):
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab is required for PDF generation. " "Install with: pip install reportlab"
            )

        self.config = config or ReportConfig()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.sections: List[ReportSection] = []

    def _setup_custom_styles(self) -> None:
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Title"],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue,
            )
        )

        # Section header style
        self.styles.add(
            ParagraphStyle(
                name="SectionHeader",
                parent=self.styles["Heading1"],
                fontSize=16,
                spaceAfter=12,
                spaceBefore=20,
                textColor=colors.darkblue,
                borderWidth=1,
                borderColor=colors.darkblue,
                borderPadding=5,
            )
        )

        # Subsection header style
        self.styles.add(
            ParagraphStyle(
                name="SubsectionHeader",
                parent=self.styles["Heading2"],
                fontSize=14,
                spaceAfter=8,
                spaceBefore=12,
                textColor=colors.darkgreen,
            )
        )

        # Code style
        self.styles.add(
            ParagraphStyle(
                name="Code",
                parent=self.styles["Normal"],
                fontName="Courier",
                fontSize=10,
                leftIndent=20,
                backgroundColor=colors.lightgrey,
                borderWidth=1,
                borderColor=colors.grey,
                borderPadding=5,
            )
        )

        # Caption style
        self.styles.add(
            ParagraphStyle(
                name="Caption",
                parent=self.styles["Normal"],
                fontSize=10,
                alignment=TA_CENTER,
                textColor=colors.grey,
                spaceAfter=12,
            )
        )

    def add_title_page(
        self,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        authors: Optional[List[str]] = None,
        date: Optional[str] = None,
    ) -> None:
        """Add a title page to the report."""
        title = title or self.config.title
        date = date or datetime.datetime.now().strftime("%B %d, %Y")

        content = []

        # Main title
        content.append(Paragraph(title, self.styles["CustomTitle"]))
        content.append(Spacer(1, 0.5 * inch))

        # Subtitle
        if subtitle:
            content.append(Paragraph(subtitle, self.styles["Heading2"]))
            content.append(Spacer(1, 0.3 * inch))

        # Authors
        if authors:
            author_text = "<br/>".join(authors)
            content.append(Paragraph(author_text, self.styles["Normal"]))
            content.append(Spacer(1, 0.3 * inch))

        # Date
        content.append(Paragraph(date, self.styles["Normal"]))
        content.append(Spacer(1, 1 * inch))

        # Add APGI Framework info
        framework_info = """
        <b>APGI Framework</b><br/>
        Allostatic Precision-Gated Ignition<br/>
        Consciousness Research Platform
        """
        content.append(Paragraph(framework_info, self.styles["Normal"]))

        self.sections.append(ReportSection("Title Page", content, page_break=True))

    def add_table_of_contents(self) -> None:
        """Add a table of contents (placeholder for now)."""
        content = []
        content.append(Paragraph("Table of Contents", self.styles["SectionHeader"]))
        content.append(Spacer(1, 0.2 * inch))

        # This is a simplified TOC - in a full implementation,
        # you'd track section titles and page numbers
        toc_items = [
            "1. Executive Summary",
            "2. Experimental Setup",
            "3. Results",
            "4. Statistical Analysis",
            "5. Visualizations",
            "6. Conclusions",
        ]

        for item in toc_items:
            content.append(Paragraph(item, self.styles["Normal"]))

        self.sections.append(ReportSection("Table of Contents", content, page_break=True))

    def add_section(self, title: str, content_items: List[Any], page_break: bool = False) -> None:
        """Add a section to the report."""
        content = []
        content.append(Paragraph(title, self.styles["SectionHeader"]))
        content.append(Spacer(1, 0.1 * inch))

        for item in content_items:
            if isinstance(item, str):
                # Text content
                content.append(Paragraph(item, self.styles["Normal"]))
                content.append(Spacer(1, 0.1 * inch))
            elif isinstance(item, dict):
                # Handle different content types
                if item.get("type") == "table":
                    content.append(self._create_table(item))
                elif item.get("type") == "image":
                    content.append(self._create_image(item))
                elif item.get("type") == "chart":
                    content.append(self._create_chart(item))
                elif item.get("type") == "code":
                    content.append(Paragraph(item["content"], self.styles["Code"]))
                    content.append(Spacer(1, 0.1 * inch))
            else:
                # Direct ReportLab flowable
                content.append(item)

        self.sections.append(ReportSection(title, content, page_break))

    def add_experimental_results(self, results: Dict[str, Any]) -> None:
        """Add experimental results section."""
        content_items: List[Union[str, Dict[str, Any]]] = []

        # Summary statistics
        if "summary" in results:
            summary = results["summary"]
            content_items.append("Experimental Summary:")

            summary_text = f"""
            <b>Total Participants:</b> {summary.get('n_participants', 'N/A')}<br/>
            <b>Total Trials:</b> {summary.get('n_trials', 'N/A')}<br/>
            <b>Success Rate:</b> {summary.get('success_rate', 'N/A'):.2%}<br/>
            <b>Mean Response Time:</b> {summary.get('mean_rt', 'N/A'):.3f} seconds
            """
            content_items.append(summary_text)

        # Parameter estimates
        if "parameters" in results:
            params = results["parameters"]
            content_items.append(
                {
                    "type": "table",
                    "data": [
                        ["Parameter", "Estimate", "CI Lower", "CI Upper"],
                        [
                            "θ₀ (Threshold)",
                            f"{params.get('theta0', 0):.3f}",
                            f"{params.get('theta0_ci_lower', 0):.3f}",
                            f"{params.get('theta0_ci_upper', 0):.3f}",
                        ],
                        [
                            "πᵢ (Interoceptive Precision)",
                            f"{params.get('pi_i', 0):.3f}",
                            f"{params.get('pi_i_ci_lower', 0):.3f}",
                            f"{params.get('pi_i_ci_upper', 0):.3f}",
                        ],
                        [
                            "β (Somatic Marker)",
                            f"{params.get('beta', 0):.3f}",
                            f"{params.get('beta_ci_lower', 0):.3f}",
                            f"{params.get('beta_ci_upper', 0):.3f}",
                        ],
                    ],
                    "title": "Parameter Estimates",
                }
            )

        self.add_section("Experimental Results", content_items)

    def add_statistical_analysis(self, stats: Dict[str, Any]) -> None:
        """Add statistical analysis section."""
        content_items: List[Union[str, Dict[str, Any]]] = []

        # Hypothesis tests
        if "tests" in stats:
            content_items.append("Statistical Tests:")

            for test_name, test_result in stats["tests"].items():
                test_text = f"""
                <b>{test_name}:</b><br/>
                Statistic: {test_result.get('statistic', 'N/A'):.4f}<br/>
                p-value: {test_result.get('p_value', 'N/A'):.4f}<br/>
                Effect Size: {test_result.get('effect_size', 'N/A'):.4f}<br/>
                """
                content_items.append(test_text)

        # Model comparison
        if "model_comparison" in stats:
            comparison = stats["model_comparison"]
            content_items.append(
                {
                    "type": "table",
                    "data": [
                        ["Model", "AIC", "BIC", "Log Likelihood"],
                        *[
                            [
                                model,
                                f"{metrics['aic']:.2f}",
                                f"{metrics['bic']:.2f}",
                                f"{metrics['log_likelihood']:.2f}",
                            ]
                            for model, metrics in comparison.items()
                        ],
                    ],
                    "title": "Model Comparison",
                }
            )

        self.add_section("Statistical Analysis", content_items)

    def add_matplotlib_figure(
        self, fig: Any, title: Optional[str] = None, caption: Optional[str] = None
    ) -> None:
        """Add a matplotlib figure to the report."""
        # Save figure to bytes
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)

        content_items: List[Union[str, Dict[str, Any]]] = []

        if title:
            content_items.append(
                {
                    "type": "image",
                    "buffer": img_buffer,
                    "width": 6 * inch,
                    "height": 4 * inch,
                    "title": title,
                }
            )

        if caption:
            content_items.append(f"<i>{caption}</i>")

        self.add_section("Figure", content_items)

    def _create_table(self, table_info: Dict[str, Any]) -> Any:
        """Create a ReportLab table."""
        data = table_info["data"]
        title = table_info.get("title", "")

        # Create table
        table = Table(data)

        # Style the table
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]

        table.setStyle(TableStyle(table_style))

        # Add title if provided
        elements = []
        if title:
            elements.append(Paragraph(title, self.styles["SubsectionHeader"]))
            elements.append(Spacer(1, 0.1 * inch))

        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))

        return KeepTogether(elements)

    def _create_image(self, image_info: Dict[str, Any]) -> Any:
        """Create a ReportLab image."""
        if "buffer" in image_info:
            # Image from buffer
            img = Image(image_info["buffer"])
        elif "path" in image_info:
            # Image from file path
            img = Image(image_info["path"])
        else:
            return Spacer(1, 0.1 * inch)

        # Set dimensions
        if "width" in image_info and "height" in image_info:
            img.drawWidth = image_info["width"]
            img.drawHeight = image_info["height"]

        elements = []

        # Add title if provided
        if "title" in image_info:
            elements.append(Paragraph(image_info["title"], self.styles["SubsectionHeader"]))
            elements.append(Spacer(1, 0.1 * inch))

        elements.append(img)

        # Add caption if provided
        if "caption" in image_info:
            elements.append(Spacer(1, 0.05 * inch))
            elements.append(Paragraph(image_info["caption"], self.styles["Caption"]))

        elements.append(Spacer(1, 0.2 * inch))

        return KeepTogether(elements)

    def _create_chart(self, chart_info: Dict[str, Any]) -> Any:
        """Create a ReportLab chart (placeholder)."""
        # This is a simplified implementation
        # In practice, you'd create proper ReportLab charts
        return Paragraph("Chart placeholder", self.styles["Normal"])

    def generate_pdf(
        self,
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> bool:
        """Generate the PDF report."""
        try:
            output_path = Path(output_path)

            # Create document
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=self.config.page_size,
                topMargin=self.config.margins["top"],
                bottomMargin=self.config.margins["bottom"],
                leftMargin=self.config.margins["left"],
                rightMargin=self.config.margins["right"],
                title=self.config.title,
                author=self.config.author,
                subject=self.config.subject,
                creator=self.config.creator,
            )

            # Build story (all content)
            story = []
            total_sections = len(self.sections)

            for i, section in enumerate(self.sections):
                if progress_callback:
                    progress_callback(
                        int((i / total_sections) * 100),
                        f"Processing section: {section.title}",
                    )

                # Add section content
                story.extend(section.content)

                # Add page break if requested
                if section.page_break and i < total_sections - 1:
                    story.append(PageBreak())

            # Build PDF
            if progress_callback:
                progress_callback(90, "Building PDF document...")

            doc.build(story)

            if progress_callback:
                progress_callback(100, "PDF generation completed!")

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(0, f"PDF generation failed: {str(e)}")
            return False

    def clear_sections(self) -> None:
        """Clear all sections."""
        self.sections = []


class APGIReportTemplate:
    """Pre-configured report template for APGI Framework results."""

    def __init__(self) -> None:
        self.generator = PDFReportGenerator()

    def create_experiment_report(
        self, experiment_data: Dict[str, Any], output_path: Union[str, Path]
    ) -> bool:
        """Create a complete experiment report."""
        # Clear any existing sections
        self.generator.clear_sections()

        # Add title page
        self.generator.add_title_page(
            title="APGI Framework Experiment Report",
            subtitle=experiment_data.get("experiment_name", "Consciousness Study"),
            authors=experiment_data.get("authors", ["APGI Framework User"]),
            date=experiment_data.get("date"),
        )

        # Add table of contents
        self.generator.add_table_of_contents()

        # Add executive summary
        if "summary" in experiment_data:
            self.generator.add_section("Executive Summary", [experiment_data["summary"]])

        # Add experimental setup
        if "setup" in experiment_data:
            setup_items: List[Any] = []
            setup = experiment_data["setup"]

            setup_text = f"""
            <b>Paradigm:</b> {setup.get('paradigm', 'N/A')}<br/>
            <b>Participants:</b> {setup.get('n_participants', 'N/A')}<br/>
            <b>Trials per condition:</b> {setup.get('n_trials', 'N/A')}<br/>
            <b>Duration:</b> {setup.get('duration', 'N/A')}<br/>
            """
            setup_items.append(setup_text)

            if "parameters" in setup:
                setup_items.append(
                    {
                        "type": "table",
                        "data": [
                            ["Parameter", "Value"],
                            *[[k, str(v)] for k, v in setup["parameters"].items()],
                        ],
                        "title": "Experimental Parameters",
                    }
                )

            self.generator.add_section("Experimental Setup", setup_items)

        # Add results
        if "results" in experiment_data:
            self.generator.add_experimental_results(experiment_data["results"])

        # Add statistical analysis
        if "statistics" in experiment_data:
            self.generator.add_statistical_analysis(experiment_data["statistics"])

        # Add figures
        if "figures" in experiment_data:
            for fig_name, fig_data in experiment_data["figures"].items():
                if "matplotlib_fig" in fig_data:
                    self.generator.add_matplotlib_figure(
                        fig_data["matplotlib_fig"],
                        title=fig_data.get("title", fig_name),
                        caption=fig_data.get("caption"),
                    )

        # Add conclusions
        if "conclusions" in experiment_data:
            self.generator.add_section("Conclusions", [experiment_data["conclusions"]])

        # Generate PDF
        result = self.generator.generate_pdf(output_path)
        return cast(bool, result)


# Convenience functions
def create_pdf_generator(config: Optional[ReportConfig] = None) -> PDFReportGenerator:
    """Create a PDF report generator."""
    return PDFReportGenerator(config)


def generate_experiment_report(
    experiment_data: Dict[str, Any], output_path: Union[str, Path]
) -> bool:
    """Generate a complete experiment report using the default template."""
    template = APGIReportTemplate()
    return template.create_experiment_report(experiment_data, output_path)
