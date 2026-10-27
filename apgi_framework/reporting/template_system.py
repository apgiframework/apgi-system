"""Custom Report Template System for APGI Framework.

Provides a flexible template system for generating customized reports
with user-defined layouts, content sections, and styling options.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jinja2


class TemplateFormat(Enum):
    """Supported template formats."""

    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    JSON = "json"


@dataclass
class TemplateVariable:
    """Represents a template variable with metadata."""

    name: str
    description: str
    type: str = "string"  # string, number, boolean, list, dict
    required: bool = True
    default: Any = None
    validation: Optional[Dict[str, Any]] = None


@dataclass
class TemplateSection:
    """Represents a section in a report template."""

    name: str
    title: str
    content_template: str
    variables: List[TemplateVariable] = field(default_factory=list)
    optional: bool = False
    order: int = 0
    conditions: Optional[Dict[str, Any]] = None  # Conditions for including section


@dataclass
class ReportTemplate:
    """Represents a complete report template."""

    name: str
    description: str
    format: TemplateFormat
    sections: List[TemplateSection] = field(default_factory=list)
    global_variables: List[TemplateVariable] = field(default_factory=list)
    styles: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "format": self.format.value,
            "sections": [
                {
                    "name": section.name,
                    "title": section.title,
                    "content_template": section.content_template,
                    "variables": [
                        {
                            "name": var.name,
                            "description": var.description,
                            "type": var.type,
                            "required": var.required,
                            "default": var.default,
                            "validation": var.validation,
                        }
                        for var in section.variables
                    ],
                    "optional": section.optional,
                    "order": section.order,
                    "conditions": section.conditions,
                }
                for section in self.sections
            ],
            "global_variables": [
                {
                    "name": var.name,
                    "description": var.description,
                    "type": var.type,
                    "required": var.required,
                    "default": var.default,
                    "validation": var.validation,
                }
                for var in self.global_variables
            ],
            "styles": self.styles,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportTemplate":
        """Create template from dictionary."""
        sections = []
        for section_data in data.get("sections", []):
            variables = [
                TemplateVariable(**var_data) for var_data in section_data.get("variables", [])
            ]
            sections.append(
                TemplateSection(
                    name=section_data["name"],
                    title=section_data["title"],
                    content_template=section_data["content_template"],
                    variables=variables,
                    optional=section_data.get("optional", False),
                    order=section_data.get("order", 0),
                    conditions=section_data.get("conditions"),
                )
            )

        global_variables = [
            TemplateVariable(**var_data) for var_data in data.get("global_variables", [])
        ]

        return cls(
            name=data["name"],
            description=data["description"],
            format=TemplateFormat(data["format"]),
            sections=sections,
            global_variables=global_variables,
            styles=data.get("styles"),
            metadata=data.get("metadata"),
        )


class TemplateManager:
    """Manages report templates and template operations."""

    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir or Path("templates")
        self.template_dir.mkdir(exist_ok=True)

        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

        # Add custom filters
        self._setup_custom_filters()

        # Built-in templates
        self._create_builtin_templates()

    def _setup_custom_filters(self) -> None:
        """Setup custom Jinja2 filters for APGI-specific formatting."""

        def format_parameter(value: Any, precision: int = 3) -> str:
            """Format parameter values with appropriate precision."""
            if isinstance(value, (int, float)):
                return f"{value:.{precision}f}"
            return str(value)

        def format_pvalue(value: Any) -> str:
            """Format p-values with appropriate precision."""
            if isinstance(value, (int, float)):
                if value < 0.001:
                    return "p < 0.001"
                elif value < 0.01:
                    return f"{value:.3f}"
                else:
                    return f"{value:.2f}"
            return str(value)

        def format_percentage(value: Any) -> str:
            """Format values as percentages."""
            if isinstance(value, (int, float)):
                return f"{value * 100:.1f}%"
            return str(value)

        def format_ci(lower: float, upper: float, precision: int = 3) -> str:
            """Format confidence intervals."""
            return f"[{lower:.{precision}f}, {upper:.{precision}f}]"

        # Register filters
        self.jinja_env.filters["param"] = format_parameter
        self.jinja_env.filters["pvalue"] = format_pvalue
        self.jinja_env.filters["percentage"] = format_percentage
        self.jinja_env.filters["ci"] = format_ci

    def _create_builtin_templates(self) -> None:
        """Create built-in templates."""
        # APGI Experiment Report Template
        apgi_template = ReportTemplate(
            name="apgi_experiment",
            description="Standard APGI Framework experiment report",
            format=TemplateFormat.HTML,
            global_variables=[
                TemplateVariable("title", "Report title", default="APGI Experiment Report"),
                TemplateVariable("author", "Report author", default="APGI Framework User"),
                TemplateVariable(
                    "date", "Report date", default=datetime.now().strftime("%Y-%m-%d")
                ),
            ],
            sections=[
                TemplateSection(
                    name="header",
                    title="Report Header",
                    content_template="""
                    <h1>{{ title }}</h1>
                    <p><strong>Author:</strong> {{ author }}</p>
                    <p><strong>Date:</strong> {{ date }}</p>
                    <hr>
                    """,
                    order=0,
                ),
                TemplateSection(
                    name="summary",
                    title="Executive Summary",
                    content_template="""
                    <h2>Executive Summary</h2>
                    <p>{{ summary_text }}</p>
                    {% if key_findings %}
                    <h3>Key Findings</h3>
                    <ul>
                    {% for finding in key_findings %}
                        <li>{{ finding }}</li>
                    {% endfor %}
                    </ul>
                    {% endif %}
                    """,
                    variables=[
                        TemplateVariable("summary_text", "Summary text"),
                        TemplateVariable(
                            "key_findings",
                            "List of key findings",
                            type="list",
                            required=False,
                        ),
                    ],
                    order=1,
                ),
                TemplateSection(
                    name="parameters",
                    title="Parameter Estimates",
                    content_template="""
                    <h2>Parameter Estimates</h2>
                    <table border="1" style="border-collapse: collapse; width: 100%;">
                        <tr>
                            <th>Parameter</th>
                            <th>Estimate</th>
                            <th>95% CI</th>
                            <th>Description</th>
                        </tr>
                        {% for param in parameters %}
                        <tr>
                            <td>{{ param.symbol }}</td>
                            <td>{{ param.estimate | param }}</td>
                            <td>{{ param.ci_lower | ci(param.ci_upper) }}</td>
                            <td>{{ param.description }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                    """,
                    variables=[
                        TemplateVariable("parameters", "List of parameter estimates", type="list")
                    ],
                    order=2,
                ),
                TemplateSection(
                    name="statistics",
                    title="Statistical Tests",
                    content_template="""
                    <h2>Statistical Analysis</h2>
                    {% for test in statistical_tests %}
                    <h3>{{ test.name }}</h3>
                    <p><strong>Statistic:</strong> {{ test.statistic | param }}</p>
                    <p><strong>p-value:</strong> {{ test.p_value | pvalue }}</p>
                    {% if test.effect_size %}
                    <p><strong>Effect Size:</strong> {{ test.effect_size | param }}</p>
                    {% endif %}
                    <p><strong>Interpretation:</strong> {{ test.interpretation }}</p>
                    {% endfor %}
                    """,
                    variables=[
                        TemplateVariable(
                            "statistical_tests",
                            "List of statistical tests",
                            type="list",
                        )
                    ],
                    order=3,
                    optional=True,
                ),
                TemplateSection(
                    name="conclusions",
                    title="Conclusions",
                    content_template="""
                    <h2>Conclusions</h2>
                    <p>{{ conclusions_text }}</p>
                    {% if limitations %}
                    <h3>Limitations</h3>
                    <p>{{ limitations }}</p>
                    {% endif %}
                    {% if future_work %}
                    <h3>Future Work</h3>
                    <p>{{ future_work }}</p>
                    {% endif %}
                    """,
                    variables=[
                        TemplateVariable("conclusions_text", "Main conclusions"),
                        TemplateVariable("limitations", "Study limitations", required=False),
                        TemplateVariable("future_work", "Future work suggestions", required=False),
                    ],
                    order=4,
                ),
            ],
            styles={"css": """
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
                h2 { color: #34495e; margin-top: 30px; }
                h3 { color: #7f8c8d; }
                table { margin: 20px 0; }
                th { background-color: #3498db; color: white; padding: 10px; }
                td { padding: 8px; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                """},
        )

        self.save_template(apgi_template)

    def create_template(
        self, name: str, description: str, format: TemplateFormat
    ) -> ReportTemplate:
        """Create a new empty template."""
        return ReportTemplate(name=name, description=description, format=format)

    def save_template(self, template: ReportTemplate) -> bool:
        """Save a template to disk."""
        try:
            template_file = self.template_dir / f"{template.name}.json"
            with open(template_file, "w") as f:
                json.dump(template.to_dict(), f, indent=2)

            # Also save the content templates as separate files
            for section in template.sections:
                section_file = self.template_dir / f"{template.name}_{section.name}.html"
                with open(section_file, "w") as f:
                    f.write(section.content_template)

            return True
        except Exception:
            return False

    def load_template(self, name: str) -> Optional[ReportTemplate]:
        """Load a template from disk."""
        try:
            template_file = self.template_dir / f"{name}.json"
            if not template_file.exists():
                return None

            with open(template_file, "r") as f:
                data = json.load(f)

            return ReportTemplate.from_dict(data)
        except Exception:
            return None

    def list_templates(self) -> List[str]:
        """List available templates."""
        templates = []
        for template_file in self.template_dir.glob("*.json"):
            templates.append(template_file.stem)
        return templates

    def delete_template(self, name: str) -> bool:
        """Delete a template."""
        try:
            template_file = self.template_dir / f"{name}.json"
            if template_file.exists():
                template_file.unlink()

            # Delete associated section files
            for section_file in self.template_dir.glob(f"{name}_*.html"):
                section_file.unlink()

            return True
        except Exception:
            return False

    def validate_data(self, template: ReportTemplate, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate data against template requirements."""
        errors: Dict[str, List[str]] = {}

        # Validate global variables
        for var in template.global_variables:
            if var.required and var.name not in data:
                if "global" not in errors:
                    errors["global"] = []
                errors["global"].append(f"Required variable '{var.name}' is missing")

        # Validate section variables
        for section in template.sections:
            if not section.optional or self._check_section_conditions(section, data):
                for var in section.variables:
                    if var.required and var.name not in data:
                        if section.name not in errors:
                            errors[section.name] = []
                        errors[section.name].append(f"Required variable '{var.name}' is missing")

        return errors

    def _check_section_conditions(self, section: TemplateSection, data: Dict[str, Any]) -> bool:
        """Check if section conditions are met."""
        if not section.conditions:
            return True

        # Simple condition checking (can be extended)
        for key, expected_value in section.conditions.items():
            if key not in data or data[key] != expected_value:
                return False

        return True

    def render_template(self, template: ReportTemplate, data: Dict[str, Any]) -> str:
        """Render a template with provided data."""
        # Validate data first
        errors = self.validate_data(template, data)
        if errors:
            raise ValueError(f"Data validation failed: {errors}")

        # Sort sections by order
        sections = sorted(template.sections, key=lambda s: s.order)

        rendered_sections = []

        for section in sections:
            # Check if section should be included
            if section.optional and not self._check_section_conditions(section, data):
                continue

            # Render section template
            template_obj = self.jinja_env.from_string(section.content_template)
            rendered_content = template_obj.render(**data)
            rendered_sections.append(rendered_content)

        # Combine all sections
        full_content = "\n".join(rendered_sections)

        # Add styles if HTML format
        if template.format == TemplateFormat.HTML and template.styles:
            css = template.styles.get("css", "")
            full_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>{data.get('title', 'Report')}</title>
                <style>{css}</style>
            </head>
            <body>
            {full_content}
            </body>
            </html>
            """

        return full_content

    def generate_report(
        self, template_name: str, data: Dict[str, Any], output_path: Union[str, Path]
    ) -> bool:
        """Generate a report using a template."""
        try:
            template = self.load_template(template_name)
            if not template:
                return False

            rendered_content = self.render_template(template, data)

            output_path = Path(output_path)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rendered_content)

            return True
        except Exception:
            return False


class TemplateBuilder:
    """Interactive template builder for creating custom templates."""

    def __init__(self, manager: TemplateManager):
        self.manager = manager
        self.current_template: Optional[ReportTemplate] = None

    def start_template(
        self, name: str, description: str, format: TemplateFormat
    ) -> Optional[ReportTemplate]:
        """Start building a new template."""
        self.current_template = self.manager.create_template(name, description, format)
        return self.current_template

    def add_global_variable(
        self,
        name: str,
        description: str,
        var_type: str = "string",
        required: bool = True,
        default: Any = None,
    ) -> bool:
        """Add a global variable to the current template."""
        if not self.current_template:
            return False

        var = TemplateVariable(name, description, var_type, required, default)
        self.current_template.global_variables.append(var)
        return True

    def add_section(
        self,
        name: str,
        title: str,
        content_template: str,
        order: int = 0,
        optional: bool = False,
    ) -> Optional[TemplateSection]:
        """Add a section to the current template."""
        if not self.current_template:
            return None

        section = TemplateSection(name, title, content_template, order=order, optional=optional)
        self.current_template.sections.append(section)
        return section

    def add_section_variable(
        self,
        section_name: str,
        var_name: str,
        description: str,
        var_type: str = "string",
        required: bool = True,
        default: Any = None,
    ) -> bool:
        """Add a variable to a specific section."""
        if not self.current_template:
            return False

        section = next((s for s in self.current_template.sections if s.name == section_name), None)
        if not section:
            return False

        var = TemplateVariable(var_name, description, var_type, required, default)
        section.variables.append(var)
        return True

    def set_styles(self, styles: Dict[str, Any]) -> bool:
        """Set styles for the current template."""
        if not self.current_template:
            return False

        self.current_template.styles = styles
        return True

    def save_current_template(self) -> bool:
        """Save the current template."""
        if not self.current_template:
            return False

        return self.manager.save_template(self.current_template)


# Convenience functions
def create_template_manager(template_dir: Optional[Path] = None) -> TemplateManager:
    """Create a template manager."""
    return TemplateManager(template_dir)


def create_template_builder(manager: TemplateManager) -> TemplateBuilder:
    """Create a template builder."""
    return TemplateBuilder(manager)
