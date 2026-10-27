"""In-System Help System with context-sensitive guidance and tooltips.

Provides real-time help and guidance within the APGI Framework GUI.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

from apgi_framework.logging.standardized_logging import get_logger

logger = get_logger(__name__)


class HelpContext(Enum):
    """Context for help system."""

    MAIN_WINDOW = "main_window"
    SESSION_SETUP = "session_setup"
    HARDWARE_CONFIG = "hardware_config"
    DETECTION_TASK = "detection_task"
    HEARTBEAT_TASK = "heartbeat_task"
    ODDBALL_TASK = "oddball_task"
    MONITORING = "monitoring"
    QUALITY_CONTROL = "quality_control"
    PARAMETER_RESULTS = "parameter_results"
    DATA_EXPORT = "data_export"
    TROUBLESHOOTING = "troubleshooting"


@dataclass
class HelpContent:
    """Help content for a specific context."""

    context: HelpContext
    title: str
    description: str
    steps: Optional[List[str]] = None
    tips: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    related_topics: Optional[List[str]] = None
    video_url: Optional[str] = None


@dataclass
class Tooltip:
    """Tooltip content for UI element."""

    element_id: str
    text: str
    detailed_help: Optional[str] = None


class InSystemHelpSystem:
    """Provides context-sensitive help within the APGI Framework GUI.

    Features:
    - Context-aware help content
    - Interactive tooltips
    - Step-by-step guides
    - Video tutorials
    - Quick tips
    """

    def __init__(self) -> None:
        """Initialize in-system help system."""
        self.logger = logging.getLogger(__name__)

        self.help_content: Dict[HelpContext, HelpContent] = self._define_help_content()
        self.tooltips: Dict[str, Tooltip] = self._define_tooltips()

        # Callback for showing help
        self.show_help_callback: Optional[Callable] = None

    def _define_help_content(self) -> Dict[HelpContext, HelpContent]:
        """Define help content for all contexts."""
        content = {}

        # Main window help
        content[HelpContext.MAIN_WINDOW] = HelpContent(
            context=HelpContext.MAIN_WINDOW,
            title="APGI Framework Main Window",
            description="The main window provides access to all parameter estimation functions.",
            steps=[
                "Click 'New Session' to start a new participant session",
                "Use 'Configure Hardware' to set up EEG, eye tracker, and cardiac sensors",
                "Select 'View Results' to review completed sessions",
                "Access 'Settings' to customize system parameters",
            ],
            tips=[
                "Always validate hardware configuration before starting a session",
                "Check system requirements in the status bar",
                "Use keyboard shortcuts for faster navigation (F1 for help)",
            ],
            related_topics=["Session Setup", "Hardware Configuration"],
        )

        # Session setup help
        content[HelpContext.SESSION_SETUP] = HelpContent(
            context=HelpContext.SESSION_SETUP,
            title="Session Setup",
            description="Configure a new parameter estimation session.",
            steps=[
                "Enter participant ID (required)",
                "Select session type (full protocol or individual tasks)",
                "Configure task parameters (use defaults for standard protocol)",
                "Verify hardware connections (green indicators)",
                "Click 'Start Session' when ready",
            ],
            tips=[
                "Use consistent participant ID format (e.g., P001, P002)",
                "Full protocol takes 45-60 minutes",
                "Individual tasks can be run separately for troubleshooting",
            ],
            warnings=[
                "Ensure all hardware is connected before starting",
                "Do not change hardware configuration during session",
                "Participant ID cannot be changed after session starts",
            ],
            related_topics=["Hardware Configuration", "Task Protocols"],
        )

        # Hardware configuration help
        content[HelpContext.HARDWARE_CONFIG] = HelpContent(
            context=HelpContext.HARDWARE_CONFIG,
            title="Hardware Configuration",
            description="Configure EEG, eye tracker, and cardiac sensors.",
            steps=[
                "Select EEG system from dropdown",
                "Configure channel layout and sampling rate",
                "Select eye tracker model",
                "Run calibration for eye tracker",
                "Select cardiac sensor type",
                "Test all connections",
                "Save configuration",
            ],
            tips=[
                "Use predefined configurations for common hardware",
                "Test connections before each session",
                "Save custom configurations for reuse",
                "Check LSL stream names match hardware output",
            ],
            warnings=[
                "Incorrect configuration may cause data loss",
                "Always test configuration before participant arrival",
                "Some settings require hardware restart",
            ],
            related_topics=["LSL Configuration", "Troubleshooting"],
        )

        # Detection task help
        content[HelpContext.DETECTION_TASK] = HelpContent(
            context=HelpContext.DETECTION_TASK,
            title="Detection Task (θ₀ Estimation)",
            description="Adaptive visual/auditory detection task for baseline ignition threshold.",
            steps=[
                "Explain task to participant: 'Press SPACE when you see/hear stimulus'",
                "Run practice trials (5-10 trials)",
                "Start main task (200 trials, ~10 minutes)",
                "Monitor response rate (target: 40-60%)",
                "Monitor EEG quality during task",
                "Task completes automatically",
            ],
            tips=[
                "Encourage fast responses",
                "Response rate indicates proper threshold",
                "Pause if participant needs break",
                "Monitor P3b amplitude in real-time",
            ],
            warnings=[
                "Do not interrupt during critical trials",
                "High response rate (>80%) indicates threshold too low",
                "Low response rate (<20%) indicates threshold too high",
            ],
            related_topics=["Task Protocols", "Quality Monitoring"],
        )

        # Heartbeat detection task help
        content[HelpContext.HEARTBEAT_TASK] = HelpContent(
            context=HelpContext.HEARTBEAT_TASK,
            title="Heartbeat Detection Task (Πᵢ Estimation)",
            description="Heartbeat detection with confidence ratings for interoceptive precision.",
            steps=[
                "Explain task: 'Judge if tone is synchronized with heartbeat'",
                "Instruct confidence ratings (1=guessing, 4=certain)",
                "Run practice trials (5 trials)",
                "Start main task (60 trials, ~8 minutes)",
                "Monitor cardiac signal quality",
                "Monitor pupil tracking quality",
                "Task completes automatically",
            ],
            tips=[
                "Emphasize importance of confidence ratings",
                "Encourage focus on internal sensations",
                "Monitor HEP amplitude in real-time",
                "Check pupil dilation responses",
            ],
            warnings=[
                "Ensure good cardiac signal before starting",
                "Poor cardiac quality invalidates results",
                "Pupil data loss >20% may affect estimates",
            ],
            related_topics=["Cardiac Monitoring", "Pupillometry"],
        )

        # Oddball task help
        content[HelpContext.ODDBALL_TASK] = HelpContent(
            context=HelpContext.ODDBALL_TASK,
            title="Dual-Modality Oddball Task (β Estimation)",
            description="Oddball detection for interoceptive vs exteroceptive bias.",
            steps=[
                "Run stimulus calibration first (5-10 minutes)",
                "Explain task: 'Press SPACE for oddball stimuli'",
                "Emphasize two types of oddballs (internal and external)",
                "Run practice trials (10 trials)",
                "Start main task (120 trials, ~12 minutes)",
                "Monitor detection accuracy for both types",
                "Monitor P3b to both deviant types",
                "Task completes automatically",
            ],
            tips=[
                "Calibration ensures equal detectability",
                "Monitor detection rates (target: 60-90%)",
                "Compare P3b amplitudes between deviant types",
                "Ensure participant understands both oddball types",
            ],
            warnings=[
                "CO₂ puff safety: brief duration (<500ms)",
                "Monitor participant comfort with interoceptive stimuli",
                "Unequal detection rates may indicate calibration issues",
            ],
            related_topics=["Stimulus Calibration", "P3b Analysis"],
        )

        # Monitoring help
        content[HelpContext.MONITORING] = HelpContent(
            context=HelpContext.MONITORING,
            title="Real-Time Monitoring",
            description="Monitor data quality and task progress in real-time.",
            steps=[
                "Check EEG impedance panel (green = good)",
                "Monitor artifact detection rate (target: <30%)",
                "Check pupil tracking quality (target: >80%)",
                "Monitor cardiac signal quality",
                "Review task progress and trial count",
                "Watch for quality alerts",
            ],
            tips=[
                "Green indicators = good quality",
                "Yellow = warning, may need attention",
                "Red = critical, requires immediate action",
                "Pause task if quality degrades significantly",
            ],
            warnings=[
                "Do not ignore quality alerts",
                "Poor quality data cannot be recovered",
                "Address issues immediately when possible",
            ],
            related_topics=["Quality Control", "Troubleshooting"],
        )

        # Quality control help
        content[HelpContext.QUALITY_CONTROL] = HelpContent(
            context=HelpContext.QUALITY_CONTROL,
            title="Data Quality Control",
            description="Ensure high-quality data collection.",
            steps=[
                "Check all quality metrics before starting",
                "Monitor continuously during tasks",
                "Address alerts immediately",
                "Document any quality issues",
                "Review quality summary after session",
            ],
            tips=[
                "Prevention is better than correction",
                "Regular breaks improve quality",
                "Participant comfort reduces artifacts",
                "Good setup saves time later",
            ],
            warnings=[
                "Cannot recover from poor quality data",
                "Wide parameter CIs indicate quality issues",
                "May need to repeat session if quality poor",
            ],
            related_topics=["Monitoring", "Troubleshooting"],
        )

        # Parameter results help
        content[HelpContext.PARAMETER_RESULTS] = HelpContent(
            context=HelpContext.PARAMETER_RESULTS,
            title="Parameter Results",
            description="Interpret parameter estimates and credible intervals.",
            steps=[
                "Review parameter estimates (θ₀, Πᵢ, β)",
                "Check credible interval widths",
                "Review data quality summary",
                "Compare to normative ranges",
                "Export results if satisfactory",
            ],
            tips=[
                "Narrow CIs (<0.3) indicate high confidence",
                "Wide CIs (>0.5) suggest retest",
                "Check quality metrics if estimates seem unusual",
                "Compare to previous sessions for reliability",
            ],
            warnings=[
                "Do not over-interpret single estimates",
                "Consider test-retest reliability",
                "Clinical interpretation requires expertise",
            ],
            related_topics=["Parameter Interpretation", "Data Export"],
        )

        return content

    def _define_tooltips(self) -> Dict[str, Tooltip]:
        """Define tooltips for UI elements."""
        tooltips = {}

        # Main window tooltips
        tooltips["btn_new_session"] = Tooltip(
            element_id="btn_new_session",
            text="Start a new parameter estimation session",
            detailed_help="Opens session setup dialog to configure participant ID and task parameters.",
        )

        tooltips["btn_hardware_config"] = Tooltip(
            element_id="btn_hardware_config",
            text="Configure hardware devices",
            detailed_help="Set up EEG system, eye tracker, and cardiac sensors. Test connections before starting session.",
        )

        # Session setup tooltips
        tooltips["input_participant_id"] = Tooltip(
            element_id="input_participant_id",
            text="Enter unique participant identifier",
            detailed_help="Use consistent format (e.g., P001). Cannot be changed after session starts.",
        )

        tooltips["combo_session_type"] = Tooltip(
            element_id="combo_session_type",
            text="Select session type",
            detailed_help="Full Protocol: All three tasks (~45-60 min). Individual Tasks: Run tasks separately.",
        )

        # Hardware config tooltips
        tooltips["combo_eeg_system"] = Tooltip(
            element_id="combo_eeg_system",
            text="Select EEG system",
            detailed_help="Choose your EEG hardware. Predefined configurations available for common systems.",
        )

        tooltips["spin_sampling_rate"] = Tooltip(
            element_id="spin_sampling_rate",
            text="EEG sampling rate (Hz)",
            detailed_help="Minimum 1000 Hz recommended. Higher rates improve temporal resolution.",
        )

        # Monitoring tooltips
        tooltips["indicator_eeg_quality"] = Tooltip(
            element_id="indicator_eeg_quality",
            text="EEG signal quality",
            detailed_help="Green: Good (<30% artifacts). Yellow: Acceptable (30-40%). Red: Poor (>40%).",
        )

        tooltips["indicator_pupil_quality"] = Tooltip(
            element_id="indicator_pupil_quality",
            text="Pupil tracking quality",
            detailed_help="Shows percentage of valid pupil data. Target: >80%.",
        )

        tooltips["indicator_cardiac_quality"] = Tooltip(
            element_id="indicator_cardiac_quality",
            text="Cardiac signal quality",
            detailed_help="Shows R-peak detection confidence. Green: >90%, Yellow: 70-90%, Red: <70%.",
        )

        # Parameter results tooltips
        tooltips["value_theta0"] = Tooltip(
            element_id="value_theta0",
            text="Baseline ignition threshold (θ₀)",
            detailed_help="Individual threshold for conscious access. Lower values indicate higher sensitivity.",
        )

        tooltips["value_pi_i"] = Tooltip(
            element_id="value_pi_i",
            text="Interoceptive precision (Πᵢ)",
            detailed_help="Sensitivity to internal bodily signals. Higher values indicate better interoceptive awareness.",
        )

        tooltips["value_beta"] = Tooltip(
            element_id="value_beta",
            text="Somatic bias (β)",
            detailed_help="Relative weighting of interoceptive vs exteroceptive information. Higher values indicate interoceptive bias.",
        )

        return tooltips

    def get_help_content(self, context: HelpContext) -> Optional[HelpContent]:
        """Get help content for specific context.

        Args:
            context: Help context.

        Returns:
            Help content if available.
        """
        return self.help_content.get(context)

    def get_tooltip(self, element_id: str) -> Optional[Tooltip]:
        """Get tooltip for UI element.

        Args:
            element_id: UI element identifier.

        Returns:
            Tooltip if available.
        """
        return self.tooltips.get(element_id)

    def show_help(self, context: HelpContext) -> None:
        """Show help for specific context.

        Args:
            context: Help context to display.
        """
        content = self.get_help_content(context)

        if content:
            self.logger.info(f"Showing help for: {content.title}")

            if self.show_help_callback:
                self.show_help_callback(content)
            else:
                # Fallback: print to console
                self._print_help_content(content)
        else:
            self.logger.warning(f"No help content available for context: {context}")

    def _print_help_content(self, content: HelpContent) -> None:
        """Print help content to console (fallback)."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"HELP: {content.title}")
        logger.info(f"{'=' * 60}")
        logger.info(f"\n{content.description}\n")

        if content.steps:
            logger.info("Steps:")
            for i, step in enumerate(content.steps, 1):
                logger.info(f"  {i}. {step}")
            logger.info("")

        if content.tips:
            logger.info("Tips:")
            for tip in content.tips:
                logger.info(f"  💡 {tip}")
            logger.info("")

        if content.warnings:
            logger.info("Warnings:")
            for warning in content.warnings:
                logger.info(f"  ⚠️  {warning}")
            logger.info("")

        if content.related_topics:
            logger.info("Related Topics:")
            for topic in content.related_topics:
                logger.info(f"  → {topic}")
            logger.info("")

        logger.info(f"{'=' * 60}\n")

    def search_help(self, query: str) -> List[HelpContent]:
        """Search help content.

        Args:
            query: Search query.

        Returns:
            List of matching help content.
        """
        query_lower = query.lower()
        results = []

        for content in self.help_content.values():
            # Search in title and description
            if query_lower in content.title.lower() or query_lower in content.description.lower():
                results.append(content)
                continue

            # Search in steps
            if content.steps:
                if any(query_lower in step.lower() for step in content.steps):
                    results.append(content)
                    continue

            # Search in tips
            if content.tips:
                if any(query_lower in tip.lower() for tip in content.tips):
                    results.append(content)
                    continue

        return results

    def register_help_callback(self, callback: Callable) -> None:
        """Register callback for showing help.

        Args:
            callback: Function to call when showing help.
        """
        self.show_help_callback = callback
        self.logger.info("Help callback registered")

    def generate_help_index(self) -> str:
        """Generate help index for all contexts.

        Returns:
            Help index as formatted string.
        """
        index = "# APGI Framework Help Index\n\n"

        for context, content in sorted(self.help_content.items(), key=lambda x: x[1].title):
            index += f"## {content.title}\n"
            index += f"{content.description}\n\n"

            if content.related_topics:
                index += "Related: " + ", ".join(content.related_topics) + "\n\n"

        return index
