"""Troubleshooting Guide with hardware-specific solutions and FAQ.

Provides comprehensive troubleshooting information for common issues
encountered during APGI Framework parameter estimation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TroubleshootingEntry:
    """Single troubleshooting entry."""

    category: str
    problem: str
    symptoms: List[str]
    causes: List[str]
    solutions: List[str]
    hardware_specific: Optional[Dict[str, List[str]]] = None


class TroubleshootingGuide:
    """Generates comprehensive troubleshooting guide.

    Includes hardware-specific solutions, common issues, and FAQ.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize troubleshooting guide generator.

        Args:
            output_dir: Directory for saving guide files.
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir or Path.home() / ".apgi_framework" / "documentation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.entries = self._define_troubleshooting_entries()

    def _define_troubleshooting_entries(self) -> List[TroubleshootingEntry]:
        """Define all troubleshooting entries."""
        entries = []

        # Installation issues
        entries.append(
            TroubleshootingEntry(
                category="Installation",
                problem="Dependency installation fails",
                symptoms=[
                    "pip install errors",
                    "Module not found errors",
                    "Version conflicts",
                ],
                causes=[
                    "Incompatible Python version",
                    "Missing system libraries",
                    "Network connectivity issues",
                    "Insufficient permissions",
                ],
                solutions=[
                    "Verify Python version (3.8+ required): python --version",
                    "Update pip: python -m pip install --upgrade pip",
                    "Install system dependencies (Linux): sudo apt-get install python3-dev",
                    "Use virtual environment: python -m venv apgi_env",
                    "Install with --user flag: pip install --user apgi-framework",
                    "Check firewall/proxy settings",
                ],
            )
        )

        # EEG issues
        entries.append(
            TroubleshootingEntry(
                category="EEG",
                problem="High impedance readings",
                symptoms=[
                    "Impedance > 10 kΩ",
                    "Noisy EEG signal",
                    "Frequent artifacts",
                ],
                causes=[
                    "Insufficient gel",
                    "Poor electrode contact",
                    "Dry skin",
                    "Hair interference",
                ],
                solutions=[
                    "Apply more conductive gel",
                    "Part hair at electrode sites",
                    "Gently abrade skin with prep pad",
                    "Ensure electrode pins penetrate gel",
                    "Check electrode connections",
                    "Replace damaged electrodes",
                ],
                hardware_specific={
                    "BioSemi": [
                        "Check CMS/DRL reference electrodes",
                        "Verify battery charge",
                        "Ensure proper grounding",
                    ],
                    "EGI": [
                        "Soak sponges thoroughly",
                        "Check net tension",
                        "Verify amplifier connection",
                    ],
                    "BrainProducts": [
                        "Check active electrode batteries",
                        "Verify ground electrode",
                        "Test with impedance meter",
                    ],
                },
            )
        )

        entries.append(
            TroubleshootingEntry(
                category="EEG",
                problem="No EEG data streaming",
                symptoms=[
                    "LSL stream not found",
                    "No data in monitoring window",
                    "Connection timeout",
                ],
                causes=[
                    "Amplifier not powered on",
                    "LSL not configured",
                    "Network issues",
                    "Driver problems",
                ],
                solutions=[
                    "Verify amplifier power and connection",
                    "Check LSL stream name in configuration",
                    "Test LSL with LabRecorder",
                    "Restart amplifier and software",
                    "Check firewall settings",
                    "Update device drivers",
                    "Verify USB/network connection",
                ],
                hardware_specific={
                    "BioSemi": [
                        "Check USB receiver connection",
                        "Verify ActiView is running",
                        "Check battery level",
                    ],
                    "EGI": [
                        "Verify Net Station connection",
                        "Check network settings",
                        "Restart Net Amps",
                    ],
                },
            )
        )

        # Eye tracking issues
        entries.append(
            TroubleshootingEntry(
                category="Eye Tracking",
                problem="Poor calibration accuracy",
                symptoms=[
                    "Calibration error > 1°",
                    "Validation fails",
                    "Inconsistent gaze tracking",
                ],
                causes=[
                    "Participant movement",
                    "Poor lighting",
                    "Reflections on glasses",
                    "Incorrect distance",
                ],
                solutions=[
                    "Ensure stable head position (chin rest)",
                    "Adjust lighting (avoid glare)",
                    "Remove glasses if possible (or use contact lenses)",
                    "Verify 60-70 cm distance",
                    "Clean tracker lens",
                    "Recalibrate with 9 points",
                    "Check pupil detection quality",
                ],
                hardware_specific={
                    "Tobii": [
                        "Use Tobii Eye Tracker Manager for diagnostics",
                        "Check track status indicators",
                        "Verify firmware version",
                    ],
                    "EyeLink": [
                        "Adjust camera focus",
                        "Check illuminator settings",
                        "Use drift correction",
                    ],
                    "Pupil Labs": [
                        "Adjust camera exposure",
                        "Check pupil detection confidence",
                        "Recalibrate world camera",
                    ],
                },
            )
        )

        entries.append(
            TroubleshootingEntry(
                category="Eye Tracking",
                problem="Excessive data loss",
                symptoms=[
                    "Pupil data loss > 20%",
                    "Frequent tracking loss",
                    "Blink detection issues",
                ],
                causes=[
                    "Participant fatigue",
                    "Dry eyes",
                    "Poor lighting",
                    "Tracker positioning",
                ],
                solutions=[
                    "Provide breaks",
                    "Use artificial tears (if approved)",
                    "Adjust lighting conditions",
                    "Reposition tracker",
                    "Check for obstructions",
                    "Verify sampling rate settings",
                ],
            )
        )

        # Cardiac issues
        entries.append(
            TroubleshootingEntry(
                category="Cardiac",
                problem="Poor R-peak detection",
                symptoms=["Missed heartbeats", "False R-peaks", "Irregular HRV"],
                causes=[
                    "Poor electrode contact",
                    "Movement artifacts",
                    "Incorrect lead placement",
                    "Low signal amplitude",
                ],
                solutions=[
                    "Clean skin thoroughly",
                    "Apply electrodes firmly",
                    "Verify Lead II placement",
                    "Minimize participant movement",
                    "Adjust R-peak detection threshold",
                    "Check electrode gel freshness",
                    "Try alternative lead configuration",
                ],
                hardware_specific={
                    "ECG": [
                        "Check electrode expiration date",
                        "Verify ground electrode",
                        "Test with different lead placement",
                    ],
                    "PPG": [
                        "Ensure good finger/earlobe contact",
                        "Minimize movement",
                        "Check sensor positioning",
                        "Verify adequate perfusion",
                    ],
                },
            )
        )

        # Task execution issues
        entries.append(
            TroubleshootingEntry(
                category="Task Execution",
                problem="Task crashes or freezes",
                symptoms=[
                    "Application stops responding",
                    "Python error messages",
                    "Task won't start",
                ],
                causes=[
                    "Insufficient memory",
                    "Hardware connection lost",
                    "Software bug",
                    "Configuration error",
                ],
                solutions=[
                    "Check system resources (RAM, CPU)",
                    "Verify all hardware connections",
                    "Review error logs",
                    "Restart application",
                    "Check configuration files",
                    "Update to latest version",
                    "Contact support with error logs",
                ],
            )
        )

        entries.append(
            TroubleshootingEntry(
                category="Task Execution",
                problem="Stimulus timing issues",
                symptoms=[
                    "Inconsistent stimulus presentation",
                    "Timing warnings in logs",
                    "Synchronization errors",
                ],
                causes=[
                    "High CPU load",
                    "Display refresh rate",
                    "Background processes",
                    "Hardware latency",
                ],
                solutions=[
                    "Close unnecessary applications",
                    "Set process priority to high",
                    "Verify display refresh rate (60+ Hz)",
                    "Disable power saving modes",
                    "Check LSL clock synchronization",
                    "Use dedicated computer for testing",
                ],
            )
        )

        # Data quality issues
        entries.append(
            TroubleshootingEntry(
                category="Data Quality",
                problem="Excessive artifacts",
                symptoms=["Artifact rate > 40%", "Noisy signals", "Quality alerts"],
                causes=[
                    "Participant movement",
                    "Muscle tension",
                    "Environmental noise",
                    "Equipment issues",
                ],
                solutions=[
                    "Instruct participant to relax",
                    "Minimize jaw clenching",
                    "Reduce eye movements during critical periods",
                    "Check for electromagnetic interference",
                    "Verify equipment grounding",
                    "Consider break for participant",
                ],
            )
        )

        # Parameter estimation issues
        entries.append(
            TroubleshootingEntry(
                category="Parameter Estimation",
                problem="Wide credible intervals",
                symptoms=[
                    "CI width > 0.5",
                    "Low confidence estimates",
                    "Poor model convergence",
                ],
                causes=[
                    "Insufficient data quality",
                    "Inconsistent responses",
                    "Model convergence issues",
                    "Extreme parameter values",
                ],
                solutions=[
                    "Review data quality metrics",
                    "Check for response patterns",
                    "Increase number of trials",
                    "Verify model priors",
                    "Consider retest session",
                    "Check for participant understanding",
                ],
            )
        )

        return entries

    def generate_guide(self) -> Path:
        """Generate complete troubleshooting guide.

        Returns:
            Path to generated guide file.
        """
        self.logger.info("Generating troubleshooting guide...")

        guide_path = self.output_dir / "Troubleshooting_Guide.md"

        with open(guide_path, "w", encoding="utf-8") as f:
            f.write(self._generate_header())
            f.write(self._generate_toc())

            # Group entries by category
            categories: Dict[str, List[Any]] = {}
            for entry in self.entries:
                if entry.category not in categories:
                    categories[entry.category] = []
                categories[entry.category].append(entry)

            # Write each category
            for category, entries in sorted(categories.items()):
                f.write(f"\n## {category} Issues\n\n")
                for entry in entries:
                    f.write(self._format_entry(entry))

            f.write(self._generate_faq())
            f.write(self._generate_contact_info())

        self.logger.info(f"Troubleshooting guide generated: {guide_path}")
        return guide_path

    def _generate_header(self) -> str:
        """Generate guide header."""
        return """# APGI Framework Troubleshooting Guide

This guide provides solutions to common issues encountered during APGI Framework parameter estimation.

**Quick Navigation:**
- [Installation Issues](#installation-issues)
- [EEG Issues](#eeg-issues)
- [Eye Tracking Issues](#eye-tracking-issues)
- [Cardiac Issues](#cardiac-issues)
- [Task Execution Issues](#task-execution-issues)
- [Data Quality Issues](#data-quality-issues)
- [Parameter Estimation Issues](#parameter-estimation-issues)
- [FAQ](#frequently-asked-questions)

---

"""

    def _generate_toc(self) -> str:
        """Generate table of contents."""
        return ""  # Already included in header

    def _format_entry(self, entry: TroubleshootingEntry) -> str:
        """Format a single troubleshooting entry."""
        text = f"### {entry.problem}\n\n"

        text += "**Symptoms:**\n"
        for symptom in entry.symptoms:
            text += f"- {symptom}\n"
        text += "\n"

        text += "**Possible Causes:**\n"
        for cause in entry.causes:
            text += f"- {cause}\n"
        text += "\n"

        text += "**Solutions:**\n"
        for i, solution in enumerate(entry.solutions, 1):
            text += f"{i}. {solution}\n"
        text += "\n"

        if entry.hardware_specific:
            text += "**Hardware-Specific Solutions:**\n\n"
            for hardware, solutions in entry.hardware_specific.items():
                text += f"*{hardware}:*\n"
                for solution in solutions:
                    text += f"- {solution}\n"
                text += "\n"

        text += "---\n\n"
        return text

    def _generate_faq(self) -> str:
        """Generate FAQ section."""
        return """## Frequently Asked Questions

### General Questions

**Q: How long does a complete parameter estimation session take?**  
A: Approximately 45-60 minutes including setup, calibration, and all three tasks.

**Q: Can I run tasks in a different order?**  
A: Yes, but the recommended order (Detection → Heartbeat → Oddball) minimizes fatigue effects.

**Q: What if a participant needs to stop mid-session?**  
A: Use the pause function to save state. Resume within 30 minutes if possible.

**Q: How many participants can I test per day?**  
A: Typically 4-6 with proper setup time and equipment cleaning between sessions.

### Technical Questions

**Q: What sampling rates are required?**  
A: Minimum: EEG 1000 Hz, Eye tracker 250 Hz, Cardiac 250 Hz. Recommended: EEG 1000 Hz, Eye tracker 1000 Hz, Cardiac 1000 Hz.

**Q: Can I use fewer EEG channels?**  
A: Yes, but 64+ channels recommended. Minimum 32 channels with Pz, Fz, Cz coverage.

**Q: Do I need all three hardware types?**  
A: EEG is required. Eye tracker and cardiac sensor are highly recommended but optional for some analyses.

**Q: Can I run without LSL?**  
A: Not recommended. LSL ensures proper synchronization across data streams.

### Data Questions

**Q: Where is data stored?**  
A: Default: `~/.apgi_framework/data/`. Configurable in settings.

**Q: What format is the output?**  
A: Parameter estimates in JSON/HDF5. Raw data in BIDS-compliant format.

**Q: How do I export data for analysis?**  
A: Use the export function in GUI or `DataExporter` class in Python.

**Q: What if parameter estimates seem unrealistic?**  
A: Check data quality metrics, review task performance, consider retest.

### Troubleshooting Questions

**Q: What if I can't resolve an issue?**  
A: Contact support@apgi-framework.org with:
- Error logs
- System information
- Hardware configuration
- Steps to reproduce

**Q: How do I report a bug?**  
A: Use GitHub issues: https://github.com/apgi-framework/issues

**Q: Where can I find updates?**  
A: Check https://apgi-framework.org/updates

---

"""

    def _generate_contact_info(self) -> str:
        """Generate contact information."""
        return """## Support Contact

**Technical Support:**  
Email: support@apgi-framework.org  
Response time: 24-48 hours

**Bug Reports:**  
GitHub: https://github.com/apgi-framework/issues

**Documentation:**  
Website: https://apgi-framework.org/docs

**Community Forum:**  
Forum: https://forum.apgi-framework.org

**Emergency Contact:**  
For critical issues during data collection, call: +1-555-0123-4567

---

**Last Updated:** {datetime.now().strftime('%B %d, %Y')}

"""

    def generate_quick_reference(self) -> Path:
        """Generate quick reference card."""
        ref_path = self.output_dir / "Quick_Reference.md"

        with open(ref_path, "w", encoding="utf-8") as f:
            f.write("""# Quick Reference Card

## Common Issues & Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| High EEG impedance | Add more gel, check contact |
| Eye tracker calibration fails | Clean lens, adjust distance |
| No cardiac signal | Check electrode placement |
| Task won't start | Verify hardware connections |
| Excessive artifacts | Participant relaxation, check environment |
| Wide parameter CIs | Review data quality, consider retest |

## Emergency Procedures

1. **Participant discomfort:** Stop immediately, remove equipment
2. **Equipment malfunction:** Save data, document issue, contact support
3. **Data loss:** Check automatic backups in `~/.apgi_framework/backups/`

## Quick Commands

```bash
# Check system status
python -m apgi_framework.deployment.system_check

# Validate installation
python -m apgi_framework.deployment.deployment_validator

# Test hardware
python -m apgi_framework.deployment.hardware_test

# View logs
tail -f ~/.apgi_framework/logs/apgi_framework.log
```

## Support Contacts

- Technical: support@apgi-framework.org
- Emergency: +1-555-0123-4567
- Documentation: https://apgi-framework.org/docs
""")

        self.logger.info(f"Quick reference generated: {ref_path}")
        return ref_path
