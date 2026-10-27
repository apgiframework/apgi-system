"""Parameter Estimation User Manual generator.

Generates comprehensive user manual with detailed protocols for all three tasks
(detection, heartbeat detection, dual-modality oddball).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


class ParameterEstimationUserManual:
    """Generates comprehensive user manual for APGI Framework parameter estimation.

    Includes detailed protocols, setup instructions, and best practices
    for all three parameter estimation tasks.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize user manual generator.

        Args:
            output_dir: Directory for saving manual files.
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir or Path.home() / ".apgi_framework" / "documentation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_complete_manual(self) -> Path:
        """Generate complete user manual.

        Returns:
            Path to generated manual file.
        """
        self.logger.info("Generating complete user manual...")

        manual_path = self.output_dir / "APGI_Parameter_Estimation_User_Manual.md"

        with open(manual_path, "w", encoding="utf-8") as f:
            f.write(self._generate_title_page())
            f.write(self._generate_table_of_contents())
            f.write(self._generate_introduction())
            f.write(self._generate_system_overview())
            f.write(self._generate_hardware_setup())
            f.write(self._generate_software_installation())
            f.write(self._generate_task_protocols())
            f.write(self._generate_data_quality())
            f.write(self._generate_parameter_interpretation())
            f.write(self._generate_best_practices())
            f.write(self._generate_appendices())

        self.logger.info(f"User manual generated: {manual_path}")
        return manual_path

    def _generate_title_page(self) -> str:
        """Generate title page."""
        return f"""# APGI Framework Parameter Estimation System
## User Manual

**Version:** 1.0  
**Date:** {datetime.now().strftime('%B %d, %Y')}  
**Authors:** APGI Framework Development Team

---

"""

    def _generate_table_of_contents(self) -> str:
        """Generate table of contents."""
        return """## Table of Contents

1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Hardware Setup](#hardware-setup)
4. [Software Installation](#software-installation)
5. [Task Protocols](#task-protocols)
   - 5.1 [Detection Task (θ₀ Estimation)](#detection-task)
   - 5.2 [Heartbeat Detection Task (Πᵢ Estimation)](#heartbeat-detection-task)
   - 5.3 [Dual-Modality Oddball Task (β Estimation)](#dual-modality-oddball-task)
6. [Data Quality Monitoring](#data-quality-monitoring)
7. [Parameter Interpretation](#parameter-interpretation)
8. [Best Practices](#best-practices)
9. [Appendices](#appendices)

---

"""

    def _generate_introduction(self) -> str:
        """Generate introduction section."""
        return """## 1. Introduction

The APGI Framework Parameter Estimation System is a comprehensive neuroscience research platform designed to rapidly quantify three key computational parameters within a standardized 45-60 minute protocol:

- **θ₀ (Baseline Ignition Threshold)**: Individual differences in conscious perception thresholds
- **Πᵢ (Interoceptive Precision)**: Sensitivity to internal bodily signals
- **β (Somatic Bias)**: Relative weighting of interoceptive versus exteroceptive information

### Purpose

This system combines behavioral tasks, high-density EEG analysis, and pupillometry to extract individualized parameters through hierarchical Bayesian modeling. The parameters have demonstrated clinical utility in predicting anxiety, somatic symptoms, and attentional control.

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores, 2.0 GHz
- RAM: 8 GB
- Storage: 10 GB free space
- Python 3.8+
- Operating System: Windows 10, macOS 10.14+, or Linux

**Recommended Requirements:**
- CPU: 8+ cores, 3.0 GHz
- RAM: 16 GB
- Storage: 50 GB free space
- Python 3.9+
- Dedicated GPU (optional, for accelerated processing)

---

"""

    def _generate_system_overview(self) -> str:
        """Generate system overview section."""
        return """## 2. System Overview

### Architecture

The APGI Framework consists of several integrated components:

1. **Task Control Layer**: Manages stimulus presentation and response collection
2. **Data Acquisition Layer**: Interfaces with EEG, eye tracker, and cardiac sensors
3. **Signal Processing Layer**: Real-time processing of multi-modal data
4. **Statistical Modeling Layer**: Hierarchical Bayesian parameter estimation
5. **User Interface Layer**: Experiment control and monitoring

### Workflow

```
Participant Setup → Hardware Calibration → Task Execution → 
Real-time Processing → Parameter Estimation → Report Generation
```

### Data Flow

- **Input**: Behavioral responses, EEG (128+ channels), pupillometry (1000 Hz), cardiac (ECG/PPG)
- **Processing**: Real-time artifact detection, feature extraction, quality monitoring
- **Output**: Individual parameter estimates (θ₀, Πᵢ, β) with 95% credible intervals

---

"""

    def _generate_hardware_setup(self) -> str:
        """Generate hardware setup section."""
        return """## 3. Hardware Setup

### Required Hardware

1. **EEG System** (128+ channels recommended)
   - Supported: BioSemi ActiveTwo, EGI HydroCel, Brain Products actiCHamp
   - Sampling rate: 1000 Hz minimum
   - Impedance: < 10 kΩ

2. **Eye Tracker** (1000 Hz recommended)
   - Supported: Tobii Pro Spectrum, EyeLink 1000 Plus, Pupil Labs Core
   - Tracking mode: Binocular preferred
   - Accuracy: < 1° visual angle

3. **Cardiac Sensor**
   - ECG (preferred) or PPG
   - Sampling rate: 250 Hz minimum
   - R-peak detection capability

### Setup Procedure

#### EEG Setup

1. **Electrode Preparation**
   - Clean electrode sites with alcohol prep pads
   - Apply conductive gel to electrodes
   - Ensure even gel distribution

2. **Cap Placement**
   - Measure head circumference
   - Position cap with Cz at vertex
   - Align Fpz and Oz on midline
   - Secure chin strap

3. **Impedance Check**
   - Target: < 10 kΩ for all channels
   - Re-apply gel if impedance high
   - Document problematic channels

#### Eye Tracker Setup

1. **Positioning**
   - Place tracker 60-70 cm from participant
   - Align at eye level
   - Ensure stable mounting

2. **Calibration**
   - Run 9-point calibration
   - Validation accuracy: < 1° target
   - Recalibrate if needed

3. **Verification**
   - Check pupil detection quality
   - Verify gaze tracking accuracy
   - Test blink detection

#### Cardiac Sensor Setup

1. **Electrode Placement** (ECG)
   - Lead II configuration recommended
   - Clean skin with alcohol
   - Apply electrodes firmly
   - Check signal quality

2. **Alternative: PPG**
   - Finger or earlobe placement
   - Ensure good contact
   - Minimize movement artifacts

### Lab Streaming Layer (LSL) Configuration

All devices should stream data via LSL for synchronization:

```python
# Example LSL configuration
eeg_stream = "BioSemi"
eyetracker_stream = "TobiiSpectrum"
cardiac_stream = "ECG"
```

---

"""

    def _generate_software_installation(self) -> str:
        """Generate software installation section."""
        return """## 4. Software Installation

### Installation Steps

1. **Install Python**
   ```bash
   # Download Python 3.9+ from python.org
   # Verify installation
   python --version
   ```

2. **Install APGI Framework**
   ```bash
   # Clone repository
   git clone https://github.com/apgi-framework/parameter-estimation.git
   cd parameter-estimation
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install framework
   pip install -e .
   ```

3. **Verify Installation**
   ```bash
   # Run system validation
   python -m apgi_framework.deployment.deployment_validator
   ```

4. **Configure Hardware**
   ```python
   from apgi_framework.deployment import HardwareConfigurationManager
   
   hw_manager = HardwareConfigurationManager()
   hw_manager.configure_eeg_system('biosemi_128')
   hw_manager.configure_eye_tracker('tobii_spectrum')
   hw_manager.configure_cardiac_sensor('biosemi_ecg')
   hw_manager.save_configuration('my_lab_config.json')
   ```

### Troubleshooting Installation

See [Troubleshooting Guide](#troubleshooting-guide) for common installation issues.

---

"""

    def _generate_task_protocols(self) -> str:
        """Generate task protocols section."""
        return """## 5. Task Protocols

### 5.1 Detection Task (θ₀ Estimation)

**Duration:** ~10 minutes  
**Trials:** 200  
**Purpose:** Estimate baseline ignition threshold

#### Protocol

1. **Participant Instructions**
   ```
   "You will see brief flashes or hear brief tones. Some will be very faint.
   Press the SPACE bar as quickly as possible when you detect a stimulus.
   If you're not sure, make your best guess."
   ```

2. **Task Parameters**
   - Stimulus type: Gabor patches (visual) or pure tones (auditory)
   - Duration: 50 ms
   - ISI: 1000-1500 ms (jittered)
   - Adaptive staircase: QUEST+ algorithm

3. **Data Collection**
   - Behavioral: Detection responses, reaction times
   - EEG: Continuous recording, P3b extraction (250-500 ms at Pz)
   - Quality monitoring: Real-time impedance, artifact detection

4. **Quality Criteria**
   - Response rate: 40-60% (indicates proper threshold)
   - Reaction time: 200-800 ms
   - EEG artifact rate: < 30%

#### Running the Task

```python
from apgi_framework.tasks import DetectionTask

task = DetectionTask(
    participant_id='P001',
    modality='visual',
    n_trials=200
)

results = task.run()
```

### 5.2 Heartbeat Detection Task (Πᵢ Estimation)

**Duration:** ~8 minutes  
**Trials:** 60  
**Purpose:** Estimate interoceptive precision

#### Protocol

1. **Participant Instructions**
   ```
   "You will hear brief tones. Some tones will be synchronized with your
   heartbeat, others will not. After each tone, indicate whether it was
   synchronized (press 'S') or not (press 'N'), and rate your confidence
   from 1 (guessing) to 4 (certain)."
   ```

2. **Task Parameters**
   - Tone duration: 100 ms
   - Synchronous: Presented at R-peak + 200 ms
   - Asynchronous: R-peak + 400-600 ms
   - Confidence scale: 1-4

3. **Data Collection**
   - Behavioral: Synchrony judgments, confidence ratings
   - EEG: HEP extraction (250-400 ms post R-peak)
   - Pupillometry: Dilation response (200-500 ms post-tone)
   - Cardiac: Continuous ECG/PPG

4. **Quality Criteria**
   - Heartbeat detection d': 0.3-2.0
   - Cardiac signal quality: > 90% valid R-peaks
   - Pupil data loss: < 20%

#### Running the Task

```python
from apgi_framework.tasks import HeartbeatDetectionTask

task = HeartbeatDetectionTask(
    participant_id='P001',
    n_trials=60
)

results = task.run()
```

### 5.3 Dual-Modality Oddball Task (β Estimation)

**Duration:** ~12 minutes  
**Trials:** 120  
**Purpose:** Estimate somatic bias

#### Protocol

1. **Participant Instructions**
   ```
   "You will experience a series of stimuli. Most will be standard, but
   occasionally you'll experience an oddball stimulus. Press SPACE whenever
   you detect an oddball. There are two types of oddballs - internal
   sensations and external stimuli."
   ```

2. **Task Parameters**
   - Standard trials: 80% (visual gratings, no response)
   - Interoceptive deviants: 10% (CO₂ puffs or heartbeat flashes)
   - Exteroceptive deviants: 10% (rare Gabor orientations)
   - ISI: 2000-3000 ms

3. **Stimulus Calibration**
   - Run separate staircases to match Πₑ ≈ Πᵢ
   - Ensure equal detectability across modalities

4. **Data Collection**
   - Behavioral: Oddball detection responses
   - EEG: P3b to interoceptive and exteroceptive deviants
   - Safety: CO₂ concentration monitoring

5. **Quality Criteria**
   - Detection accuracy: 60-90% for both deviant types
   - P3b amplitude: > 5 μV for conscious deviants
   - EEG artifact rate: < 30%

#### Running the Task

```python
from apgi_framework.tasks import DualModalityOddballTask

task = DualModalityOddballTask(
    participant_id='P001',
    n_trials=120
)

# Calibrate stimuli first
task.calibrate_stimuli()

# Run main task
results = task.run()
```

---

"""

    def _generate_data_quality(self) -> str:
        """Generate data quality monitoring section."""
        return """## 6. Data Quality Monitoring

### Real-Time Quality Metrics

The system provides continuous quality monitoring:

1. **EEG Quality**
   - Impedance levels (< 10 kΩ target)
   - Artifact detection rate
   - Signal-to-noise ratio
   - Channel dropout detection

2. **Pupillometry Quality**
   - Pupil detection rate (> 80% target)
   - Blink frequency
   - Tracking loss events
   - Calibration drift

3. **Cardiac Quality**
   - R-peak detection confidence
   - Heart rate variability
   - Ectopic beat detection
   - Signal amplitude

### Quality Alerts

The system generates alerts for:
- High impedance (> 15 kΩ)
- Excessive artifacts (> 40%)
- Tracking loss (> 30%)
- Cardiac signal issues

### Corrective Actions

**High Impedance:**
1. Pause task
2. Re-apply gel to affected electrodes
3. Check electrode contact
4. Resume when impedance < 10 kΩ

**Excessive Artifacts:**
1. Check participant comfort
2. Reduce muscle tension
3. Minimize movement
4. Consider break

**Tracking Loss:**
1. Recalibrate eye tracker
2. Adjust participant position
3. Check lighting conditions
4. Clean tracker lens

---

"""

    def _generate_parameter_interpretation(self) -> str:
        """Generate parameter interpretation section."""
        return """## 7. Parameter Interpretation

### θ₀ (Baseline Ignition Threshold)

**Range:** Typically 2.5 - 4.5  
**Interpretation:**
- **Low θ₀ (< 3.0)**: High sensitivity to stimuli, lower threshold for conscious access
- **High θ₀ (> 4.0)**: Reduced sensitivity, higher threshold for conscious access

**Clinical Relevance:**
- Predicts attentional lapses
- Associated with vigilance performance
- Related to arousal regulation

### Πᵢ (Interoceptive Precision)

**Range:** Typically 1.0 - 2.5  
**Interpretation:**
- **Low Πᵢ (< 1.5)**: Poor interoceptive awareness, low confidence in bodily signals
- **High Πᵢ (> 2.0)**: Strong interoceptive awareness, high confidence in bodily signals

**Clinical Relevance:**
- Predicts anxiety sensitivity
- Associated with panic disorder
- Related to emotional regulation

### β (Somatic Bias)

**Range:** Typically 0.8 - 1.8  
**Interpretation:**
- **Low β (< 1.0)**: Exteroceptive bias, prioritizes external information
- **High β (> 1.5)**: Interoceptive bias, prioritizes internal sensations

**Clinical Relevance:**
- Predicts somatic symptom severity
- Associated with health anxiety
- Related to body vigilance

### Credible Intervals

All parameters include 95% credible intervals:
- **Narrow CI (< 0.3)**: High confidence in estimate
- **Wide CI (> 0.5)**: Lower confidence, consider retest

---

"""

    def _generate_best_practices(self) -> str:
        """Generate best practices section."""
        return """## 8. Best Practices

### Participant Preparation

1. **Pre-Session**
   - Explain study procedures clearly
   - Obtain informed consent
   - Screen for exclusion criteria
   - Avoid caffeine 2 hours before

2. **During Session**
   - Ensure comfortable seating
   - Minimize distractions
   - Provide breaks between tasks
   - Monitor participant fatigue

3. **Post-Session**
   - Debrief participant
   - Clean equipment thoroughly
   - Back up data immediately
   - Document any issues

### Data Collection

1. **Environment**
   - Quiet, temperature-controlled room
   - Consistent lighting (avoid glare)
   - Minimize electromagnetic interference
   - Stable furniture (no wobbling)

2. **Timing**
   - Same time of day for test-retest
   - Avoid immediately after meals
   - Consider circadian effects
   - Document time of day

3. **Quality Control**
   - Check all equipment before session
   - Run test trials first
   - Monitor quality metrics continuously
   - Document all issues

### Data Management

1. **Organization**
   - Use consistent naming conventions
   - Separate raw and processed data
   - Maintain detailed logs
   - Version control for analysis scripts

2. **Backup**
   - Immediate backup after session
   - Multiple backup locations
   - Verify backup integrity
   - Regular backup testing

3. **Privacy**
   - De-identify data promptly
   - Secure storage (encrypted)
   - Access controls
   - Compliance with regulations

---

"""

    def _generate_appendices(self) -> str:
        """Generate appendices section."""
        return """## 9. Appendices

### Appendix A: Hardware Specifications

**Supported EEG Systems:**
- BioSemi ActiveTwo (128/256 channels)
- EGI HydroCel (128/256 channels)
- Brain Products actiCHamp (64/128 channels)
- ANT Neuro eego (64/128 channels)

**Supported Eye Trackers:**
- Tobii Pro Spectrum (1200 Hz)
- EyeLink 1000 Plus (1000 Hz)
- Pupil Labs Core (200 Hz)
- SR Research EyeLink Portable Duo

**Supported Cardiac Sensors:**
- BioSemi ECG
- Polar H10 PPG
- Brain Products ExG
- Generic ECG/PPG via LSL

### Appendix B: File Formats

**Data Storage:**
- Raw EEG: `.bdf`, `.edf`, `.fif`
- Processed EEG: `.fif`, `.set`
- Behavioral: `.csv`, `.json`
- Parameters: `.json`, `.hdf5`

**Export Formats:**
- BIDS-compliant structure
- CSV for statistical analysis
- HDF5 for large datasets
- JSON for metadata

### Appendix C: Glossary

- **θ₀**: Baseline ignition threshold
- **Πᵢ**: Interoceptive precision
- **β**: Somatic bias
- **P3b**: Event-related potential component (250-500 ms)
- **HEP**: Heartbeat-evoked potential
- **LSL**: Lab Streaming Layer
- **ICC**: Intraclass correlation coefficient
- **QUEST+**: Adaptive staircase algorithm

### Appendix D: References

1. Seth, A. K., & Friston, K. J. (2016). Active interoceptive inference and the emotional brain. *Philosophical Transactions of the Royal Society B*, 371(1708), 20160007.

2. Dehaene, S., & Changeux, J. P. (2011). Experimental and theoretical approaches to conscious processing. *Neuron*, 70(2), 200-227.

3. Garfinkel, S. N., et al. (2015). Knowing your own heart: Distinguishing interoceptive accuracy from interoceptive awareness. *Biological Psychology*, 104, 65-74.

---

**End of User Manual**

For technical support, contact: support@apgi-framework.org  
For updates and documentation: https://apgi-framework.org/docs

"""

    def generate_quick_start_guide(self) -> Path:
        """Generate quick start guide."""
        guide_path = self.output_dir / "Quick_Start_Guide.md"

        with open(guide_path, "w", encoding="utf-8") as f:
            f.write("""# APGI Framework Quick Start Guide

## 5-Minute Setup

### 1. Install Software
```bash
pip install apgi-framework
```

### 2. Configure Hardware
```python
from apgi_framework.deployment import HardwareConfigurationManager

hw = HardwareConfigurationManager()
hw.configure_eeg_system('biosemi_128')
hw.configure_eye_tracker('tobii_spectrum')
```

### 3. Run Validation
```python
from apgi_framework.deployment import DeploymentValidator

validator = DeploymentValidator()
report = validator.validate_deployment()
print(report.generate_summary())
```

### 4. Start First Session
```python
from apgi_framework.gui import ParameterEstimationGUI

gui = ParameterEstimationGUI()
gui.start_session(participant_id='P001')
```

## Task Sequence

1. **Detection Task** (10 min) → θ₀
2. **Heartbeat Detection** (8 min) → Πᵢ
3. **Dual-Modality Oddball** (12 min) → β

Total: ~30 minutes + setup

## Need Help?

- Full Manual: `APGI_Parameter_Estimation_User_Manual.md`
- Troubleshooting: `Troubleshooting_Guide.md`
- Support: support@apgi-framework.org
""")

        self.logger.info(f"Quick start guide generated: {guide_path}")
        return guide_path
