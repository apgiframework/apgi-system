"""Pharmacological Simulator for APGI Framework Testing.

This module implements the PharmacologicalSimulator class that models drug effects
on ignition threshold modulation and physiological control measures for testing
neuromodulatory falsification criteria.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..exceptions import ValidationError


class DrugClass(Enum):
    """Classification of drug types by mechanism."""

    BETA_BLOCKER = "beta_blocker"
    DOPAMINE_PRECURSOR = "dopamine_precursor"
    SEROTONIN_REUPTAKE_INHIBITOR = "serotonin_reuptake_inhibitor"
    CHOLINESTERASE_INHIBITOR = "cholinesterase_inhibitor"
    PLACEBO = "placebo"


class NeuromodulatorSystem(Enum):
    """Neuromodulatory systems affected by drugs."""

    NOREPINEPHRINE = "norepinephrine"
    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"
    ACETYLCHOLINE = "acetylcholine"


@dataclass
class DrugProfile:
    """Pharmacological profile of a drug."""

    name: str
    drug_class: DrugClass
    target_system: NeuromodulatorSystem
    half_life: float  # hours
    peak_effect_time: float  # hours
    bioavailability: float  # 0.0-1.0
    dose_response_curve: str  # "linear", "logarithmic", "sigmoid"

    # Expected threshold modulation
    threshold_effect_magnitude: float  # Expected Δθₜ at standard dose
    threshold_effect_direction: int  # +1 for increase, -1 for decrease

    # Physiological effects for validation
    physiological_effects: Dict[str, Tuple[float, float]]  # effect_name: (min_change, max_change)


@dataclass
class DrugAdministration:
    """Drug administration parameters."""

    drug_profile: DrugProfile
    dosage: float  # mg or standardized units
    administration_time: datetime
    route: str  # "oral", "iv", "im"
    participant_weight: float  # kg for dose adjustment

    # Individual factors
    metabolism_rate: float  # 0.5-2.0 (normal = 1.0)
    sensitivity_factor: float  # 0.5-2.0 (normal = 1.0)


@dataclass
class PharmacologicalResponse:
    """Complete pharmacological response simulation."""

    drug_administration: DrugAdministration
    time_course: Dict[float, float]  # time_hours: plasma_concentration

    # Threshold modulation over time
    threshold_modulation: Dict[float, float]  # time_hours: threshold_change
    peak_threshold_effect: float
    peak_effect_time: float

    # Physiological control measures
    physiological_responses: Dict[str, Dict[float, float]]  # measure: {time: value}

    # Validation measures
    drug_level_confirmed: bool
    physiological_effects_confirmed: bool
    expected_vs_observed_correlation: float


class PharmacologicalSimulator:
    """Simulator for drug effects on ignition threshold and physiological measures.

    Models pharmacokinetics, pharmacodynamics, and individual variability
    for testing neuromodulatory falsification criteria.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize pharmacological simulator with drug profiles."""
        self.drug_profiles = self._initialize_drug_profiles()

        # Simulation parameters
        self.simulation_duration = 8.0  # hours
        self.time_resolution = 0.25  # hours (15 minutes)

        # Individual variability parameters
        self.default_metabolism_range = (0.7, 1.3)
        self.default_sensitivity_range = (0.8, 1.2)

        # Seeded RNG for reproducibility
        self.rng = np.random.RandomState(random_seed if random_seed is not None else 42)

    def _initialize_drug_profiles(self) -> Dict[str, DrugProfile]:
        """Initialize standard drug profiles for testing."""
        profiles = {}

        # Propranolol (beta-blocker)
        profiles["propranolol"] = DrugProfile(
            name="propranolol",
            drug_class=DrugClass.BETA_BLOCKER,
            target_system=NeuromodulatorSystem.NOREPINEPHRINE,
            half_life=4.0,  # hours
            peak_effect_time=1.5,  # hours
            bioavailability=0.25,  # Low oral bioavailability
            dose_response_curve="logarithmic",
            threshold_effect_magnitude=0.5,  # Increase threshold
            threshold_effect_direction=1,
            physiological_effects={
                "heart_rate": (-25, -10),  # Decreased HR
                "blood_pressure_systolic": (-15, -5),  # Decreased BP
                "pupil_light_response": (-0.3, -0.1),  # Reduced response
                "exercise_tolerance": (-0.2, -0.1),  # Reduced tolerance
            },
        )

        # L-DOPA (dopamine precursor)
        profiles["l_dopa"] = DrugProfile(
            name="l_dopa",
            drug_class=DrugClass.DOPAMINE_PRECURSOR,
            target_system=NeuromodulatorSystem.DOPAMINE,
            half_life=1.5,  # hours (short half-life)
            peak_effect_time=1.0,  # hours
            bioavailability=0.30,
            dose_response_curve="sigmoid",
            threshold_effect_magnitude=0.3,  # Decrease threshold
            threshold_effect_direction=-1,
            physiological_effects={
                "motor_response_time": (-0.15, -0.05),  # Faster responses
                "attention_score": (0.05, 0.20),  # Enhanced attention
                "pupil_dilation": (0.10, 0.25),  # Increased dilation
                "dyskinesia_score": (0.0, 0.1),  # Potential side effect
            },
        )

        # Sertraline (SSRI)
        profiles["sertraline"] = DrugProfile(
            name="sertraline",
            drug_class=DrugClass.SEROTONIN_REUPTAKE_INHIBITOR,
            target_system=NeuromodulatorSystem.SEROTONIN,
            half_life=26.0,  # hours (long half-life)
            peak_effect_time=6.0,  # hours (delayed effect)
            bioavailability=0.44,
            dose_response_curve="linear",
            threshold_effect_magnitude=0.2,  # Decrease threshold
            threshold_effect_direction=-1,
            physiological_effects={
                "mood_rating": (0.02, 0.15),  # Improved mood
                "anxiety_score": (-0.15, -0.02),  # Reduced anxiety
                "sleep_latency": (-0.1, 0.1),  # Variable sleep effect
                "gastrointestinal_score": (0.0, 0.2),  # Potential side effect
            },
        )

        # Physostigmine (cholinesterase inhibitor)
        profiles["physostigmine"] = DrugProfile(
            name="physostigmine",
            drug_class=DrugClass.CHOLINESTERASE_INHIBITOR,
            target_system=NeuromodulatorSystem.ACETYLCHOLINE,
            half_life=0.5,  # hours (very short)
            peak_effect_time=0.25,  # hours (15 minutes)
            bioavailability=0.12,  # Low bioavailability
            dose_response_curve="linear",
            threshold_effect_magnitude=0.4,  # Decrease threshold
            threshold_effect_direction=-1,
            physiological_effects={
                "memory_score": (0.10, 0.30),  # Enhanced memory
                "attention_score": (0.15, 0.35),  # Enhanced attention
                "salivation": (0.20, 0.50),  # Increased salivation
                "muscle_fasciculation": (0.0, 0.15),  # Potential side effect
            },
        )

        # Placebo
        profiles["placebo"] = DrugProfile(
            name="placebo",
            drug_class=DrugClass.PLACEBO,
            target_system=NeuromodulatorSystem.NOREPINEPHRINE,  # Dummy assignment
            half_life=0.0,
            peak_effect_time=0.0,
            bioavailability=0.0,
            dose_response_curve="linear",
            threshold_effect_magnitude=0.0,
            threshold_effect_direction=0,
            physiological_effects={"placebo_response": (-0.05, 0.05)},  # Minimal placebo effects
        )

        return profiles

    def simulate_drug_administration(
        self,
        drug_name: str,
        dosage: float,
        participant_weight: float = 70.0,
        route: str = "oral",
        metabolism_rate: Optional[float] = None,
        sensitivity_factor: Optional[float] = None,
    ) -> PharmacologicalResponse:
        """Simulate complete pharmacological response to drug administration.

        Args:
            drug_name: Name of drug to simulate
            dosage: Dose in mg
            participant_weight: Participant weight in kg
            route: Administration route
            metabolism_rate: Individual metabolism rate (1.0 = normal)
            sensitivity_factor: Individual sensitivity (1.0 = normal)

        Returns:
            Complete pharmacological response simulation
        """
        if drug_name not in self.drug_profiles:
            raise ValidationError(f"Unknown drug: {drug_name}")

        drug_profile = self.drug_profiles[drug_name]

        # Set individual factors if not provided
        if metabolism_rate is None:
            metabolism_rate = self.rng.uniform(*self.default_metabolism_range)
        if sensitivity_factor is None:
            sensitivity_factor = self.rng.uniform(*self.default_sensitivity_range)

        # Create drug administration
        administration = DrugAdministration(
            drug_profile=drug_profile,
            dosage=dosage,
            administration_time=datetime.now(),
            route=route,
            participant_weight=participant_weight,
            metabolism_rate=metabolism_rate,
            sensitivity_factor=sensitivity_factor,
        )

        # Simulate pharmacokinetics (plasma concentration over time)
        time_course = self._simulate_pharmacokinetics(administration)

        # Simulate threshold modulation
        threshold_modulation = self._simulate_threshold_modulation(administration, time_course)

        # Simulate physiological responses
        physiological_responses = self._simulate_physiological_responses(
            administration, time_course
        )

        # Calculate peak effects
        peak_threshold_effect = max(abs(effect) for effect in threshold_modulation.values())
        peak_effect_time = max(
            threshold_modulation.keys(),
            key=lambda t: abs(threshold_modulation[t]),
        )

        # Validate drug effects
        drug_level_confirmed = self._confirm_drug_levels(time_course)
        physiological_effects_confirmed = self._confirm_physiological_effects(
            physiological_responses, drug_profile
        )

        # Calculate correlation between expected and observed effects
        expected_vs_observed_correlation = self._calculate_effect_correlation(
            administration, threshold_modulation, physiological_responses
        )

        return PharmacologicalResponse(
            drug_administration=administration,
            time_course=time_course,
            threshold_modulation=threshold_modulation,
            peak_threshold_effect=peak_threshold_effect,
            peak_effect_time=peak_effect_time,
            physiological_responses=physiological_responses,
            drug_level_confirmed=drug_level_confirmed,
            physiological_effects_confirmed=physiological_effects_confirmed,
            expected_vs_observed_correlation=expected_vs_observed_correlation,
        )

    def _simulate_pharmacokinetics(self, administration: DrugAdministration) -> Dict[float, float]:
        """Simulate plasma concentration over time."""
        drug_profile = administration.drug_profile

        # Adjust dose for weight and bioavailability
        effective_dose = (
            administration.dosage
            * drug_profile.bioavailability
            * (administration.participant_weight / 70.0)
        )

        # Adjust for individual metabolism
        adjusted_half_life = drug_profile.half_life / administration.metabolism_rate

        time_points = np.arange(0, self.simulation_duration, self.time_resolution)
        concentrations: Dict[float, float] = {}

        for t in time_points:
            time_point = float(t)  # Convert numpy scalar to float for dict key
            if drug_profile.name == "placebo":
                concentration = 0.0
            else:
                # First-order absorption and elimination
                ka = 2.0 / drug_profile.peak_effect_time  # Absorption rate constant
                ke = 0.693 / adjusted_half_life  # Elimination rate constant

                if time_point == 0:
                    concentration = 0.0
                else:
                    # One-compartment model with first-order absorption
                    concentration = (
                        effective_dose
                        * ka
                        / (ka - ke)
                        * (np.exp(-ke * time_point) - np.exp(-ka * time_point))
                    )

            concentrations[time_point] = float(max(0.0, concentration))

        return concentrations

    def _simulate_threshold_modulation(
        self,
        administration: DrugAdministration,
        time_course: Dict[float, float],
    ) -> Dict[float, float]:
        """Simulate ignition threshold modulation over time."""
        drug_profile = administration.drug_profile
        threshold_changes = {}

        for time_point, concentration in time_course.items():
            if drug_profile.name == "placebo":
                # Placebo may have small random effects
                threshold_change = self.rng.normal(0, 0.05)
            else:
                # Calculate threshold change based on concentration and dose-response
                normalized_concentration = (
                    concentration / max(time_course.values())
                    if max(time_course.values()) > 0
                    else 0
                )

                if drug_profile.dose_response_curve == "linear":
                    effect_magnitude = normalized_concentration
                elif drug_profile.dose_response_curve == "logarithmic":
                    effect_magnitude = np.log(1 + normalized_concentration) / np.log(2)
                elif drug_profile.dose_response_curve == "sigmoid":
                    effect_magnitude = normalized_concentration / (0.5 + normalized_concentration)
                else:
                    effect_magnitude = normalized_concentration

                # Apply individual sensitivity
                effect_magnitude *= administration.sensitivity_factor

                # Calculate threshold change
                threshold_change = (
                    drug_profile.threshold_effect_direction
                    * drug_profile.threshold_effect_magnitude
                    * effect_magnitude
                )

                # Add noise
                noise = (
                    self.rng.normal(0, 0.1 * abs(threshold_change)) if threshold_change != 0 else 0
                )
                threshold_change += noise

            threshold_changes[time_point] = threshold_change

        return threshold_changes

    def _simulate_physiological_responses(
        self,
        administration: DrugAdministration,
        time_course: Dict[float, float],
    ) -> Dict[str, Dict[float, float]]:
        """Simulate physiological control measures over time."""
        drug_profile = administration.drug_profile
        responses = {}

        for effect_name, (
            min_change,
            max_change,
        ) in drug_profile.physiological_effects.items():
            effect_time_course = {}

            for time_point, concentration in time_course.items():
                if drug_profile.name == "placebo":
                    # Placebo effects are minimal and random
                    effect = self.rng.normal(0, 0.02)
                else:
                    # Calculate effect based on concentration
                    normalized_concentration = (
                        concentration / max(time_course.values())
                        if max(time_course.values()) > 0
                        else 0
                    )

                    # Scale effect between min and max
                    effect_magnitude = (
                        min_change + (max_change - min_change) * normalized_concentration
                    )

                    # Apply individual sensitivity
                    effect_magnitude *= administration.sensitivity_factor

                    # Add noise
                    noise_std = 0.1 * abs(effect_magnitude) if effect_magnitude != 0 else 0.01
                    effect = effect_magnitude + self.rng.normal(0, noise_std)

                effect_time_course[time_point] = effect

            responses[effect_name] = effect_time_course

        return responses

    def _confirm_drug_levels(self, time_course: Dict[float, float]) -> bool:
        """Confirm that drug reached detectable levels."""
        if not time_course:  # Handle empty time course
            return False

        max_concentration = max(time_course.values())

        # Drug is confirmed if peak concentration exceeds detection threshold
        detection_threshold = 0.1  # Arbitrary units
        return max_concentration > detection_threshold

    def _confirm_physiological_effects(
        self,
        physiological_responses: Dict[str, Dict[float, float]],
        drug_profile: DrugProfile,
    ) -> bool:
        """Confirm that physiological effects match expected patterns."""
        confirmed_effects = 0
        total_effects = len(drug_profile.physiological_effects)

        if total_effects == 0:
            return True

        for (
            effect_name,
            expected_range,
        ) in drug_profile.physiological_effects.items():
            if effect_name in physiological_responses:
                effect_values = list(physiological_responses[effect_name].values())
                max_effect = max(abs(v) for v in effect_values)
                expected_max = max(abs(expected_range[0]), abs(expected_range[1]))

                # Effect is confirmed if it reaches at least 50% of expected magnitude
                if max_effect > 0.5 * expected_max:
                    confirmed_effects += 1

        # Require at least 50% of effects to be confirmed
        return confirmed_effects / total_effects >= 0.5

    def _calculate_effect_correlation(
        self,
        administration: DrugAdministration,
        threshold_modulation: Dict[float, float],
        physiological_responses: Dict[str, Dict[float, float]],
    ) -> float:
        """Calculate correlation between expected and observed effects."""
        drug_profile = administration.drug_profile

        if drug_profile.name == "placebo":
            return 1.0  # Perfect correlation for placebo (no expected effects)

        # Calculate expected threshold effect at peak time
        expected_threshold_effect = (
            drug_profile.threshold_effect_direction * drug_profile.threshold_effect_magnitude
        )

        # Find observed threshold effect at peak time
        peak_time = drug_profile.peak_effect_time
        closest_time = min(threshold_modulation.keys(), key=lambda t: abs(t - peak_time))
        observed_threshold_effect = threshold_modulation[closest_time]

        # Calculate correlation for threshold effect
        if expected_threshold_effect == 0:
            threshold_correlation = 1.0 if abs(observed_threshold_effect) < 0.1 else 0.0
        else:
            threshold_correlation = max(
                0.0,
                1.0
                - abs(expected_threshold_effect - observed_threshold_effect)
                / abs(expected_threshold_effect),
            )

        physiological_correlations = []
        for effect_name, expected_range in drug_profile.physiological_effects.items():
            if effect_name in physiological_responses:
                expected_effect = (expected_range[0] + expected_range[1]) / 2
                effect_values = list(physiological_responses[effect_name].values())
                if effect_values:
                    observed_effect = np.mean(effect_values)
                    if expected_effect != 0:
                        correlation = 1.0 - min(
                            1.0,
                            float(abs(expected_effect - observed_effect) / abs(expected_effect)),
                        )
                    else:
                        correlation = 1.0 if abs(observed_effect) < 0.1 else 0.0
                    physiological_correlations.append(correlation)

        # Calculate overall correlation
        if physiological_correlations:
            overall_correlation = 0.6 * threshold_correlation + 0.4 * float(
                np.mean(physiological_correlations)
            )
        else:
            overall_correlation = threshold_correlation

        return overall_correlation

    def simulate_drug_interaction(
        self,
        drug_names: List[str],
        dosages: List[float],
        participant_weight: float = 70.0,
        route: str = "oral",
    ) -> Dict[str, PharmacologicalResponse]:
        """Simulate multiple drug administration with potential interactions.

        Args:
            drug_names: List of drugs to administer
            dosages: List of dosages for each drug
            participant_weight: Participant weight in kg
            route: Administration route (default: "oral")

        Returns:
            Dictionary of drug responses with interaction effects
        """
        if len(drug_names) != len(dosages):
            raise ValidationError("Number of drugs must match number of dosages")

        responses = {}

        # Simulate each drug individually first
        for drug_name, dosage in zip(drug_names, dosages):
            try:
                response = self.simulate_drug_administration(
                    drug_name, dosage, participant_weight, route
                )
                responses[drug_name] = response
            except ValidationError:
                # Handle unknown drugs gracefully by creating a placeholder response
                from datetime import datetime

                placeholder_profile = DrugProfile(
                    name=drug_name,
                    drug_class=DrugClass.PLACEBO,
                    target_system=NeuromodulatorSystem.NOREPINEPHRINE,
                    half_life=0.0,
                    peak_effect_time=0.0,
                    bioavailability=0.0,
                    dose_response_curve="linear",
                    threshold_effect_magnitude=0.0,
                    threshold_effect_direction=0,
                    physiological_effects={},
                )

                placeholder_admin = DrugAdministration(
                    drug_profile=placeholder_profile,
                    dosage=dosage,
                    administration_time=datetime.now(),
                    route=route,
                    participant_weight=participant_weight,
                    metabolism_rate=1.0,
                    sensitivity_factor=1.0,
                )

                responses[drug_name] = PharmacologicalResponse(
                    drug_administration=placeholder_admin,
                    time_course={0.0: 0.0},
                    threshold_modulation={0.0: 0.0},
                    peak_threshold_effect=0.0,
                    peak_effect_time=0.0,
                    physiological_responses={},
                    drug_level_confirmed=False,
                    physiological_effects_confirmed=False,
                    expected_vs_observed_correlation=0.0,
                )

        # Apply interaction effects (simplified)
        if len(drug_names) > 1:
            responses = self._apply_drug_interactions(responses)

        return responses

    def _apply_drug_interactions(
        self, responses: Dict[str, PharmacologicalResponse]
    ) -> Dict[str, PharmacologicalResponse]:
        """Apply drug interaction effects (simplified model)."""
        # This is a simplified interaction model
        # In reality, drug interactions are complex and drug-specific

        drug_names = list(responses.keys())

        # Apply interaction factors
        for i, drug1 in enumerate(drug_names):
            for j, drug2 in enumerate(drug_names[i + 1 :], i + 1):
                interaction_factor = self._get_interaction_factor(drug1, drug2)

                # Modify threshold effects
                for time_point in responses[drug1].threshold_modulation:
                    responses[drug1].threshold_modulation[time_point] *= interaction_factor
                    responses[drug2].threshold_modulation[time_point] *= interaction_factor

        return responses

    def _get_interaction_factor(self, drug1: str, drug2: str) -> float:
        """Get interaction factor between two drugs."""
        # Simplified interaction matrix
        interactions = {
            ("propranolol", "l_dopa"): 0.9,  # Slight antagonism
            ("propranolol", "sertraline"): 1.1,  # Slight synergy
            ("l_dopa", "physostigmine"): 1.2,  # Synergy
            ("sertraline", "physostigmine"): 0.95,  # Slight antagonism
        }

        # Check both orders
        factor = interactions.get((drug1, drug2), interactions.get((drug2, drug1), 1.0))
        return factor

    def get_drug_profile(self, drug_name: str) -> DrugProfile:
        """Get drug profile for a specific drug."""
        if drug_name not in self.drug_profiles:
            raise ValidationError(f"Unknown drug: {drug_name}")
        return self.drug_profiles[drug_name]

    def list_available_drugs(self) -> List[str]:
        """List all available drugs in the simulator."""
        return list(self.drug_profiles.keys())

    def validate_dosage(self, drug_name: str, dosage: float) -> Tuple[bool, str]:
        """Validate if dosage is within safe/realistic range.

        Args:
            drug_name: Name of drug
            dosage: Proposed dosage in mg

        Returns:
            Tuple of (is_valid, message)
        """
        # Define safe dosage ranges (mg)
        safe_ranges = {
            "propranolol": (10, 160),
            "l_dopa": (50, 400),
            "sertraline": (25, 200),
            "physostigmine": (0.5, 4),
            "placebo": (0, 1000),  # Any amount is "safe" for placebo
        }

        if drug_name not in safe_ranges:
            return False, f"Unknown drug: {drug_name}"

        min_dose, max_dose = safe_ranges[drug_name]

        if dosage < min_dose:
            return (
                False,
                f"Dosage {dosage} mg below minimum safe dose {min_dose} mg",
            )
        elif dosage > max_dose:
            return (
                False,
                f"Dosage {dosage} mg exceeds maximum safe dose {max_dose} mg",
            )
        else:
            return True, f"Dosage {dosage} mg is within safe range"
