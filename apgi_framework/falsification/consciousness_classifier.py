"""Consciousness Classifier for APGI Framework.

Detects whether an AI/ACC activation pattern represents genuine conscious engagement
or automated falsification patterns, helping validate falsification testing results.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from ..falsification.ai_acc_validation import AIACCActivation

logger = logging.getLogger(__name__)


class ConsciousnessLevel(Enum):
    """Classification of consciousness level in AI/ACC patterns."""

    NONE = "none"
    QUESTIONABLE = "questionable"
    LIKELY_CONSCIOUS = "likely_conscious"
    CONSCIOUS = "conscious"


class ConsciousnessIndicator(Enum):
    """Specific indicators of consciousness in AI/ACC patterns."""

    # Temporal dynamics indicators
    CONSISTENT_ANTICIPATION = "consistent_anticipation"
    ADAPTIVE_TIMING = "adaptive_timing"
    PHASE_TRANSITIONS = "phase_transitions"

    # Spatial complexity indicators
    COMPLEX_SPATIAL_PATTERNS = "complex_spatial_patterns"
    APPROPRIATE_CONNECTIVITY = "appropriate_connectivity"

    # Behavioral authenticity indicators
    RESPONSE_VARIABILITY = "response_variability"
    CONTEXTUAL_AWARENESS = "contextual_awareness"
    METACOGNITIVE_AWARENESS = "metacognitive_awareness"


@dataclass
class ConsciousnessPattern:
    """Pattern characteristics for consciousness classification."""

    pattern_type: str
    confidence: float
    indicators: List[ConsciousnessIndicator]
    description: str
    requires_ai_acc: bool = True
    threshold_met: bool = False
    matched_patterns: List[Any] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.matched_patterns is None:
            self.matched_patterns = []


@dataclass
class ConsciousnessResult:
    """Result of consciousness classification analysis."""

    level: ConsciousnessLevel
    confidence: float
    primary_indicators: List[ConsciousnessIndicator]
    missing_indicators: List[ConsciousnessIndicator]
    explanation: str
    is_falsification: bool
    recommendations: List[str]


class ConsciousnessClassifier:
    """Classifies AI/ACC activation patterns to determine likelihood of genuine consciousness
    versus automated falsification patterns.

    Uses multi-criteria analysis based on:
    - Temporal dynamics and anticipation
    - Spatial complexity and organization
    - Behavioral authenticity and variability
    - Cross-regional coherence and connectivity
    - Phase transition characteristics
    """

    def __init__(self) -> None:
        """Initialize consciousness classifier."""
        self.logger = logging.getLogger(__name__)

        # Define consciousness patterns for genuine consciousness
        self.consciousness_patterns = {
            # High consciousness patterns
            ConsciousnessPattern(
                pattern_type="adaptive_timing",
                confidence=0.9,
                indicators=[
                    ConsciousnessIndicator.CONSISTENT_ANTICIPATION,
                    ConsciousnessIndicator.PHASE_TRANSITIONS,
                    ConsciousnessIndicator.RESPONSE_VARIABILITY,
                ],
                description="Consistent adaptive timing with appropriate phase transitions",
                requires_ai_acc=True,
            ),
            ConsciousnessPattern(
                pattern_type="complex_spatial_patterns",
                confidence=0.85,
                indicators=[
                    ConsciousnessIndicator.COMPLEX_SPATIAL_PATTERNS,
                    ConsciousnessIndicator.APPROPRIATE_CONNECTIVITY,
                    ConsciousnessIndicator.CONTEXTUAL_AWARENESS,
                ],
                description="Complex spatial patterns with appropriate connectivity",
                requires_ai_acc=True,
            ),
            ConsciousnessPattern(
                pattern_type="contextual_awareness",
                confidence=0.8,
                indicators=[
                    ConsciousnessIndicator.CONTEXTUAL_AWARENESS,
                    ConsciousnessIndicator.METACOGNITIVE_AWARENESS,
                ],
                description="Contextual responses with metacognitive awareness",
                requires_ai_acc=True,
            ),
            # Low consciousness/automated patterns
            ConsciousnessPattern(
                pattern_type="consistent_anticipation",
                confidence=0.3,
                indicators=[
                    ConsciousnessIndicator.CONSISTENT_ANTICIPATION,
                ],
                description="Consistent but rigid timing (likely automated)",
                requires_ai_acc=False,
            ),
            ConsciousnessPattern(
                pattern_type="response_variability",
                confidence=0.2,
                indicators=[
                    ConsciousnessIndicator.RESPONSE_VARIABILITY,
                ],
                description="Highly variable responses (indicating randomness)",
                requires_ai_acc=False,
            ),
        }

        # Define falsification patterns
        self.falsification_patterns: List[ConsciousnessPattern] = [
            ConsciousnessPattern(
                pattern_type="consistent_anticipation",
                confidence=0.4,
                indicators=[
                    ConsciousnessIndicator.CONSISTENT_ANTICIPATION,
                ],
                description="Consistent artificial timing without phase transitions",
                requires_ai_acc=False,
            ),
            ConsciousnessPattern(
                pattern_type="response_variability",
                confidence=0.6,
                indicators=[
                    ConsciousnessIndicator.RESPONSE_VARIABILITY,
                ],
                description="Extremely variable responses (clear automation)",
                requires_ai_acc=False,
            ),
        ]

    def classify_activation_pattern(
        self, activation: AIACCActivation, regions_data: Dict[str, Any]
    ) -> ConsciousnessResult:
        """Classify a single AI/ACC activation pattern.

        Args:
            activation: AI/ACC activation data
            regions_data: Dictionary of region activations

        Returns:
            ConsciousnessResult with classification and confidence
        """
        self.logger.info(f"Classifying activation pattern for regions: {list(regions_data.keys())}")

        # Check for AI/ACC data
        if not activation or not regions_data:
            return ConsciousnessResult(
                level=ConsciousnessLevel.NONE,
                confidence=0.0,
                primary_indicators=[],
                missing_indicators=[ConsciousnessIndicator.RESPONSE_VARIABILITY],
                explanation="No AI/ACC data available",
                is_falsification=False,
                recommendations=["Enable AI/ACC data collection"],
            )

        # Analyze each region
        region_scores = []
        total_confidence = 0.0
        matched_patterns = []

        for region_name, region_data in regions_data.items():
            if not region_data:
                continue

            region_score = self._analyze_region_activation(region_name, region_data)
            region_scores.append(region_score)

            # Add to overall confidence with weighting
            if region_score.confidence > 0:
                total_confidence += region_score.confidence * 0.25  # Weight regions more heavily
            else:
                total_confidence += 0.1  # Minimum weight for valid regions

            matched_patterns.extend(region_score.matched_patterns)

        # Determine overall classification
        consciousness_level = self._determine_consciousness_level(
            total_confidence, matched_patterns
        )

        # Check for falsification indicators
        is_falsification = self._detect_falsification_indicators(matched_patterns)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            consciousness_level, is_falsification, matched_patterns
        )

        return ConsciousnessResult(
            level=consciousness_level,
            confidence=min(total_confidence, 1.0),
            primary_indicators=matched_patterns[:3],  # Top 3 indicators
            missing_indicators=self._get_missing_indicators(matched_patterns),
            explanation=self._generate_explanation(
                consciousness_level, matched_patterns, regions_data
            ),
            is_falsification=is_falsification,
            recommendations=recommendations,
        )

    def _analyze_region_activation(
        self, region_name: str, region_data: Any
    ) -> ConsciousnessPattern:
        """Analyze activation in a specific AI/ACC region.

        Args:
            region_name: Name of the AI/ACC region
            region_data: Activation data for the region

        Returns:
            Best matching consciousness pattern
        """
        self.logger.debug(f"Analyzing {region_name} region: {region_data}")

        best_pattern = None
        best_score = 0.0

        # Check against consciousness patterns
        for pattern in self.consciousness_patterns:
            if pattern.requires_ai_acc and not region_data:
                continue

            score = self._calculate_pattern_score(pattern, region_data)
            self.logger.debug(f"Pattern '{pattern.pattern_type}' score: {score}")

            if score > best_score:
                best_score = score
                best_pattern = pattern

        if best_pattern:
            best_pattern.threshold_met = True
        else:
            # Default to low consciousness pattern if nothing matches
            best_pattern = self.falsification_patterns[0]  # Low automation pattern
            best_score = 0.1

        return best_pattern

    def _calculate_pattern_score(self, pattern: ConsciousnessPattern, region_data: Any) -> float:
        """Calculate how well region data matches a consciousness pattern."""
        score = pattern.confidence

        # Check each indicator
        for indicator in pattern.indicators:
            if self._check_indicator(region_data, indicator):
                score += 0.2  # Each matching indicator adds 0.2

        return min(score, 1.0)

    def _check_indicator(self, region_data: Any, indicator: ConsciousnessIndicator) -> bool:
        """Check if a specific consciousness indicator is present in region data."""
        # This is a simplified check - in practice, would need more sophisticated
        # pattern matching based on data characteristics

        if indicator == ConsciousnessIndicator.CONSISTENT_ANTICIPATION:
            return self._has_consistent_timing(region_data)
        elif indicator == ConsciousnessIndicator.PHASE_TRANSITIONS:
            return self._has_phase_transitions(region_data)
        elif indicator == ConsciousnessIndicator.RESPONSE_VARIABILITY:
            return self._has_appropriate_variability(region_data)
        elif indicator == ConsciousnessIndicator.COMPLEX_SPATIAL_PATTERNS:
            return self._has_complex_spatial_patterns(region_data)
        elif indicator == ConsciousnessIndicator.APPROPRIATE_CONNECTIVITY:
            return self._has_appropriate_connectivity(region_data)
        elif indicator == ConsciousnessIndicator.CONTEXTUAL_AWARENESS:
            return self._has_contextual_awareness(region_data)
        elif indicator == ConsciousnessIndicator.METACOGNITIVE_AWARENESS:
            return self._has_metacognitive_awareness(region_data)

        return False

    def _has_consistent_timing(self, region_data: Any) -> bool:
        """Check for consistent adaptive timing patterns."""
        # Would check for regular intervals, appropriate latency, etc.
        # Simplified check for demonstration
        return hasattr(region_data, "peak_latency") and hasattr(region_data, "activation_duration")

    def _has_phase_transitions(self, region_data: Any) -> bool:
        """Check for phase transition characteristics."""
        # Would check for smooth transitions between states
        return hasattr(region_data, "phase_coupling") and hasattr(
            region_data, "functional_connectivity"
        )

    def _has_appropriate_variability(self, region_data: Any) -> bool:
        """Check for appropriate response variability."""
        # Would check for coefficient of variation within reasonable bounds
        return hasattr(region_data, "response_variability") and hasattr(
            region_data, "baseline_corrected"
        )

    def _has_complex_spatial_patterns(self, region_data: Any) -> bool:
        """Check for complex spatial activation patterns."""
        # Would check for multi-region coherence, complex topographies
        return hasattr(region_data, "spatial_coherence") and hasattr(
            region_data, "topographic_complexity"
        )

    def _has_appropriate_connectivity(self, region_data: Any) -> bool:
        """Check for appropriate connectivity patterns."""
        # Would check for physiologically plausible connectivity
        return hasattr(region_data, "effective_connectivity") and hasattr(
            region_data, "anatomical_plausibility"
        )

    def _has_contextual_awareness(self, region_data: Any) -> bool:
        """Check for contextual awareness indicators."""
        # Would check for task-appropriate responses, context integration
        return hasattr(region_data, "contextual_integration") and hasattr(
            region_data, "adaptive_responses"
        )

    def _has_metacognitive_awareness(self, region_data: Any) -> bool:
        """Check for metacognitive awareness indicators."""
        # Would check for self-monitoring, strategic planning, insight
        return hasattr(region_data, "metacognitive_monitoring") and hasattr(
            region_data, "strategic_adaptation"
        )

    def _determine_consciousness_level(
        self, total_confidence: float, matched_patterns: List[ConsciousnessPattern]
    ) -> ConsciousnessLevel:
        """Determine overall consciousness level from pattern matches."""
        if not matched_patterns:
            return ConsciousnessLevel.NONE

        # Check for high consciousness indicators
        high_consciousness_indicators = [
            ConsciousnessIndicator.ADAPTIVE_TIMING,
            ConsciousnessIndicator.COMPLEX_SPATIAL_PATTERNS,
            ConsciousnessIndicator.CONTEXTUAL_AWARENESS,
            ConsciousnessIndicator.METACOGNITIVE_AWARENESS,
        ]

        has_high_indicators = any(
            indicator in high_consciousness_indicators
            for pattern in matched_patterns
            for indicator in pattern.indicators
        )

        # Check for likely consciousness
        likely_consciousness_indicators = [
            ConsciousnessIndicator.CONSISTENT_ANTICIPATION,
            ConsciousnessIndicator.PHASE_TRANSITIONS,
        ]

        has_likely_indicators = any(
            indicator in likely_consciousness_indicators
            for pattern in matched_patterns
            for indicator in pattern.indicators
        )

        # Check for questionable
        if total_confidence >= 0.3:
            return ConsciousnessLevel.QUESTIONABLE

        # Determine overall classification
        if has_high_indicators:
            return ConsciousnessLevel.CONSCIOUS

        # Check for likely consciousness
        elif has_likely_indicators and total_confidence >= 0.5:
            return ConsciousnessLevel.LIKELY_CONSCIOUS

        # Check for questionable
        if total_confidence >= 0.3:
            return ConsciousnessLevel.QUESTIONABLE

        return ConsciousnessLevel.NONE

    def _detect_falsification_indicators(
        self, matched_patterns: List[ConsciousnessPattern]
    ) -> bool:
        """Detect whether matched patterns indicate falsification.

        Args:
            matched_patterns: List of matched consciousness patterns

        Returns:
            True if falsification indicators are detected
        """
        if not matched_patterns:
            return False

        # Check for low-confidence patterns that suggest automation
        low_confidence_count = sum(1 for p in matched_patterns if p.confidence < 0.3)

        # If mostly low confidence patterns, likely falsification
        if low_confidence_count >= len(matched_patterns) * 0.5:
            return True

        # Check for specific falsification indicators
        for pattern in matched_patterns:
            if pattern.pattern_type in {
                "consistent_anticipation",
                "response_variability",
            }:
                if pattern.confidence > 0.5:
                    return True

        return False

    def _get_missing_indicators(
        self, matched_patterns: List[ConsciousnessPattern]
    ) -> List[ConsciousnessIndicator]:
        """Get indicators that should be present but are missing."""
        all_indicators = set()
        for pattern in matched_patterns:
            all_indicators.update(pattern.indicators)

        # High consciousness indicators that should be present
        expected_indicators = {
            ConsciousnessIndicator.ADAPTIVE_TIMING,
            ConsciousnessIndicator.COMPLEX_SPATIAL_PATTERNS,
            ConsciousnessIndicator.CONTEXTUAL_AWARENESS,
            ConsciousnessIndicator.METACOGNITIVE_AWARENESS,
        }

        missing = list(expected_indicators - all_indicators)

        return missing

    def _generate_explanation(
        self,
        level: ConsciousnessLevel,
        matched_patterns: List[ConsciousnessPattern],
        regions_data: Dict[str, Any],
    ) -> str:
        """Generate explanation for consciousness classification result."""
        if level == ConsciousnessLevel.CONSCIOUS:
            explanation = "Strong evidence of genuine consciousness: "
            for pattern in matched_patterns:
                explanation += f"{pattern.description}; "

        elif level == ConsciousnessLevel.LIKELY_CONSCIOUS:
            explanation = "Moderate evidence of consciousness: "
            for pattern in matched_patterns:
                explanation += f"{pattern.description}; "

        elif level == ConsciousnessLevel.QUESTIONABLE:
            explanation = "Limited evidence - requires further investigation: "
            for pattern in matched_patterns:
                explanation += f"{pattern.description}; "

        else:
            explanation = "No clear consciousness patterns detected"

        # Add region-specific details
        if regions_data:
            explanation += (
                f"\n\nAnalyzed {len(regions_data)} AI/ACC regions: {', '.join(regions_data.keys())}"
            )

        return explanation

    def _generate_recommendations(
        self,
        level: ConsciousnessLevel,
        is_falsification: bool,
        matched_patterns: List[ConsciousnessPattern],
    ) -> List[str]:
        """Generate recommendations based on consciousness classification."""
        recommendations = []

        if level == ConsciousnessLevel.NONE:
            recommendations.extend(
                [
                    "Enable AI/ACC data collection for consciousness analysis",
                    "Review experimental design for proper consciousness indicators",
                    "Consider adding ground truth consciousness measures",
                ]
            )

        elif level == ConsciousnessLevel.QUESTIONABLE:
            recommendations.extend(
                [
                    "Investigate potential automation or data quality issues",
                    "Add additional consciousness validation measures",
                    "Review timing and variability parameters",
                ]
            )

        elif not is_falsification and level in [
            ConsciousnessLevel.LIKELY_CONSCIOUS,
            ConsciousnessLevel.CONSCIOUS,
        ]:
            recommendations.extend(
                [
                    "Continue monitoring for consciousness development",
                    "Document consciousness indicators for reproducibility",
                    "Consider adaptive consciousness protocols",
                ]
            )

        else:  # is_falsification
            recommendations.extend(
                [
                    "Investigate potential falsification mechanisms",
                    "Review automation parameters and randomness",
                    "Implement stricter consciousness validation protocols",
                    "Add control conditions with known consciousness patterns",
                ]
            )

        return recommendations
