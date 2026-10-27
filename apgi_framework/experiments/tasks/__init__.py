"""Experimental tasks module for the APGI Framework.

This module contains various cognitive and experimental tasks
for testing the APGI falsification framework.
"""

from .attentional_blink import AttentionalBlinkTask, StimulusType
from .binocular_rivalry import BinocularRivalryTask
from .change_blindness import ChangeBlindnessTask, ChangeType
from .iowa_gambling import DECK_SCHEDULES, DeckType, IowaGamblingTask
from .masking_paradigm import MaskingParadigmTask, MaskType
from .nback import NBackLevel, NBackTask
from .nback import StimulusType as NBackStimulusType
from .stroop import StimulusType as StroopStimulusType
from .stroop import StroopTask

__all__ = [
    "AttentionalBlinkTask",
    "StimulusType",
    "BinocularRivalryTask",
    "ChangeBlindnessTask",
    "ChangeType",
    "IowaGamblingTask",
    "DeckType",
    "DECK_SCHEDULES",
    "MaskingParadigmTask",
    "MaskType",
    "StroopTask",
    "StroopStimulusType",
    "NBackTask",
    "NBackLevel",
    "NBackStimulusType",
]
