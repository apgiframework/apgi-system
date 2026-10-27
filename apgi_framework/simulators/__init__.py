"""Neural signature simulators for APGI Framework falsification testing.

This package provides simulators for various neural signatures associated with
consciousness, including P3b ERP, gamma-band synchrony, BOLD activation, and
Perturbational Complexity Index (PCI).
"""

from .bold_simulator import BOLDResult, BOLDSignature, BOLDSimulator
from .bold_simulator import BrainRegion as BOLDBrainRegion
from .gamma_simulator import BrainRegion as GammaBrainRegion
from .gamma_simulator import GammaResult, GammaSignature, GammaSimulator
from .p3b_simulator import P3bSignature, P3bSimulator
from .pci_calculator import PCICalculator, PCISignature
from .signature_validator import (
    CombinedSignature,
    ConsciousnessLevel,
    SignatureType,
    SignatureValidator,
)

__all__ = [
    # P3b ERP simulator
    "P3bSimulator",
    "P3bSignature",
    # Gamma-band synchrony simulator
    "GammaSimulator",
    "GammaSignature",
    "GammaResult",
    "GammaBrainRegion",
    # BOLD activation simulator
    "BOLDSimulator",
    "BOLDSignature",
    "BOLDResult",
    "BOLDBrainRegion",
    # PCI calculator
    "PCICalculator",
    "PCISignature",
    # Signature validation and combination
    "SignatureValidator",
    "CombinedSignature",
    "ConsciousnessLevel",
    "SignatureType",
]
