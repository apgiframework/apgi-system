"""Vectorized Global Workspace Implementation."""

from enum import Enum
from typing import Dict, Optional

import numpy as np

from ..types import ConfigDict


class WorkspaceState(Enum):
    IDLE = 0
    IGNITING = 1
    BROADCASTING = 2
    MAINTAINING = 3
    FADING = 4


class GlobalWorkspace:
    """Vectorized Global Workspace for multi-agent conscious broadcasting."""

    def __init__(self, config: ConfigDict, rng: Optional[np.random.Generator] = None) -> None:
        """Initialize vectorized global workspace."""
        self.config = config
        self.batch_size = config.get("active_inference", {}).get("batch_size", 1)

        # Determine content dimensions from config or default
        self.content_dim = config.get("ignition", {}).get("workspace_dim", 256)

        # Workspace parameters
        self.amplification_duration_ms = config.get("ignition", {}).get(
            "amplification_duration_ms", 300
        )
        self.maintenance_duration_ms = 1000.0
        self.fade_duration_ms = 200.0
        self.competition_duration_ms = 50.0

        # States (B,)
        self.states = np.full(self.batch_size, WorkspaceState.IDLE.value)
        self.state_times = np.zeros(self.batch_size)

        # Content (B, D)
        self.current_content = np.zeros((self.batch_size, self.content_dim))
        self.content_priority = np.zeros(self.batch_size)

        # Competition pool: each agent can have its own candidate
        self.has_candidate = np.zeros(self.batch_size, dtype=bool)
        self.candidate_content = np.zeros((self.batch_size, self.content_dim))
        self.candidate_priority = np.zeros(self.batch_size)

        self.amplification_gain = 2.0
        self.rng = rng if rng is not None else np.random.default_rng()

    def update(
        self,
        ignition_mask: np.ndarray,
        candidates: Optional[np.ndarray] = None,
        priorities: Optional[np.ndarray] = None,
        dt: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """Update workspace for batch (B,)."""
        # 1. Update candidates for agents
        if candidates is not None:
            # Mask where we have valid candidates
            # Here we assume candidates is (B, D)
            self.candidate_content = candidates.copy()
            self.candidate_priority = (
                priorities if priorities is not None else np.ones(self.batch_size)
            )
            self.has_candidate[:] = True

        # 2. State Machine Transitions

        # IDLE -> IGNITING
        idle_mask = self.states == WorkspaceState.IDLE.value
        start_igniting = idle_mask & ignition_mask & self.has_candidate
        self.states[start_igniting] = WorkspaceState.IGNITING.value
        self.state_times[start_igniting] = 0.0

        # IGNITING -> BROADCASTING
        igniting_mask = self.states == WorkspaceState.IGNITING.value
        self.state_times[igniting_mask] += dt
        finish_igniting = igniting_mask & (self.state_times >= self.competition_duration_ms)
        if np.any(finish_igniting):
            self.states[finish_igniting] = WorkspaceState.BROADCASTING.value
            self.state_times[finish_igniting] = 0.0
            # Select winner (simple: current candidate becomes content)
            self.current_content[finish_igniting] = self.candidate_content[finish_igniting]
            self.content_priority[finish_igniting] = self.candidate_priority[finish_igniting]
            self.has_candidate[finish_igniting] = False

        # BROADCASTING -> MAINTAINING
        broadcasting_mask = self.states == WorkspaceState.BROADCASTING.value
        self.state_times[broadcasting_mask] += dt

        # Amplification
        if np.any(broadcasting_mask):
            self._amplify_batch(broadcasting_mask)

        finish_broadcasting = broadcasting_mask & (
            self.state_times >= self.amplification_duration_ms
        )
        self.states[finish_broadcasting] = WorkspaceState.MAINTAINING.value
        self.state_times[finish_broadcasting] = 0.0

        # MAINTAINING -> FADING
        maintaining_mask = self.states == WorkspaceState.MAINTAINING.value
        self.state_times[maintaining_mask] += dt
        finish_maintaining = maintaining_mask & (self.state_times >= self.maintenance_duration_ms)
        self.states[finish_maintaining] = WorkspaceState.FADING.value
        self.state_times[finish_maintaining] = 0.0

        # FADING -> IDLE
        fading_mask = self.states == WorkspaceState.FADING.value
        self.state_times[fading_mask] += dt
        # Apply fading
        fade_factors = 1.0 - (self.state_times[fading_mask] / self.fade_duration_ms)
        self.current_content[fading_mask] *= np.maximum(0, fade_factors)[:, np.newaxis]

        finish_fading = fading_mask & (self.state_times >= self.fade_duration_ms)
        self.states[finish_fading] = WorkspaceState.IDLE.value
        self.state_times[finish_fading] = 0.0
        self.current_content[finish_fading] = 0.0

        is_reportable = (self.states == WorkspaceState.BROADCASTING.value) | (
            self.states == WorkspaceState.MAINTAINING.value
        )

        return {
            "states": self.states.copy(),
            "is_reportable": is_reportable,
            "content": self.current_content.copy(),
        }

    def _amplify_batch(self, mask: np.ndarray) -> None:
        """Vectorized recurrent amplification."""
        batch_indices = np.where(mask)[0]
        if len(batch_indices) == 0:
            return

        # Time-based gain factor
        time_factors = self.state_times[mask] / self.amplification_duration_ms
        gain_factors = 1.0 + (self.amplification_gain - 1.0) * np.tanh(time_factors * 2.5)

        # Apply gain
        # We need to preserve direction, so normalize and resScale
        # Target magnitude is original magnitude (at start of broadcast) * gain
        # For simplicity in vectorized version, we'll just scale the current magnitudes toward a target
        # But since magnitudes grow, we'll just scale directly
        self.current_content[mask] *= gain_factors[:, np.newaxis] / (
            1.0 + 0.1 * time_factors[:, np.newaxis]
        )  # damping

        # Add stochastic noise
        noise = self.rng.normal(scale=0.01, size=self.current_content[mask].shape)
        self.current_content[mask] += noise

    def reset(self) -> None:
        """Reset all agent workspaces."""
        self.states.fill(WorkspaceState.IDLE.value)
        self.state_times.fill(0.0)
        self.current_content.fill(0.0)
        self.has_candidate.fill(False)

    def is_reportable(self) -> bool:
        """Check if workspace is currently broadcasting."""
        is_reportable = (self.states == WorkspaceState.BROADCASTING.value) | (
            self.states == WorkspaceState.MAINTAINING.value
        )
        return bool(np.any(is_reportable))
