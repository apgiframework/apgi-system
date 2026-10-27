"""Motor planning by expected free energy (EFE) minimisation.

:class:`GenerativeLikelihood` is a small deterministic forward model: a hidden state evolves
through an orthogonal transition ``B`` and an action drive ``C`` (squashed by ``tanh``), and
is read out through a row-orthonormal likelihood ``A`` followed by a softmax, so every predicted
observation is a probability distribution. :class:`MotorPlanner` samples (or accepts) candidate
actions, rolls each one out over the planning horizon and scores it by the expected free energy
of the predicted observations against the agent's preferences.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _orthonormal(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """A ``rows x cols`` matrix with orthonormal rows (if rows <= cols) or columns."""
    gaussian = rng.standard_normal((rows, cols))
    if rows <= cols:
        q, _ = np.linalg.qr(gaussian.T)  # (cols, rows) with orthonormal columns
        return np.asarray(q.T)
    q, _ = np.linalg.qr(gaussian)
    return np.asarray(q)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return np.asarray(exp / exp.sum())


class GenerativeLikelihood:
    """Deterministic generative model mapping (state, action) to predicted observations.

    Parameters
    ----------
    state_dim, obs_dim, action_dim : int
        Dimensions of the hidden state, the observation and the action vector.
    seed : int
        Seed of the random (orthonormal) weight matrices; equal seeds give equal models.

    Attributes
    ----------
    A : ndarray (obs_dim, state_dim)
        Likelihood weights, orthonormal rows (columns when ``obs_dim > state_dim``).
    B : ndarray (state_dim, state_dim)
        Orthogonal state transition.
    C : ndarray (state_dim, action_dim)
        Action-to-state drive, orthonormal along its smaller dimension.
    """

    def __init__(
        self, state_dim: int = 10, obs_dim: int = 10, action_dim: int = 10, seed: int = 0
    ) -> None:
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        rng = np.random.default_rng(seed)
        self.A = _orthonormal(obs_dim, state_dim, rng)
        self.B = _orthonormal(state_dim, state_dim, rng)
        self.C = _orthonormal(state_dim, action_dim, rng)

    def _pad_action(self, action: np.ndarray) -> np.ndarray:
        """Zero-pad or truncate ``action`` to ``action_dim``."""
        action = np.asarray(action, dtype=float).ravel()
        padded = np.zeros(self.action_dim)
        n = min(len(action), self.action_dim)
        padded[:n] = action[:n]
        return padded

    def rollout(
        self, s0: np.ndarray, a: np.ndarray, horizon: int = 3
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Predict ``horizon`` steps of a constant action.

        Returns
        -------
        states : list of ndarray
            Predicted hidden states, each inside (-1, 1); empty for ``horizon == 0``.
        observations : ndarray (horizon, obs_dim)
            Predicted observation distributions (each row sums to one).
        """
        drive = self.C @ self._pad_action(a)
        state = np.asarray(s0, dtype=float).copy()
        states: List[np.ndarray] = []
        observations = np.zeros((max(horizon, 0), self.obs_dim))
        for t in range(max(horizon, 0)):
            state = np.tanh(self.B @ state + drive)
            states.append(state)
            observations[t] = _softmax(self.A @ state)
        return states, observations


class MotorPlanner:
    """Choose an action by minimising expected free energy over candidate policies.

    Configuration (all optional): ``system.random_seed`` / ``action_dim`` / ``obs_dim`` and
    ``active_inference.planning.horizon`` / ``num_policies`` / ``precision``.
    """

    EPS = 1e-10

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        state_dim: int = 10,
        action_dim: int = 5,
    ) -> None:
        self.config: Dict[str, Any] = config or {}
        system = self.config.get("system", {})
        planning = self.config.get("active_inference", {}).get("planning", {})

        self.seed: int = int(system.get("random_seed", 0))
        self.action_dim: int = int(system.get("action_dim", action_dim))
        self.obs_dim: int = int(system.get("obs_dim", state_dim))
        self.state_dim = state_dim
        self.horizon: int = int(planning.get("horizon", 3))
        self.num_policies: int = int(planning.get("num_policies", 5))
        self.precision: float = float(planning.get("precision", 1.0))
        # Weight of the observation-entropy (ambiguity) term. With weight 1 the sum of risk
        # and ambiguity is the cross-entropy against the preferences, which is constant for
        # uniform preferences, so the default scores policies by risk alone.
        self.ambiguity_weight: float = float(planning.get("ambiguity_weight", 0.0))

        self.rng = np.random.default_rng(self.seed)
        self.preferences: np.ndarray = np.ones(self.obs_dim) / self.obs_dim
        self._generative_model: Optional[GenerativeLikelihood] = None

    def set_preferences(self, preferences: np.ndarray) -> None:
        """Set the preferred observation distribution (normalised internally when planning)."""
        self.preferences = preferences

    def _normalised_preferences(self) -> np.ndarray:
        prefs = np.clip(np.asarray(self.preferences, dtype=float).ravel(), 0.0, None)
        if prefs.size != self.obs_dim:
            prefs = np.resize(prefs, self.obs_dim)
        total = prefs.sum()
        if total <= 0.0:
            return np.ones(self.obs_dim) / self.obs_dim
        return prefs / total

    def _model_for(self, state_dim: int) -> GenerativeLikelihood:
        """The generative model for ``state_dim``, built lazily and rebuilt when it changes."""
        if self._generative_model is None or self._generative_model.state_dim != state_dim:
            self._generative_model = GenerativeLikelihood(
                state_dim=state_dim,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                seed=self.seed,
            )
        return self._generative_model

    def _expected_free_energy(self, observations: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Risk (KL to preferences) plus weighted ambiguity (entropy), summed over the horizon."""
        prefs = self._normalised_preferences()
        q = np.clip(observations, self.EPS, 1.0)
        risk = float(np.sum(q * (np.log(q) - np.log(prefs + self.EPS))))
        ambiguity = float(-np.sum(q * np.log(q)))
        return risk + self.ambiguity_weight * ambiguity, {"risk": risk, "ambiguity": ambiguity}

    def plan(
        self, beliefs: List[Any], available_actions: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select an action.

        Parameters
        ----------
        beliefs : list
            Belief states per hierarchy level (objects with a ``mean``); the top level
            (last) provides the initial hidden state.
        available_actions : list of ndarray, optional
            Candidate actions; sampled from the planner's seeded RNG when omitted.

        Returns
        -------
        action : ndarray (action_dim,)
            Action of the policy with the lowest expected free energy.
        info : dict
            ``best_idx``, ``policy_probs`` (softmax of ``-precision * EFE``), ``policy_efes``,
            ``selected_efe`` and ``components`` (risk/ambiguity of the selected policy).
        """
        s0 = np.asarray(beliefs[-1].mean, dtype=float)
        model = self._model_for(s0.shape[0])

        if available_actions is None:
            candidates = [
                self.rng.standard_normal(self.action_dim) for _ in range(self.num_policies)
            ]
        else:
            candidates = [np.asarray(a, dtype=float) for a in available_actions]

        efes: List[float] = []
        components: List[Dict[str, float]] = []
        for action in candidates:
            _, observations = model.rollout(s0, action, self.horizon)
            efe, parts = self._expected_free_energy(observations)
            efes.append(efe)
            components.append(parts)

        best_idx = int(np.argmin(efes))
        probs = _softmax(-self.precision * np.asarray(efes))
        action_out = model._pad_action(candidates[best_idx])
        return action_out, {
            "best_idx": best_idx,
            "policy_probs": probs.tolist(),
            "policy_efes": efes,
            "selected_efe": efes[best_idx],
            "components": components[best_idx],
        }

    def reset(self) -> None:
        """Reset per-run state; the generative model's weights are kept."""
        self.rng = np.random.default_rng(self.seed)
