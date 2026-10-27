"""Stub module for active inference.

This module provides type stubs for active inference functionality
that is being migrated from the old apgi_simulation structure.
"""

import asyncio
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class HierarchicalGaussianFilter:
    """Hierarchical Gaussian filter for active inference."""

    time: float
    filter: Any
    beliefs: List[Any]

    def __init__(
        self,
        num_levels: int = 3,
        state_dims: Optional[List[int]] = None,
        observation_dim: int = 16,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize hierarchical Gaussian filter.

        Parameters
        ----------
        num_levels : int, optional
            Number of hierarchical levels, by default 3
        state_dims : List[int], optional
            Dimension of each level, by default None
        observation_dim : int, optional
            Dimension of observation space, by default 16
        config : Dict[str, Any], optional
            Configuration dictionary, by default None
        """
        self.num_levels = num_levels
        self.state_dims = state_dims or [32, 16, 8]
        self.observation_dim = observation_dim
        self.config = config or {}
        self.time = 0.0

        # Initialize belief states for each level
        self.belief_states = [np.zeros(dim) for dim in self.state_dims]
        self.precisions = [np.ones(dim) for dim in self.state_dims]

        # Create belief objects for compatibility
        class Belief:
            def __init__(self, dim: int):
                self.mean = np.zeros(dim)
                self.covariance = np.eye(dim)
                self.precision = np.ones(dim)
                self.prediction = np.zeros(dim)
                self.prediction_error = np.zeros(dim)

        self.beliefs = [Belief(dim) for dim in self.state_dims]
        self.filter = self  # Self-reference for compatibility

        # Cache attributes for projection matrices
        self._projection_cache: Dict[str, np.ndarray] = {}
        self._cache_access_order: List[str] = []
        self._cache_lock = Lock()
        self._projection_cache_size: int = self.config.get("projection_cache_size", 10)

    def update(self, observation: np.ndarray) -> tuple[List[np.ndarray], float]:
        """Update belief state given observation.

        Parameters
        ----------
        observation : np.ndarray
            New observation

        Returns
        -------
        tuple[List[np.ndarray], float]
            Tuple of (belief states per level, free energy)
        """
        # Accept a single observation (D,) or a batch (B, D); a batch is averaged to one belief
        observation = np.asarray(observation, dtype=np.float64)
        if observation.ndim > 1:
            observation = observation.reshape(-1, self.belief_states[0].shape[-1]).mean(axis=0)

        # Simplified update
        prediction_error = observation - self.belief_states[0]
        self.belief_states[0] += prediction_error * 0.1
        free_energy = float(np.sum(prediction_error**2))
        self.time += 1.0
        return self.belief_states, free_energy

    def step(
        self, observation: np.ndarray, actions: List[Any] | None = None
    ) -> tuple[Any, Dict[str, Any]]:
        """Step the active inference engine.

        Parameters
        ----------
        observation : np.ndarray
            Current observation
        actions : List[Any], optional
            Available actions (not used in simplified implementation)

        Returns
        -------
        tuple[Any, Dict[str, Any]]
            Tuple of (selected action, info dict)
        """
        # Ensure 2D shape
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        belief_states, free_energy = self.update(observation[0])
        info = {
            "free_energy": free_energy,
            "belief_states": belief_states,
        }
        return None, info

    def get_belief_state(self) -> np.ndarray:
        """Get current belief state.

        Returns
        -------
        np.ndarray
            Current belief state (lowest level)
        """
        return self.belief_states[0].copy()

    def _map_down(self, level: int, state: np.ndarray) -> np.ndarray:
        """Map state down from higher to lower level.

        Parameters
        ----------
        level : int
            Source level
        state : np.ndarray
            State to map down

        Returns
        -------
        np.ndarray
            Mapped state
        """
        # Simplified mapping
        if level == 0:
            return state
        target_dim = self.state_dims[level - 1]
        return np.zeros((state.shape[0], target_dim))

    def _project_up(self, level: int, state: np.ndarray) -> np.ndarray:
        """Project state up from lower to higher level.

        Parameters
        ----------
        level : int
            Source level
        state : np.ndarray
            State to project up

        Returns
        -------
        np.ndarray
            Projected state
        """
        # Top level returns unchanged
        if level >= self.num_levels - 1:
            return state
        target_dim = self.state_dims[level + 1]
        return np.zeros(target_dim)


class BeliefState:
    """Container for belief state information."""

    def __init__(self, mean: np.ndarray, covariance: np.ndarray):
        """Initialize belief state.

        Parameters
        ----------
        mean : np.ndarray
            Mean of belief state
        covariance : np.ndarray
            Covariance matrix
        """
        self.mean = mean
        self.covariance = covariance


class ActiveInferenceEngine:
    """Active inference engine for decision making."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize active inference engine.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        """
        self.config = config
        hierarchy_config = config.get("hierarchy", {})
        self.num_levels = hierarchy_config.get("num_levels", 3)
        level_configs = hierarchy_config.get("level_configs", [])
        self.state_dims = [lc.get("nodes", 16) for lc in level_configs]
        self.observation_dim = self.state_dims[0]

        self.filter = HierarchicalGaussianFilter(
            num_levels=self.num_levels,
            state_dims=self.state_dims,
            observation_dim=self.observation_dim,
            config=config,
        )

        ai_config = config.get("active_inference", {})
        planning_config = ai_config.get("planning", {})
        self.num_policies = planning_config.get("num_policies", 5)
        self.planning_horizon = planning_config.get("horizon", 2)

        system_config = config.get("system", {})
        self.timestep = system_config.get("timestep_ms", 1.0) / 1000.0
        self.time = 0.0

        self.fe_calc = None  # Stub for free energy calculator

    def step(
        self, observation: np.ndarray, actions: Optional[List[np.ndarray]] = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Perform one inference step.

        Parameters
        ----------
        observation : np.ndarray
            Current observation
        actions : List[np.ndarray], optional
            Available actions

        Returns
        -------
        tuple[np.ndarray, Dict[str, Any]]
            Tuple of (selected action, info dict)
        """
        beliefs, fe = self.filter.update(observation)
        self.time += self.timestep

        # Simplified action selection
        if actions:
            action = actions[0]
        else:
            action = np.zeros(1)

        info = {
            "time": self.time,
            "free_energy": fe,
            "beliefs": beliefs,
            "precisions": self.filter.precisions,
            "prediction_errors": observation - beliefs[0],
            "efe_components": {},
        }

        return action, info

    def vectorized_batch_step(self, observations: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a batch of observations for multiple agents.

        Parameters
        ----------
        observations : np.ndarray
            Batch of observations (batch_size, obs_dim)

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Tuple of (actions, info_dict) where info_dict contains
            'per_agent_free_energy' array of shape (batch_size,).

        Raises
        ------
        ValueError
            If batch size doesn't match engine's batch_size.
        """
        batch_size = self.config.get("system", {}).get("batch_size", 1)
        if observations.shape[0] != batch_size:
            raise ValueError(
                f"Observation batch size ({observations.shape[0]}) "
                f"doesn't match engine batch_size ({batch_size})"
            )

        # Process each observation and collect results
        all_actions = []
        all_free_energies = []

        for i in range(batch_size):
            obs = observations[i]
            # Simple action: broadcast same action for all
            action = np.zeros(1)
            # Compute free energy from observation
            fe = float(np.sum(obs**2))
            all_actions.append(action)
            all_free_energies.append(fe)

        actions = np.array(all_actions)
        per_agent_fe = np.array(all_free_energies)

        info = {
            "per_agent_free_energy": per_agent_fe,
            "beliefs": [self.filter.belief_states],
            "free_energy": float(np.mean(per_agent_fe)),
        }

        return actions, info

    async def async_step(
        self, observation: np.ndarray | Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Async version of step().

        Parameters
        ----------
        observation : np.ndarray or dict
            Current observation. Can be a single array or dict of modalities.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Tuple of (action, info_dict).
        """
        # Handle dict observations (multi-modal)
        if isinstance(observation, dict):
            # Extract first modality for simplicity
            observation = next(iter(observation.values()))

        # Run the synchronous step in an executor to make it async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.step, observation)


class ActiveInferenceAgent:
    """Agent that uses active inference for decision making."""

    def __init__(self, agent_id: int, config: Optional[Dict[str, Any]] = None):
        """Initialize active inference agent.

        Parameters
        ----------
        agent_id : int
            Unique identifier for this agent.
        config : dict, optional
            Agent-specific configuration, by default None.
        """
        self.agent_id = agent_id
        self.config = config if config is not None else {}
        self.free_energy_history: List[float] = []

    def record_step(self, free_energy: float) -> None:
        """Record free energy for this step.

        Parameters
        ----------
        free_energy : float
            Free energy value to record.
        """
        self.free_energy_history.append(free_energy)

    def reset(self) -> None:
        """Clear the free energy history."""
        self.free_energy_history = []

    def select_action(self, observation: np.ndarray) -> int:
        """Select action based on current belief state.

        Parameters
        ----------
        observation : np.ndarray
            Current observation

        Returns
        -------
        int
            Selected action index
        """
        # Free energy of the observation under the current (flat) belief is logged per step
        self.record_step(self.get_free_energy(observation))

        # Simplified action selection
        return int(np.argmax(observation[:5]) if len(observation) >= 5 else 0)

    def get_free_energy(self, observation: Optional[np.ndarray] = None) -> float:
        """Get current free energy.

        Parameters
        ----------
        observation : np.ndarray, optional
            Optional observation to compute free energy from.

        Returns
        -------
        float
            Current free energy value
        """
        if observation is not None:
            return float(np.sum(observation**2))
        return 0.0


class VectorizedAgentPool:
    """Pool of vectorized active inference agents."""

    def __init__(
        self,
        engine: "ActiveInferenceEngine",
        num_agents: int,
        agent_configs: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize vectorized agent pool.

        Parameters
        ----------
        engine : ActiveInferenceEngine
            The engine that will process batch steps.
        num_agents : int
            Number of agents in pool.
        agent_configs : List[dict], optional
            Per-agent configurations. If None, empty configs are used.

        Raises
        ------
        ValueError
            If num_agents doesn't match engine's batch_size.
        """
        batch_size = engine.config.get("system", {}).get("batch_size", 1)
        if num_agents != batch_size:
            raise ValueError(
                f"num_agents ({num_agents}) must match engine batch_size ({batch_size})"
            )

        self.engine = engine
        self.num_agents = num_agents

        if agent_configs is not None and len(agent_configs) != num_agents:
            raise ValueError(
                f"agent_configs length ({len(agent_configs)}) must match num_agents ({num_agents})"
            )

        configs: List[Dict[str, Any]] = (
            agent_configs if agent_configs is not None else [{} for _ in range(num_agents)]
        )
        self.agents = [
            ActiveInferenceAgent(agent_id=i, config=cfg) for i, cfg in enumerate(configs)
        ]

    def step(self, observations: np.ndarray) -> List[Dict[str, Any]]:
        """Step all agents with observations.

        Parameters
        ----------
        observations : np.ndarray
            Observations for all agents (num_agents, obs_dim)

        Returns
        -------
        List[dict]
            Results for each agent with keys: agent_id, free_energy, action, belief_means.
        """
        if observations.shape[0] != self.num_agents:
            raise ValueError(
                f"Observation batch size ({observations.shape[0]}) "
                f"doesn't match pool size ({self.num_agents})"
            )

        actions, info = self.engine.vectorized_batch_step(observations)
        per_agent_fe = info.get("per_agent_free_energy", np.zeros(self.num_agents))

        results = []
        for i, agent in enumerate(self.agents):
            agent.record_step(float(per_agent_fe[i]))
            result = {
                "agent_id": agent.agent_id,
                "free_energy": float(per_agent_fe[i]),
                "action": actions[i] if actions.ndim > 1 else actions,
                "belief_means": info.get("beliefs", [[]])[0] if "beliefs" in info else [],
            }
            results.append(result)
        return results

    async def async_step(self, observations: np.ndarray) -> List[Dict[str, Any]]:
        """Async step all agents with observations.

        Parameters
        ----------
        observations : np.ndarray
            Observations for all agents (num_agents, obs_dim)

        Returns
        -------
        List[dict]
            Results for each agent.
        """
        # For now, run synchronously in async context
        return self.step(observations)

    def reset(self) -> None:
        """Reset all agents' free energy histories."""
        for agent in self.agents:
            agent.reset()


def simulate_active_inference(
    observations: List[np.ndarray],
    agent: ActiveInferenceAgent,
) -> Dict[str, Any]:
    """Simulate active inference process.

    Parameters
    ----------
    observations : List[np.ndarray]
        List of observations
    agent : ActiveInferenceAgent
        Agent to use for simulation

    Returns
    -------
    Dict[str, Any]
        Simulation results
    """
    actions = []
    free_energies = []

    for obs in observations:
        action = agent.select_action(obs)
        free_energy = agent.get_free_energy()
        actions.append(action)
        free_energies.append(free_energy)

    return {
        "actions": actions,
        "free_energies": free_energies,
        "total_free_energy": sum(free_energies),
    }
