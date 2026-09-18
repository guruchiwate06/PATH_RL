"""
simulation.py
-------------
Discrete-timestep simulation loop coordinating environment and agents.

Responsibilities
~~~~~~~~~~~~~~~~
- Initialise all components (environment, navigation graph, agents,
  metrics collector) from a ``SimulationConfig``.
- Advance the simulation by one timestep at a time.
- Detect termination conditions (all evacuated or max_timesteps reached).
- Delegate movement decisions to a pluggable ``MovementStrategy``.
- Report events to the ``MetricsCollector``.

Non-responsibilities (intentionally excluded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Pathfinding algorithms           → pathfinding.py (later stage)
- Agent decision-making            → movement strategy (later stage)
- Visualization / rendering        → outside the core package

Design notes
~~~~~~~~~~~~
- The ``MovementStrategy`` protocol defines a single method that the
  simulation calls each timestep.  The foundation ships with a
  ``NullMovementStrategy`` (agents do not move) so that the simulation
  loop can be exercised without any movement logic.
- The simulation exposes a ``run()`` convenience method *and* a
  ``step()`` method so that external drivers (dashboards, debuggers,
  tests) can control the loop granularity.
- Random state is isolated in ``numpy.random.Generator`` (seeded from
  the scenario config) and passed to the strategy; it is never global.
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

import numpy as np

from evacuation_simulation.simulation.agent import Agent, AgentState
from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.metrics import MetricsCollector, SimulationResult
from evacuation_simulation.simulation.pathfinding import NavigationGraph


# ---------------------------------------------------------------------------
# Movement strategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MovementStrategy(Protocol):
    """
    Protocol defining the interface for agent movement logic.

    Implementors decide how each active agent moves on a given timestep.
    The strategy receives the full environment, navigation graph, and
    agent list so it can implement any algorithm without coupling to the
    simulation internals.

    Future concrete implementations might include:
    - ``ShortestPathStrategy``   (greedy BFS / A*)
    - ``PotentialFieldStrategy`` (vector-field following)
    - ``ReinforcementLearningStrategy`` (policy network, far future)
    """

    def move_agents(
        self,
        agents: list[Agent],
        environment: Environment,
        nav_graph: NavigationGraph,
        rng: np.random.Generator,
        timestep: int,
    ) -> None:
        """
        Update agent positions in-place for a single timestep.

        Implementations must not modify *environment* or *nav_graph*.
        Only active (non-evacuated) agents should be considered.
        """
        ...


# ---------------------------------------------------------------------------
# Null movement strategy (foundation stub)
# ---------------------------------------------------------------------------


class NullMovementStrategy:
    """
    A no-op movement strategy used during the foundation stage.

    Agents remain stationary.  This allows the simulation loop to be
    fully exercised without requiring pathfinding or movement logic.
    """

    def move_agents(
        self,
        agents: list[Agent],
        environment: Environment,
        nav_graph: NavigationGraph,
        rng: np.random.Generator,
        timestep: int,
    ) -> None:
        """Do nothing — agents do not move."""
        pass


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------


class SimulationState:
    """
    Snapshot of a simulation at the current timestep.

    Attributes
    ----------
    timestep:
        The current (just-completed) timestep index.
    is_terminated:
        True if the simulation has reached a termination condition.
    termination_reason:
        Human-readable reason for termination, or empty string.
    """

    def __init__(self) -> None:
        self.timestep: int = 0
        self.is_terminated: bool = False
        self.termination_reason: str = ""


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class Simulation:
    """
    Discrete-timestep evacuation simulation.

    Parameters
    ----------
    config:
        Fully-validated scenario configuration.
    movement_strategy:
        Strategy object responsible for moving agents each timestep.
        Defaults to ``NullMovementStrategy``.

    Attributes
    ----------
    config:
        The scenario configuration (read-only after init).
    environment:
        The grid environment.
    nav_graph:
        Navigation graph over traversable cells.
    agents:
        List of all agents (active and evacuated).
    metrics:
        The metrics collector accumulating events during the run.
    state:
        Mutable simulation state (current timestep, termination info).
    _rng:
        NumPy random generator seeded from the scenario config.
    """

    def __init__(
        self,
        config: SimulationConfig,
        movement_strategy: MovementStrategy | None = None,
    ) -> None:
        self.config: SimulationConfig = config
        self.environment: Environment = Environment.from_config(config)
        self.nav_graph: NavigationGraph = NavigationGraph.from_environment(
            self.environment
        )
        self.agents: list[Agent] = [
            Agent.from_config(agent_cfg) for agent_cfg in config.agents
        ]
        self.metrics: MetricsCollector = MetricsCollector(
            scenario_name=config.scenario_name,
            total_agents=len(self.agents),
        )
        self.state: SimulationState = SimulationState()
        self._strategy: MovementStrategy = movement_strategy or NullMovementStrategy()

        # Seeded random generator — isolated, not global
        seed = config.parameters.random_seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------

    def step(self) -> bool:
        """
        Advance the simulation by one discrete timestep.

        Returns
        -------
        bool
            True if the simulation should continue, False if it has
            reached a termination condition.

        Side effects
        ------------
        - Delegates agent movement to the strategy.
        - Checks if any agent is now on an exit cell and marks them
          evacuated.
        - Checks termination conditions.
        - Increments ``state.timestep``.
        """
        if self.state.is_terminated:
            return False

        self.state.timestep += 1
        current_step = self.state.timestep

        # Let the strategy move active agents
        active_agents = [a for a in self.agents if a.is_active]
        self._strategy.move_agents(
            agents=active_agents,
            environment=self.environment,
            nav_graph=self.nav_graph,
            rng=self._rng,
            timestep=current_step,
        )

        # Check for evacuations (agents now on exit cells)
        for agent in active_agents:
            if self.environment.is_exit(agent.row, agent.col):
                agent.mark_evacuated(timestep=current_step)
                self.metrics.record_evacuation(
                    agent_id=agent.agent_id, timestep=current_step
                )

        # Check termination conditions
        return self._check_termination()

    def run(self) -> SimulationResult:
        """
        Run the simulation to completion.

        The loop continues until all agents have evacuated or the
        maximum number of timesteps is reached.

        Returns
        -------
        SimulationResult
            Aggregated outcome statistics.
        """
        max_steps = self.config.parameters.max_timesteps
        while not self.state.is_terminated:
            should_continue = self.step()
            if not should_continue:
                break
            if self.state.timestep >= max_steps:
                self.state.is_terminated = True
                self.state.termination_reason = (
                    f"Max timesteps ({max_steps}) reached."
                )
                break

        return self.metrics.compute_result(
            total_timesteps=self.state.timestep
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_termination(self) -> bool:
        """
        Check termination conditions and update ``state`` accordingly.

        Returns True if the simulation should continue, False if done.
        """
        if self.metrics.is_fully_evacuated():
            self.state.is_terminated = True
            self.state.termination_reason = "All agents evacuated."
            return False

        max_steps = self.config.parameters.max_timesteps
        if self.state.timestep >= max_steps:
            self.state.is_terminated = True
            self.state.termination_reason = (
                f"Max timesteps ({max_steps}) reached."
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def active_agents(self) -> list[Agent]:
        """Return agents that have not yet evacuated."""
        return [a for a in self.agents if a.is_active]

    @property
    def evacuated_agents(self) -> list[Agent]:
        """Return agents that have evacuated."""
        return [a for a in self.agents if not a.is_active]

    def __repr__(self) -> str:
        return (
            f"Simulation("
            f"scenario={self.config.scenario_name!r}, "
            f"timestep={self.state.timestep}, "
            f"agents={len(self.agents)}, "
            f"evacuated={len(self.evacuated_agents)}, "
            f"terminated={self.state.is_terminated})"
        )
