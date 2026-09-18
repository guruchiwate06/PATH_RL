"""
simulation.py
-------------
Discrete-timestep simulation loop coordinating environment and agents.

Responsibilities
~~~~~~~~~~~~~~~~
- Initialise all components (environment, navigation graph, agents,
  metrics collector) from a ``SimulationConfig``.
- Advance the simulation by one timestep at a time.
- Coordinate the full movement pipeline each step:
    1. Collect movement requests from the strategy.
    2. Validate requests against the environment.
    3. Resolve conflicts deterministically.
    4. Apply approved movements.
    5. Detect and record evacuations.
    6. Check termination conditions.
- Report events to the ``MetricsCollector``.

Non-responsibilities (intentionally excluded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Pathfinding algorithms       → pathfinding.py
- Movement decision-making     → strategy.py
- Request validation/conflicts → movement.py
- Visualization / rendering    → outside the core package

Design notes
~~~~~~~~~~~~
- The ``MovementStrategy`` protocol defines a ``move_agents`` method that
  returns a ``dict[str, GridCell]`` (agent_id → requested next cell).
  Strategies must NOT mutate agents directly — the simulation owns all
  state changes.  This makes strategies independently testable.
- ``NullMovementStrategy`` returns an empty dict (agents stay put).
- Random state is isolated in a seeded ``numpy.random.Generator`` and
  passed to the strategy; it is never global.
- The simulation exposes both ``step()`` (single timestep) and ``run()``
  (run to completion) so external drivers can control granularity.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from evacuation_simulation.simulation.agent import Agent, AgentState
from evacuation_simulation.simulation.config import GridCell, SimulationConfig
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.metrics import MetricsCollector, SimulationResult
from evacuation_simulation.simulation.movement import (
    MovementRequest,
    apply_movements,
    resolve_conflicts,
    validate_request,
)
from evacuation_simulation.simulation.pathfinding import NavigationGraph


# ---------------------------------------------------------------------------
# Movement strategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MovementStrategy(Protocol):
    """
    Protocol defining the interface for agent movement strategies.

    A strategy receives the full environment, navigation graph, and active
    agent list for the current timestep, and returns a mapping of
    ``agent_id → requested_next_cell``.

    **Strategies must not mutate agents, the environment, or the graph.**
    All state changes are applied by the simulation layer after conflict
    resolution.

    Returning an empty dict is valid (e.g. ``NullMovementStrategy``).

    Future concrete implementations might include:
    - ``ShortestPathStrategy``        (greedy BFS — Stage 2)
    - ``PotentialFieldStrategy``      (vector-field following)
    - ``ReinforcementLearningStrategy`` (policy network, far future)
    """

    def move_agents(
        self,
        agents: list[Agent],
        environment: Environment,
        nav_graph: NavigationGraph,
        rng: np.random.Generator,
        timestep: int,
    ) -> dict[str, GridCell]:
        """
        Compute movement requests for active agents.

        Parameters
        ----------
        agents : list[Agent]
            Active (non-evacuated) agents for this timestep.
        environment : Environment
            The grid environment (read-only).
        nav_graph : NavigationGraph
            Navigation graph (read-only).
        rng : np.random.Generator
            Seeded random generator for stochastic strategies.
        timestep : int
            Current simulation timestep.

        Returns
        -------
        dict[str, GridCell]
            Mapping from agent_id to the requested next cell.
            Absent agents are not requesting a move.
        """
        ...


# ---------------------------------------------------------------------------
# Null movement strategy
# ---------------------------------------------------------------------------


class NullMovementStrategy:
    """
    A no-op movement strategy.

    Agents produce no movement requests and remain stationary.  Useful
    for testing the simulation loop independently of movement logic, and
    as the default strategy in the foundation stage.
    """

    def move_agents(
        self,
        agents: list[Agent],
        environment: Environment,
        nav_graph: NavigationGraph,
        rng: np.random.Generator,
        timestep: int,
    ) -> dict[str, GridCell]:
        """Return an empty dict — no movement requests."""
        return {}


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------


class SimulationState:
    """
    Mutable snapshot of a simulation's current progress.

    Attributes
    ----------
    timestep : int
        The index of the most recently completed timestep (0 = not started).
    is_terminated : bool
        True once a termination condition has been reached.
    termination_reason : str
        Human-readable reason for termination, or empty string if still
        running.
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

    Coordinates the full movement pipeline each timestep:
    collect → validate → resolve → apply → evacuate → check termination.

    Parameters
    ----------
    config : SimulationConfig
        Fully-validated scenario configuration.
    movement_strategy : MovementStrategy, optional
        Strategy responsible for generating movement requests each
        timestep.  Defaults to ``NullMovementStrategy``.

    Attributes
    ----------
    config : SimulationConfig
        Scenario configuration (read-only after construction).
    environment : Environment
        The grid environment.
    nav_graph : NavigationGraph
        Navigation graph over traversable cells.
    agents : list[Agent]
        All agents (active and evacuated).
    metrics : MetricsCollector
        Accumulates evacuation events during the run.
    state : SimulationState
        Mutable simulation state (timestep, termination info).
    _rng : np.random.Generator
        Seeded random generator, passed to the strategy each step.
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

        # Seeded random generator — isolated, never global
        seed = config.parameters.random_seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------

    def step(self) -> bool:
        """
        Advance the simulation by one discrete timestep.

        Each call executes the full coordinated movement pipeline:

        1. Collect movement requests from the strategy (dict).
        2. Build ``MovementRequest`` objects from the dict.
        3. Validate each request against the environment.
        4. Resolve conflicts deterministically (lex-first agent_id wins).
        5. Apply approved movements (update agent positions).
        6. Detect agents now on exit cells and mark them evacuated.
        7. Record evacuations in the metrics collector.
        8. Check termination conditions.

        Returns
        -------
        bool
            ``True`` if the simulation should continue running.
            ``False`` if a termination condition has been reached.
        """
        if self.state.is_terminated:
            return False

        self.state.timestep += 1
        current_step = self.state.timestep

        # --- 1. Identify active agents ------------------------------------
        active_agents = [a for a in self.agents if a.is_active]
        agents_by_id: dict[str, Agent] = {
            a.agent_id: a for a in active_agents
        }

        # --- 2. Collect raw movement requests from strategy ---------------
        raw_requests: dict[str, GridCell] = (
            self._strategy.move_agents(
                agents=active_agents,
                environment=self.environment,
                nav_graph=self.nav_graph,
                rng=self._rng,
                timestep=current_step,
            )
            or {}  # guard against strategies that return None
        )

        # --- 3. Build and validate MovementRequest objects ----------------
        valid_requests: list[MovementRequest] = []
        for agent_id, to_cell in raw_requests.items():
            if agent_id not in agents_by_id:
                continue
            agent = agents_by_id[agent_id]
            req = MovementRequest(
                agent_id=agent_id,
                from_cell=agent.position,
                to_cell=to_cell,
            )
            if validate_request(req, self.environment):
                valid_requests.append(req)

        # --- 4. Resolve conflicts deterministically -----------------------
        approved, _rejected = resolve_conflicts(valid_requests)

        # --- 5. Apply approved movements ----------------------------------
        apply_movements(approved, agents_by_id)

        # --- 6 & 7. Detect evacuations and record -------------------------
        for agent in active_agents:
            if self.environment.is_exit(agent.row, agent.col):
                agent.mark_evacuated(timestep=current_step)
                self.metrics.record_evacuation(
                    agent_id=agent.agent_id, timestep=current_step
                )

        # --- 8. Check termination -----------------------------------------
        return self._check_termination()

    def run(self) -> SimulationResult:
        """
        Run the simulation to completion.

        Continues stepping until all agents have evacuated or the
        maximum timestep limit is reached.

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
        Evaluate termination conditions and update ``state``.

        Returns
        -------
        bool
            ``True`` if the simulation should continue, ``False`` if done.
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
