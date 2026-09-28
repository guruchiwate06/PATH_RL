"""
simulation.py
-------------
Discrete-timestep simulation loop coordinating environment and agents.

Responsibilities
~~~~~~~~~~~~~~~~
- Initialise all components (environment, navigation graph, agents,
  occupancy map, metrics collector) from a ``SimulationConfig``.
- Advance the simulation by one timestep at a time.
- Coordinate the full movement pipeline each step:
    1. Rebuild occupancy from active agent positions.
    2. Collect movement requests from the strategy.
    3. Validate requests against the environment.
    4. Compute vacating cells (agents leaving their current cell).
    5. Resolve conflicts with capacity constraints (deterministic).
    6. Apply approved movements.
    7. Rebuild occupancy after movements.
    8. Detect and record evacuations.
    9. Record waiting and congestion to the metrics collector.
   10. Store a per-timestep trace for inspection.
   11. Check termination conditions.
- Report events to the ``MetricsCollector``.

Non-responsibilities (intentionally excluded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Pathfinding algorithms       -> pathfinding.py
- Movement decision-making     -> strategy.py
- Request validation/conflicts -> movement.py
- Occupancy tracking           -> occupancy.py
- Visualization / rendering    -> outside the core package

Design notes
~~~~~~~~~~~~
- The ``MovementStrategy`` protocol defines a ``move_agents`` method that
  returns a ``dict[str, GridCell]`` (agent_id -> requested next cell).
  Strategies must NOT mutate agents directly -- the simulation owns all
  state changes.  This makes strategies independently testable.
- ``NullMovementStrategy`` returns an empty dict (agents stay put).
- Random state is isolated in a seeded ``numpy.random.Generator`` and
  passed to the strategy; it is never global.
- The simulation exposes both ``step()`` (single timestep) and ``run()``
  (run to completion) so external drivers can control granularity.
- ``last_step_trace`` holds structured information about the most
  recently completed timestep for inspection and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from evacuation_simulation.simulation.agent import Agent, AgentState
from evacuation_simulation.simulation.config import GridCell, SimulationConfig
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.metrics import MetricsCollector, SimulationResult
from evacuation_simulation.simulation.movement import (
    MovementRequest,
    RejectionReason,
    apply_movements,
    resolve_conflicts,
    validate_request,
)
from evacuation_simulation.simulation.occupancy import OccupancyMap
from evacuation_simulation.simulation.pathfinding import NavigationGraph


# ---------------------------------------------------------------------------
# Timestep trace dataclasses (Stage 3)
# ---------------------------------------------------------------------------


@dataclass
class AgentStepResult:
    """
    Record of a single agent's movement attempt for one timestep.

    Attributes
    ----------
    agent_id : str
        Agent identifier.
    position_before : GridCell
        Agent's position at the start of the timestep.
    requested : Optional[GridCell]
        The cell the agent tried to move to, or ``None`` if no request
        was generated.
    position_after : GridCell
        Agent's position at the end of the timestep.
    result : str
        One of: ``"APPROVED"``, ``"REJECTED_CAPACITY"``,
        ``"REJECTED_CONFLICT"``, ``"NO_REQUEST"``, ``"EVACUATED"``.
    rejection_reason : Optional[RejectionReason]
        The rejection reason enum value if rejected, else ``None``.
    """

    agent_id: str
    position_before: GridCell
    requested: Optional[GridCell]
    position_after: GridCell
    result: str
    rejection_reason: Optional[RejectionReason] = None


@dataclass
class TimestepTrace:
    """
    Structured snapshot of everything that happened in one timestep.

    Attributes
    ----------
    timestep : int
        The timestep number.
    agent_results : list[AgentStepResult]
        Per-agent movement outcomes.
    occupancy_before : dict[GridCell, int]
        Occupancy map at the start of the timestep (pre-move).
    occupancy_after : dict[GridCell, int]
        Occupancy map at the end of the timestep (post-move).
    capacity : int
        Uniform cell capacity for this run.
    congested_cells_after : dict[GridCell, float]
        Cells that were congested after this timestep, with ratios.
    """

    timestep: int
    agent_results: list[AgentStepResult] = field(default_factory=list)
    occupancy_before: dict[GridCell, int] = field(default_factory=dict)
    occupancy_after: dict[GridCell, int] = field(default_factory=dict)
    capacity: int = 1
    congested_cells_after: dict[GridCell, float] = field(default_factory=dict)

    def format_text(self) -> str:
        """
        Return a human-readable text report of this timestep.

        Example output::

            Timestep 5
            ----------
            Agent a0:  (4,3) -> (4,4)   APPROVED
            Agent a1:  (4,2) -> (4,3)   REJECTED_CAPACITY
            Occupancy (post-move):
              (4,3): 0/1
              (4,4): 1/1   [CONGESTED]
        """
        lines = [f"Timestep {self.timestep}", "-" * 20]
        for ar in self.agent_results:
            req_str = f"-> {ar.requested}" if ar.requested else "(no request)"
            lines.append(
                f"  Agent {ar.agent_id}:  {ar.position_before} {req_str}  "
                f"{ar.result}"
            )
        lines.append("Occupancy (post-move):")
        all_cells = set(self.occupancy_before) | set(self.occupancy_after)
        for cell in sorted(all_cells):
            occ = self.occupancy_after.get(cell, 0)
            cap = self.capacity
            congested = "  [CONGESTED]" if occ >= cap else ""
            lines.append(f"  {cell}: {occ}/{cap}{congested}")
        return "\n".join(lines)


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
    occupancy -> collect -> validate -> resolve (capacity) -> apply ->
    occupancy (post) -> evacuate -> metrics -> check termination.

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
    occupancy : OccupancyMap
        Dynamic occupancy tracker for the current timestep.
    metrics : MetricsCollector
        Accumulates evacuation events during the run.
    state : SimulationState
        Mutable simulation state (timestep, termination info).
    last_step_trace : Optional[TimestepTrace]
        Structured trace of the most recently completed timestep.
        ``None`` before the first step.
    _rng : np.random.Generator
        Seeded random generator, passed to the strategy each step.
    """

    def __init__(
        self,
        config: SimulationConfig,
        movement_strategy: MovementStrategy | None = None,
    ) -> None:
        self.config: SimulationConfig = config
        self.environment: Environment = Environment.from_floor_plan_and_exits(
            config.floor_plan, config.exit_configuration
        )
        self.nav_graph: NavigationGraph = NavigationGraph.from_environment(
            self.environment
        )
        self.agents: list[Agent] = [
            Agent.from_config(agent_cfg) for agent_cfg in config.occupants.agents
        ]
        # Stage 3: dynamic occupancy map (derived from agent positions)
        self.occupancy: OccupancyMap = OccupancyMap(
            default_cell_capacity=config.parameters.default_cell_capacity
        )
        # Initialise occupancy from starting positions
        self.occupancy.rebuild(self.agents)

        self.metrics: MetricsCollector = MetricsCollector(
            scenario_name=config.scenario_name,
            total_agents=len(self.agents),
        )
        self.state: SimulationState = SimulationState()
        self._strategy: MovementStrategy = movement_strategy or NullMovementStrategy()
        self.last_step_trace: Optional[TimestepTrace] = None

        # Seeded random generator -- isolated, never global
        seed = config.parameters.random_seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------

    def step(self) -> bool:
        """
        Advance the simulation by one discrete timestep.

        Each call executes the full coordinated movement pipeline:

        1.  Rebuild occupancy from active agent positions (pre-move).
        2.  Collect movement requests from the strategy (dict).
        3.  Build ``MovementRequest`` objects from the dict.
        4.  Validate each request against the environment.
        5.  Compute the set of cells being vacated this timestep.
        6.  Resolve conflicts + capacity constraints deterministically.
        7.  Apply approved movements (update agent positions).
        8.  Rebuild occupancy from agent positions (post-move).
        9.  Detect agents now on exit cells and mark them evacuated.
        10. Record evacuations, waiting steps, and occupancy in metrics.
        11. Store a ``TimestepTrace`` for external inspection.
        12. Check termination conditions.

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

        # --- 1. Rebuild occupancy (pre-move) --------------------------------
        active_agents = [a for a in self.agents if a.is_active]
        self.occupancy.rebuild(active_agents)
        occupancy_before = self.occupancy.get_occupied_cells()

        agents_by_id: dict[str, Agent] = {
            a.agent_id: a for a in active_agents
        }
        positions_before: dict[str, GridCell] = {
            a.agent_id: a.position for a in active_agents
        }

        # --- 2. Collect raw movement requests from strategy -----------------
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

        # --- 3 & 4. Build and validate MovementRequest objects --------------
        valid_requests: list[MovementRequest] = []
        invalid_agent_ids: set[str] = set()
        reaction_delayed_ids: set[str] = set()
        speed_delayed_ids: set[str] = set()

        for agent_id, to_cell in raw_requests.items():
            if agent_id not in agents_by_id:
                continue
            agent = agents_by_id[agent_id]

            # Stage 6: Profile eligibility check (reaction delay & speed)
            if agent.profile.is_reaction_delayed(current_step):
                reaction_delayed_ids.add(agent_id)
                continue
            if not agent.profile.is_speed_eligible(current_step):
                speed_delayed_ids.add(agent_id)
                continue

            req = MovementRequest(
                agent_id=agent_id,
                from_cell=agent.position,
                to_cell=to_cell,
            )
            if validate_request(req, self.environment):
                valid_requests.append(req)
            else:
                invalid_agent_ids.add(agent_id)

        # --- 5. Compute vacating cells (still tracked for trace, not used in resolution) -
        # Kept for possible future use; not passed to resolve_conflicts because
        # vacating credit leads to over-occupancy when the vacating agent is itself
        # rejected at its destination.

        # --- 6. Resolve conflicts + capacity constraints --------------------
        approved, rejected, rejection_reasons = resolve_conflicts(
            valid_requests,
            capacity_fn=self.occupancy.get_capacity,
            current_occupancy=occupancy_before,
        )

        # --- 7. Apply approved movements ------------------------------------
        apply_movements(approved, agents_by_id)

        # --- 8. Detect evacuations and record -------------------------------
        for agent in active_agents:
            if self.environment.is_exit(agent.row, agent.col):
                agent.mark_evacuated(timestep=current_step)
                self.metrics.record_evacuation(
                    agent_id=agent.agent_id,
                    timestep=current_step,
                    exit_cell=(agent.row, agent.col),  # Stage 7: track which exit was used
                )

        # --- 9. Rebuild occupancy (post-move, post-evacuation) ---------------
        self.occupancy.rebuild(self.agents)
        occupancy_after = self.occupancy.get_occupied_cells()

        # --- 10. Record waiting and occupancy in metrics --------------------
        # Waiting: only capacity-blocked agents count (not conflict losers)
        capacity_waiting_ids: set[str] = {
            agent_id
            for agent_id, reason in rejection_reasons.items()
            if reason == RejectionReason.DESTINATION_CAPACITY
        }
        if capacity_waiting_ids:
            self.metrics.record_step_waiting(
                waiting_agent_ids=capacity_waiting_ids,
                timestep=current_step,
            )
        # Occupancy snapshot (post-move, excluding just-evacuated)
        self.metrics.record_step_occupancy(
            occupancy_snapshot=occupancy_after,
            capacity=self.occupancy.default_capacity,
        )

        # --- 11. Build timestep trace for inspection ------------------------
        approved_ids = {req.agent_id for req in approved}

        agent_results: list[AgentStepResult] = []
        for agent in active_agents:
            aid = agent.agent_id
            requested = raw_requests.get(aid)
            pos_after = agent.position
            # Determine position_before from agents_by_id snapshot
            ag_before = agents_by_id[aid]

            if agent.state == AgentState.EVACUATED:
                result_str = "EVACUATED"
                rej_reason = None
            elif aid in approved_ids:
                result_str = "APPROVED"
                rej_reason = None
            elif aid in rejection_reasons:
                rr = rejection_reasons[aid]
                result_str = (
                    "REJECTED_CAPACITY"
                    if rr == RejectionReason.DESTINATION_CAPACITY
                    else "REJECTED_CONFLICT"
                )
                rej_reason = rr
            elif aid in invalid_agent_ids:
                result_str = "INVALID_REQUEST"
                rej_reason = None
            elif aid in reaction_delayed_ids:
                result_str = "REACTION_DELAY"
                rej_reason = None
            elif aid in speed_delayed_ids:
                result_str = "SPEED_DELAY"
                rej_reason = None
            else:
                result_str = "NO_REQUEST"
                rej_reason = None

            agent_results.append(
                AgentStepResult(
                    agent_id=aid,
                    position_before=positions_before[aid],
                    requested=requested,
                    position_after=pos_after,
                    result=result_str,
                    rejection_reason=rej_reason,
                )
            )

        self.last_step_trace = TimestepTrace(
            timestep=current_step,
            agent_results=agent_results,
            occupancy_before=occupancy_before,
            occupancy_after=occupancy_after,
            capacity=self.occupancy.default_capacity,
            congested_cells_after=self.occupancy.congested_cells(),
        )

        # --- 12. Check termination ------------------------------------------
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

    def get_agent_planned_path(self, agent_id: str) -> Optional[list[GridCell]]:
        """
        Return the planned shortest path for *agent_id* if available.

        Returns ``None`` if the agent does not exist, is evacuated, or no
        exit is reachable.
        """
        agent = next((a for a in self.agents if a.agent_id == agent_id), None)
        if agent is None or not agent.is_active:
            return None
        if hasattr(self._strategy, "get_planned_path"):
            return self._strategy.get_planned_path(
                agent, self.environment, self.nav_graph
            )
        return None

    def get_agent_info(self, agent_id: str) -> Optional[dict]:
        """
        Return inspection metadata for *agent_id*.

        Returns a dictionary containing:
        - ``agent_id``: str
        - ``position``: (row, col)
        - ``state``: "MOVING", "WAITING", or "EVACUATED"
        - ``evacuation_timestep``: int or None
        - ``waiting_steps``: int
        - ``target_exit``: (row, col) or None
        - ``path_length``: int or None
        - ``planned_path``: list of GridCell or None
        - ``last_result``: str or None
        - ``last_rejection_reason``: str or None
        """
        agent = next((a for a in self.agents if a.agent_id == agent_id), None)
        if agent is None:
            return None

        last_result = None
        last_reason = None
        if self.last_step_trace is not None:
            for ar in self.last_step_trace.agent_results:
                if ar.agent_id == agent_id:
                    last_result = ar.result
                    last_reason = (
                        ar.rejection_reason.name if ar.rejection_reason else None
                    )
                    break

        if not agent.is_active:
            state_str = "EVACUATED"
        elif last_result in ("REJECTED_CAPACITY", "REJECTED_CONFLICT"):
            state_str = "WAITING"
        else:
            state_str = "MOVING"

        path = self.get_agent_planned_path(agent_id)
        target_exit = path[-1] if path and len(path) > 0 else None
        path_length = len(path) - 1 if path and len(path) > 1 else (0 if path else None)
        waiting_steps = self.metrics.get_agent_waiting_steps(agent_id)

        return {
            "agent_id": agent.agent_id,
            "position": agent.position,
            "state": state_str,
            "evacuation_timestep": agent.evacuation_timestep,
            "waiting_steps": waiting_steps,
            "speed": agent.speed,
            "reaction_delay": agent.reaction_delay,
            "profile": agent.profile.to_dict(),
            "target_exit": target_exit,
            "path_length": path_length,
            "planned_path": path,
            "last_result": last_result,
            "last_rejection_reason": last_reason,
        }

    def format_last_trace(self) -> str:
        """
        Return a human-readable text report of the last completed timestep.

        Returns an empty string if no step has been taken yet.
        """
        if self.last_step_trace is None:
            return "(no steps taken yet)"
        return self.last_step_trace.format_text()

    def __repr__(self) -> str:
        return (
            f"Simulation("
            f"scenario={self.config.scenario_name!r}, "
            f"timestep={self.state.timestep}, "
            f"agents={len(self.agents)}, "
            f"evacuated={len(self.evacuated_agents)}, "
            f"terminated={self.state.is_terminated})"
        )
