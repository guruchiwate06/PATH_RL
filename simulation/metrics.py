"""
metrics.py
----------
Collects and computes outcome statistics from a simulation run.

Responsibilities
~~~~~~~~~~~~~~~~
- Accept events from the simulation (agent evacuated, timestep elapsed).
- Compute aggregate statistics at the end of a run.
- Return structured results (a plain dataclass — no Pydantic needed here
  because this is output-only, not user-supplied configuration).

Non-responsibilities (intentionally excluded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Simulation control                → simulation.py
- Visualization / plotting          → outside the core package
- Database writes / file I/O        → caller's responsibility

Design notes
~~~~~~~~~~~~
- ``MetricsCollector`` is stateful — one instance per simulation run.
- All data is stored in simple Python structures to keep dependencies
  minimal and serialisation trivial.
- The ``SimulationResult`` dataclass is the canonical output of a run;
  downstream code (reports, visualizations) should depend on it rather
  than on the collector directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """
    Immutable summary of a completed simulation run.

    Attributes
    ----------
    scenario_name:
        Label carried forward from the scenario configuration.
    total_agents:
        Number of agents that participated in the simulation.
    evacuated_count:
        Number of agents that reached an exit.
    non_evacuated_count:
        Number of agents that did NOT reach an exit by the time the
        simulation ended.
    total_timesteps:
        Number of timesteps the simulation executed.
    evacuation_rate:
        Fraction of agents that evacuated (0.0 - 1.0).
    mean_evacuation_time:
        Mean timestep at which evacuated agents reached an exit.
        ``None`` if no agent evacuated.
    min_evacuation_time:
        Earliest evacuation timestep.  ``None`` if no agent evacuated.
    max_evacuation_time:
        Latest evacuation timestep.  ``None`` if no agent evacuated.
    evacuation_times:
        Per-agent evacuation timestep mapping ``{agent_id: timestep}``.

    -- Stage 3: Waiting statistics --

    total_waiting_steps:
        Sum of capacity-caused waiting steps across all agents.  A
        "waiting step" is a timestep where an active agent submitted a
        valid movement request that was rejected because the destination
        cell had no free capacity slots (``RejectionReason.DESTINATION_CAPACITY``).
    mean_waiting_steps:
        ``total_waiting_steps / total_agents``.  0.0 if no agents.
    max_waiting_steps:
        Maximum number of waiting steps recorded for any single agent.
    per_agent_waiting_steps:
        Per-agent waiting step counts ``{agent_id: steps}``.

    -- Stage 3: Congestion statistics --

    max_cell_occupancy:
        Highest single-cell occupancy observed at the end of any timestep.
    max_occupancy_ratio:
        ``max_cell_occupancy / cell_capacity`` -- the highest occupancy
        ratio observed.  A ratio of 1.0 means at least one cell was
        completely full at some point.
    congested_cell_steps:
        Cumulative count of (cell, timestep) pairs where a cell was
        congested (occupancy >= capacity) at the end of a timestep.
        Useful for quantifying how much congestion existed overall.
    """

    scenario_name: str
    total_agents: int
    evacuated_count: int
    non_evacuated_count: int
    total_timesteps: int
    evacuation_rate: float
    mean_evacuation_time: Optional[float]
    min_evacuation_time: Optional[int]
    max_evacuation_time: Optional[int]
    evacuation_times: dict[str, int] = field(default_factory=dict)

    # -- Stage 3: Waiting --
    total_waiting_steps: int = 0
    mean_waiting_steps: float = 0.0
    max_waiting_steps: int = 0
    per_agent_waiting_steps: dict[str, int] = field(default_factory=dict)

    # -- Stage 3: Congestion --
    max_cell_occupancy: int = 0
    max_occupancy_ratio: float = 0.0
    congested_cell_steps: int = 0

    def __str__(self) -> str:
        lines = [
            f"=== Simulation Result: {self.scenario_name} ===",
            f"  Agents           : {self.total_agents}",
            f"  Evacuated        : {self.evacuated_count} / {self.total_agents}",
            f"  Evacuation rate  : {self.evacuation_rate:.1%}",
            f"  Timesteps run    : {self.total_timesteps}",
        ]
        if self.mean_evacuation_time is not None:
            lines += [
                f"  Mean evac. time  : {self.mean_evacuation_time:.1f}",
                f"  Min  evac. time  : {self.min_evacuation_time}",
                f"  Max  evac. time  : {self.max_evacuation_time}",
            ]
        else:
            lines.append("  No agents evacuated.")
        lines += [
            f"  Total wait steps : {self.total_waiting_steps}",
            f"  Mean  wait steps : {self.mean_waiting_steps:.2f}",
            f"  Max   wait steps : {self.max_waiting_steps}",
            f"  Max cell occupancy: {self.max_cell_occupancy}",
            f"  Max occupancy ratio: {self.max_occupancy_ratio:.2f}",
            f"  Congested cell-steps: {self.congested_cell_steps}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """
    Accumulates per-event data during a simulation run and computes
    aggregate statistics at the end.

    Usage
    ----:
    ::

        collector = MetricsCollector(scenario_name="my_scenario", total_agents=10)
        # ... during simulation loop:
        collector.record_evacuation(agent_id="agent_1", timestep=42)
        collector.record_step_waiting(waiting_agent_ids={"agent_2"}, timestep=3)
        collector.record_step_occupancy({(1, 2): 1, (3, 4): 2}, capacity=1)
        # ... at the end:
        result = collector.compute_result(total_timesteps=200)
        print(result)

    Attributes
    ----------
    scenario_name:
        Carried from the scenario config for labelling results.
    total_agents:
        Total agent count registered at initialisation.
    _evacuation_times:
        Internal dict mapping agent_id -> evacuation timestep.
    _waiting_steps:
        Internal dict mapping agent_id -> cumulative capacity-caused
        waiting steps (``RejectionReason.DESTINATION_CAPACITY`` rejections).
    _max_cell_occupancy:
        Running maximum single-cell occupancy seen so far.
    _max_occupancy_ratio:
        Running maximum occupancy ratio seen so far.
    _congested_cell_steps:
        Running total of (cell, timestep) pairs where occupancy >= capacity.
    """

    def __init__(self, scenario_name: str, total_agents: int) -> None:
        if total_agents < 0:
            raise ValueError("total_agents must be non-negative.")
        self.scenario_name: str = scenario_name
        self.total_agents: int = total_agents
        self._evacuation_times: dict[str, int] = {}
        # Stage 3: waiting
        self._waiting_steps: dict[str, int] = {}
        # Stage 3: congestion
        self._max_cell_occupancy: int = 0
        self._max_occupancy_ratio: float = 0.0
        self._congested_cell_steps: int = 0

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record_evacuation(self, agent_id: str, timestep: int) -> None:
        """
        Record that *agent_id* evacuated at *timestep*.

        Parameters
        ----------
        agent_id:
            The ID of the evacuated agent.
        timestep:
            The simulation timestep at which evacuation occurred.

        Raises
        ------
        ValueError
            If *agent_id* has already been recorded.
        """
        if agent_id in self._evacuation_times:
            raise ValueError(
                f"Evacuation for agent '{agent_id}' has already been recorded."
            )
        self._evacuation_times[agent_id] = timestep

    def record_step_waiting(
        self,
        waiting_agent_ids: set[str],
        timestep: int,  # noqa: ARG002  (reserved for future per-timestep analysis)
    ) -> None:
        """
        Record which agents experienced a capacity-caused wait this timestep.

        A "waiting step" is defined as a timestep where an active agent
        submitted a valid movement request that was rejected due to
        insufficient destination capacity (``RejectionReason.DESTINATION_CAPACITY``).
        Conflict-rejected agents (``RejectionReason.DESTINATION_CONFLICT``) are
        NOT counted as waiting -- they were outcompeted, not capacity-blocked.

        Parameters
        ----------
        waiting_agent_ids : set[str]
            Agent IDs whose requests were rejected this step with reason
            ``DESTINATION_CAPACITY``.
        timestep : int
            The current simulation timestep (reserved for future use).
        """
        for agent_id in waiting_agent_ids:
            self._waiting_steps[agent_id] = self._waiting_steps.get(agent_id, 0) + 1

    def record_step_occupancy(
        self,
        occupancy_snapshot: dict,
        capacity: int,
    ) -> None:
        """
        Record the post-move occupancy state for one timestep.

        Updates running maximums and congestion counters.

        Parameters
        ----------
        occupancy_snapshot : dict[GridCell, int]
            Mapping ``{cell: occupancy}`` for all occupied cells at the
            end of this timestep.
        capacity : int
            The uniform cell capacity for this run.
        """
        for cell, occ in occupancy_snapshot.items():
            ratio = occ / capacity if capacity > 0 else 0.0
            if occ > self._max_cell_occupancy:
                self._max_cell_occupancy = occ
            if ratio > self._max_occupancy_ratio:
                self._max_occupancy_ratio = ratio
            if occ >= capacity:
                self._congested_cell_steps += 1

    # ------------------------------------------------------------------
    # Result computation
    # ------------------------------------------------------------------

    def compute_result(self, total_timesteps: int) -> SimulationResult:
        """
        Compute and return the final ``SimulationResult``.

        Parameters
        ----------
        total_timesteps:
            The number of timesteps the simulation executed (used for
            context in the result; does not affect calculations).

        Returns
        -------
        SimulationResult
        """
        evacuated_count = len(self._evacuation_times)
        non_evacuated_count = self.total_agents - evacuated_count
        evacuation_rate = (
            evacuated_count / self.total_agents if self.total_agents > 0 else 0.0
        )

        evac_times = list(self._evacuation_times.values())

        mean_time: Optional[float] = (
            sum(evac_times) / len(evac_times) if evac_times else None
        )
        min_time: Optional[int] = min(evac_times) if evac_times else None
        max_time: Optional[int] = max(evac_times) if evac_times else None

        # Stage 3: waiting aggregates
        total_waiting = sum(self._waiting_steps.values())
        mean_waiting = total_waiting / self.total_agents if self.total_agents > 0 else 0.0
        max_waiting = max(self._waiting_steps.values()) if self._waiting_steps else 0

        return SimulationResult(
            scenario_name=self.scenario_name,
            total_agents=self.total_agents,
            evacuated_count=evacuated_count,
            non_evacuated_count=non_evacuated_count,
            total_timesteps=total_timesteps,
            evacuation_rate=evacuation_rate,
            mean_evacuation_time=mean_time,
            min_evacuation_time=min_time,
            max_evacuation_time=max_time,
            evacuation_times=dict(self._evacuation_times),
            # Stage 3
            total_waiting_steps=total_waiting,
            mean_waiting_steps=mean_waiting,
            max_waiting_steps=max_waiting,
            per_agent_waiting_steps=dict(self._waiting_steps),
            max_cell_occupancy=self._max_cell_occupancy,
            max_occupancy_ratio=self._max_occupancy_ratio,
            congested_cell_steps=self._congested_cell_steps,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def evacuated_count(self) -> int:
        """Number of agents recorded as evacuated so far."""
        return len(self._evacuation_times)

    def get_agent_waiting_steps(self, agent_id: str) -> int:
        """Return cumulative waiting steps for a specific agent."""
        return self._waiting_steps.get(agent_id, 0)

    def is_fully_evacuated(self) -> bool:
        """Return True if all agents have evacuated."""
        return self.evacuated_count == self.total_agents

    def __repr__(self) -> str:
        return (
            f"MetricsCollector("
            f"scenario={self.scenario_name!r}, "
            f"agents={self.total_agents}, "
            f"evacuated={self.evacuated_count})"
        )
