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
        Fraction of agents that evacuated (0.0 – 1.0).
    mean_evacuation_time:
        Mean timestep at which evacuated agents reached an exit.
        ``None`` if no agent evacuated.
    min_evacuation_time:
        Earliest evacuation timestep.  ``None`` if no agent evacuated.
    max_evacuation_time:
        Latest evacuation timestep.  ``None`` if no agent evacuated.
    evacuation_times:
        Per-agent evacuation timestep mapping ``{agent_id: timestep}``.
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
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """
    Accumulates per-event data during a simulation run and computes
    aggregate statistics at the end.

    Usage
    -----
    ::

        collector = MetricsCollector(scenario_name="my_scenario", total_agents=10)
        # ... during simulation loop:
        collector.record_evacuation(agent_id="agent_1", timestep=42)
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
        Internal dict mapping agent_id → evacuation timestep.
    """

    def __init__(self, scenario_name: str, total_agents: int) -> None:
        if total_agents < 0:
            raise ValueError("total_agents must be non-negative.")
        self.scenario_name: str = scenario_name
        self.total_agents: int = total_agents
        self._evacuation_times: dict[str, int] = {}

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
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def evacuated_count(self) -> int:
        """Number of agents recorded as evacuated so far."""
        return len(self._evacuation_times)

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
