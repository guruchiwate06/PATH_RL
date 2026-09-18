"""
agent.py
--------
Agent entity representing a single building occupant.

Responsibilities
~~~~~~~~~~~~~~~~
- Encapsulate agent identity (ID, name).
- Track current grid position as a mutable (row, col) pair.
- Track lifecycle state (WAITING, MOVING, EVACUATED).
- Record the timestep at which evacuation occurred.

Non-responsibilities (intentionally excluded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Pathfinding / movement decisions → pathfinding.py + simulation.py
- Visualization                    → outside the core package
- Inter-agent communication        → future stage

Design notes
~~~~~~~~~~~~
- ``AgentState`` is an ``IntEnum`` so that state comparisons are readable
  *and* NumPy-friendly (states can be stored in integer arrays later).
- ``Agent`` uses ``__slots__`` to minimise per-instance memory overhead,
  important when hundreds of agents are simulated.
- All mutation of agent state happens through named methods rather than
  direct attribute assignment, making call sites self-documenting and
  enabling future validation / event hooks.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

from evacuation_simulation.simulation.config import GridCell


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class AgentState(IntEnum):
    """
    Lifecycle states for an evacuation agent.

    States
    ------
    WAITING:
        Agent has not yet started moving (pre-simulation or paused).
        Reserved for future use (e.g. agents that react to a trigger).
    MOVING:
        Agent is actively navigating toward an exit.
    EVACUATED:
        Agent has reached an exit and left the building.
    """

    WAITING = 0
    MOVING = 1
    EVACUATED = 2


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """
    Represents a single building occupant in the simulation.

    Attributes
    ----------
    agent_id:
        Unique string identifier.  Used as a key in simulation data
        structures and metric reports.
    row:
        Current row position (zero-indexed).
    col:
        Current column position (zero-indexed).
    state:
        Current lifecycle state (see ``AgentState``).
    evacuation_timestep:
        Timestep at which the agent reached an exit, or ``None`` if the
        agent has not yet evacuated.
    """

    __slots__ = ("agent_id", "row", "col", "state", "evacuation_timestep")

    def __init__(
        self,
        agent_id: str,
        row: int,
        col: int,
        initial_state: AgentState = AgentState.MOVING,
    ) -> None:
        if not agent_id:
            raise ValueError("agent_id must be a non-empty string.")
        if row < 0 or col < 0:
            raise ValueError(
                f"Agent '{agent_id}': row and col must be non-negative, "
                f"got row={row}, col={col}."
            )

        self.agent_id: str = agent_id
        self.row: int = row
        self.col: int = col
        self.state: AgentState = initial_state
        self.evacuation_timestep: Optional[int] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, agent_cfg: object) -> "Agent":
        """
        Construct an ``Agent`` from an ``AgentConfig`` Pydantic model.

        Parameters
        ----------
        agent_cfg:
            An ``AgentConfig`` instance (typed as ``object`` here to avoid
            importing ``AgentConfig`` into this module, preventing coupling
            between the data-layer config and the agent entity).

        Returns
        -------
        Agent
        """
        return cls(
            agent_id=agent_cfg.agent_id,  # type: ignore[attr-defined]
            row=agent_cfg.row,            # type: ignore[attr-defined]
            col=agent_cfg.col,            # type: ignore[attr-defined]
        )

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def mark_evacuated(self, timestep: int) -> None:
        """
        Transition the agent to the EVACUATED state.

        Parameters
        ----------
        timestep:
            The current simulation timestep.

        Raises
        ------
        ValueError
            If the agent is already evacuated.
        """
        if self.state == AgentState.EVACUATED:
            raise ValueError(
                f"Agent '{self.agent_id}' is already evacuated "
                f"(at timestep {self.evacuation_timestep})."
            )
        self.state = AgentState.EVACUATED
        self.evacuation_timestep = timestep

    def set_position(self, row: int, col: int) -> None:
        """
        Update the agent's grid position.

        Parameters
        ----------
        row, col:
            New zero-indexed grid coordinates.

        Raises
        ------
        ValueError
            If the agent is already evacuated (evacuated agents no longer
            occupy a grid cell).
        """
        if self.state == AgentState.EVACUATED:
            raise ValueError(
                f"Cannot move evacuated agent '{self.agent_id}'."
            )
        if row < 0 or col < 0:
            raise ValueError(
                f"Agent '{self.agent_id}': coordinates must be non-negative, "
                f"got row={row}, col={col}."
            )
        self.row = row
        self.col = col

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def position(self) -> GridCell:
        """Return current position as an immutable (row, col) tuple."""
        return (self.row, self.col)

    @property
    def is_active(self) -> bool:
        """Return True if the agent is still present in the building."""
        return self.state != AgentState.EVACUATED

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        evac = (
            f", evacuated_at={self.evacuation_timestep}"
            if self.evacuation_timestep is not None
            else ""
        )
        return (
            f"Agent(id={self.agent_id!r}, "
            f"pos=({self.row},{self.col}), "
            f"state={self.state.name}"
            f"{evac})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Agent):
            return NotImplemented
        return self.agent_id == other.agent_id

    def __hash__(self) -> int:
        return hash(self.agent_id)
