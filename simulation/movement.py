"""
movement.py
-----------
Movement request lifecycle: creation, validation, conflict resolution,
and application.

Responsibilities
~~~~~~~~~~~~~~~~
- Define ``MovementRequest`` — the data carrier for a single agent's
  desired move.
- Validate that a request is physically legal (in bounds, passable,
  exactly one cell away).
- Resolve conflicts when multiple agents want the same cell, using a
  deterministic rule (lexicographically first agent_id wins).
- Apply approved requests by updating agent positions in-place.

Non-responsibilities (intentionally excluded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Deciding *where* an agent wants to go  → strategy.py
- Simulation loop coordination           → simulation.py
- Pathfinding                            → pathfinding.py

Design notes
~~~~~~~~~~~~
- All four functions are pure / stateless so they are independently
  testable without instantiating a ``Simulation``.
- ``resolve_conflicts`` is deterministic and order-independent: the
  winner is always the lexicographically smallest ``agent_id`` among
  competing agents, regardless of the order they appear in the input
  list.  This means the same scenario + seed always produces the same
  outcome.
- The ``MovementRequest`` dataclass is frozen (immutable) to prevent
  accidental mutation after creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from evacuation_simulation.simulation.config import GridCell

if TYPE_CHECKING:
    from evacuation_simulation.simulation.agent import Agent
    from evacuation_simulation.simulation.environment import Environment


# ---------------------------------------------------------------------------
# MovementRequest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MovementRequest:
    """
    An immutable record of a single agent's desired move for one timestep.

    Attributes
    ----------
    agent_id : str
        Unique identifier of the requesting agent.
    from_cell : GridCell
        The agent's current position at the time the request was made.
    to_cell : GridCell
        The requested destination cell.
    """

    agent_id: str
    from_cell: GridCell
    to_cell: GridCell


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_request(
    request: MovementRequest,
    environment: "Environment",
) -> bool:
    """
    Return True if a movement request is physically legal.

    Rules enforced
    --------------
    1. Destination must be within grid bounds.
    2. Destination must not be a wall (passable).
    3. Movement must be exactly one cell in a cardinal direction
       (Manhattan distance == 1).  Diagonal moves and staying in place
       are both rejected.

    Parameters
    ----------
    request : MovementRequest
        The request to validate.
    environment : Environment
        The environment used to check bounds and passability.

    Returns
    -------
    bool
        ``True`` if all rules pass, ``False`` otherwise.
    """
    r_to, c_to = request.to_cell
    r_from, c_from = request.from_cell

    # Rule 1 & 2: destination must be within bounds and passable
    if not environment.is_passable(r_to, c_to):
        return False

    # Rule 3: exactly one cardinal step (Manhattan distance == 1)
    row_delta = abs(r_to - r_from)
    col_delta = abs(c_to - c_from)
    if row_delta + col_delta != 1:
        return False

    return True


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------


def resolve_conflicts(
    requests: list[MovementRequest],
) -> tuple[list[MovementRequest], list[MovementRequest]]:
    """
    Resolve competing movement requests deterministically.

    When multiple agents request the same destination cell, exactly one
    is approved and the rest are rejected.  The winner is the agent whose
    ``agent_id`` is lexicographically smallest among the competitors.

    This rule is:
    - **Deterministic**: the same set of requests always yields the same
      winner, regardless of input ordering.
    - **Order-independent**: processing Agent A before Agent B never gives
      A an inherent advantage unless A wins on ``agent_id`` ordering.
    - **Isolated**: the logic is here and nowhere else, making it easy
      to replace with a more sophisticated model in a future stage.

    Parameters
    ----------
    requests : list[MovementRequest]
        All validated movement requests for a single timestep.

    Returns
    -------
    approved : list[MovementRequest]
        Requests that were granted.
    rejected : list[MovementRequest]
        Requests that were denied due to cell conflict.
    """
    # Group by destination
    by_destination: dict[GridCell, list[MovementRequest]] = {}
    for req in requests:
        by_destination.setdefault(req.to_cell, []).append(req)

    approved: list[MovementRequest] = []
    rejected: list[MovementRequest] = []

    for _destination, competing in by_destination.items():
        if len(competing) == 1:
            approved.append(competing[0])
        else:
            # Sort by agent_id for a deterministic, reproducible winner
            sorted_competing = sorted(competing, key=lambda r: r.agent_id)
            approved.append(sorted_competing[0])
            rejected.extend(sorted_competing[1:])

    return approved, rejected


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def apply_movements(
    approved: list[MovementRequest],
    agents_by_id: dict[str, "Agent"],
) -> None:
    """
    Apply approved movement requests by updating agent positions in-place.

    Only active (non-evacuated) agents are moved.  Evacuated agents that
    appear in the approved list (which should not happen under normal
    operation) are silently skipped.

    Parameters
    ----------
    approved : list[MovementRequest]
        Movement requests that have passed validation and conflict
        resolution.
    agents_by_id : dict[str, Agent]
        Mapping from ``agent_id`` to ``Agent`` instance.
    """
    for request in approved:
        agent = agents_by_id.get(request.agent_id)
        if agent is not None and agent.is_active:
            agent.set_position(request.to_cell[0], request.to_cell[1])
