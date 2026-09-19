"""
movement.py
-----------
Movement request lifecycle: creation, validation, conflict resolution,
and application.

Responsibilities
~~~~~~~~~~~~~~~~
- Define ``MovementRequest`` -- the data carrier for a single agent's
  desired move.
- Validate that a request is physically legal (in bounds, passable,
  exactly one cell away).
- Resolve conflicts when multiple agents want the same cell, using a
  deterministic rule (lexicographically first agent_id wins).
- Enforce cell-capacity constraints during conflict resolution (Stage 3).
- Apply approved requests by updating agent positions in-place.

Non-responsibilities (intentionally excluded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Deciding *where* an agent wants to go  -> strategy.py
- Simulation loop coordination           -> simulation.py
- Pathfinding                            -> pathfinding.py

Design notes
~~~~~~~~~~~~
- All functions are pure / stateless so they are independently
  testable without instantiating a ``Simulation``.
- ``resolve_conflicts`` is deterministic and order-independent: the
  winner is always the lexicographically smallest ``agent_id`` among
  competing agents, regardless of the order they appear in the input
  list.  This means the same scenario + seed always produces the same
  outcome.
- Stage 3 adds capacity awareness to ``resolve_conflicts``.  The new
  parameters are all optional with defaults that reproduce Stage 2
  behaviour exactly when omitted -- full backward compatibility.
- Cell capacity rule: ``free_slots = capacity(D) - current_occupancy(D)``.
  A cell is enterable only if its current occupancy is strictly below its
  capacity.  Simultaneous vacating is NOT credited -- if an agent's
  departure request is later rejected (due to conflict at its own
  destination), it would remain, so crediting the slot in advance would
  be unsafe and lead to over-occupancy.
- The ``MovementRequest`` dataclass is frozen (immutable) to prevent
  accidental mutation after creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

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
# Rejection reasons (Stage 3)
# ---------------------------------------------------------------------------


class RejectionReason(Enum):
    """
    Reason a movement request was rejected during conflict resolution.

    Values
    ------
    DESTINATION_CONFLICT:
        Another agent (with a lexicographically smaller agent_id) was
        approved for the same destination cell.  (Stage 2 behaviour.)
    DESTINATION_CAPACITY:
        The destination cell does not have sufficient free slots to
        accommodate this agent given current occupancy and simultaneous
        vacating movements.  (Stage 3 addition.)
    """

    DESTINATION_CONFLICT = "DESTINATION_CONFLICT"
    DESTINATION_CAPACITY = "DESTINATION_CAPACITY"


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------


def resolve_conflicts(
    requests: list[MovementRequest],
    *,
    capacity_fn: Optional[Callable[[GridCell], int]] = None,
    current_occupancy: Optional[dict[GridCell, int]] = None,
) -> tuple[list[MovementRequest], list[MovementRequest], dict[str, RejectionReason]]:
    """
    Resolve competing movement requests deterministically.

    When multiple agents request the same destination cell, candidates are
    sorted by ``agent_id`` (lexicographic ascending).  Requests are approved
    in order until the cell's available capacity is exhausted; the remainder
    are rejected.

    **Stage 3 capacity rule**::

        free_slots = capacity(D) - current_occupancy(D)

    Simultaneous vacating is NOT credited.  If an agent's departure request
    is rejected (due to conflict at its own destination), it remains in its
    current cell.  Crediting its departure in advance would be unsafe and
    lead to over-occupancy.

    **Backward compatibility**: all new keyword-only parameters default to
    ``None``.  When all are ``None`` (the Stage 2 call signature),
    the behaviour is identical to Stage 2: capacity defaults to 1, occupancy
    defaults to 0 -- so exactly one request per destination is approved.

    Parameters
    ----------
    requests : list[MovementRequest]
        All validated movement requests for a single timestep.
    capacity_fn : Callable[[GridCell], int] or None
        Function returning the capacity of a given cell.  Defaults to
        ``lambda _: 1`` when ``None`` (Stage 2 equivalent).
    current_occupancy : dict[GridCell, int] or None
        Pre-movement occupancy counts.  Missing keys are treated as 0.
        Defaults to empty dict when ``None``.

    Returns
    -------
    approved : list[MovementRequest]
        Requests that were granted.
    rejected : list[MovementRequest]
        Requests that were denied.
    rejection_reasons : dict[str, RejectionReason]
        Mapping ``{agent_id: reason}`` for every rejected request.
    """
    # Resolve defaults
    _cap_fn: Callable[[GridCell], int] = capacity_fn if capacity_fn is not None else (lambda _: 1)
    _occ: dict[GridCell, int] = current_occupancy if current_occupancy is not None else {}

    # Group by destination
    by_destination: dict[GridCell, list[MovementRequest]] = {}
    for req in requests:
        by_destination.setdefault(req.to_cell, []).append(req)

    approved: list[MovementRequest] = []
    rejected: list[MovementRequest] = []
    rejection_reasons: dict[str, RejectionReason] = {}

    for destination, competing in by_destination.items():
        cap = _cap_fn(destination)
        occ = _occ.get(destination, 0)
        # No vacating credit: conservative and correct.
        free_slots = max(0, cap - occ)

        # Sort deterministically by agent_id (lex ascending)
        sorted_competing = sorted(competing, key=lambda r: r.agent_id)

        # Approve up to free_slots requests; reject the rest
        for i, req in enumerate(sorted_competing):
            if i < free_slots:
                approved.append(req)
            else:
                rejected.append(req)
                # Distinguish capacity rejection from pure conflict rejection
                if free_slots == 0:
                    rejection_reasons[req.agent_id] = RejectionReason.DESTINATION_CAPACITY
                else:
                    # free_slots > 0 but already consumed by earlier agents
                    rejection_reasons[req.agent_id] = RejectionReason.DESTINATION_CONFLICT

    return approved, rejected, rejection_reasons


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
