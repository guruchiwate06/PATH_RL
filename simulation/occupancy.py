"""
occupancy.py
------------
Dynamic cell-occupancy tracking for the evacuation simulation.

Responsibilities
~~~~~~~~~~~~~~~~
- Maintain a count of active agents in each grid cell.
- Expose occupancy queries used by the movement resolver and metrics layer.
- Provide capacity and congestion queries based on the configured cell capacity.

Non-responsibilities (intentionally excluded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Static building representation   -> environment.py
- Agent lifecycle / movement       -> agent.py, simulation.py
- Pathfinding                      -> pathfinding.py
- Visualization                    -> outside the core package

Design notes
~~~~~~~~~~~~
- ``OccupancyMap`` is **derived state**: it is rebuilt from actual agent
  positions every timestep via ``rebuild()``.  This guarantees consistency
  with agent state at the cost of an O(active_agents) rebuild each step,
  which is acceptable for the simulation sizes targeted.
- Dynamic state is kept entirely separate from ``Environment`` (the static
  building representation), satisfying the Stage 3 architectural requirement.
- Capacity is uniform across all passable cells in Stage 3.  The architecture
  allows per-cell overrides in future stages by replacing ``_capacity``
  with a ``dict[GridCell, int]``.
- **Congestion definition (Stage 3):**
  A cell is congested when its occupancy equals or exceeds its capacity,
  i.e. ``occupancy_ratio >= 1.0``.  This is a simple, measurable threshold
  with no ambiguity.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from evacuation_simulation.simulation.config import GridCell

if TYPE_CHECKING:
    from evacuation_simulation.simulation.agent import Agent


# ---------------------------------------------------------------------------
# OccupancyMap
# ---------------------------------------------------------------------------


class OccupancyMap:
    """
    Tracks the number of active agents occupying each grid cell.

    Occupancy is derived from agent positions and must be kept in sync
    with the simulation state by calling ``rebuild()`` after every
    movement application.

    Parameters
    ----------
    default_cell_capacity : int
        Maximum number of active agents allowed in a single cell.
        Must be >= 1.  Applies uniformly to all passable cells in Stage 3.

    Attributes
    ----------
    _capacity : int
        Uniform cell capacity for this simulation run.
    _occupancy : dict[GridCell, int]
        Maps (row, col) -> current active agent count.  Cells with zero
        occupancy are absent (sparse representation).
    """

    def __init__(self, default_cell_capacity: int = 1) -> None:
        if default_cell_capacity < 1:
            raise ValueError(
                f"default_cell_capacity must be >= 1, got {default_cell_capacity}."
            )
        self._capacity: int = default_cell_capacity
        self._occupancy: dict[GridCell, int] = {}

    # ------------------------------------------------------------------
    # State synchronisation
    # ------------------------------------------------------------------

    def rebuild(self, agents: list["Agent"]) -> None:
        """
        Recompute occupancy from scratch using current agent positions.

        Only active (non-evacuated) agents contribute to occupancy.
        This method must be called after applying movements each timestep
        to keep the map consistent with simulation state.

        Parameters
        ----------
        agents : list[Agent]
            All agents (active and evacuated).  Evacuated agents are
            automatically filtered out.
        """
        counts: dict[GridCell, int] = defaultdict(int)
        for agent in agents:
            if agent.is_active:
                counts[agent.position] += 1
        # Replace the map atomically -- no partial updates.
        self._occupancy = dict(counts)

    # ------------------------------------------------------------------
    # Occupancy queries
    # ------------------------------------------------------------------

    def get_occupancy(self, cell: GridCell) -> int:
        """
        Return the number of active agents currently in *cell*.

        Returns 0 for any cell with no active occupants (including walls --
        this method does not validate that *cell* is passable; the caller
        is responsible for meaningful queries).

        Parameters
        ----------
        cell : GridCell
            The (row, col) cell to query.

        Returns
        -------
        int
            Active agent count.  Always >= 0.
        """
        return self._occupancy.get(cell, 0)

    def get_occupied_cells(self) -> dict[GridCell, int]:
        """
        Return a copy of the occupancy mapping for all currently occupied cells.

        Only cells with at least one active agent are included.

        Returns
        -------
        dict[GridCell, int]
            Mapping ``{(row, col): occupancy_count}`` for occupied cells.
        """
        return dict(self._occupancy)

    # ------------------------------------------------------------------
    # Capacity queries
    # ------------------------------------------------------------------

    def get_capacity(self, cell: GridCell) -> int:  # noqa: ARG002
        """
        Return the capacity of *cell*.

        In Stage 3 all passable cells share the same uniform capacity.
        The ``cell`` parameter is accepted for API forward-compatibility
        with future per-cell capacity models.

        Parameters
        ----------
        cell : GridCell
            The cell to query.

        Returns
        -------
        int
            Maximum number of agents allowed in *cell*.
        """
        return self._capacity

    @property
    def default_capacity(self) -> int:
        """Return the uniform cell capacity used by this occupancy map."""
        return self._capacity

    # ------------------------------------------------------------------
    # Congestion queries
    # ------------------------------------------------------------------

    def get_occupancy_ratio(self, cell: GridCell) -> float:
        """
        Return the occupancy ratio for *cell*.

        Defined as ``occupancy / capacity``.  A ratio >= 1.0 means the
        cell is at or beyond capacity (congested).

        Parameters
        ----------
        cell : GridCell
            The cell to query.

        Returns
        -------
        float
            Value in [0.0, +inf).  Values > 1.0 indicate over-capacity,
            which should not occur after correct movement resolution.
        """
        return self.get_occupancy(cell) / self._capacity

    def is_congested(self, cell: GridCell) -> bool:
        """
        Return True if *cell* is congested.

        **Congestion definition (Stage 3):** A cell is congested when its
        occupancy equals or exceeds its capacity (``occupancy_ratio >= 1.0``).

        Parameters
        ----------
        cell : GridCell
            The cell to query.

        Returns
        -------
        bool
        """
        return self.get_occupancy(cell) >= self._capacity

    def congested_cells(self) -> dict[GridCell, float]:
        """
        Return all currently congested cells with their occupancy ratios.

        Returns
        -------
        dict[GridCell, float]
            Mapping ``{cell: occupancy_ratio}`` for all cells where
            ``occupancy >= capacity``.
        """
        return {
            cell: self.get_occupancy_ratio(cell)
            for cell in self._occupancy
            if self.is_congested(cell)
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        occupied = len(self._occupancy)
        congested = len(self.congested_cells())
        return (
            f"OccupancyMap("
            f"capacity={self._capacity}, "
            f"occupied_cells={occupied}, "
            f"congested_cells={congested})"
        )
