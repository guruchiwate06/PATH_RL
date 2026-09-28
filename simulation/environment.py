"""
environment.py
--------------
Grid-based environment representing the simulated building.

Responsibilities
~~~~~~~~~~~~~~~~
- Store the grid dimensions.
- Track which cells are walls (impassable) and which are exits.
- Provide query helpers used by pathfinding and simulation.

Non-responsibilities (intentionally excluded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Pathfinding / graph construction  → pathfinding.py
- Agent state                       → agent.py
- Visualization                     → outside the core package

Design notes
~~~~~~~~~~~~
- ``Environment`` is constructed from a ``SimulationConfig`` so that all
  initial conditions come from one authoritative source.
- The grid is stored as a NumPy boolean array (``_passable``) where True
  means a cell can be traversed.  This gives O(1) lookup and will support
  future vectorised operations (e.g. flood-fill, density maps).
- Walls and exits are also stored as ``frozenset`` for fast membership
  checks in simulation hot-loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import numpy.typing as npt

from evacuation_simulation.simulation.config import GridCell, SimulationConfig


# ---------------------------------------------------------------------------
# Cell type enum-like constants
# ---------------------------------------------------------------------------


class CellType:
    """
    Integer codes for grid cell types.

    Using simple integer constants (rather than an Enum) keeps NumPy
    array operations straightforward without type-casting overhead.
    """

    OPEN: int = 0   # Traversable, not an exit
    WALL: int = 1   # Impassable
    EXIT: int = 2   # Traversable exit cell


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class Environment:
    """
    Represents the simulated building as a discrete 2-D grid.

    Attributes
    ----------
    rows:
        Number of rows in the grid (height).
    cols:
        Number of columns in the grid (width).
    _cell_type:
        Integer array of shape (rows, cols) encoding each cell's type
        using ``CellType`` constants.
    _walls:
        Frozenset of (row, col) wall cells for O(1) membership checks.
    _exits:
        Frozenset of (row, col) exit cells for O(1) membership checks.
    """

    rows: int
    cols: int
    _cell_type: npt.NDArray[np.int8] = field(init=False, repr=False)
    _walls: frozenset[GridCell] = field(init=False, repr=False)
    _exits: frozenset[GridCell] = field(init=False, repr=False)
    _exit_widths: dict[GridCell, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # Default: all cells are open
        self._cell_type = np.zeros((self.rows, self.cols), dtype=np.int8)
        self._walls = frozenset()
        self._exits = frozenset()
        self._exit_widths = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "Environment":
        """
        Construct an ``Environment`` from a validated ``SimulationConfig``.
        (Backward compatibility wrapper).

        Parameters
        ----------
        config:
            A fully-validated scenario configuration.

        Returns
        -------
        Environment
            A ready-to-use environment instance.
        """
        return cls.from_floor_plan_and_exits(config.floor_plan, config.exit_configuration)

    @classmethod
    def from_floor_plan_and_exits(
        cls, floor_plan: "FloorPlan", exit_config: "ExitConfiguration"
    ) -> "Environment":
        """
        Construct an ``Environment`` from an explicit floor plan and exit configuration.

        Parameters
        ----------
        floor_plan:
            Static building geometry.
        exit_config:
            Authoritative configuration for exit cells.

        Returns
        -------
        Environment
            A ready-to-use environment instance.
        """
        env = cls(rows=floor_plan.grid.rows, cols=floor_plan.grid.cols)

        # Apply walls
        walls: list[GridCell] = []
        for row, col in floor_plan.walls:
            env._cell_type[row, col] = CellType.WALL
            walls.append((row, col))
        env._walls = frozenset(walls)

        # Apply exits (exits override open cells, never walls — config
        # validation already ensures no overlap)
        exits: list[GridCell] = []
        for row, col in exit_config.exits:
            env._cell_type[row, col] = CellType.EXIT
            exits.append((row, col))
        env._exits = frozenset(exits)
        env._exit_widths = {tuple(pos): exit_config.get_width(pos) for pos in exit_config.exits}

        return env

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def is_within_bounds(self, row: int, col: int) -> bool:
        """Return True if (row, col) is inside the grid."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_wall(self, row: int, col: int) -> bool:
        """Return True if (row, col) is a wall cell."""
        return (row, col) in self._walls

    def is_exit(self, row: int, col: int) -> bool:
        """Return True if (row, col) is an exit cell."""
        return (row, col) in self._exits

    def get_exit_width(self, row: int, col: int) -> float:
        """Return the physical width in meters of an exit cell (default 1.0m)."""
        return self._exit_widths.get((row, col), 1.0)

    def is_passable(self, row: int, col: int) -> bool:
        """
        Return True if an agent can occupy (row, col).

        A cell is passable if it is within bounds and not a wall.
        Exit cells are passable (agents must be able to reach them).
        """
        return self.is_within_bounds(row, col) and not self.is_wall(row, col)

    def get_cell_type(self, row: int, col: int) -> int:
        """
        Return the ``CellType`` integer code for cell (row, col).

        Raises
        ------
        IndexError
            If (row, col) is outside the grid.
        """
        if not self.is_within_bounds(row, col):
            raise IndexError(
                f"Cell ({row}, {col}) is outside the grid "
                f"({self.rows}×{self.cols})."
            )
        return int(self._cell_type[row, col])

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def iter_passable_cells(self) -> Iterator[GridCell]:
        """Yield all (row, col) cells that are passable (not walls)."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self._cell_type[r, c] != CellType.WALL:
                    yield (r, c)

    def iter_exit_cells(self) -> Iterator[GridCell]:
        """Yield all (row, col) cells designated as exits."""
        yield from self._exits

    def iter_wall_cells(self) -> Iterator[GridCell]:
        """Yield all (row, col) wall cells."""
        yield from self._walls

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def neighbours(
        self,
        row: int,
        col: int,
        *,
        diagonal: bool = False,
    ) -> list[GridCell]:
        """
        Return passable neighbours of cell (row, col).

        Parameters
        ----------
        row, col:
            Target cell coordinates.
        diagonal:
            If True, include diagonal neighbours (8-connectivity).
            Default is False (4-connectivity / cardinal directions only).

        Returns
        -------
        list[GridCell]
            Passable neighbour cells.
        """
        deltas: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal:
            deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        result: list[GridCell] = []
        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if self.is_passable(nr, nc):
                result.append((nr, nc))
        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        wall_count = len(self._walls)
        exit_count = len(self._exits)
        return (
            f"Environment(rows={self.rows}, cols={self.cols}, "
            f"walls={wall_count}, exits={exit_count})"
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return grid shape as (rows, cols)."""
        return (self.rows, self.cols)

    @property
    def exits(self) -> frozenset[GridCell]:
        """Read-only view of exit cells."""
        return self._exits

    @property
    def walls(self) -> frozenset[GridCell]:
        """Read-only view of wall cells."""
        return self._walls
