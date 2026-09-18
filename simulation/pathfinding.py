"""
pathfinding.py
--------------
Navigation graph construction over traversable grid cells.

Responsibilities
~~~~~~~~~~~~~~~~
- Build a NetworkX undirected graph whose nodes are passable (row, col)
  cells and whose edges connect 4-connected (cardinal) neighbours.
- Expose graph queries: node existence, edge existence, graph statistics.
- Provide a stub for path queries (to be filled in a later stage).

Non-responsibilities (intentionally excluded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- A* or other search algorithms    → later stage
- Agent movement                   → simulation.py (later stage)
- Visualization                    → outside the core package

Design notes
~~~~~~~~~~~~
- NetworkX is used because it provides a well-tested graph data structure
  with built-in algorithms (BFS, shortest-path, connectivity checks) that
  will be exercised in future stages without requiring new dependencies.
- The graph is built once per scenario and treated as immutable during
  a simulation run (dynamic obstacles are a future concern).
- Edge weights default to 1.0 (unit cost per cardinal step).  This field
  will support terrain-cost and hazard penalties in later stages.
- Diagonal movement is disabled by default because 4-connectivity is
  simpler to reason about during the foundation stage.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx

from evacuation_simulation.simulation.config import GridCell, SimulationConfig
from evacuation_simulation.simulation.environment import Environment


# ---------------------------------------------------------------------------
# NavigationGraph
# ---------------------------------------------------------------------------


class NavigationGraph:
    """
    A graph of traversable grid cells used for navigation.

    Each node is a ``(row, col)`` tuple.  Edges connect 4-connected
    passable neighbours and carry a ``weight`` attribute (default 1.0).

    Attributes
    ----------
    _graph:
        The underlying ``networkx.Graph`` instance.
    _exits:
        Frozenset of exit cells copied from the environment for fast
        lookup without keeping a reference to the full environment.
    """

    def __init__(self) -> None:
        self._graph: nx.Graph = nx.Graph()
        self._exits: frozenset[GridCell] = frozenset()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_environment(cls, environment: Environment) -> "NavigationGraph":
        """
        Build a ``NavigationGraph`` from a constructed ``Environment``.

        Parameters
        ----------
        environment:
            A ready-to-use ``Environment`` instance.

        Returns
        -------
        NavigationGraph
            A graph whose nodes are all passable cells and whose edges
            connect 4-connected passable neighbours.
        """
        nav = cls()
        nav._exits = environment.exits

        # Add all passable nodes
        for cell in environment.iter_passable_cells():
            nav._graph.add_node(cell)

        # Add edges between passable 4-connected neighbours
        for cell in environment.iter_passable_cells():
            row, col = cell
            for neighbour in environment.neighbours(row, col, diagonal=False):
                if not nav._graph.has_edge(cell, neighbour):
                    nav._graph.add_edge(cell, neighbour, weight=1.0)

        return nav

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "NavigationGraph":
        """
        Convenience factory: build graph directly from a config object.

        Internally constructs an ``Environment`` and delegates to
        ``from_environment``.
        """
        env = Environment.from_config(config)
        return cls.from_environment(env)

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """Number of traversable nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Number of edges (traversal links) in the graph."""
        return self._graph.number_of_edges()

    def has_node(self, cell: GridCell) -> bool:
        """Return True if *cell* is a node in the navigation graph."""
        return self._graph.has_node(cell)

    def has_edge(self, source: GridCell, target: GridCell) -> bool:
        """Return True if there is a direct edge between *source* and *target*."""
        return self._graph.has_edge(source, target)

    def neighbours_of(self, cell: GridCell) -> list[GridCell]:
        """Return all graph neighbours of *cell*."""
        if not self._graph.has_node(cell):
            raise KeyError(f"Cell {cell} is not in the navigation graph.")
        return list(self._graph.neighbors(cell))

    def is_connected(self) -> bool:
        """
        Return True if the navigation graph is fully connected.

        A disconnected graph means some regions of the building cannot
        reach any exit — useful for scenario validation.
        """
        if self._graph.number_of_nodes() == 0:
            return True  # Vacuously connected
        return nx.is_connected(self._graph)

    def all_exits_reachable_from(self, cell: GridCell) -> bool:
        """
        Return True if at least one exit is reachable from *cell*.

        Uses NetworkX's built-in BFS via ``has_path``.
        """
        for exit_cell in self._exits:
            if self._graph.has_node(exit_cell) and nx.has_path(
                self._graph, cell, exit_cell
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # Path query stub (to be implemented in a later stage)
    # ------------------------------------------------------------------

    def shortest_path(
        self,
        source: GridCell,
        target: GridCell,
    ) -> Optional[list[GridCell]]:
        """
        Return the shortest path from *source* to *target*, or None.

        .. note::
            This currently uses NetworkX's built-in BFS shortest-path.
            A* with heuristics will replace this in a later stage.

        Parameters
        ----------
        source:
            Starting cell.
        target:
            Destination cell.

        Returns
        -------
        list[GridCell] or None
            Ordered list of cells from source to target (inclusive), or
            None if no path exists.
        """
        try:
            path: list[GridCell] = nx.shortest_path(
                self._graph, source=source, target=target, weight="weight"
            )
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NavigationGraph("
            f"nodes={self.node_count}, "
            f"edges={self.edge_count}, "
            f"exits={len(self._exits)}, "
            f"connected={self.is_connected()})"
        )
