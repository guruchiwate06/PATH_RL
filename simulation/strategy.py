"""
strategy.py
-----------
Concrete movement strategies for the evacuation simulation.

Responsibilities
~~~~~~~~~~~~~~~~
- Implement the ``MovementStrategy`` protocol defined in ``simulation.py``.
- Decide where each active agent should move next.
- Return movement requests as a ``dict[str, GridCell]`` — strategies must
  NOT mutate agents directly.

Non-responsibilities (intentionally excluded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Physical movement execution      → movement.py
- Conflict resolution              → movement.py
- Agent state mutation             → simulation.py
- Navigation graph construction    → pathfinding.py
- Visualization                    → outside the core package

Design notes
~~~~~~~~~~~~
- Strategies return a ``dict[str, GridCell]`` mapping agent_id to the
  requested next cell.  The simulation layer owns all mutation.
- ``ShortestPathStrategy`` uses a path cache keyed by agent_id.  The
  cache stores the full planned path (including the agent's current
  position as element [0]).  If the agent's actual position no longer
  matches the expected path head (e.g. a conflict rejection moved them
  differently), the cache is invalidated and a fresh path is computed.
- Exit selection uses actual graph path cost (number of hops) rather
  than Euclidean distance, so walls are correctly accounted for.
- BFS is the underlying algorithm (via ``NavigationGraph.shortest_path``).
  A* can be substituted later by changing that single method in
  ``pathfinding.py`` without touching this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from evacuation_simulation.simulation.config import GridCell

if TYPE_CHECKING:
    from evacuation_simulation.simulation.agent import Agent
    from evacuation_simulation.simulation.environment import Environment
    from evacuation_simulation.simulation.pathfinding import NavigationGraph


# ---------------------------------------------------------------------------
# ShortestPathStrategy
# ---------------------------------------------------------------------------


class ShortestPathStrategy:
    """
    Moves each active agent one step along the shortest path to the
    nearest reachable exit.

    Algorithm (per agent per timestep)
    -----------------------------------
    1. If the agent is already on an exit cell, produce no request
       (the simulation's evacuation check handles it).
    2. Check the path cache for a valid planned path.
    3. If no valid cached path exists, query the navigation graph for
       paths to every exit and select the shortest one.
    4. Return the first step of the chosen path as the movement request.
    5. If no exit is reachable, produce no request — the agent stays put.

    Exit selection
    --------------
    All exits are evaluated.  Unreachable exits (no path in the graph)
    are silently skipped.  The exit requiring the fewest hops is chosen.
    Ties are broken by the natural iteration order of ``environment.exits``
    (a ``frozenset``), which is deterministic for a fixed graph.

    Path caching
    ------------
    The cache maps ``agent_id → full_path`` where ``full_path[0]`` is the
    agent's expected current position.  On each call the cache head is
    verified against the agent's actual position.  A mismatch (caused by
    conflict rejection in a prior step) invalidates the cache and triggers
    a fresh BFS query.

    Attributes
    ----------
    _path_cache : dict[str, list[GridCell]]
        Per-agent planned path cache.
    """

    def __init__(self) -> None:
        self._path_cache: dict[str, list[GridCell]] = {}

    # ------------------------------------------------------------------
    # MovementStrategy protocol
    # ------------------------------------------------------------------

    def move_agents(
        self,
        agents: list["Agent"],
        environment: "Environment",
        nav_graph: "NavigationGraph",
        rng: np.random.Generator,
        timestep: int,
    ) -> dict[str, GridCell]:
        """
        Compute movement requests for all active agents.

        Parameters
        ----------
        agents : list[Agent]
            Active (non-evacuated) agents.
        environment : Environment
            Used to check whether an agent is already at an exit.
        nav_graph : NavigationGraph
            Used for path queries.
        rng : np.random.Generator
            Random generator (not used by this deterministic strategy;
            reserved for future stochastic extensions).
        timestep : int
            Current simulation timestep (reserved for time-dependent
            strategies in future stages).

        Returns
        -------
        dict[str, GridCell]
            Mapping ``{agent_id: requested_next_cell}`` for agents that
            have a valid move.  Agents with no reachable exit are absent.
        """
        requests: dict[str, GridCell] = {}

        for agent in agents:
            if not agent.is_active:
                continue

            # Agent already at an exit: simulation handles evacuation,
            # no movement request needed.
            if environment.is_exit(agent.row, agent.col):
                self._path_cache.pop(agent.agent_id, None)
                continue

            next_step = self._get_next_step(agent, environment, nav_graph)
            if next_step is not None:
                requests[agent.agent_id] = next_step

        return requests

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_next_step(
        self,
        agent: "Agent",
        environment: "Environment",
        nav_graph: "NavigationGraph",
    ) -> Optional[GridCell]:
        """
        Return the next cell the agent should move toward, or None.

        Validates the cache; recomputes if stale or absent.
        """
        current = agent.position
        cached_path = self._path_cache.get(agent.agent_id)

        # Cache hit: path[0] must match the agent's current position and
        # there must be at least one more step to take.
        if (
            cached_path
            and cached_path[0] == current
            and len(cached_path) > 1
        ):
            return cached_path[1]

        # Cache miss or stale — recompute from scratch.
        path = self._find_nearest_exit_path(current, environment, nav_graph)

        if path is None or len(path) <= 1:
            # No reachable exit (or agent is already at exit — handled above).
            self._path_cache.pop(agent.agent_id, None)
            return None

        self._path_cache[agent.agent_id] = path
        return path[1]  # path[0] == current position

    def _find_nearest_exit_path(
        self,
        position: GridCell,
        environment: "Environment",
        nav_graph: "NavigationGraph",
    ) -> Optional[list[GridCell]]:
        """
        Return the shortest path from *position* to any reachable exit.

        Among all exits for which a path exists, the one with the minimum
        number of hops is returned.  Returns ``None`` if no exit is
        reachable.

        Uses ``environment.exits`` (public API) to enumerate exits and
        ``nav_graph.shortest_path`` for BFS queries.
        """
        best_path: Optional[list[GridCell]] = None

        for exit_cell in environment.exits:
            path = nav_graph.shortest_path(position, exit_cell)
            if path is None:
                continue
            if best_path is None or len(path) < len(best_path):
                best_path = path

        return best_path

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache_for(self, agent_id: str) -> None:
        """Evict the cached path for a specific agent."""
        self._path_cache.pop(agent_id, None)

    def clear_all_caches(self) -> None:
        """
        Evict all cached paths.

        Call this if the environment changes (e.g. a dynamic obstacle is
        added) to force all agents to replan.
        """
        self._path_cache.clear()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ShortestPathStrategy("
            f"cached_agents={len(self._path_cache)})"
        )
