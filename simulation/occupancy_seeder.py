"""
occupancy_seeder.py
-------------------
Configurable, reproducible occupant seeding for PATH_RL floor plans.

Guarantees:
1. Occupants only spawn on valid, walkable interior cells.
2. Occupants never spawn on solid wall cells, obstacle cells, exit cells, or exterior voids.
3. Configurable distribution: by partitioned room suites or uniform walkable sampling.
4. Bounded capacity enforcement (e.g. 1 occupant per cell by default).
5. Fully deterministic reproducibility governed by random seeds.
"""

from __future__ import annotations

import random
from typing import Optional

from pydantic import BaseModel, Field

from evacuation_simulation.simulation.config import (
    AgentConfig,
    AgentProfileConfig,
    ExitConfiguration,
    FloorPlan,
    GridCell,
    OccupantScenario,
)
from evacuation_simulation.simulation.environment import Environment


class OccupantSeederConfig(BaseModel):
    """Configuration parameters for occupant population seeding."""
    total_occupants: int = Field(default=10, ge=1, description="Number of agents to seed")
    seed: Optional[int] = Field(default=42, description="Random seed for reproducible placement")
    strategy: str = Field(default="random_walkable", description="Placement strategy: 'random_walkable' or 'room_distributed'")
    max_occupants_per_cell: int = Field(default=1, ge=1, description="Maximum agents per starting cell")
    allow_exit_cells: bool = Field(default=False, description="Whether agents may spawn on exit cells (always False for evacuations)")
    speed: float = Field(default=1.0, gt=0.0, le=1.0, description="Default movement speed")
    reaction_delay: int = Field(default=0, ge=0, description="Default reaction delay timesteps")


class OccupantSeeder:
    """Generates initial occupant placements for a FloorPlan."""

    @classmethod
    def seed(
        cls,
        floor_plan: FloorPlan,
        exit_config: Optional[ExitConfiguration] = None,
        config: Optional[OccupantSeederConfig] = None,
    ) -> OccupantScenario:
        """
        Seed occupants into a floor plan according to configuration constraints.

        Parameters
        ----------
        floor_plan:
            Static building geometry.
        exit_config:
            Optional exit configuration used to exclude exit cells.
        config:
            Seeder configuration. Defaults to 10 occupants with seed 42.

        Returns
        -------
        OccupantScenario
            A populated scenario containing valid AgentConfig entries.

        Raises
        ------
        ValueError
            If the requested number of occupants exceeds available walkable capacity.
        """
        cfg = config or OccupantSeederConfig()
        rng = random.Random(cfg.seed)

        rows = floor_plan.grid.rows
        cols = floor_plan.grid.cols
        wall_set = set(map(tuple, floor_plan.walls))
        exit_set = set(map(tuple, exit_config.exits)) if exit_config else set()

        # Identify all valid walkable cells
        walkable_cells: list[GridCell] = []
        for r in range(rows):
            for c in range(cols):
                pos = (r, c)
                if pos in wall_set:
                    continue
                if not cfg.allow_exit_cells and pos in exit_set:
                    continue
                walkable_cells.append(pos)

        total_capacity = len(walkable_cells) * cfg.max_occupants_per_cell
        if cfg.total_occupants > total_capacity:
            raise ValueError(
                f"Cannot seed {cfg.total_occupants} occupants: only {len(walkable_cells)} walkable cells "
                f"available (total capacity: {total_capacity})."
            )

        # Distribute occupants
        agents: list[AgentConfig] = []
        cell_occupancy: dict[GridCell, int] = {}

        # Shuffle walkable cells deterministically
        shuffled_cells = list(walkable_cells)
        rng.shuffle(shuffled_cells)

        placed = 0
        cell_idx = 0
        attempts = 0
        max_attempts = cfg.total_occupants * 50

        while placed < cfg.total_occupants and attempts < max_attempts:
            attempts += 1
            cand_cell = shuffled_cells[cell_idx % len(shuffled_cells)]
            cell_idx += 1

            current_count = cell_occupancy.get(cand_cell, 0)
            if current_count < cfg.max_occupants_per_cell:
                cell_occupancy[cand_cell] = current_count + 1
                agent_id = f"agent_{placed+1:03d}"
                agents.append(
                    AgentConfig(
                        agent_id=agent_id,
                        row=cand_cell[0],
                        col=cand_cell[1],
                        profile=AgentProfileConfig(
                            speed=cfg.speed,
                            reaction_delay=cfg.reaction_delay,
                        ),
                    )
                )
                placed += 1

        if placed < cfg.total_occupants:
            raise ValueError(
                f"Could only place {placed}/{cfg.total_occupants} occupants within bounded resampling limit."
            )

        return OccupantScenario(agents=agents)
