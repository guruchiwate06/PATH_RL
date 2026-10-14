"""
commercial.py
-------------
Procedural Commercial and Institutional Floor-Plan Generator for PATH_RL.

Generates synthetic, physically consistent commercial layouts (central spine,
H-corridor, or perimeter suites) containing:
1. An enclosed exterior boundary.
2. Multiple rooms (offices, labs, meeting rooms) and wide corridors.
3. Connected walkable regions.
4. Exterior walls, interior partitions, and structural obstacles.
5. Multiple candidate exits (>= 2) adhering to NBC 2016 Part 4 minimum width,
   count, and separation rules.
6. 100% deterministic reproducibility given a random seed.
7. Explicit validation and bounded retry attempts (no infinite loops or silent rule relaxing).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx
from pydantic import BaseModel, Field

from evacuation_simulation.simulation.config import (
    DoorOpening,
    ExitConfiguration,
    FloorPlan,
    GridCell,
    GridConfig,
    OccupantScenario,
)
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.pathfinding import NavigationGraph
from evacuation_simulation.simulation.validator import ExitValidator


class GenerationError(Exception):
    """Raised when a valid procedural layout cannot be generated within max_attempts."""
    pass


class CommercialGeneratorConfig(BaseModel):
    """
    Configuration parameters for procedural commercial layout generation.
    """
    grid_rows: int = Field(default=24, ge=16, le=200, description="Total rows in grid")
    grid_cols: int = Field(default=40, ge=20, le=200, description="Total columns in grid")
    cell_size_m: float = Field(default=0.5, gt=0.0, description="Physical size of one cell in metres")
    corridor_width_cells: int = Field(default=4, ge=2, le=10, description="Width of corridor in cells (4 cells = 2.0m)")
    min_room_width_cells: int = Field(default=6, ge=4, description="Minimum room width along corridor in cells")
    max_room_width_cells: int = Field(default=10, ge=6, description="Maximum room width along corridor in cells")
    num_exterior_exits: int = Field(default=2, ge=2, le=8, description="Number of exterior candidate exits to place")
    exit_door_width_m: float = Field(default=1.5, ge=1.0, description="Width of exterior candidate exits in metres")
    interior_door_width_m: float = Field(default=1.0, ge=0.8, description="Width of interior doors in metres")
    add_columns: bool = Field(default=True, description="Whether to include occasional structural columns")
    max_attempts: int = Field(default=20, ge=1, le=100, description="Maximum generation attempts before raising GenerationError")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible layout generation")


@dataclass
class CommercialRoom:
    """Represents a partitioned room on the generated floor plan."""
    room_id: str
    room_type: str
    row_range: tuple[int, int]  # [r_min, r_max] inclusive
    col_range: tuple[int, int]  # [c_min, c_max] inclusive
    door_cell: GridCell
    cells: list[GridCell] = field(default_factory=list)


@dataclass
class CommercialLayout:
    """A generated and validated commercial floor-plan scenario."""
    floor_plan: FloorPlan
    exit_configuration: ExitConfiguration
    rooms: list[CommercialRoom]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert layout summary and metadata to a dictionary."""
        return {
            "metadata": self.metadata,
            "grid": {
                "rows": self.floor_plan.grid.rows,
                "cols": self.floor_plan.grid.cols,
                "cell_size_m": self.metadata.get("cell_size_m", 0.5),
            },
            "wall_cells_count": len(self.floor_plan.walls),
            "door_count": len(self.floor_plan.doors),
            "exterior_exits_count": len(self.exit_configuration.exits),
            "room_count": len(self.rooms),
            "rooms": [
                {
                    "room_id": r.room_id,
                    "room_type": r.room_type,
                    "bounds": [r.row_range[0], r.row_range[1], r.col_range[0], r.col_range[1]],
                    "door_cell": list(r.door_cell),
                    "area_cells": len(r.cells),
                }
                for r in self.rooms
            ],
        }


class CommercialLayoutGenerator:
    """
    Procedural generator that synthesizes compliant commercial/institutional floor plans.
    """

    ROOM_TYPES = ["Office", "Laboratory", "Conference", "Restroom", "Breakroom"]

    def __init__(self, validator: Optional[ExitValidator] = None) -> None:
        self.validator = validator or ExitValidator()

    def generate(self, config: Optional[CommercialGeneratorConfig] = None) -> CommercialLayout:
        """
        Generate a single-floor commercial layout adhering to all hard constraints.

        Parameters
        ----------
        config:
            Generator configuration. Uses default parameters if None.

        Returns
        -------
        CommercialLayout
            A validated layout with FloorPlan, ExitConfiguration, and room topology.

        Raises
        ------
        GenerationError
            If a valid layout cannot be generated within config.max_attempts.
        """
        cfg = config or CommercialGeneratorConfig()
        base_seed = cfg.seed if cfg.seed is not None else random.randint(1, 1_000_000_000)

        for attempt in range(cfg.max_attempts):
            current_seed = base_seed + attempt * 7919
            rng = random.Random(current_seed)

            try:
                layout = self._attempt_generate(cfg, rng, current_seed)
                # Full validation against structural and regulatory checks
                if self._validate_layout(layout, cfg):
                    return layout
            except Exception as e:
                # Log or continue to next attempt
                continue

        raise GenerationError(
            f"Failed to generate a valid commercial layout within {cfg.max_attempts} attempts "
            f"(base_seed={base_seed}). Ensure grid dimensions and room parameters are feasible."
        )

    def _attempt_generate(
        self, cfg: CommercialGeneratorConfig, rng: random.Random, seed_val: int
    ) -> CommercialLayout:
        rows, cols = cfg.grid_rows, cfg.grid_cols
        walls: set[GridCell] = set()
        doors: list[DoorOpening] = []
        rooms: list[CommercialRoom] = []

        # 1. Enclosed exterior perimeter
        for c in range(cols):
            walls.add((0, c))
            walls.add((rows - 1, c))
        for r in range(rows):
            walls.add((r, 0))
            walls.add((r, cols - 1))

        # 2. Central horizontal corridor
        # Position corridor approximately midway vertically
        corr_h = cfg.corridor_width_cells
        corr_start_r = (rows - corr_h) // 2
        corr_end_r = corr_start_r + corr_h  # exclusive

        # Corridor walls separating north and south suites
        north_corr_wall_r = corr_start_r - 1
        south_corr_wall_r = corr_end_r

        for c in range(1, cols - 1):
            walls.add((north_corr_wall_r, c))
            walls.add((south_corr_wall_r, c))

        # 3. Partition North Wing Rooms
        north_r_min = 1
        north_r_max = north_corr_wall_r - 1
        if north_r_max >= north_r_min + 2:
            self._partition_wing(
                wing_name="North",
                r_min=north_r_min,
                r_max=north_r_max,
                cols=cols,
                door_wall_r=north_corr_wall_r,
                cfg=cfg,
                rng=rng,
                walls=walls,
                doors=doors,
                rooms=rooms,
            )

        # 4. Partition South Wing Rooms
        south_r_min = south_corr_wall_r + 1
        south_r_max = rows - 2
        if south_r_max >= south_r_min + 2:
            self._partition_wing(
                wing_name="South",
                r_min=south_r_min,
                r_max=south_r_max,
                cols=cols,
                door_wall_r=south_corr_wall_r,
                cfg=cfg,
                rng=rng,
                walls=walls,
                doors=doors,
                rooms=rooms,
            )

        # 5. Add optional structural columns in large corridor or rooms
        if cfg.add_columns:
            # Place 2-4 isolated column cells along corridor center line
            mid_corr_r = (corr_start_r + corr_end_r) // 2
            col_step = max(8, cols // 4)
            for c_col in range(col_step, cols - col_step + 1, col_step):
                # Ensure it doesn't block door alignments
                cand_col = (mid_corr_r, c_col)
                if cand_col not in walls:
                    walls.add(cand_col)

        # 6. Place Exterior Candidate Exits
        # We place candidate exits on the perimeter. For reference exit_configuration,
        # we select the primary opposing West and East exits to guarantee R3 diagonal separation.
        exit_doors: list[DoorOpening] = []
        mid_corr_r = (corr_start_r + corr_end_r) // 2

        # Candidate 1: West corridor facade
        exit1_pos = (mid_corr_r, 0)
        walls.discard(exit1_pos)
        ex1 = DoorOpening(
            door_id="EXIT_WEST",
            position=exit1_pos,
            width=cfg.exit_door_width_m,
            exterior=True,
        )
        doors.append(ex1)
        exit_doors.append(ex1)

        # Candidate 2: East corridor facade
        exit2_pos = (mid_corr_r, cols - 1)
        walls.discard(exit2_pos)
        ex2 = DoorOpening(
            door_id="EXIT_EAST",
            position=exit2_pos,
            width=cfg.exit_door_width_m,
            exterior=True,
        )
        doors.append(ex2)
        exit_doors.append(ex2)

        # Optional additional candidate exits (e.g. Northwest, Southeast)
        if cfg.num_exterior_exits >= 3:
            nw_col = max(3, cols // 4)
            exit3_pos = (0, nw_col)
            walls.discard(exit3_pos)
            for r in range(1, north_corr_wall_r + 1):
                walls.discard((r, nw_col))
            ex3 = DoorOpening(
                door_id="EXIT_NORTHWEST",
                position=exit3_pos,
                width=cfg.exit_door_width_m,
                exterior=True,
            )
            doors.append(ex3)
            exit_doors.append(ex3)

        if cfg.num_exterior_exits >= 4:
            se_col = min(cols - 4, (3 * cols) // 4)
            exit4_pos = (rows - 1, se_col)
            walls.discard(exit4_pos)
            for r in range(south_corr_wall_r, rows - 1):
                walls.discard((r, se_col))
            ex4 = DoorOpening(
                door_id="EXIT_SOUTHEAST",
                position=exit4_pos,
                width=cfg.exit_door_width_m,
                exterior=True,
            )
            doors.append(ex4)
            exit_doors.append(ex4)

        # 7. Ensure absolute invariant: doors and walls must not overlap
        for d in doors:
            walls.discard(tuple(d.position))

        # Build FloorPlan
        fp = FloorPlan(
            grid=GridConfig(rows=rows, cols=cols),
            walls=sorted(list(walls)),
            doors=doors,
        )

        # Reference exit configuration activates primary opposing exits (West + East)
        # to ensure the initial baseline configuration is compliant under R1-R5.
        exit_cfg = ExitConfiguration(
            configuration_id="reference_compliant_exits",
            exits=[ex1.position, ex2.position],
            widths=[ex1.width, ex2.width],
        )

        metadata = {
            "generator": "CommercialLayoutGenerator",
            "seed": seed_val,
            "style": "central_corridor_spine",
            "cell_size_m": cfg.cell_size_m,
            "dimensions_m": [cols * cfg.cell_size_m, rows * cfg.cell_size_m],
            "num_exits_configured": len(exit_doors),
        }

        return CommercialLayout(
            floor_plan=fp,
            exit_configuration=exit_cfg,
            rooms=rooms,
            metadata=metadata,
        )

    def _partition_wing(
        self,
        wing_name: str,
        r_min: int,
        r_max: int,
        cols: int,
        door_wall_r: int,
        cfg: CommercialGeneratorConfig,
        rng: random.Random,
        walls: set[GridCell],
        doors: list[DoorOpening],
        rooms: list[CommercialRoom],
    ) -> None:
        """Divide a wing into partitioned rooms with door openings."""
        cur_c = 1
        room_idx = 1

        while cur_c < cols - 2:
            remaining_c = (cols - 1) - cur_c
            if remaining_c < cfg.min_room_width_cells:
                # Room absorbs remaining width
                room_w = remaining_c
            else:
                room_w = rng.randint(cfg.min_room_width_cells, cfg.max_room_width_cells)
                room_w = min(room_w, remaining_c)

            room_end_c = cur_c + room_w - 1  # inclusive

            # Build partition wall on the right of this room if not reaching perimeter
            if room_end_c < cols - 2:
                wall_c = room_end_c + 1
                for r in range(r_min, r_max + 1):
                    walls.add((r, wall_c))
                cur_end_c = room_end_c
            else:
                cur_end_c = cols - 2

            # Place door between room and corridor along door_wall_r
            # Pick a column near the room center
            door_c = (cur_c + cur_end_c) // 2
            door_pos = (door_wall_r, door_c)
            walls.discard(door_pos)

            door_id = f"DOOR_{wing_name.upper()}_{room_idx:02d}"
            doors.append(
                DoorOpening(
                    door_id=door_id,
                    position=door_pos,
                    width=cfg.interior_door_width_m,
                    exterior=False,
                )
            )

            # Record room cells
            room_cells = [
                (r, c)
                for r in range(r_min, r_max + 1)
                for c in range(cur_c, cur_end_c + 1)
            ]
            r_type = rng.choice(self.ROOM_TYPES)
            rooms.append(
                CommercialRoom(
                    room_id=f"Room_{wing_name}_{room_idx:02d}",
                    room_type=r_type,
                    row_range=(r_min, r_max),
                    col_range=(cur_c, cur_end_c),
                    door_cell=door_pos,
                    cells=room_cells,
                )
            )

            cur_c = cur_end_c + 2
            room_idx += 1

    def _validate_layout(self, layout: CommercialLayout, cfg: CommercialGeneratorConfig) -> bool:
        """
        Validate generated layout against all hard constraints.
        Returns True only if all checks pass.
        """
        fp = layout.floor_plan
        exit_cfg = layout.exit_configuration

        # 1. Structural checks
        exterior_doors = [d for d in fp.doors if d.exterior]
        if len(exterior_doors) < cfg.num_exterior_exits:
            return False

        if len(exit_cfg.exits) < 2:
            return False

        if len(layout.rooms) < 2:
            return False

        wall_set = set(map(tuple, fp.walls))
        for d in fp.doors:
            if tuple(d.position) in wall_set:
                return False

        # 2. Connectivity check via Environment & NavigationGraph
        env = Environment.from_floor_plan_and_exits(fp, exit_cfg)
        nav = NavigationGraph.from_environment(env)
        g = nav._graph

        # All exits must exist in the navigation graph
        for exit_pos in exit_cfg.exits:
            if not g.has_node(tuple(exit_pos)):
                return False

        # Check BFS reachability: every room must have at least one cell that can reach an exit
        exit_nodes = [tuple(p) for p in exit_cfg.exits]
        for room in layout.rooms:
            # Pick a sample cell in room
            sample_cell = (
                (room.row_range[0] + room.row_range[1]) // 2,
                (room.col_range[0] + room.col_range[1]) // 2,
            )
            if not g.has_node(sample_cell):
                return False
            # Check if any path to any exit
            can_reach = any(nx.has_path(g, sample_cell, ex) for ex in exit_nodes)
            if not can_reach:
                return False

        # 3. Regulatory check via ExitValidator
        res = self.validator.check(fp, exit_cfg, OccupantScenario(agents=[]))
        # Note: res.is_compliant must be True
        if not res.is_compliant:
            return False

        return True
