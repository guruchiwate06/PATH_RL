"""
grid_discretizer.py
-------------------
Stage 15A: Continuous-to-Discrete Grid Discretization.

Rasterizes continuous SVG polygons into a discrete 2D grid matrix of (row, col) cells.
Ensures:
1. Strict boundary containment (0 <= r < rows, 0 <= c < cols).
2. Contiguous wall barriers.
3. Passable carved door cells (satisfies doors_and_walls_must_not_overlap).
4. Configurable spatial resolution (e.g., 0.5m vs 1.0m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from evacuation_simulation.simulation.config import DoorOpening, GridCell
from evacuation_simulation.simulation.converters.cubicasa_parser import (
    CubiCasaSVGData,
    ParsedSpace,
    ParsedWall,
    Polygon2D,
)
from evacuation_simulation.simulation.converters.door_classifier import (
    DoorReviewRecord,
    point_in_polygon,
)


@dataclass
class DiscretizationResult:
    """Output of continuous-to-grid conversion."""
    grid_rows: int
    grid_cols: int
    cell_size_m: float
    walls: list[GridCell]
    doors: list[DoorOpening]
    door_records: list[DoorReviewRecord]


class GridDiscretizer:
    """
    Converts vector SVG floor-plan geometry into discrete PATH_RL grid representation.
    """

    @classmethod
    def discretize(
        cls,
        svg_data: CubiCasaSVGData,
        door_records: list[DoorReviewRecord],
        scale_px_per_m: float,
        cell_size_m: float = 0.5,
        sub_samples_per_cell: int = 5,
    ) -> DiscretizationResult:
        """
        Discretize vector SVG geometry onto a 2D orthogonal grid.

        Parameters
        ----------
        svg_data:
            Parsed SVG data containing walls, columns, and spaces.
        door_records:
            Classified door records with SVG centers and widths.
        scale_px_per_m:
            Scale factor in pixels per meter.
        cell_size_m:
            Physical size of one grid cell in meters (default 0.5m).
        sub_samples_per_cell:
            Number of sample points along each axis inside a grid cell (default 3x3).

        Returns
        -------
        DiscretizationResult
            Grid dimensions, wall cells, and DoorOpening objects.
        """
        if cell_size_m <= 0.0:
            raise ValueError(f"cell_size_m must be strictly positive, got {cell_size_m}")
        if scale_px_per_m <= 0.0:
            raise ValueError(f"scale_px_per_m must be strictly positive, got {scale_px_per_m}")

        cell_px = cell_size_m * scale_px_per_m
        cols = int(math.ceil(svg_data.viewbox_width / cell_px))
        rows = int(math.ceil(svg_data.viewbox_height / cell_px))

        # Collect impassable obstacle polygons: walls and structural columns
        obstacle_polygons: list[Polygon2D] = [w.polygon for w in svg_data.walls]
        obstacle_polygons.extend(c.polygon for c in svg_data.columns)

        # 1. Rasterize obstacle polygons
        wall_cells: set[GridCell] = set()
        sub_n = max(1, sub_samples_per_cell)

        for r in range(rows):
            for c in range(cols):
                is_wall = False
                for si in range(sub_n):
                    for sj in range(sub_n):
                        px = (c + (si + 0.5) / sub_n) * cell_px
                        py = (r + (sj + 0.5) / sub_n) * cell_px
                        for poly in obstacle_polygons:
                            if point_in_polygon(px, py, poly):
                                is_wall = True
                                break
                        if is_wall:
                            break
                    if is_wall:
                        break
                if is_wall:
                    wall_cells.add((r, c))

        # 2. Carve all cells intersecting door threshold openings from wall_cells
        for door_elem in svg_data.doors:
            xs = [p[0] for p in door_elem.polygon]
            ys = [p[1] for p in door_elem.polygon]
            min_c = max(0, int(min(xs) / cell_px))
            max_c = min(cols - 1, int(max(xs) / cell_px))
            min_r = max(0, int(min(ys) / cell_px))
            max_r = min(rows - 1, int(max(ys) / cell_px))
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    # Check if cell center or sub-samples intersect door opening
                    inter = False
                    for si in range(sub_n):
                        for sj in range(sub_n):
                            px = (c + (si + 0.5) / sub_n) * cell_px
                            py = (r + (sj + 0.5) / sub_n) * cell_px
                            if point_in_polygon(px, py, door_elem.polygon):
                                inter = True
                                break
                        if inter:
                            break
                    if inter:
                        wall_cells.discard((r, c))

        # 3. Map door records to primary grid positions
        door_openings: list[DoorOpening] = []
        assigned_door_positions: set[GridCell] = set()

        for record in door_records:
            cx, cy = record.svg_center
            dr = min(rows - 1, max(0, int(cy / cell_px)))
            dc = min(cols - 1, max(0, int(cx / cell_px)))
            door_pos: GridCell = (dr, dc)

            # Handle duplicate cell positions by local search if resolution is coarse
            if door_pos in assigned_door_positions:
                # Search 1-neighbourhood for nearest unassigned cell along opening
                placed = False
                for dr_adj, dc_adj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    cand_pos = (dr + dr_adj, dc + dc_adj)
                    if (
                        0 <= cand_pos[0] < rows
                        and 0 <= cand_pos[1] < cols
                        and cand_pos not in assigned_door_positions
                    ):
                        door_pos = cand_pos
                        placed = True
                        break
                if not placed:
                    pass

            assigned_door_positions.add(door_pos)
            record.grid_position = door_pos

            # Ensure primary position is never a wall
            wall_cells.discard(door_pos)

            door_openings.append(
                DoorOpening(
                    door_id=record.door_id,
                    position=door_pos,
                    width=max(0.1, round(record.estimated_width_m, 2)),
                    exterior=record.is_candidate_exit,
                )
            )

        # Sort wall cells deterministically
        sorted_walls = sorted(list(wall_cells))

        return DiscretizationResult(
            grid_rows=rows,
            grid_cols=cols,
            cell_size_m=cell_size_m,
            walls=sorted_walls,
            doors=door_openings,
            door_records=door_records,
        )


def discretize_plan(
    svg_data: CubiCasaSVGData,
    door_records: list[DoorReviewRecord],
    scale_px_per_m: float,
    cell_size_m: float = 0.5,
) -> DiscretizationResult:
    """Convenience functional wrapper for GridDiscretizer.discretize."""
    return GridDiscretizer.discretize(
        svg_data,
        door_records,
        scale_px_per_m,
        cell_size_m=cell_size_m,
    )
