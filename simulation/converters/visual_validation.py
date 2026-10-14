"""
visual_validation.py
--------------------
Stage 15B: Visual & Quantitative Validation Tool for Converted Floor Plans.

Overlays source SVG vector geometry with the discretized 2D simulation grid.
Performs concrete, measurable checks:
1. Room interior reachability to exterior exit.
2. Doorway traversal connectivity across wall boundaries.
3. Disappearing wall polygon detection.
4. Continuous-to-discrete quantization error per door.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from evacuation_simulation.simulation.config import ExitConfiguration, FloorPlan
from evacuation_simulation.simulation.converters.cubicasa_parser import (
    CubiCasaParser,
    CubiCasaSVGData,
    ParsedSpace,
)
from evacuation_simulation.simulation.converters.door_classifier import (
    DoorClassification,
    DoorReviewRecord,
    point_in_polygon,
)
from evacuation_simulation.simulation.converters.importer import (
    CubiCasaImportResult,
    import_cubicasa_plan,
)
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.pathfinding import NavigationGraph


def validate_and_visualize_sample(
    svg_path: str | Path,
    output_png: str | Path,
    output_report_json: Optional[str | Path] = None,
    cell_size_m: float = 0.5,
) -> dict[str, Any]:
    """
    Generate an overlay visualization and quantitative validation report.
    """
    svg_path = Path(svg_path)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    # 1. Run import pipeline
    res = import_cubicasa_plan(svg_path, cell_size_m=cell_size_m)
    svg_data = CubiCasaParser.parse_file(svg_path)
    scale = res.scale_info.scale_px_per_m
    cell_px = cell_size_m * scale
    rows = res.floor_plan.grid.rows
    cols = res.floor_plan.grid.cols

    # Build environment and navigation graph
    cand_exit = res.candidate_exits[0] if res.candidate_exits else None
    exit_pos = cand_exit.position if cand_exit else res.floor_plan.doors[0].position
    exit_cfg = ExitConfiguration(exits=[exit_pos], widths=[1.0])
    env = Environment.from_floor_plan_and_exits(res.floor_plan, exit_cfg)
    nav = NavigationGraph.from_environment(env)

    # 2. Measurable geometric & topological checks
    report: dict[str, Any] = {
        "source_file": str(svg_path),
        "cell_size_m": cell_size_m,
        "grid_shape": [rows, cols],
        "scale_px_per_m": scale,
        "wall_cells_count": len(res.floor_plan.walls),
        "door_count": len(res.floor_plan.doors),
        "discrepancies": [],
        "room_connectivity": {},
        "door_alignment": {},
    }

    # Check 1: Disappearing wall polygons
    wall_cells_set = set(map(tuple, res.floor_plan.walls))
    for idx, wall in enumerate(svg_data.walls):
        has_rasterized_cell = False
        xs = [p[0] for p in wall.polygon]
        ys = [p[1] for p in wall.polygon]
        min_c = max(0, int(min(xs) / cell_px))
        max_c = min(cols - 1, int(max(xs) / cell_px))
        min_r = max(0, int(min(ys) / cell_px))
        max_r = min(rows - 1, int(max(ys) / cell_px))
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) in wall_cells_set:
                    has_rasterized_cell = True
                    break
            if has_rasterized_cell:
                break
        if not has_rasterized_cell:
            report["discrepancies"].append(
                f"Disappearing wall: Wall #{idx} ({wall.wall_class}) produced 0 rasterized wall cells."
            )

    # Check 2: Room interior connectivity
    for space in svg_data.spaces:
        # Calculate polygon interior sampling point
        xs = [p[0] for p in space.polygon]
        ys = [p[1] for p in space.polygon]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        # Verify centroid is inside polygon; if not, sample bounding box
        target_pt = (cx, cy)
        if not point_in_polygon(cx, cy, space.polygon):
            for test_x in np.linspace(min(xs), max(xs), 9):
                for test_y in np.linspace(min(ys), max(ys), 9):
                    if point_in_polygon(test_x, test_y, space.polygon):
                        target_pt = (test_x, test_y)
                        break
                if point_in_polygon(target_pt[0], target_pt[1], space.polygon):
                    break

        cr = min(rows - 1, max(0, int(target_pt[1] / cell_px)))
        cc = min(cols - 1, max(0, int(target_pt[0] / cell_px)))
        is_passable = env.is_passable(cr, cc)

        # Check path to exit
        path = nav.shortest_path((cr, cc), exit_pos) if is_passable else None
        sp_name = space.name_label or space.space_class.replace("Space ", "")

        report["room_connectivity"][f"{sp_name} ({space.space_class})"] = {
            "interior_cell": [cr, cc],
            "is_passable": is_passable,
            "can_reach_exit": path is not None,
            "path_length_steps": len(path) if path else None,
        }

        # Balconies / outdoor spaces may legitimately not reach exit if enclosed
        is_outdoor = "Outdoor" in space.space_class or "ULKOTILA" in sp_name
        if not is_outdoor and (not is_passable or path is None):
            report["discrepancies"].append(
                f"Disconnected room: Interior space '{sp_name}' at cell ({cr}, {cc}) cannot reach candidate exit {exit_pos}."
            )

    # Check 3: Door alignment & neighbor connectivity
    for record in res.door_records:
        dr, dc = record.grid_position
        cx, cy = record.svg_center
        # Quantization error (distance from SVG center to grid cell center in meters)
        cell_cx_px = (dc + 0.5) * cell_px
        cell_cy_px = (dr + 0.5) * cell_px
        dist_err_m = math.hypot(cx - cell_cx_px, cy - cell_cy_px) / scale

        # Cardinal open neighbours
        cardinals = [((dr - 1, dc), "N"), ((dr + 1, dc), "S"), ((dr, dc - 1), "W"), ((dr, dc + 1), "E")]
        open_dirs = [d for pos, d in cardinals if env.is_passable(pos[0], pos[1])]

        report["door_alignment"][record.door_id] = {
            "grid_position": [dr, dc],
            "quantization_error_m": round(dist_err_m, 3),
            "open_directions": open_dirs,
            "is_carved_passable": env.is_passable(dr, dc),
        }

        if len(open_dirs) == 0:
            report["discrepancies"].append(
                f"Blocked doorway: Door '{record.door_id}' at ({dr}, {dc}) has 0 passable cardinal neighbours."
            )

    # 3. Create Multi-Panel Matplotlib Figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=150)
    fig.suptitle(
        f"PATH_RL Stage 15B: Floor-Plan Overlay & Validation (Sample 333, Resolution: {cell_size_m}m)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    door_colors = {
        DoorClassification.POTENTIAL_EXTERIOR: "#2ecc71",  # Emerald Green
        DoorClassification.BALCONY_OUTDOOR: "#e67e22",     # Orange
        DoorClassification.UNCERTAIN: "#f1c40f",           # Yellow
        DoorClassification.INTERIOR: "#3498db",            # Blue
    }

    # Panel 1: Continuous SVG Vector Geometry
    ax1 = axes[0]
    ax1.set_title("1. Continuous Vector Geometry (SVG)", fontsize=13, fontweight="semibold")
    ax1.set_xlim(0, svg_data.viewbox_width)
    ax1.set_ylim(svg_data.viewbox_height, 0)  # Invert Y to match SVG coordinates
    ax1.set_aspect("equal")

    # Plot Spaces
    for s in svg_data.spaces:
        poly_pts = np.array(s.polygon)
        color = "#eaf2f8" if "Outdoor" not in s.space_class else "#fcf3cf"
        ax1.fill(poly_pts[:, 0], poly_pts[:, 1], color=color, alpha=0.6)
        ax1.plot(poly_pts[:, 0], poly_pts[:, 1], color="#aeb6bf", linewidth=0.8, linestyle="--")
        cx = np.mean(poly_pts[:, 0])
        cy = np.mean(poly_pts[:, 1])
        lbl = s.name_label or s.space_class.replace("Space ", "")
        ax1.text(cx, cy, lbl, fontsize=9, ha="center", va="center", color="#2c3e50", fontweight="bold")

    # Plot Walls
    for w in svg_data.walls:
        w_pts = np.array(w.polygon)
        edge_col = "#17202a" if "External" in w.wall_class else "#566573"
        ax1.fill(w_pts[:, 0], w_pts[:, 1], color=edge_col, alpha=0.85)

    # Plot Doors
    for r in res.door_records:
        cx, cy = r.svg_center
        col = door_colors.get(r.classification, "#95a5a6")
        ax1.scatter(cx, cy, color=col, s=80, edgecolors="black", zorder=5)
        ax1.text(cx, cy - 12, f"{r.door_id}\n({r.estimated_width_m}m)", fontsize=7, ha="center", color=col, fontweight="bold")

    # Panel 2: Discretized Grid Representation
    ax2 = axes[1]
    ax2.set_title(f"2. Discretized Simulation Grid ({rows}x{cols}, {cell_size_m}m)", fontsize=13, fontweight="semibold")
    ax2.set_xlim(0, cols)
    ax2.set_ylim(rows, 0)
    ax2.set_aspect("equal")

    # Grid background & walls
    grid_img = np.zeros((rows, cols, 3))
    grid_img[:, :] = [0.96, 0.96, 0.96]  # Light grey walkable
    for r, c in res.floor_plan.walls:
        grid_img[r, c] = [0.15, 0.15, 0.15]  # Dark grey wall

    ax2.imshow(grid_img, origin="upper", extent=[0, cols, rows, 0])

    # Grid lines
    ax2.set_xticks(np.arange(0, cols + 1, 2))
    ax2.set_yticks(np.arange(0, rows + 1, 2))
    ax2.grid(True, color="#d5d8dc", linewidth=0.5)

    # Plot Doors on Grid
    for r in res.door_records:
        dr, dc = r.grid_position
        col = door_colors.get(r.classification, "#95a5a6")
        rect = patches.Rectangle((dc, dr), 1, 1, linewidth=1.5, edgecolor=col, facecolor=col, alpha=0.7)
        ax2.add_patch(rect)
        ax2.text(dc + 0.5, dr + 0.5, r.door_id.replace("door_", "D"), fontsize=8, ha="center", va="center", color="white", fontweight="bold")

    # Panel 3: Direct Visual Overlay
    ax3 = axes[2]
    ax3.set_title("3. Overlay: Vector Geometry over Discretized Grid", fontsize=13, fontweight="semibold")
    ax3.set_xlim(0, cols)
    ax3.set_ylim(rows, 0)
    ax3.set_aspect("equal")

    # Background grid
    ax3.imshow(grid_img, origin="upper", extent=[0, cols, rows, 0], alpha=0.55)
    ax3.grid(True, color="#bdc3c7", linewidth=0.5, linestyle=":")

    # Overlay continuous walls scaled to grid coordinates
    for w in svg_data.walls:
        w_pts = np.array(w.polygon) / cell_px  # Convert px to grid coordinates
        ax3.plot(w_pts[:, 0], w_pts[:, 1], color="#e74c3c", linewidth=1.2)

    # Overlay doors
    for r in res.door_records:
        dr, dc = r.grid_position
        cx_grid = r.svg_center[0] / cell_px
        cy_grid = r.svg_center[1] / cell_px
        col = door_colors.get(r.classification, "#95a5a6")
        # Line from continuous center to discrete cell center
        ax3.plot([cx_grid, dc + 0.5], [cy_grid, dr + 0.5], color=col, linewidth=1.5, linestyle="-")
        ax3.scatter([cx_grid], [cy_grid], color=col, marker="o", s=40, edgecolors="black", zorder=6)
        ax3.scatter([dc + 0.5], [dr + 0.5], color=col, marker="s", s=60, edgecolors="black", zorder=6)

    # Legend on Panel 3
    legend_elements = [
        patches.Patch(facecolor="#2ecc71", edgecolor="black", label="Potential Exterior Exit"),
        patches.Patch(facecolor="#e67e22", edgecolor="black", label="Balcony/Outdoor Door"),
        patches.Patch(facecolor="#f1c40f", edgecolor="black", label="Uncertain Opening"),
        patches.Patch(facecolor="#3498db", edgecolor="black", label="Interior Door"),
        patches.Patch(facecolor="#151515", edgecolor="black", label="Discretized Wall Cell"),
        patches.Patch(facecolor="#e74c3c", edgecolor="black", label="Source SVG Wall Contour"),
    ]
    ax3.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    if output_report_json:
        p_json = Path(output_report_json)
        p_json.parent.mkdir(parents=True, exist_ok=True)
        with p_json.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    return report
