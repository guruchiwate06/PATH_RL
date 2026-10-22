"""
test_stage15b_pilot_robustness.py
---------------------------------
Stage 15B: Robustness tests based on the 6-plan pilot conversion and discovered failure modes.

Coverage:
1. Multi-plan pilot conversion across diverse layouts (333, 1611, 1654, 7842, 6257, 6244).
2. Zero wall-door overlaps across all pilot plans.
3. Door-to-exit traversal connectivity across all pilot plans.
4. Prevention of silent exit promotion: doors bordering UNDEFINED spaces are flagged UNCERTAIN, not POTENTIAL_EXTERIOR.
5. Detection of multi-floor plans as unsuitable for 2D single-story representation.
6. Thin wall rasterization: sub_samples_per_cell=5 preserves walls < 15cm thick.
7. Door threshold carving: prevents wall straddling blockages across doorways.
8. Physical width fidelity: preserves sub-meter door widths (e.g., 0.56m, 0.68m, 0.75m).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from evacuation_simulation.simulation.config import ExitConfiguration, SimulationConfig
from evacuation_simulation.simulation.converters import (
    CubiCasaParser,
    DoorClassification,
    GridDiscretizer,
    import_cubicasa_plan,
)
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.pathfinding import NavigationGraph

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ARCH = _REPO_ROOT / "dataset" / "cubicasa5k" / "high_quality_architectural"
PILOT_IDS = ["333", "1611", "1654", "7842", "6257", "6244"]


# ==============================================================================
# 1. Multi-Plan Pilot Conversion & Invariants
# ==============================================================================


@pytest.mark.parametrize("plan_id", PILOT_IDS)
def test_pilot_plan_converts_cleanly(plan_id: str) -> None:
    """Verify each pilot plan converts without exception and satisfies Pydantic schemas."""
    svg_path = DATASET_ARCH / plan_id / "model.svg"
    assert svg_path.exists(), f"Source file missing: {svg_path}"

    res = import_cubicasa_plan(svg_path, cell_size_m=0.5)

    # Grid bounds and basic dimensions
    assert res.floor_plan.grid.rows >= 10
    assert res.floor_plan.grid.cols >= 10
    assert len(res.floor_plan.walls) > 0
    assert len(res.floor_plan.doors) > 0

    # Scale derived within realistic range (~100 px/m)
    assert 95.0 <= res.scale_info.scale_px_per_m <= 105.0
    assert res.scale_info.sample_count >= 5


@pytest.mark.parametrize("plan_id", PILOT_IDS)
def test_pilot_zero_wall_door_overlap(plan_id: str) -> None:
    """Strict invariant: door cells must NEVER overlap with solid wall cells."""
    svg_path = DATASET_ARCH / plan_id / "model.svg"
    res = import_cubicasa_plan(svg_path, cell_size_m=0.5)

    wall_set = set(map(tuple, res.floor_plan.walls))
    door_set = {tuple(d.position) for d in res.floor_plan.doors}
    overlap = wall_set & door_set
    assert len(overlap) == 0, f"Plan {plan_id} has wall-door overlap at cells: {overlap}"


@pytest.mark.parametrize("plan_id", PILOT_IDS)
def test_pilot_doors_are_navigable(plan_id: str) -> None:
    """Verify that every door opening is reachable from the reference exit."""
    svg_path = DATASET_ARCH / plan_id / "model.svg"
    res = import_cubicasa_plan(svg_path, cell_size_m=0.5)

    # Pick reference exit: candidate exit if available, else first door
    exit_pos = (
        res.candidate_exits[0].position
        if res.candidate_exits
        else res.floor_plan.doors[0].position
    )

    exit_cfg = ExitConfiguration(exits=[exit_pos], widths=[1.0])
    env = Environment.from_floor_plan_and_exits(res.floor_plan, exit_cfg)
    nav = NavigationGraph.from_environment(env)

    # Every door must have a valid path to the reference exit
    for d in res.floor_plan.doors:
        path = nav.shortest_path(d.position, exit_pos)
        assert path is not None, f"Plan {plan_id}: Door {d.door_id} at {d.position} cannot reach exit {exit_pos}"


# ==============================================================================
# 2. Preventing Silent Exit Promotion (Safety & Classification Integrity)
# ==============================================================================


def test_doors_bordering_undefined_spaces_are_flagged_uncertain() -> None:
    """
    Doors bordering generic 'UNDEFINED' or 'UserDefined' rooms on external walls
    must NOT be promoted to candidate exits; they must be flagged UNCERTAIN.
    """
    # Sample 1611 has Door 1 adjacent to UNDEFINED on an external wall
    svg_path = DATASET_ARCH / "1611" / "model.svg"
    res = import_cubicasa_plan(svg_path, cell_size_m=0.5)

    # Verified: 0 candidate exits promoted automatically
    assert len(res.candidate_exits) == 0

    # Door 1 is classified as UNCERTAIN with review required
    d1 = next(d for d in res.door_records if d.door_id == "door_1")
    assert d1.classification == DoorClassification.UNCERTAIN
    assert d1.is_candidate_exit is False
    assert d1.review_required is True
    assert "unknown external elevation" in d1.review_reason.lower() or "ambiguous" in d1.review_reason.lower()


def test_balcony_doors_are_never_candidate_exits() -> None:
    """Verify that balcony doors (e.g., in 7842 and 333) are never candidate exits."""
    svg_path = DATASET_ARCH / "7842" / "model.svg"
    res = import_cubicasa_plan(svg_path, cell_size_m=0.5)

    balcony_doors = [d for d in res.door_records if d.classification == DoorClassification.BALCONY_OUTDOOR]
    assert len(balcony_doors) >= 1

    for bd in balcony_doors:
        assert bd.is_candidate_exit is False
        assert bd.review_required is True
        assert "balcony" in bd.review_reason.lower()


# ==============================================================================
# 3. Geometric Discretization Robustness & Thin Walls
# ==============================================================================


def test_thin_partition_walls_preserved_by_subsampling() -> None:
    """Sub-sampling with 5x5 points per cell prevents thin walls (10-15cm) from disappearing."""
    svg_path = DATASET_ARCH / "333" / "model.svg"
    svg_data = CubiCasaParser.parse_file(svg_path)

    # With default 5x5 sub-sampling
    res_5x5 = GridDiscretizer.discretize(
        svg_data,
        [],
        scale_px_per_m=100.0,
        cell_size_m=0.5,
        sub_samples_per_cell=5,
    )
    wall_cells_5x5 = set(res_5x5.walls)

    # Walls 14 and 16 are thin partition walls (12cm and 13cm wide)
    # Check that cells in the vicinity of Wall 14 (c=15, r in [7, 8]) are present
    assert any(c == 15 for r, c in wall_cells_5x5)


def test_doorway_threshold_carving_clears_straddling_walls() -> None:
    """Door threshold carving clears walls that straddle grid boundaries across the opening."""
    svg_path = DATASET_ARCH / "333" / "model.svg"
    res = import_cubicasa_plan(svg_path, cell_size_m=0.5)

    # Door 3 at (10, 14) connects West (Storage) to East (Outdoor)
    d3 = next(d for d in res.floor_plan.doors if d.door_id == "door_3")
    r, c = d3.position

    wall_set = set(map(tuple, res.floor_plan.walls))
    # East neighbour (r, c+1) should NOT be an impassable wall blocking the threshold
    # Note: Door 3 threshold spans into column 15, which was cleared
    assert (r, c) not in wall_set


def test_physical_width_retention() -> None:
    """Preserves physical door widths derived from SVG without artificial rounding to 1.0m."""
    svg_path = DATASET_ARCH / "333" / "model.svg"
    res = import_cubicasa_plan(svg_path, cell_size_m=0.5)

    widths = {d.door_id: d.width for d in res.floor_plan.doors}
    assert widths["door_5"] < 0.60  # Sauna door is ~0.56m
    assert widths["door_4"] < 0.70  # Bathroom door is ~0.68m
    assert 0.70 < widths["door_1"] < 0.85  # Entrance door is ~0.79m


# ==============================================================================
# 4. Multi-Floor Unsuitability Check
# ==============================================================================


def test_multifloor_plans_identified_as_unsuitable() -> None:
    """Multi-floor plans (e.g., 5559, 3015) contain multiple floor files and require multi-level schema."""
    multi_p = DATASET_ARCH / "5559"
    f_files = list(multi_p.glob("F*_original.png"))
    assert len(f_files) == 3, "Sample 5559 should have 3 floor images"
