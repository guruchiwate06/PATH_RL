"""
test_stage15_cubicasa_import.py
-------------------------------
Stage 15A: Comprehensive tests for the CubiCasa5K single-plan import pipeline.

Verifies:
1. Parsing sample 333 (walls, spaces, doors, columns, viewbox).
2. Physical scale extraction from usable dimension annotations (~100 px/m).
3. ScaleDerivationError when dimension annotations are missing or insufficient.
4. Manual scale override handling and validation.
5. Door classification:
   - Interior doors (PH->ET, PH->Sauna, ET->TK).
   - Balcony/outdoor door (OH->ULKOTILA) marked non-exit with review reason.
   - Uncertain door (VAR->ULKOTILA) marked non-exit.
   - Potential exterior opening (TK->exterior) marked candidate exit.
6. Grid bounds containment (0 <= r < rows, 0 <= c < cols).
7. Door carving and strict door-wall non-overlap.
8. Navigation connectivity preservation across interior rooms.
9. Candidate exit derivation strictly selects justified exterior openings.
10. Review manifest structure, content, and JSON serializability.
11. Exported scenario JSON validity via SimulationConfig.from_json.
12. Deterministic conversion across multiple invocations.
13. Robust handling of invalid/corrupted inputs.
14. Comparison between 0.5m and 1.0m resolutions.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from evacuation_simulation.simulation.config import ExitConfiguration, FloorPlan, SimulationConfig
from evacuation_simulation.simulation.converters import (
    CubiCasaParser,
    CubiCasaSVGData,
    DoorClassification,
    DoorClassifier,
    GridDiscretizer,
    ParsedSpace,
    ScaleDerivationError,
    ScaleEstimator,
    import_cubicasa_plan,
)
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.generation import generate_candidate_exits
from evacuation_simulation.simulation.pathfinding import NavigationGraph

SAMPLE_333_PATH = Path(r"F:\Projects\RL_ENV\dataset\cubicasa5k\high_quality_architectural\333\model.svg")


# ==============================================================================
# 1. Parsing & Geometry Extraction
# ==============================================================================


def test_parse_sample_333_structure() -> None:
    """Parse sample 333 and verify elements are cleanly extracted."""
    data = CubiCasaParser.parse_file(SAMPLE_333_PATH)

    assert data.viewbox_width > 1000.0
    assert data.viewbox_height > 500.0
    assert len(data.spaces) == 10
    assert len(data.walls) == 19
    assert len(data.doors) == 6
    assert len(data.columns) > 0


def test_extract_wall_and_space_polygons() -> None:
    """Verify extracted wall and space polygons have valid coordinates."""
    data = CubiCasaParser.parse_file(SAMPLE_333_PATH)

    space_names = [s.name_label for s in data.spaces]
    assert "OH" in space_names
    assert "K" in space_names
    assert "ET" in space_names
    assert "TK" in space_names
    assert "ALKOVI" in space_names
    assert "PH" in space_names
    assert "VAR" in space_names

    # Check all polygons have at least 3 vertices
    for s in data.spaces:
        assert len(s.polygon) >= 3
    for w in data.walls:
        assert len(w.polygon) >= 3
    for d in data.doors:
        assert len(d.polygon) >= 3


# ==============================================================================
# 2. Physical Scale Derivation
# ==============================================================================


def test_scale_extraction_sample_333() -> None:
    """Derive physical scale from sample 333 dimension annotations (~100 px/m)."""
    data = CubiCasaParser.parse_file(SAMPLE_333_PATH)
    scale_info = ScaleEstimator.estimate_scale(data.spaces)

    assert scale_info.is_derived is True
    assert scale_info.sample_count == 20
    # Expected scale is approx 100.0 px per meter
    assert 99.0 <= scale_info.scale_px_per_m <= 101.0
    assert scale_info.std_dev is not None
    assert scale_info.std_dev < 1.0


def test_scale_extraction_failure_on_missing_data() -> None:
    """Raise ScaleDerivationError when spaces lack dimension annotations."""
    dummy_spaces = [
        ParsedSpace(
            space_class="Space LivingRoom",
            name_label="OH",
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            dimension_texts=[],
            dimension_marks={},
        )
    ]
    with pytest.raises(ScaleDerivationError, match="no usable metric dimension labels"):
        ScaleEstimator.estimate_scale(dummy_spaces)


def test_manual_scale_override() -> None:
    """Verify manual scale override bypasses derivation and enforces validation."""
    dummy_spaces: list[ParsedSpace] = []
    scale_info = ScaleEstimator.estimate_scale(dummy_spaces, manual_override=120.0)

    assert scale_info.is_derived is False
    assert scale_info.scale_px_per_m == 120.0
    assert scale_info.sample_count == 0

    with pytest.raises(ScaleDerivationError, match="must be positive"):
        ScaleEstimator.estimate_scale(dummy_spaces, manual_override=-10.0)

    with pytest.raises(ScaleDerivationError, match="must be positive"):
        ScaleEstimator.estimate_scale(dummy_spaces, manual_override=0.0)


# ==============================================================================
# 3. Contextual Door Classification & Review
# ==============================================================================


def test_door_classification_sample_333() -> None:
    """Verify door classifications distinguish interior, balcony, and potential exterior."""
    data = CubiCasaParser.parse_file(SAMPLE_333_PATH)
    records = DoorClassifier.classify(data.doors, data.spaces, scale_px_per_m=100.0)

    assert len(records) == 6

    # Count by classification
    by_class: dict[DoorClassification, list] = {}
    for r in records:
        by_class.setdefault(r.classification, []).append(r)

    # 3 interior partition doors
    assert len(by_class[DoorClassification.INTERIOR]) == 3
    for r in by_class[DoorClassification.INTERIOR]:
        assert r.is_candidate_exit is False
        assert r.review_required is False

    # 1 balcony outdoor door (Living Room -> ULKOTILA)
    assert len(by_class[DoorClassification.BALCONY_OUTDOOR]) == 1
    balcony_door = by_class[DoorClassification.BALCONY_OUTDOOR][0]
    assert balcony_door.is_candidate_exit is False
    assert balcony_door.review_required is True
    assert "balcony" in balcony_door.review_reason.lower()

    # 1 potential exterior opening (TK DraughtLobby -> exterior)
    assert len(by_class[DoorClassification.POTENTIAL_EXTERIOR]) == 1
    ext_door = by_class[DoorClassification.POTENTIAL_EXTERIOR][0]
    assert ext_door.is_candidate_exit is True
    assert ext_door.review_required is True
    assert "TK" in ext_door.adjacent_spaces[0] or "TK" in ext_door.adjacent_spaces[1]

    # 1 uncertain door (VAR Storage -> ULKOTILA)
    assert len(by_class[DoorClassification.UNCERTAIN]) == 1
    uncertain_door = by_class[DoorClassification.UNCERTAIN][0]
    assert uncertain_door.is_candidate_exit is False
    assert uncertain_door.review_required is True


# ==============================================================================
# 4. Continuous-to-Grid Discretization & Wall-Door Non-Overlap
# ==============================================================================


def test_grid_discretization_and_carving() -> None:
    """Verify wall rasterization, bounds containment, and door carving."""
    res = import_cubicasa_plan(SAMPLE_333_PATH, cell_size_m=0.5)
    fp = res.floor_plan

    # Grid dimensions
    assert fp.grid.rows == 12
    assert fp.grid.cols == 27
    assert len(fp.walls) > 0
    assert len(fp.doors) == 6

    # Verify bounds
    for r, c in fp.walls:
        assert 0 <= r < fp.grid.rows
        assert 0 <= c < fp.grid.cols

    for d in fp.doors:
        r, c = d.position
        assert 0 <= r < fp.grid.rows
        assert 0 <= c < fp.grid.cols

    # Critical PATH_RL constraint: door cells must NOT overlap with walls
    wall_set = set(map(tuple, fp.walls))
    door_set = {tuple(d.position) for d in fp.doors}
    overlap = wall_set & door_set
    assert len(overlap) == 0, f"Detected wall-door overlap at cells: {overlap}"


# ==============================================================================
# 5. Candidate Exits & Regulatory Integration
# ==============================================================================


def test_candidate_exits_strictly_justified() -> None:
    """Verify candidate-exit derivation strictly excludes balcony and interior doors."""
    res = import_cubicasa_plan(SAMPLE_333_PATH, cell_size_m=0.5)

    # Exactly 1 candidate exit derived (from Door 1)
    assert len(res.candidate_exits) == 1
    cand = res.candidate_exits[0]
    assert cand.door_id == "door_1"
    assert cand.position == (9, 16)
    assert cand.width > 0.7


# ==============================================================================
# 6. Navigation Connectivity Preservation
# ==============================================================================


def test_navigation_connectivity_sample_333() -> None:
    """Verify indoor rooms can navigate across door passages to the candidate exit."""
    res = import_cubicasa_plan(SAMPLE_333_PATH, cell_size_m=0.5)
    cand_exit = res.candidate_exits[0]

    exit_cfg = ExitConfiguration(exits=[cand_exit.position], widths=[cand_exit.width])
    env = Environment.from_floor_plan_and_exits(res.floor_plan, exit_cfg)
    nav = NavigationGraph.from_environment(env)

    # Pick an interior point in the living room: OH is roughly at r=3, c=9
    living_room_cell = (3, 9)
    assert env.is_passable(living_room_cell[0], living_room_cell[1])

    # Find shortest path from living room to candidate exit
    path = nav.shortest_path(living_room_cell, cand_exit.position)
    assert path is not None
    assert len(path) > 0
    assert path[0] == living_room_cell
    assert path[-1] == cand_exit.position


# ==============================================================================
# 7. Manifest & Scenario Export Validation
# ==============================================================================


def test_review_manifest_and_json_export(tmp_path: Path) -> None:
    """Verify review manifest and scenario JSON can be exported and reloaded cleanly."""
    res = import_cubicasa_plan(SAMPLE_333_PATH, cell_size_m=0.5)

    # Export manifest
    manifest_file = tmp_path / "review_manifest.json"
    res.export_review_manifest(manifest_file)
    assert manifest_file.exists()

    with manifest_file.open("r", encoding="utf-8") as fh:
        loaded_manifest = json.load(fh)
    assert loaded_manifest["total_doors_detected"] == 6
    assert loaded_manifest["candidate_exterior_exits_derived"] == 1
    assert len(loaded_manifest["doors"]) == 6

    # Export scenario JSON
    scenario_file = tmp_path / "imported_scenario.json"
    res.export_scenario_json(scenario_file)
    assert scenario_file.exists()

    # Validate that exported scenario JSON loads cleanly in PATH_RL SimulationConfig
    sim_config = SimulationConfig.from_json(scenario_file)
    assert sim_config.floor_plan.grid.rows == 12
    assert sim_config.floor_plan.grid.cols == 27
    assert len(sim_config.exit_configuration.exits) == 1


# ==============================================================================
# 8. Determinism & Robustness
# ==============================================================================


def test_conversion_is_deterministic() -> None:
    """Running conversion twice on the same plan produces identical output."""
    res1 = import_cubicasa_plan(SAMPLE_333_PATH, cell_size_m=0.5)
    res2 = import_cubicasa_plan(SAMPLE_333_PATH, cell_size_m=0.5)

    assert res1.floor_plan.grid.rows == res2.floor_plan.grid.rows
    assert res1.floor_plan.grid.cols == res2.floor_plan.grid.cols
    assert res1.floor_plan.walls == res2.floor_plan.walls
    assert len(res1.floor_plan.doors) == len(res2.floor_plan.doors)
    for d1, d2 in zip(res1.floor_plan.doors, res2.floor_plan.doors):
        assert d1.position == d2.position
        assert d1.width == d2.width
        assert d1.exterior == d2.exterior


def test_error_handling_invalid_inputs(tmp_path: Path) -> None:
    """Verify clear error handling for non-existent files and malformed SVG."""
    with pytest.raises(FileNotFoundError):
        import_cubicasa_plan(tmp_path / "nonexistent.svg")

    # Malformed XML
    bad_svg = tmp_path / "bad.svg"
    bad_svg.write_text("<svg><unclosed>", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed SVG XML"):
        import_cubicasa_plan(bad_svg)

    # Invalid cell size
    with pytest.raises(ValueError, match="strictly positive"):
        import_cubicasa_plan(SAMPLE_333_PATH, cell_size_m=-0.5)


# ==============================================================================
# 9. Resolution Comparison (0.5m vs 1.0m)
# ==============================================================================


def test_resolution_tradeoffs_05m_vs_10m() -> None:
    """Evaluate grid resolution trade-offs between 0.5m and 1.0m."""
    res_05 = import_cubicasa_plan(SAMPLE_333_PATH, cell_size_m=0.5)
    res_10 = import_cubicasa_plan(SAMPLE_333_PATH, cell_size_m=1.0)

    # 0.5m resolution has 4x area cells
    assert res_05.floor_plan.grid.rows == 12
    assert res_05.floor_plan.grid.cols == 27
    assert res_10.floor_plan.grid.rows == 6
    assert res_10.floor_plan.grid.cols == 14

    # Wall cells count: 0.5m captures wall contours with finer granularity
    assert len(res_05.floor_plan.walls) > len(res_10.floor_plan.walls)
