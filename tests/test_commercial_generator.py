"""
test_commercial_generator.py
-----------------------------
Unit tests for the procedural commercial floor-plan generator.
"""

import pytest
from evacuation_simulation.simulation.generators.commercial import (
    CommercialGeneratorConfig,
    CommercialLayoutGenerator,
    GenerationError,
)
from evacuation_simulation.simulation.config import FloorPlan


def test_generator_deterministic_seed():
    """Identical seeds must produce identical layouts."""
    generator = CommercialLayoutGenerator()
    config = CommercialGeneratorConfig(seed=12345, grid_rows=24, grid_cols=40)

    layout1 = generator.generate(config)
    layout2 = generator.generate(config)

    assert layout1.floor_plan.walls == layout2.floor_plan.walls
    assert len(layout1.floor_plan.doors) == len(layout2.floor_plan.doors)
    for d1, d2 in zip(layout1.floor_plan.doors, layout2.floor_plan.doors):
        assert d1.position == d2.position
        assert d1.width == d2.width
        assert d1.exterior == d2.exterior


def test_generator_hard_invariants():
    """Generated layout must strictly satisfy FloorPlan invariants."""
    generator = CommercialLayoutGenerator()
    config = CommercialGeneratorConfig(seed=42, grid_rows=24, grid_cols=40, num_exterior_exits=2)

    layout = generator.generate(config)
    fp = layout.floor_plan

    # 1. Bounds invariant
    for r, c in fp.walls:
        assert 0 <= r < fp.grid.rows
        assert 0 <= c < fp.grid.cols

    for door in fp.doors:
        assert 0 <= door.position[0] < fp.grid.rows
        assert 0 <= door.position[1] < fp.grid.cols

    # 2. No overlap between walls and doors
    wall_set = set(map(tuple, fp.walls))
    for door in fp.doors:
        assert tuple(door.position) not in wall_set

    # 3. Unique door positions
    door_positions = [tuple(d.position) for d in fp.doors]
    assert len(door_positions) == len(set(door_positions))

    # 4. Meets configured minimum exits (>= 2)
    assert len(layout.exit_configuration.exits) >= 2


def test_generator_multi_exit_styles():
    """Supports 2, 3, and 4 exterior exits."""
    generator = CommercialLayoutGenerator()
    for n_exits in [2, 3, 4]:
        config = CommercialGeneratorConfig(
            seed=100 + n_exits,
            grid_rows=26,
            grid_cols=44,
            num_exterior_exits=n_exits,
        )
        layout = generator.generate(config)
        exterior_doors = [d for d in layout.floor_plan.doors if d.exterior]
        assert len(exterior_doors) == n_exits


def test_generator_bounded_attempts():
    """Unfeasible configuration raises GenerationError without hanging."""
    generator = CommercialLayoutGenerator()
    # Unfeasible grid dimensions (too small for corridor + rooms)
    config = CommercialGeneratorConfig(
        seed=999,
        grid_rows=16,
        grid_cols=20,
        corridor_width_cells=10,  # leaves no room for walls/rooms
        max_attempts=2,
    )
    with pytest.raises(GenerationError):
        generator.generate(config)
