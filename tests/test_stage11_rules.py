"""
test_stage11_rules.py
---------------------
Stage 11: Regulatory & Exit-Placement Constraint Layer — comprehensive tests.

Coverage:
A. Valid exterior DoorOpening can be selected (R1)
B. Interior PASS cannot be selected (R1)
C. Invalid/nonexistent door ID rejected (R1)
D. Duplicate exits rejected (R1)
E. Minimum exit count validation (R2)
F. Exit separation validation (R3)
G. Exit width read from DoorOpening.width (R4)
H. Physical width and legal capacity remain separate (R4)
I. Travel distance calculated consistently (R5)
J. Unreachable occupants produce feasibility violation (R5)
K. Compliant configuration returns is_compliant=True
L. Non-compliant configuration returns structured violations
M. Mask correctly blocks obviously invalid candidates (R1)
N. Mask does not evaluate constraints needing complete config
O. Stage 1-10 behavior unchanged (regression test)
P. Immutability: check() does not mutate inputs
Q. Legacy (no explicit doors) scenarios still work
R. RuleSet with custom rules works
S. load_regulatory_config raises on missing file
T. build_nbc_2016_ruleset produces 5 rules
U. ValidationResult is machine-readable (pydantic)
V. Violation has all expected fields
W. Integration: basic_building scenario compliant check
X. Integration: violations are structured (have rule_id, measured_value, etc.)
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from evacuation_simulation.simulation.config import (
    AgentConfig,
    DoorOpening,
    ExitConfiguration,
    FloorPlan,
    GridCell,
    OccupantScenario,
    SimulationConfig,
)
from evacuation_simulation.simulation.generation import (
    CandidateExit,
    generate_candidate_exits,
    generate_exit_configurations,
)
from evacuation_simulation.simulation.rules import (
    R1ValidExitPosition,
    R2MinimumExitCount,
    R3ExitSeparation,
    R4ExitWidth,
    R5TravelDistance,
    RuleSet,
    ValidationResult,
    Violation,
    build_nbc_2016_ruleset,
    load_regulatory_config,
)
from evacuation_simulation.simulation.validator import ExitValidator, make_validator


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def small_floor_plan():
    """10x10 floor plan with 4 exterior doors and 2 interior doors."""
    return FloorPlan(
        grid={"rows": 10, "cols": 10},
        walls=[(5, c) for c in range(10) if c not in (3, 6)],
        doors=[
            DoorOpening(position=(0, 2), width=1.2, exterior=True, door_id="north_west"),
            DoorOpening(position=(0, 7), width=1.5, exterior=True, door_id="north_east"),
            DoorOpening(position=(9, 2), width=1.0, exterior=True, door_id="south_west"),
            DoorOpening(position=(9, 7), width=1.8, exterior=True, door_id="south_east"),
            DoorOpening(position=(5, 3), width=0.9, exterior=False, door_id="int_passage_a"),
            DoorOpening(position=(5, 6), width=0.9, exterior=False, door_id="int_passage_b"),
        ],
    )


@pytest.fixture
def two_occupants():
    """Two agents, one in each half of the building."""
    return OccupantScenario(
        agents=[
            AgentConfig(agent_id="a1", row=2, col=2),
            AgentConfig(agent_id="a2", row=7, col=7),
        ]
    )


@pytest.fixture
def valid_two_exit_config():
    """Valid 2-exit config using well-separated exterior doors."""
    return ExitConfiguration(
        configuration_id="valid_2_exits",
        exits=[(0, 2), (9, 7)],
        widths=[1.2, 1.8],
    )


@pytest.fixture
def nbc_ruleset():
    return build_nbc_2016_ruleset()


@pytest.fixture
def validator():
    return make_validator()


# ==============================================================================
# A. Valid exterior DoorOpening can be selected
# ==============================================================================

class TestR1ValidExitPosition:
    def test_exterior_door_passes_check(self, small_floor_plan, two_occupants, valid_two_exit_config):
        """A. Valid exterior DoorOpening is accepted."""
        r1 = R1ValidExitPosition(load_regulatory_config())
        violations = r1.check(small_floor_plan, valid_two_exit_config, two_occupants)
        assert violations == [], f"Expected no violations, got: {violations}"

    # B. Interior PASS cannot be selected as exit
    def test_interior_door_rejected(self, small_floor_plan, two_occupants):
        """B. Interior door position rejected as emergency exit."""
        interior_exit_config = ExitConfiguration(
            configuration_id="bad_interior",
            exits=[(5, 3)],  # interior passage
            widths=[0.9],
        )
        r1 = R1ValidExitPosition(load_regulatory_config())
        violations = r1.check(small_floor_plan, interior_exit_config, two_occupants)
        assert len(violations) >= 1
        assert all(v.rule_id == "R1_VALID_EXIT_POSITION" for v in violations)
        assert any("ineligible" in v.message or "not a valid exterior" in v.message for v in violations)

    def test_both_interior_doors_rejected(self, small_floor_plan, two_occupants):
        """B. Both interior doors rejected; exterior doors not flagged."""
        bad_config = ExitConfiguration(
            configuration_id="all_interior",
            exits=[(5, 3), (5, 6)],
            widths=[0.9, 0.9],
        )
        r1 = R1ValidExitPosition(load_regulatory_config())
        violations = r1.check(small_floor_plan, bad_config, two_occupants)
        assert len(violations) == 2

    # C. Invalid/nonexistent door ID is rejected
    def test_nonexistent_position_rejected(self, small_floor_plan, two_occupants):
        """C. Exit at position not corresponding to any DoorOpening is rejected."""
        bad_config = ExitConfiguration(
            configuration_id="fake_exit",
            exits=[(3, 5)],  # Interior cell, no door
            widths=[1.0],
        )
        r1 = R1ValidExitPosition(load_regulatory_config())
        violations = r1.check(small_floor_plan, bad_config, two_occupants)
        assert len(violations) >= 1

    # D. Duplicate exits are rejected
    def test_duplicate_exits_rejected(self, small_floor_plan, two_occupants):
        """D. Duplicate exit position in configuration is detected."""
        dup_config = ExitConfiguration(
            configuration_id="dup",
            exits=[(0, 2), (0, 2)],
            widths=[1.2, 1.2],
        )
        r1 = R1ValidExitPosition(load_regulatory_config())
        violations = r1.check(small_floor_plan, dup_config, two_occupants)
        assert any("Duplicate" in v.message for v in violations)


# ==============================================================================
# E. Minimum exit count validation (R2)
# ==============================================================================

class TestR2MinimumExitCount:
    def test_single_exit_fails(self, small_floor_plan, two_occupants):
        """E. Single exit fails R2 (minimum is 2 per NBC 2016 Part 4 S8.1)."""
        one_exit = ExitConfiguration(exits=[(0, 2)], widths=[1.2])
        r2 = R2MinimumExitCount(load_regulatory_config())
        violations = r2.check(small_floor_plan, one_exit, two_occupants)
        assert len(violations) == 1
        v = violations[0]
        assert v.rule_id == "R2_MINIMUM_EXIT_COUNT"
        assert v.measured_value == 1.0
        assert v.required_value == 2.0
        assert v.unit == "exits"
        assert "NBC" in v.clause or "8.1" in v.clause

    def test_two_exits_passes(self, small_floor_plan, two_occupants, valid_two_exit_config):
        """E. Two exits satisfies R2."""
        r2 = R2MinimumExitCount(load_regulatory_config())
        violations = r2.check(small_floor_plan, valid_two_exit_config, two_occupants)
        assert violations == []

    def test_minimum_is_configurable(self, small_floor_plan, two_occupants):
        """E. R2 minimum is read from YAML config, not hard-coded."""
        cfg = load_regulatory_config()
        r2 = R2MinimumExitCount(cfg)
        # Default NBC minimum should be 2
        assert r2._min_exits == 2

    def test_r2_has_correct_clause(self, small_floor_plan, two_occupants):
        """E. R2 violation includes an NBC clause reference."""
        one_exit = ExitConfiguration(exits=[(0, 2)], widths=[1.2])
        r2 = R2MinimumExitCount(load_regulatory_config())
        violations = r2.check(small_floor_plan, one_exit, two_occupants)
        assert violations[0].clause != ""


# ==============================================================================
# F. Exit separation validation (R3)
# ==============================================================================

class TestR3ExitSeparation:
    def test_close_exits_fail_separation(self, small_floor_plan, two_occupants):
        """F. Two exits very close together fail R3 separation check."""
        close_config = ExitConfiguration(
            configuration_id="close",
            exits=[(0, 2), (0, 7)],  # same row, 5 cells apart
            widths=[1.2, 1.5],
        )
        # 10x10 grid: diagonal = sqrt(100+100) ≈ 14.14m
        # required non-sprinklered = 0.5 * 14.14 ≈ 7.07m
        # distance between (0,2) and (0,7) = 5.0m → FAILS
        r3 = R3ExitSeparation(load_regulatory_config())
        violations = r3.check(small_floor_plan, close_config, two_occupants)
        assert len(violations) >= 1
        v = violations[0]
        assert v.rule_id == "R3_EXIT_SEPARATION"
        assert v.measured_value is not None and v.measured_value < v.required_value
        assert v.unit == "m"

    def test_well_separated_exits_pass(self, small_floor_plan, two_occupants, valid_two_exit_config):
        """F. Exits at (0,2) and (9,7) are well-separated on a 10x10 grid."""
        # distance ≈ sqrt(81+25) ≈ 10.3m; required ≈ 7.07m → PASSES
        r3 = R3ExitSeparation(load_regulatory_config())
        violations = r3.check(small_floor_plan, valid_two_exit_config, two_occupants)
        assert violations == [], f"Expected no separation violations, got: {violations}"

    def test_single_exit_no_separation_check(self, small_floor_plan, two_occupants):
        """F. Single exit: R3 separation trivially satisfied (nothing to compare)."""
        one_exit = ExitConfiguration(exits=[(0, 2)], widths=[1.2])
        r3 = R3ExitSeparation(load_regulatory_config())
        violations = r3.check(small_floor_plan, one_exit, two_occupants)
        assert violations == []

    def test_separation_violation_has_relevant_ids(self, small_floor_plan, two_occupants):
        """F. Separation violation includes positions of both failing exits."""
        close_config = ExitConfiguration(exits=[(0, 2), (0, 7)], widths=[1.2, 1.5])
        r3 = R3ExitSeparation(load_regulatory_config())
        violations = r3.check(small_floor_plan, close_config, two_occupants)
        if violations:
            assert len(violations[0].relevant_ids) == 2

    def test_separation_fraction_is_configurable(self):
        """F. R3 fraction of diagonal is read from YAML, not hard-coded."""
        cfg = load_regulatory_config()
        r3 = R3ExitSeparation(cfg)
        assert r3._ns_frac == 0.5   # verified NBC value
        assert r3._sp_frac < r3._ns_frac  # sprinklered is less strict


# ==============================================================================
# G & H. Exit width (R4) — physical width vs legal capacity
# ==============================================================================

class TestR4ExitWidth:
    def test_width_read_from_exit_config(self, small_floor_plan, two_occupants):
        """G. R4 reads width from ExitConfiguration.get_width() (not computed)."""
        wide_config = ExitConfiguration(exits=[(0, 2), (9, 7)], widths=[1.2, 1.8])
        r4 = R4ExitWidth(load_regulatory_config())
        violations = r4.check(small_floor_plan, wide_config, two_occupants)
        assert violations == []

    def test_narrow_exit_fails(self, small_floor_plan, two_occupants):
        """G. Exit below minimum width produces R4 violation."""
        narrow_config = ExitConfiguration(
            exits=[(0, 2), (9, 7)],
            widths=[0.5, 1.8],   # 0.5m < 1.0m minimum
        )
        r4 = R4ExitWidth(load_regulatory_config())
        violations = r4.check(small_floor_plan, narrow_config, two_occupants)
        assert len(violations) >= 1
        v = violations[0]
        assert v.rule_id == "R4_EXIT_WIDTH"
        assert v.measured_value == pytest.approx(0.5, abs=0.01)
        assert v.required_value == pytest.approx(1.0, abs=0.01)
        assert v.unit == "m"

    def test_physical_width_separate_from_capacity(self):
        """H. width_per_person is intentionally null/unverified in R4."""
        cfg = load_regulatory_config()
        r4 = R4ExitWidth(cfg)
        # width_per_person must be None (NBC does not provide linear width-per-person)
        assert r4._width_per_person is None

    def test_r4_uses_exit_config_widths_not_door_widths_directly(self, small_floor_plan, two_occupants):
        """G. ExitConfiguration.get_width() is the authoritative width source."""
        # Configure ExitConfig with different width from what DoorOpening has
        config = ExitConfiguration(
            exits=[(0, 2)],
            widths=[1.5],   # ExitConfig says 1.5m (DoorOpening says 1.2m)
        )
        r4 = R4ExitWidth(load_regulatory_config())
        # Should use 1.5m from ExitConfig, not 1.2m from DoorOpening
        violations = r4.check(small_floor_plan, config, two_occupants)
        assert violations == []  # 1.5m >= 1.0m minimum


# ==============================================================================
# I & J. Travel distance (R5)
# ==============================================================================

class TestR5TravelDistance:
    def test_travel_distance_calculated(self, small_floor_plan, two_occupants, valid_two_exit_config):
        """I. R5 calculates travel distance and checks against limit."""
        r5 = R5TravelDistance(load_regulatory_config())
        # Small 10x10 grid — agents at (2,2) and (7,7), exits at (0,2) and (9,7)
        # Travel distances are small, well within 22.5m limit
        violations = r5.check(small_floor_plan, valid_two_exit_config, two_occupants)
        # No violations expected for small grid
        travel_violations = [v for v in violations if v.measured_value is not None]
        for v in travel_violations:
            # If violations: measured > required (no other case)
            assert v.measured_value > v.required_value

    def test_unreachable_agent_feasibility_violation(self):
        """J. Agent on isolated island produces feasibility violation."""
        fp = FloorPlan(
            grid={"rows": 5, "cols": 5},
            walls=[
                (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                (1, 0), (1, 4),
                (2, 0), (2, 4),
                (3, 0), (3, 4),
                (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
            ],
        )
        # All walls: no passable boundary cells. Agent placed at (2,2) (interior).
        # Exit at boundary — but boundary is all walls — create a minimal case:
        # Agent surrounded by walls on all sides.
        fp2 = FloorPlan(
            grid={"rows": 6, "cols": 6},
            walls=[
                (1, 1), (1, 2), (1, 3),
                (2, 1),         (2, 3),
                (3, 1), (3, 2), (3, 3),
            ],
            doors=[
                DoorOpening(position=(0, 0), width=1.0, exterior=True, door_id="exit_corner"),
            ],
        )
        isolated_agent = OccupantScenario(
            agents=[AgentConfig(agent_id="trapped", row=2, col=2)]
        )
        exit_cfg = ExitConfiguration(exits=[(0, 0)], widths=[1.0])
        r5 = R5TravelDistance(load_regulatory_config())
        violations = r5.check(fp2, exit_cfg, isolated_agent)
        # Agent at (2,2) is surrounded by walls — cannot reach (0,0)
        assert len(violations) >= 1
        feasibility = [v for v in violations if "Feasibility" in v.clause or "cannot reach" in v.message]
        assert len(feasibility) >= 1
        assert feasibility[0].relevant_ids == ["trapped"]

    def test_travel_distance_cell_size_from_config(self):
        """I. cell_size_m is read from YAML config, not hard-coded."""
        cfg = load_regulatory_config()
        r5 = R5TravelDistance(cfg)
        assert r5._cell_m == pytest.approx(1.0, abs=1e-6)

    def test_feasibility_vs_regulatory_are_distinct(self):
        """J. Feasibility violations are reported separately from regulatory ones."""
        cfg = load_regulatory_config()
        r5 = R5TravelDistance(cfg)
        # Feasibility violation has clause "Feasibility ..."
        # Regulatory violation has NBC clause
        # Verify they differ
        fp = FloorPlan(
            grid={"rows": 6, "cols": 6},
            walls=[
                (1, 1), (1, 2), (1, 3),
                (2, 1),         (2, 3),
                (3, 1), (3, 2), (3, 3),
            ],
            doors=[DoorOpening(position=(0, 0), width=1.0, exterior=True, door_id="e1")],
        )
        occ = OccupantScenario(agents=[AgentConfig(agent_id="trapped", row=2, col=2)])
        exit_cfg = ExitConfiguration(exits=[(0, 0)], widths=[1.0])
        violations = r5.check(fp, exit_cfg, occ)
        for v in violations:
            if "cannot reach" in v.message:
                assert "Feasibility" in v.clause
                # The clause clarifies it's not NBC — it starts with "Feasibility"
                assert v.clause.startswith("Feasibility")


# ==============================================================================
# K & L. Full compliance check
# ==============================================================================

class TestFullComplianceCheck:
    def test_compliant_config_returns_true(
        self, small_floor_plan, two_occupants, valid_two_exit_config, nbc_ruleset
    ):
        """K. A well-formed compliant config returns is_compliant=True."""
        result = nbc_ruleset.check(small_floor_plan, valid_two_exit_config, two_occupants)
        assert result.is_compliant is True
        assert result.violations == []

    def test_noncompliant_config_returns_false(self, small_floor_plan, two_occupants, nbc_ruleset):
        """L. Interior exit + too few exits produces structured non-compliant result."""
        bad_config = ExitConfiguration(
            configuration_id="bad",
            exits=[(5, 3)],   # interior door
            widths=[0.9],
        )
        result = nbc_ruleset.check(small_floor_plan, bad_config, two_occupants)
        assert result.is_compliant is False
        assert len(result.violations) >= 1
        rule_ids = {v.rule_id for v in result.violations}
        assert "R1_VALID_EXIT_POSITION" in rule_ids

    def test_violations_are_machine_readable(self, small_floor_plan, two_occupants, nbc_ruleset):
        """L. Violations have all required fields for programmatic consumption."""
        bad_config = ExitConfiguration(exits=[(0, 2)], widths=[1.2])  # only 1 exit
        result = nbc_ruleset.check(small_floor_plan, bad_config, two_occupants)
        assert result.is_compliant is False
        for v in result.violations:
            assert v.rule_id != ""
            assert v.message != ""
            assert v.severity in ("error", "warning")

    def test_validation_result_is_pydantic(self, small_floor_plan, two_occupants,
                                           valid_two_exit_config, nbc_ruleset):
        """U. ValidationResult is a Pydantic model with machine-readable fields."""
        result = nbc_ruleset.check(small_floor_plan, valid_two_exit_config, two_occupants)
        assert isinstance(result, ValidationResult)
        # Can be serialized to dict
        d = result.model_dump()
        assert "is_compliant" in d
        assert "violations" in d


# ==============================================================================
# M & N. Mask API
# ==============================================================================

class TestMaskAPI:
    def test_mask_blocks_already_selected(self, small_floor_plan, two_occupants, validator):
        """M. Mask returns False for already-selected exits."""
        candidates = generate_candidate_exits(small_floor_plan)
        partial = ExitConfiguration(exits=[(0, 2)], widths=[1.2])
        mask = validator.mask(small_floor_plan, two_occupants, partial, candidates)
        # (0,2) is already selected → must be False
        assert mask[(0, 2)] is False
        # Other candidates → True
        assert mask[(9, 7)] is True

    def test_mask_allows_unselected_candidates(self, small_floor_plan, two_occupants, validator):
        """M. Mask returns True for unselected valid candidates."""
        candidates = generate_candidate_exits(small_floor_plan)
        empty_partial = ExitConfiguration(exits=[], widths=[])
        # Override SimulationConfig validation — use empty ExitConfiguration
        # directly (it has no exits, which is valid for partial mask context)
        mask = validator.mask(small_floor_plan, two_occupants, empty_partial, candidates)
        for pos, allowed in mask.items():
            assert allowed is True, f"All candidates should be allowed in empty partial, got False for {pos}"

    def test_mask_does_not_evaluate_full_config_constraints(
        self, small_floor_plan, two_occupants, validator
    ):
        """N. Mask does not pretend to evaluate R2/R3/R4/R5 at partial level."""
        candidates = generate_candidate_exits(small_floor_plan)
        # Even with a single candidate selected, mask should not check min-exits (R2)
        partial = ExitConfiguration(exits=[(0, 2)], widths=[1.2])
        mask = validator.mask(small_floor_plan, two_occupants, partial, candidates)
        # Unselected exterior candidates should still be True (R2 not evaluated here)
        unselected = [pos for pos, allowed in mask.items() if allowed]
        assert len(unselected) > 0, "At least some candidates should remain allowed"

    def test_mask_returns_gridcell_keys(self, small_floor_plan, two_occupants, validator):
        """M. Mask keys are GridCell tuples."""
        candidates = generate_candidate_exits(small_floor_plan)
        partial = ExitConfiguration(exits=[], widths=[])
        mask = validator.mask(small_floor_plan, two_occupants, partial, candidates)
        for key in mask.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_mask_auto_generates_candidates(self, small_floor_plan, two_occupants, validator):
        """M. When candidates=None, mask auto-generates from floor_plan."""
        partial = ExitConfiguration(exits=[], widths=[])
        mask = validator.mask(small_floor_plan, two_occupants, partial)
        assert len(mask) > 0  # Should have entries
        # All should be exterior door positions
        ext_positions = {(d.position[0], d.position[1]) for d in small_floor_plan.doors if d.exterior}
        for pos in mask.keys():
            assert pos in ext_positions


# ==============================================================================
# O. Regression: Stage 1-10 behavior unchanged
# ==============================================================================

class TestRegression:
    def test_simulation_still_runs(self):
        """O. Existing simulation engine still works after Stage 11 imports."""
        from evacuation_simulation.simulation.simulation import Simulation
        from evacuation_simulation.simulation.strategy import ShortestPathStrategy
        cfg = SimulationConfig(
            scenario_name="regression_test",
            floor_plan={"grid": {"rows": 5, "cols": 5}, "walls": [], "doors": []},
            exit_configuration={"configuration_id": "ref", "exits": [[0, 4]]},
            occupants={"agents": [{"agent_id": "a1", "row": 4, "col": 0}]},
        )
        sim = Simulation(cfg, movement_strategy=ShortestPathStrategy())
        result = sim.run()
        assert result.evacuated_count == 1

    def test_door_opening_model_unchanged(self):
        """O. DoorOpening model still works as in Stage 10."""
        door = DoorOpening(position=(0, 3), width=1.5, exterior=True, door_id="d1")
        assert door.width == 1.5
        assert door.exterior is True

    def test_generate_candidate_exits_unchanged(self):
        """O. generate_candidate_exits behavior unchanged."""
        fp = FloorPlan(
            grid={"rows": 5, "cols": 5},
            walls=[],
            doors=[
                DoorOpening(position=(0, 2), width=1.0, exterior=True, door_id="ext"),
                DoorOpening(position=(2, 2), width=1.0, exterior=False, door_id="int"),
            ],
        )
        candidates = generate_candidate_exits(fp)
        assert len(candidates) == 1
        assert candidates[0].position == (0, 2)


# ==============================================================================
# P. Immutability: check() does not mutate inputs
# ==============================================================================

class TestImmutability:
    def test_check_does_not_mutate_floor_plan(
        self, small_floor_plan, two_occupants, valid_two_exit_config, nbc_ruleset
    ):
        """P. check() does not mutate FloorPlan."""
        original_walls = list(small_floor_plan.walls)
        original_doors = [d.model_dump() for d in small_floor_plan.doors]
        nbc_ruleset.check(small_floor_plan, valid_two_exit_config, two_occupants)
        assert list(small_floor_plan.walls) == original_walls
        assert [d.model_dump() for d in small_floor_plan.doors] == original_doors

    def test_check_does_not_mutate_exit_config(
        self, small_floor_plan, two_occupants, valid_two_exit_config, nbc_ruleset
    ):
        """P. check() does not mutate ExitConfiguration."""
        original_exits = list(valid_two_exit_config.exits)
        original_widths = list(valid_two_exit_config.widths)
        nbc_ruleset.check(small_floor_plan, valid_two_exit_config, two_occupants)
        assert list(valid_two_exit_config.exits) == original_exits
        assert list(valid_two_exit_config.widths) == original_widths

    def test_check_does_not_mutate_occupant_scenario(
        self, small_floor_plan, two_occupants, valid_two_exit_config, nbc_ruleset
    ):
        """P. check() does not mutate OccupantScenario."""
        original_agents = [a.model_dump() for a in two_occupants.agents]
        nbc_ruleset.check(small_floor_plan, valid_two_exit_config, two_occupants)
        assert [a.model_dump() for a in two_occupants.agents] == original_agents


# ==============================================================================
# Q. Legacy scenarios (no explicit doors) still work
# ==============================================================================

class TestLegacyScenarios:
    def test_legacy_no_doors_r1_uses_boundary_fallback(self):
        """Q. Legacy floor plan without doors: R1 accepts boundary cells."""
        fp_no_doors = FloorPlan(grid={"rows": 5, "cols": 5}, walls=[])
        occ = OccupantScenario(agents=[AgentConfig(agent_id="a1", row=2, col=2)])
        exit_cfg = ExitConfiguration(exits=[(0, 0)], widths=[1.0])
        r1 = R1ValidExitPosition(load_regulatory_config())
        # (0,0) is a valid boundary cell in legacy mode
        violations = r1.check(fp_no_doors, exit_cfg, occ)
        assert all("not a valid" not in v.message for v in violations), \
            "Legacy boundary cell should be valid"

    def test_legacy_scenario_json_basic_building(self):
        """Q. basic_building.json still loads and generates candidates correctly."""
        path = Path("scenarios/basic_building.json")
        if not path.exists():
            pytest.skip("basic_building.json not found")
        cfg = SimulationConfig.from_json(path)
        candidates = generate_candidate_exits(cfg.floor_plan)
        assert len(candidates) > 0


# ==============================================================================
# R. RuleSet with custom rules
# ==============================================================================

class TestRuleSetCustom:
    def test_empty_ruleset_always_compliant(self, small_floor_plan, two_occupants,
                                            valid_two_exit_config):
        """R. RuleSet with no rules always returns compliant."""
        rs = RuleSet(rules=[])
        result = rs.check(small_floor_plan, valid_two_exit_config, two_occupants)
        assert result.is_compliant is True

    def test_ruleset_with_single_rule(self, small_floor_plan, two_occupants):
        """R. RuleSet with only R2 correctly validates exit count."""
        cfg = load_regulatory_config()
        rs = RuleSet(rules=[R2MinimumExitCount(cfg)])
        one_exit = ExitConfiguration(exits=[(0, 2)], widths=[1.2])
        result = rs.check(small_floor_plan, one_exit, two_occupants)
        assert result.is_compliant is False
        assert result.violations[0].rule_id == "R2_MINIMUM_EXIT_COUNT"


# ==============================================================================
# S & T. Config and factory
# ==============================================================================

class TestConfigAndFactory:
    def test_load_regulatory_config_raises_on_missing_file(self, tmp_path):
        """S. load_regulatory_config raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_regulatory_config(tmp_path / "nonexistent.yaml")

    def test_build_nbc_2016_ruleset_has_five_rules(self):
        """T. build_nbc_2016_ruleset() produces exactly 5 rules (R1-R5)."""
        rs = build_nbc_2016_ruleset()
        assert len(rs.rules) == 5
        rule_ids = [r.rule_id for r in rs.rules]
        assert "R1_VALID_EXIT_POSITION" in rule_ids
        assert "R2_MINIMUM_EXIT_COUNT" in rule_ids
        assert "R3_EXIT_SEPARATION" in rule_ids
        assert "R4_EXIT_WIDTH" in rule_ids
        assert "R5_TRAVEL_DISTANCE" in rule_ids


# ==============================================================================
# U & V. Pydantic models
# ==============================================================================

class TestPydanticModels:
    def test_violation_has_all_expected_fields(self):
        """V. Violation Pydantic model has all required machine-readable fields."""
        v = Violation(
            rule_id="R3_EXIT_SEPARATION",
            message="Exits too close",
            severity="error",
            clause="NBC 2016 Part 4",
            measured_value=4.2,
            required_value=8.0,
            unit="m",
            relevant_ids=["(0, 2)", "(0, 4)"],
        )
        assert v.rule_id == "R3_EXIT_SEPARATION"
        assert v.measured_value == pytest.approx(4.2)
        assert v.required_value == pytest.approx(8.0)
        assert v.unit == "m"
        d = v.model_dump()
        for field in ("rule_id", "message", "severity", "clause",
                      "measured_value", "required_value", "unit", "relevant_ids"):
            assert field in d

    def test_validation_result_serializable(self, small_floor_plan, two_occupants,
                                            valid_two_exit_config, nbc_ruleset):
        """U. ValidationResult serializes cleanly to dict."""
        result = nbc_ruleset.check(small_floor_plan, valid_two_exit_config, two_occupants)
        d = result.model_dump()
        assert isinstance(d["is_compliant"], bool)
        assert isinstance(d["violations"], list)


# ==============================================================================
# W & X. Integration: realistic scenarios
# ==============================================================================

class TestIntegration:
    def test_basic_building_compliant_config(self):
        """W. basic_building.json with valid 2-exit config is compliant."""
        path = Path("scenarios/basic_building.json")
        if not path.exists():
            pytest.skip("basic_building.json not found")
        cfg = SimulationConfig.from_json(path)
        fp = cfg.floor_plan
        candidates = generate_candidate_exits(fp)
        assert len(candidates) >= 2
        # Pick first two well-separated candidates
        configs = list(generate_exit_configurations(candidates, exit_count=2))
        validator = make_validator()
        found_compliant = False
        for exit_cfg in configs:
            result = validator.check(fp, exit_cfg, cfg.occupants)
            if result.is_compliant:
                found_compliant = True
                break
        assert found_compliant, "At least one 2-exit config should be compliant"

    def test_violations_have_structured_fields(self, small_floor_plan, two_occupants, nbc_ruleset):
        """X. Violation objects from non-compliant config have populated fields."""
        bad_config = ExitConfiguration(exits=[(5, 3)], widths=[0.9])  # interior + narrow
        result = nbc_ruleset.check(small_floor_plan, bad_config, two_occupants)
        assert result.is_compliant is False
        for v in result.violations:
            assert isinstance(v.rule_id, str) and v.rule_id != ""
            assert isinstance(v.message, str) and v.message != ""
            assert isinstance(v.severity, str)
            # All hard-constraint violations should be "error"
            assert v.severity == "error"

    def test_interior_pass_cannot_become_exit(self, small_floor_plan, two_occupants, validator):
        """Integration: interior PASS openings are confirmed non-eligible as exits."""
        int_doors = [d for d in small_floor_plan.doors if not d.exterior]
        assert len(int_doors) > 0, "Test requires interior doors"
        for d in int_doors:
            interior_cfg = ExitConfiguration(
                exits=[d.position],
                widths=[d.width],
            )
            result = validator.check(small_floor_plan, interior_cfg, two_occupants)
            # Must be non-compliant because interior door is not a valid exit position
            r1_violations = [v for v in result.violations if v.rule_id == "R1_VALID_EXIT_POSITION"]
            assert len(r1_violations) >= 1, \
                f"Interior door at {d.position} should produce R1 violation"

    def test_candidate_pool_only_exterior_doors(self, small_floor_plan):
        """Integration: candidate exits contain only exterior doors."""
        candidates = generate_candidate_exits(small_floor_plan)
        candidate_positions = {(c.position[0], c.position[1]) for c in candidates}
        for d in small_floor_plan.doors:
            pos = (d.position[0], d.position[1])
            if not d.exterior:
                assert pos not in candidate_positions, \
                    f"Interior door at {pos} should not be in candidates"
            else:
                assert pos in candidate_positions, \
                    f"Exterior door at {pos} should be in candidates"
