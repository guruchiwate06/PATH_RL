"""
test_e2e_synthetic_experiment.py
----------------------------------
Integration test: complete end-to-end synthetic building evacuation pipeline.

Tests the full pipeline:
  CommercialLayoutGenerator -> ExitValidator -> OccupantSeeder
  -> ConfigurationEvaluator (Simulation) -> EvaluationResult

This is a black-box integration test; it does not re-test individual units
already covered by stage-specific tests. It verifies that the modules connect
correctly and produce meaningful, measurable evacuation results.
"""

from __future__ import annotations

import pytest

from evacuation_simulation.simulation.config import (
    OccupantScenario,
    SimulationParameters,
)
from evacuation_simulation.simulation.evaluation import ConfigurationEvaluator
from evacuation_simulation.simulation.generators.commercial import (
    CommercialGeneratorConfig,
    CommercialLayoutGenerator,
)
from evacuation_simulation.simulation.occupancy_seeder import (
    OccupantSeeder,
    OccupantSeederConfig,
)
from evacuation_simulation.simulation.validator import ExitValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compliant_layout():
    """
    Generate a compliant 20x30 commercial layout (15m x 10m, seed=42).
    Chosen to satisfy all NBC 2016 R1-R5 rules with 20 occupants and 2 exits.
    """
    cfg = CommercialGeneratorConfig(
        seed=42,
        grid_rows=20,
        grid_cols=30,
        num_exterior_exits=2,
    )
    return CommercialLayoutGenerator().generate(cfg)


@pytest.fixture(scope="module")
def occupant_scenario(compliant_layout):
    """Seed 20 occupants onto the compliant layout."""
    seeder_cfg = OccupantSeederConfig(
        total_occupants=20,
        seed=42,
        strategy="random_walkable",
        max_occupants_per_cell=1,
    )
    return OccupantSeeder.seed(
        compliant_layout.floor_plan,
        compliant_layout.exit_configuration,
        seeder_cfg,
    )


@pytest.fixture(scope="module")
def evaluation_result(compliant_layout, occupant_scenario):
    """Run the full evaluation pipeline and return the EvaluationResult."""
    params = SimulationParameters(
        max_timesteps=500,
        random_seed=42,
        default_cell_capacity=1,
    )
    evaluator = ConfigurationEvaluator()
    return evaluator.evaluate(
        compliant_layout.floor_plan,
        compliant_layout.exit_configuration,
        occupant_scenario,
        params,
    )


# ---------------------------------------------------------------------------
# Step 1: Layout generation
# ---------------------------------------------------------------------------


class TestLayoutGeneration:
    def test_layout_is_generated(self, compliant_layout):
        assert compliant_layout is not None

    def test_grid_dimensions(self, compliant_layout):
        fp = compliant_layout.floor_plan
        assert fp.grid.rows == 20
        assert fp.grid.cols == 30

    def test_has_rooms(self, compliant_layout):
        assert len(compliant_layout.rooms) >= 2

    def test_two_exterior_exits(self, compliant_layout):
        exterior = [d for d in compliant_layout.floor_plan.doors if d.exterior]
        assert len(exterior) == 2

    def test_no_wall_door_overlap(self, compliant_layout):
        fp = compliant_layout.floor_plan
        wall_set = set(map(tuple, fp.walls))
        for door in fp.doors:
            assert tuple(door.position) not in wall_set

    def test_exit_config_has_two_exits(self, compliant_layout):
        assert len(compliant_layout.exit_configuration.exits) == 2

    def test_deterministic_generation(self):
        """Same seed must produce identical layout."""
        cfg = CommercialGeneratorConfig(seed=99, grid_rows=20, grid_cols=30)
        gen = CommercialLayoutGenerator()
        l1 = gen.generate(cfg)
        l2 = gen.generate(cfg)
        assert l1.floor_plan.walls == l2.floor_plan.walls
        assert len(l1.floor_plan.doors) == len(l2.floor_plan.doors)


# ---------------------------------------------------------------------------
# Step 2: Constraint validation
# ---------------------------------------------------------------------------


class TestConstraintValidation:
    def test_structure_compliant_empty_occupants(self, compliant_layout):
        """Exit config must be NBC-compliant before occupant placement."""
        validator = ExitValidator()
        result = validator.check(
            compliant_layout.floor_plan,
            compliant_layout.exit_configuration,
            OccupantScenario(agents=[]),
        )
        assert result.is_compliant, (
            f"NBC violations (empty occupants): "
            f"{[f'{v.rule_id}: {v.message}' for v in result.violations]}"
        )

    def test_compliant_with_occupants(self, compliant_layout, occupant_scenario):
        """Exit config must be NBC-compliant with real occupant placement."""
        validator = ExitValidator()
        result = validator.check(
            compliant_layout.floor_plan,
            compliant_layout.exit_configuration,
            occupant_scenario,
        )
        assert result.is_compliant, (
            f"NBC violations (with occupants): "
            f"{[f'{v.rule_id}: {v.message}' for v in result.violations]}"
        )

    def test_no_violations(self, compliant_layout, occupant_scenario):
        validator = ExitValidator()
        result = validator.check(
            compliant_layout.floor_plan,
            compliant_layout.exit_configuration,
            occupant_scenario,
        )
        assert len(result.violations) == 0


# ---------------------------------------------------------------------------
# Step 3: Occupant seeding
# ---------------------------------------------------------------------------


class TestOccupantSeeding:
    def test_correct_occupant_count(self, occupant_scenario):
        assert len(occupant_scenario.agents) == 20

    def test_no_agent_on_wall(self, compliant_layout, occupant_scenario):
        wall_set = set(map(tuple, compliant_layout.floor_plan.walls))
        for agent in occupant_scenario.agents:
            assert (agent.row, agent.col) not in wall_set

    def test_no_agent_on_exit(self, compliant_layout, occupant_scenario):
        exit_set = set(map(tuple, compliant_layout.exit_configuration.exits))
        for agent in occupant_scenario.agents:
            assert (agent.row, agent.col) not in exit_set

    def test_agents_within_grid(self, compliant_layout, occupant_scenario):
        fp = compliant_layout.floor_plan
        for agent in occupant_scenario.agents:
            assert 0 <= agent.row < fp.grid.rows
            assert 0 <= agent.col < fp.grid.cols

    def test_unique_agent_ids(self, occupant_scenario):
        ids = [a.agent_id for a in occupant_scenario.agents]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Step 4 & 5: Simulation and evaluation
# ---------------------------------------------------------------------------


class TestEvacuationSimulation:
    def test_evaluation_runs(self, evaluation_result):
        assert evaluation_result is not None

    def test_feasible_result(self, evaluation_result):
        assert evaluation_result.is_feasible is True, (
            f"Evaluation not feasible. NBC compliant: "
            f"{evaluation_result.validation_result.is_compliant}. "
            f"Violations: {[v.message for v in evaluation_result.validation_result.violations]}"
        )

    def test_complete_evacuation(self, evaluation_result):
        assert evaluation_result.evacuation_complete is True

    def test_all_agents_evacuated(self, evaluation_result):
        m = evaluation_result.metrics
        assert m is not None
        assert m.evacuated_count == m.total_agents
        assert m.non_evacuated_count == 0

    def test_evacuation_rate_is_one(self, evaluation_result):
        assert evaluation_result.metrics.evacuation_rate == pytest.approx(1.0)

    def test_score_is_positive_integer(self, evaluation_result):
        assert evaluation_result.score is not None
        assert isinstance(evaluation_result.score, int)
        assert evaluation_result.score > 0

    def test_mean_evacuation_time_is_reasonable(self, evaluation_result):
        """Mean time must be in (0, max_timesteps)."""
        m = evaluation_result.metrics
        assert m.mean_evacuation_time is not None
        assert 0 < m.mean_evacuation_time <= 500

    def test_exit_utilization_covers_all_agents(self, evaluation_result):
        m = evaluation_result.metrics
        total_via_exits = sum(m.exit_utilization.values())
        assert total_via_exits == m.total_agents

    def test_both_exits_used(self, evaluation_result):
        """Both exits should be reachable — at least one agent per exit."""
        m = evaluation_result.metrics
        assert len(m.exit_utilization) == 2
        for count in m.exit_utilization.values():
            assert count >= 1

    def test_deterministic_score(self, compliant_layout, occupant_scenario):
        """Same seed must produce same score."""
        params = SimulationParameters(max_timesteps=500, random_seed=42)
        evaluator = ConfigurationEvaluator()
        r1 = evaluator.evaluate(
            compliant_layout.floor_plan,
            compliant_layout.exit_configuration,
            occupant_scenario,
            params,
        )
        r2 = evaluator.evaluate(
            compliant_layout.floor_plan,
            compliant_layout.exit_configuration,
            occupant_scenario,
            params,
        )
        assert r1.score == r2.score


# ---------------------------------------------------------------------------
# Full pipeline integration test (single entry point)
# ---------------------------------------------------------------------------


def test_full_e2e_pipeline_produces_real_metrics():
    """
    Black-box integration test of the complete end-to-end pipeline.
    Runs from synthetic layout generation to evacuation metrics in one call.
    """
    from evacuation_simulation.run_e2e_experiment import run_experiment

    result = run_experiment(
        seed=1,
        total_occupants=15,
        grid_rows=20,
        grid_cols=30,
        num_exterior_exits=2,
        max_timesteps=500,
        verbose=False,
    )

    # Verify structure
    assert "experiment" in result
    assert "layout" in result
    assert "metrics" in result
    assert result["metrics"] is not None

    m = result["metrics"]
    assert m["total_agents"] == 15
    assert m["evacuated_count"] == 15
    assert m["non_evacuated_count"] == 0
    assert m["evacuation_rate"] == pytest.approx(1.0)
    assert m["score_primary_objective"] is not None
    assert m["score_primary_objective"] > 0
    assert result["simulation"]["evacuation_complete"] is True
    assert result["constraint_validation"]["is_compliant"] is True
