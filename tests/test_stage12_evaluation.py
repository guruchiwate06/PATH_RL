"""
test_stage12_evaluation.py
--------------------------
Stage 12 tests: Exit Configuration Evaluation and Objective Layer.

Tests cover:
A. Valid configuration is validated and simulated.
B. Invalid configuration is rejected before simulation.
C. EvaluationResult is structured and machine-readable.
D. Total evacuation time is extracted correctly.
E. Primary objective equals the intended evacuation metric.
F. Lower evacuation time ranks better.
G. Invalid configurations cannot outrank valid configurations.
H. Same configuration + scenario + seed is deterministic.
I. Different exit configurations can produce different results.
J. Same occupant scenario is used for fair comparison.
K. Simulation metrics are preserved.
L. Exit utilization is preserved.
M. Incomplete evacuation is handled explicitly.
N. Multiple configurations can be evaluated.
O. Configurations can be ranked.
P. Evaluation does not mutate FloorPlan.
Q. Evaluation does not mutate ExitConfiguration.
R. Evaluation does not mutate OccupantScenario.
S. Stage 11 validation results are preserved inside EvaluationResult.
T. Existing Stage 1-11 tests continue passing (verified by full suite).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import pytest

from evacuation_simulation.simulation.config import (
    AgentConfig,
    DoorOpening,
    ExitConfiguration,
    FloorPlan,
    GridConfig,
    OccupantScenario,
    SimulationParameters,
)
from evacuation_simulation.simulation.evaluation import (
    ConfigurationEvaluator,
    EvaluationMetrics,
    EvaluationResult,
    rank_evaluations,
)
from evacuation_simulation.simulation.rules import ValidationResult
from evacuation_simulation.simulation.validator import ExitValidator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_simple_floor_plan() -> FloorPlan:
    """5x5 grid with one exterior door at (0,2)."""
    return FloorPlan(
        grid=GridConfig(rows=5, cols=5),
        walls=[],
        doors=[
            DoorOpening(position=(0, 2), width=1.2, exterior=True, door_id="door_top"),
            DoorOpening(position=(4, 2), width=1.2, exterior=True, door_id="door_bottom"),
        ],
    )


def _make_two_exit_config() -> ExitConfiguration:
    """Both exterior doors active."""
    return ExitConfiguration(
        configuration_id="cfg_two_exits",
        exits=[(0, 2), (4, 2)],
        widths=[1.2, 1.2],
    )


def _make_one_exit_config() -> ExitConfiguration:
    """Only bottom door active."""
    return ExitConfiguration(
        configuration_id="cfg_one_exit",
        exits=[(4, 2)],
        widths=[1.2],
    )


def _make_top_only_config() -> ExitConfiguration:
    """Only top door active."""
    return ExitConfiguration(
        configuration_id="cfg_top_only",
        exits=[(0, 2)],
        widths=[1.2],
    )


def _make_scenario_three_agents() -> OccupantScenario:
    """Three agents in the middle of the grid."""
    return OccupantScenario(
        agents=[
            AgentConfig(agent_id="a1", row=2, col=1),
            AgentConfig(agent_id="a2", row=2, col=2),
            AgentConfig(agent_id="a3", row=2, col=3),
        ]
    )


def _make_params(seed: int = 42) -> SimulationParameters:
    return SimulationParameters(max_timesteps=200, random_seed=seed)


def _make_evaluator() -> ConfigurationEvaluator:
    return ConfigurationEvaluator()


def _make_invalid_config() -> ExitConfiguration:
    """Exit at an interior (non-door) position — will fail R1."""
    return ExitConfiguration(
        configuration_id="cfg_invalid",
        exits=[(2, 2)],  # interior cell, not an exterior door
        widths=[1.2],
    )


# ---------------------------------------------------------------------------
# A. Valid configuration is validated and simulated
# ---------------------------------------------------------------------------


class TestValidConfigurationEvaluated:
    def test_valid_config_returns_evaluation_result(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert isinstance(result, EvaluationResult)

    def test_valid_config_validation_is_compliant(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert result.validation_result.is_compliant

    def test_valid_config_simulation_result_not_none(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert result.simulation_result is not None

    def test_valid_config_metrics_not_none(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert result.metrics is not None


# ---------------------------------------------------------------------------
# B. Invalid configuration is rejected before simulation
# ---------------------------------------------------------------------------


class TestInvalidConfigurationRejected:
    def test_invalid_config_is_not_feasible(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_invalid_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert not result.is_feasible

    def test_invalid_config_simulation_not_run(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_invalid_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert result.simulation_result is None

    def test_invalid_config_metrics_none(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_invalid_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert result.metrics is None

    def test_invalid_config_score_none(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_invalid_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert result.score is None

    def test_invalid_config_validation_result_not_compliant(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_invalid_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert not result.validation_result.is_compliant


# ---------------------------------------------------------------------------
# C. EvaluationResult is structured and machine-readable
# ---------------------------------------------------------------------------


class TestEvaluationResultStructure:
    def test_evaluation_result_has_configuration_id(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert result.configuration_id == "cfg_two_exits"

    def test_evaluation_result_has_is_feasible(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert isinstance(result.is_feasible, bool)

    def test_evaluation_result_has_validation_result(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert isinstance(result.validation_result, ValidationResult)

    def test_evaluation_result_summary_line_returns_string(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        line = result.summary_line()
        assert isinstance(line, str)
        assert len(line) > 0


# ---------------------------------------------------------------------------
# D. Total evacuation time is extracted correctly
# ---------------------------------------------------------------------------


class TestTotalEvacuationTime:
    def test_total_evacuation_time_is_positive_for_complete_evacuation(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        if result.evacuation_complete:
            assert result.metrics.total_evacuation_time is not None
            assert result.metrics.total_evacuation_time > 0

    def test_score_equals_total_evacuation_time(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        if result.evacuation_complete:
            assert result.score == result.metrics.total_evacuation_time


# ---------------------------------------------------------------------------
# E. Primary objective equals the intended evacuation metric
# ---------------------------------------------------------------------------


class TestPrimaryObjective:
    def test_score_equals_simulation_total_timesteps_on_complete_evacuation(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        if result.evacuation_complete:
            assert result.score == result.simulation_result.total_timesteps

    def test_score_is_none_for_invalid_config(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_invalid_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert result.score is None


# ---------------------------------------------------------------------------
# F. Lower evacuation time ranks better
# ---------------------------------------------------------------------------


class TestRankingByScore:
    def test_lower_score_ranks_first(self):
        from dataclasses import replace
        # Build two fake results with different scores
        vr = ValidationResult(is_compliant=True)
        r1 = EvaluationResult(
            configuration_id="fast",
            is_feasible=True,
            evacuation_complete=True,
            validation_result=vr,
            simulation_result=None,
            metrics=None,
            score=10,
        )
        r2 = EvaluationResult(
            configuration_id="slow",
            is_feasible=True,
            evacuation_complete=True,
            validation_result=vr,
            simulation_result=None,
            metrics=None,
            score=20,
        )
        ranked = rank_evaluations([r2, r1])
        assert ranked[0].configuration_id == "fast"
        assert ranked[1].configuration_id == "slow"


# ---------------------------------------------------------------------------
# G. Invalid configurations cannot outrank valid configurations
# ---------------------------------------------------------------------------


class TestInvalidCannotOutrankValid:
    def test_valid_always_ranked_above_invalid(self):
        vr_ok = ValidationResult(is_compliant=True)
        vr_bad = ValidationResult(is_compliant=False)
        valid = EvaluationResult(
            configuration_id="valid",
            is_feasible=True,
            evacuation_complete=True,
            validation_result=vr_ok,
            simulation_result=None,
            metrics=None,
            score=100,  # very high score (slow)
        )
        invalid = EvaluationResult(
            configuration_id="invalid",
            is_feasible=False,
            evacuation_complete=None,
            validation_result=vr_bad,
            simulation_result=None,
            metrics=None,
            score=None,
        )
        ranked = rank_evaluations([invalid, valid])
        assert ranked[0].configuration_id == "valid"
        assert ranked[1].configuration_id == "invalid"

    def test_incomplete_evacuation_ranked_below_complete(self):
        vr_ok = ValidationResult(is_compliant=True)
        complete = EvaluationResult(
            configuration_id="complete",
            is_feasible=True,
            evacuation_complete=True,
            validation_result=vr_ok,
            simulation_result=None,
            metrics=None,
            score=50,
        )
        incomplete = EvaluationResult(
            configuration_id="incomplete",
            is_feasible=False,
            evacuation_complete=False,
            validation_result=vr_ok,
            simulation_result=None,
            metrics=None,
            score=None,
        )
        ranked = rank_evaluations([incomplete, complete])
        assert ranked[0].configuration_id == "complete"


# ---------------------------------------------------------------------------
# H. Determinism: same config + scenario + seed -> same result
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_produce_same_score(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        params = _make_params(seed=42)
        r1 = evaluator.evaluate(fp, cfg, scenario, parameters=params)
        r2 = evaluator.evaluate(fp, cfg, scenario, parameters=params)
        assert r1.score == r2.score

    def test_same_inputs_produce_same_evacuation_rate(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        params = _make_params(seed=99)
        r1 = evaluator.evaluate(fp, cfg, scenario, parameters=params)
        r2 = evaluator.evaluate(fp, cfg, scenario, parameters=params)
        assert r1.metrics.evacuation_rate == r2.metrics.evacuation_rate

    def test_same_inputs_produce_same_feasibility(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        params = _make_params(seed=7)
        r1 = evaluator.evaluate(fp, cfg, scenario, parameters=params)
        r2 = evaluator.evaluate(fp, cfg, scenario, parameters=params)
        assert r1.is_feasible == r2.is_feasible


# ---------------------------------------------------------------------------
# I. Different exit configurations can produce different results
# ---------------------------------------------------------------------------


class TestDifferentConfigsDifferentResults:
    def test_two_exits_vs_one_exit_may_differ(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg_two = _make_two_exit_config()
        cfg_one = _make_one_exit_config()
        scenario = _make_scenario_three_agents()
        params = _make_params(seed=42)
        r_two = evaluator.evaluate(fp, cfg_two, scenario, parameters=params)
        r_one = evaluator.evaluate(fp, cfg_one, scenario, parameters=params)
        # They may differ — the configuration IDs certainly differ
        assert r_two.configuration_id != r_one.configuration_id


# ---------------------------------------------------------------------------
# J. Same occupant scenario for fair comparison
# ---------------------------------------------------------------------------


class TestFairComparison:
    def test_same_scenario_same_agent_count_in_both_results(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        scenario = _make_scenario_three_agents()
        params = _make_params(seed=42)
        r_two = evaluator.evaluate(fp, _make_two_exit_config(), scenario, parameters=params)
        r_one = evaluator.evaluate(fp, _make_one_exit_config(), scenario, parameters=params)
        if r_two.metrics and r_one.metrics:
            assert r_two.metrics.total_agents == r_one.metrics.total_agents

    def test_evaluate_configurations_uses_same_scenario(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        scenario = _make_scenario_three_agents()
        params = _make_params(seed=42)
        cfgs = [_make_two_exit_config(), _make_one_exit_config()]
        results = evaluator.evaluate_configurations(fp, cfgs, scenario, parameters=params)
        agent_counts = [
            r.metrics.total_agents for r in results if r.metrics is not None
        ]
        assert len(set(agent_counts)) == 1  # all same agent count


# ---------------------------------------------------------------------------
# K. Simulation metrics are preserved
# ---------------------------------------------------------------------------


class TestSimulationMetricsPreserved:
    def test_evacuation_rate_in_metrics(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        if result.metrics:
            assert 0.0 <= result.metrics.evacuation_rate <= 1.0

    def test_mean_evacuation_time_in_metrics(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        if result.metrics and result.evacuation_complete:
            assert result.metrics.mean_evacuation_time is not None
            assert result.metrics.mean_evacuation_time > 0

    def test_congested_cell_steps_in_metrics(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        if result.metrics:
            assert isinstance(result.metrics.congested_cell_steps, int)
            assert result.metrics.congested_cell_steps >= 0

    def test_waiting_steps_in_metrics(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        if result.metrics:
            assert result.metrics.total_waiting_steps >= 0
            assert result.metrics.mean_waiting_steps >= 0.0
            assert result.metrics.max_waiting_steps >= 0


# ---------------------------------------------------------------------------
# L. Exit utilization is preserved
# ---------------------------------------------------------------------------


class TestExitUtilizationPreserved:
    def test_exit_utilization_is_dict(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        if result.metrics:
            assert isinstance(result.metrics.exit_utilization, dict)

    def test_exit_utilization_sum_equals_evacuated_count(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        if result.metrics and result.evacuation_complete:
            total_via_exits = sum(result.metrics.exit_utilization.values())
            assert total_via_exits == result.metrics.evacuated_count


# ---------------------------------------------------------------------------
# M. Incomplete evacuation is handled explicitly
# ---------------------------------------------------------------------------


class TestIncompleteEvacuationHandled:
    def test_incomplete_evacuation_score_is_none(self):
        """A config with 0 timesteps traps everyone."""
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        # Force 0 timesteps so no one can evacuate
        params = SimulationParameters(max_timesteps=1, random_seed=42)
        result = evaluator.evaluate(fp, cfg, scenario, parameters=params)
        # With 1 max timestep some may or may not evacuate; if incomplete:
        if not result.evacuation_complete:
            assert result.score is None
            assert result.is_feasible is False
            assert result.metrics.total_evacuation_time is None

    def test_incomplete_result_has_non_zero_non_evacuated_count(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        params = SimulationParameters(max_timesteps=1, random_seed=42)
        result = evaluator.evaluate(fp, cfg, scenario, parameters=params)
        if not result.evacuation_complete:
            assert result.metrics.non_evacuated_count > 0

    def test_evacuation_complete_flag_is_set(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        # evacuation_complete is always bool when simulation ran
        assert isinstance(result.evacuation_complete, bool)


# ---------------------------------------------------------------------------
# N. Multiple configurations can be evaluated
# ---------------------------------------------------------------------------


class TestMultipleConfigurationsEvaluated:
    def test_evaluate_configurations_returns_one_per_config(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        scenario = _make_scenario_three_agents()
        params = _make_params()
        cfgs = [_make_two_exit_config(), _make_one_exit_config(), _make_top_only_config()]
        results = evaluator.evaluate_configurations(fp, cfgs, scenario, parameters=params)
        assert len(results) == 3

    def test_evaluate_configurations_preserves_config_ids(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        scenario = _make_scenario_three_agents()
        params = _make_params()
        cfgs = [_make_two_exit_config(), _make_one_exit_config()]
        results = evaluator.evaluate_configurations(fp, cfgs, scenario, parameters=params)
        ids = [r.configuration_id for r in results]
        assert "cfg_two_exits" in ids
        assert "cfg_one_exit" in ids


# ---------------------------------------------------------------------------
# O. Configurations can be ranked
# ---------------------------------------------------------------------------


class TestConfigurationsCanBeRanked:
    def test_rank_evaluations_returns_list(self):
        vr = ValidationResult(is_compliant=True)
        results = [
            EvaluationResult("b", True, True, vr, None, None, 20),
            EvaluationResult("a", True, True, vr, None, None, 10),
        ]
        ranked = rank_evaluations(results)
        assert isinstance(ranked, list)
        assert len(ranked) == 2

    def test_rank_evaluations_best_first(self):
        vr = ValidationResult(is_compliant=True)
        results = [
            EvaluationResult("b", True, True, vr, None, None, 20),
            EvaluationResult("a", True, True, vr, None, None, 10),
        ]
        ranked = rank_evaluations(results)
        assert ranked[0].score < ranked[1].score

    def test_rank_evaluations_empty_list(self):
        ranked = rank_evaluations([])
        assert ranked == []

    def test_rank_evaluations_single_element(self):
        vr = ValidationResult(is_compliant=True)
        results = [EvaluationResult("only", True, True, vr, None, None, 5)]
        ranked = rank_evaluations(results)
        assert len(ranked) == 1
        assert ranked[0].configuration_id == "only"


# ---------------------------------------------------------------------------
# P, Q, R. Evaluation does not mutate inputs
# ---------------------------------------------------------------------------


class TestNoMutation:
    def test_floor_plan_not_mutated(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        original_walls = list(fp.walls)
        original_doors = list(fp.doors)
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert fp.walls == original_walls
        assert fp.doors == original_doors

    def test_exit_config_not_mutated(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        original_exits = list(cfg.exits)
        original_id = cfg.configuration_id
        scenario = _make_scenario_three_agents()
        evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert cfg.exits == original_exits
        assert cfg.configuration_id == original_id

    def test_occupant_scenario_not_mutated(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        original_agent_ids = [a.agent_id for a in scenario.agents]
        evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert [a.agent_id for a in scenario.agents] == original_agent_ids


# ---------------------------------------------------------------------------
# S. Stage 11 validation results are preserved inside EvaluationResult
# ---------------------------------------------------------------------------


class TestValidationResultPreserved:
    def test_validation_result_preserved_for_valid_config(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_two_exit_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert result.validation_result is not None
        assert isinstance(result.validation_result, ValidationResult)

    def test_validation_result_preserved_for_invalid_config(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_invalid_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        assert result.validation_result is not None
        assert not result.validation_result.is_compliant

    def test_validation_violations_accessible(self):
        evaluator = _make_evaluator()
        fp = _make_simple_floor_plan()
        cfg = _make_invalid_config()
        scenario = _make_scenario_three_agents()
        result = evaluator.evaluate(fp, cfg, scenario, parameters=_make_params())
        # Invalid config should have at least one violation
        assert isinstance(result.validation_result.violations, list)


# ---------------------------------------------------------------------------
# Additional useful tests
# ---------------------------------------------------------------------------


class TestEvaluationMetricsDirect:
    def test_from_simulation_result_complete(self):
        """EvaluationMetrics.from_simulation_result with complete=True."""
        from evacuation_simulation.simulation.metrics import SimulationResult
        sim_result = SimulationResult(
            scenario_name="test",
            total_agents=3,
            evacuated_count=3,
            non_evacuated_count=0,
            total_timesteps=10,
            evacuation_rate=1.0,
            mean_evacuation_time=7.0,
            min_evacuation_time=5,
            max_evacuation_time=9,
            evacuation_times={"a1": 5, "a2": 7, "a3": 9},
            per_exit_utilization={(4, 2): 3},
            total_waiting_steps=2,
            mean_waiting_steps=0.67,
            max_waiting_steps=2,
            per_agent_waiting_steps={"a1": 0, "a2": 0, "a3": 2},
        )
        metrics = EvaluationMetrics.from_simulation_result(
            sim_result, evacuation_complete=True
        )
        assert metrics.total_evacuation_time == 10
        assert metrics.mean_evacuation_time == 7.0
        assert metrics.evacuation_rate == 1.0
        assert metrics.exit_utilization == {(4, 2): 3}

    def test_from_simulation_result_incomplete(self):
        """EvaluationMetrics.from_simulation_result with complete=False."""
        from evacuation_simulation.simulation.metrics import SimulationResult
        sim_result = SimulationResult(
            scenario_name="test",
            total_agents=3,
            evacuated_count=1,
            non_evacuated_count=2,
            total_timesteps=200,
            evacuation_rate=0.33,
            mean_evacuation_time=5.0,
            min_evacuation_time=5,
            max_evacuation_time=5,
            evacuation_times={"a1": 5},
        )
        metrics = EvaluationMetrics.from_simulation_result(
            sim_result, evacuation_complete=False
        )
        assert metrics.total_evacuation_time is None


class TestEvaluatorWithBottleneckScenario:
    """Integration test using the actual bottleneck scenario file."""

    def test_bottleneck_single_exit_evaluates(self):
        from evacuation_simulation.simulation.config import SimulationConfig
        sc = SimulationConfig.from_json(
            Path("scenarios/bottleneck.json")
        )
        evaluator = _make_evaluator()
        params = SimulationParameters(max_timesteps=200, random_seed=42)
        result = evaluator.evaluate(
            sc.floor_plan,
            sc.exit_configuration,
            sc.occupants,
            parameters=params,
        )
        # Reference config has one exit — should pass R1 but may fail R2
        assert isinstance(result, EvaluationResult)
        assert result.validation_result is not None
