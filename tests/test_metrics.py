"""
test_metrics.py
---------------
Tests for MetricsCollector and SimulationResult.
"""

from __future__ import annotations

import pytest

from evacuation_simulation.simulation.metrics import MetricsCollector, SimulationResult


# ---------------------------------------------------------------------------
# MetricsCollector construction
# ---------------------------------------------------------------------------


class TestMetricsCollectorConstruction:
    def test_basic_construction(self) -> None:
        collector = MetricsCollector(scenario_name="test", total_agents=5)
        assert collector.scenario_name == "test"
        assert collector.total_agents == 5
        assert collector.evacuated_count == 0

    def test_zero_agents_is_valid(self) -> None:
        collector = MetricsCollector(scenario_name="empty", total_agents=0)
        assert collector.total_agents == 0

    def test_negative_agents_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            MetricsCollector(scenario_name="bad", total_agents=-1)


# ---------------------------------------------------------------------------
# Evacuation recording
# ---------------------------------------------------------------------------


class TestEvacuationRecording:
    def test_record_single_evacuation(self) -> None:
        collector = MetricsCollector(scenario_name="s", total_agents=3)
        collector.record_evacuation("a1", timestep=5)
        assert collector.evacuated_count == 1

    def test_record_multiple_evacuations(self) -> None:
        collector = MetricsCollector(scenario_name="s", total_agents=3)
        collector.record_evacuation("a1", timestep=3)
        collector.record_evacuation("a2", timestep=7)
        assert collector.evacuated_count == 2

    def test_duplicate_record_raises(self) -> None:
        collector = MetricsCollector(scenario_name="s", total_agents=3)
        collector.record_evacuation("a1", timestep=5)
        with pytest.raises(ValueError, match="already been recorded"):
            collector.record_evacuation("a1", timestep=6)

    def test_is_fully_evacuated_false_partial(self) -> None:
        collector = MetricsCollector(scenario_name="s", total_agents=2)
        collector.record_evacuation("a1", timestep=1)
        assert collector.is_fully_evacuated() is False

    def test_is_fully_evacuated_true_all_done(self) -> None:
        collector = MetricsCollector(scenario_name="s", total_agents=2)
        collector.record_evacuation("a1", timestep=1)
        collector.record_evacuation("a2", timestep=2)
        assert collector.is_fully_evacuated() is True

    def test_is_fully_evacuated_zero_agents(self) -> None:
        collector = MetricsCollector(scenario_name="empty", total_agents=0)
        assert collector.is_fully_evacuated() is True


# ---------------------------------------------------------------------------
# Result computation
# ---------------------------------------------------------------------------


class TestResultComputation:
    def test_result_no_evacuations(self) -> None:
        collector = MetricsCollector(scenario_name="s", total_agents=3)
        result = collector.compute_result(total_timesteps=100)

        assert isinstance(result, SimulationResult)
        assert result.evacuated_count == 0
        assert result.non_evacuated_count == 3
        assert result.evacuation_rate == pytest.approx(0.0)
        assert result.mean_evacuation_time is None
        assert result.min_evacuation_time is None
        assert result.max_evacuation_time is None

    def test_result_all_evacuated(self) -> None:
        collector = MetricsCollector(scenario_name="s", total_agents=3)
        collector.record_evacuation("a1", timestep=10)
        collector.record_evacuation("a2", timestep=20)
        collector.record_evacuation("a3", timestep=30)
        result = collector.compute_result(total_timesteps=50)

        assert result.evacuated_count == 3
        assert result.non_evacuated_count == 0
        assert result.evacuation_rate == pytest.approx(1.0)

    def test_result_partial_evacuation_rate(self) -> None:
        collector = MetricsCollector(scenario_name="s", total_agents=4)
        collector.record_evacuation("a1", timestep=5)
        collector.record_evacuation("a2", timestep=8)
        result = collector.compute_result(total_timesteps=50)

        assert result.evacuation_rate == pytest.approx(0.5)
        assert result.non_evacuated_count == 2

    def test_result_mean_evacuation_time(self) -> None:
        collector = MetricsCollector(scenario_name="s", total_agents=2)
        collector.record_evacuation("a1", timestep=10)
        collector.record_evacuation("a2", timestep=20)
        result = collector.compute_result(total_timesteps=30)

        assert result.mean_evacuation_time == pytest.approx(15.0)

    def test_result_min_max_evacuation_time(self) -> None:
        collector = MetricsCollector(scenario_name="s", total_agents=3)
        collector.record_evacuation("a1", timestep=5)
        collector.record_evacuation("a2", timestep=12)
        collector.record_evacuation("a3", timestep=20)
        result = collector.compute_result(total_timesteps=25)

        assert result.min_evacuation_time == 5
        assert result.max_evacuation_time == 20

    def test_result_evacuation_times_dict(self) -> None:
        collector = MetricsCollector(scenario_name="s", total_agents=2)
        collector.record_evacuation("a1", timestep=7)
        collector.record_evacuation("a2", timestep=14)
        result = collector.compute_result(total_timesteps=20)

        assert result.evacuation_times == {"a1": 7, "a2": 14}

    def test_result_total_timesteps(self) -> None:
        collector = MetricsCollector(scenario_name="s", total_agents=1)
        result = collector.compute_result(total_timesteps=99)
        assert result.total_timesteps == 99

    def test_result_scenario_name(self) -> None:
        collector = MetricsCollector(scenario_name="my_scenario", total_agents=1)
        result = collector.compute_result(total_timesteps=10)
        assert result.scenario_name == "my_scenario"

    def test_result_evacuation_rate_zero_agents(self) -> None:
        """Edge case: 0 agents → evacuation rate should be 0.0 (not ZeroDivisionError)."""
        collector = MetricsCollector(scenario_name="empty", total_agents=0)
        result = collector.compute_result(total_timesteps=10)
        assert result.evacuation_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SimulationResult string representation
# ---------------------------------------------------------------------------


class TestSimulationResultStr:
    def test_str_contains_scenario_name(self) -> None:
        collector = MetricsCollector(scenario_name="alpha", total_agents=2)
        result = collector.compute_result(total_timesteps=10)
        assert "alpha" in str(result)

    def test_str_contains_evacuation_info(self) -> None:
        collector = MetricsCollector(scenario_name="alpha", total_agents=2)
        collector.record_evacuation("a1", timestep=5)
        result = collector.compute_result(total_timesteps=10)
        text = str(result)
        assert "Evacuated" in text
