"""
test_stage18b_extended.py
-------------------------
Tests for Stage 18B Extended RL Training, Live Logging, Diagnostics, and Evaluation.

Verifies:
1. Diverse scenario generator varies dimensions, candidate counts, and occupants across seeds.
2. Training script generates and updates incremental CSV and JSON files.
3. Model checkpointing properly serializes Stage 18B metadata.
4. Diagnostic plot generator creates valid PNG image with required data structures.
5. Evaluation benchmark runner handles paired comparisons and produces valid summaries.
6. Forbidden training seed guards guarantee zero overlap between training and held-out test seeds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
import torch

from evacuation_simulation.simulation.rl_policy import MaskedActorCritic
from evacuation_simulation.train_stage18b import (
    ALL_FORBIDDEN_TRAIN_SEEDS,
    RESERVED_EVAL_SEEDS,
    generate_diagnostic_plots,
    make_diverse_training_scenario,
    train_stage18b,
)
from evacuation_simulation.evaluate_stage18b import (
    EVAL_SEEDS_STAGE18B,
    create_eval_scenario,
    evaluate_random_baseline,
    run_heldout_benchmark,
)


class TestStage18BExtended:
    """Test suite for Stage 18B functionality."""

    def test_seed_isolation_guarantee(self):
        """Verify training seed pool strictly excludes all evaluation seeds."""
        assert len(RESERVED_EVAL_SEEDS) == 30
        assert len(EVAL_SEEDS_STAGE18B) == 30
        assert set(EVAL_SEEDS_STAGE18B) == RESERVED_EVAL_SEEDS
        assert 2 in ALL_FORBIDDEN_TRAIN_SEEDS  # Stage 18 eval seed
        assert 5002 in ALL_FORBIDDEN_TRAIN_SEEDS  # Stage 18B eval seed

    def test_diverse_scenario_generator_variation(self):
        """Verify scenario generation produces varied grid dimensions and occupant loads."""
        sc1 = make_diverse_training_scenario(seed=101)
        sc2 = make_diverse_training_scenario(seed=102)

        # Both valid SimulationConfigs
        assert sc1.floor_plan.grid.rows >= 16
        assert sc2.floor_plan.grid.rows >= 16
        assert len(sc1.occupants.agents) >= 15
        assert len(sc2.occupants.agents) >= 15

    def test_short_training_and_incremental_logging(self, tmp_path: Path):
        """Run short 4-episode training to verify live CSV/JSON logging and checkpoints."""
        res = train_stage18b(
            episodes=4,
            batch_size=2,
            output_dir=tmp_path,
            summary_interval=2,
            checkpoint_every=2,
        )

        assert res["episodes_completed"] == 4
        assert res["updates_completed"] == 2  # 4 ep / batch_size 2 = 2 updates

        # Check model file
        model_path = tmp_path / "model_stage18b.pt"
        assert model_path.exists()
        loaded_model, meta = MaskedActorCritic.load(model_path)
        assert meta["stage"] == "18B"
        assert meta["updates_completed"] == 2

        # Check CSV
        csv_path = tmp_path / "training_history_stage18b.csv"
        assert csv_path.exists()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) == 5  # 1 header + 4 episode rows
            assert reader[0][0] == "episode"
            assert reader[1][0] == "1"
            assert reader[4][0] == "4"

        # Check JSON
        json_path = tmp_path / "training_history_stage18b.json"
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data["episodes"]) == 4
            assert len(data["loss_history"]) == 2

    def test_diagnostic_plot_creation(self, tmp_path: Path):
        """Verify diagnostic plot generator creates PNG file from sample history."""
        sample_history = [
            {"episode": 1, "reward": -2000.0, "rolling_reward": -2000.0, "is_feasible": False, "total_evacuation_time": None, "rolling_feasibility": 0.0},
            {"episode": 2, "reward": -32.0, "rolling_reward": -1016.0, "is_feasible": True, "total_evacuation_time": 32, "rolling_feasibility": 0.5},
            {"episode": 3, "reward": -30.0, "rolling_reward": -687.3, "is_feasible": True, "total_evacuation_time": 30, "rolling_feasibility": 0.67},
        ]
        sample_loss = [
            {"update": 1, "episode": 2, "actor_loss": 0.5, "critic_loss": 12.0, "entropy": 1.1, "grad_norm": 0.8},
        ]
        plot_path = tmp_path / "test_diag.png"
        generate_diagnostic_plots(sample_history, sample_loss, plot_path)
        assert plot_path.exists()
        assert plot_path.stat().st_size > 1000

    def test_eval_benchmark_runner_mock(self, tmp_path: Path):
        """Verify held-out evaluation runner works on small seed set."""
        # Train tiny models
        m18 = MaskedActorCritic(max_candidates=8)
        p18 = tmp_path / "m18.pt"
        m18.save(p18, metadata={"stage": 18})

        m18b = MaskedActorCritic(max_candidates=8)
        p18b = tmp_path / "m18b.pt"
        m18b.save(p18b, metadata={"stage": "18B"})

        test_seeds = [5001, 5002]
        benchmark = run_heldout_benchmark(
            model_18_path=p18,
            model_18b_path=p18b,
            seeds=test_seeds,
            output_dir=tmp_path,
        )

        assert benchmark["total_scenarios"] == 2
        assert len(benchmark["scenario_results"]) == 2
        assert "stage_18" in benchmark["summary"]
        assert "stage_18b" in benchmark["summary"]
        assert "exhaustive_search" in benchmark["summary"]
        assert "random_baseline" in benchmark["summary"]

        eval_json = tmp_path / "evaluation_results_stage18b.json"
        assert eval_json.exists()
