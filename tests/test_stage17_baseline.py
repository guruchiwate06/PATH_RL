"""
test_stage17_baseline.py
------------------------
Stage 17: Exhaustive Exit-Configuration Baseline Tests.

Verifies:
1. Reproducible baseline execution connecting generator, seeder, searcher, validator, and evaluator.
2. Hard NBC constraint filtering (invalid configurations rejected before simulation).
3. Fair comparison invariance (same layout and fixed occupant scenario across all configurations).
4. Deterministic ranking order (feasible first, sorted by total evacuation time).
5. Metrics integrity (actual simulation outcomes: total_evacuation_time, evacuation_rate, exit_utilization).
6. Multi-seed baseline suite aggregation and JSON serialization.
7. Exhaustiveness status and combination limit safeguards.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from evacuation_simulation.run_baseline_experiment import (
    run_baseline_experiment,
    run_baseline_suite,
)


class TestStage17BaselineExperiment:
    """Test suite for Stage 17 Exhaustive Exit-Configuration Baseline."""

    def test_baseline_experiment_runs_successfully(self, tmp_path: Path):
        """Verify baseline runs cleanly and returns structured dictionary."""
        result = run_baseline_experiment(
            seed=42,
            total_occupants=20,
            grid_rows=20,
            grid_cols=30,
            num_candidate_exits=4,
            exit_count=2,
            output_dir=tmp_path,
            verbose=False,
        )

        assert isinstance(result, dict)
        assert result["experiment_type"] == "exhaustive_exit_configuration_baseline"
        assert result["stage"] == 17
        assert "environment" in result
        assert "search_statistics" in result
        assert "baseline_summary" in result
        assert "all_configurations_ranked" in result

    def test_hard_constraint_filtering(self, tmp_path: Path):
        """Verify that hard constraints reject non-compliant configurations."""
        result = run_baseline_experiment(
            seed=42,
            total_occupants=20,
            grid_rows=20,
            grid_cols=30,
            num_candidate_exits=4,
            exit_count=2,
            output_dir=tmp_path,
            verbose=False,
        )

        stats = result["search_statistics"]
        # In a 4-choose-2 search, 6 combinations are generated
        assert stats["total_generated"] == 6
        assert stats["rejected_by_constraints"] > 0
        assert stats["feasible"] > 0
        assert stats["total_generated"] == stats["rejected_by_constraints"] + stats["feasible"]

        # Check violation records on infeasible configurations
        ranked = result["all_configurations_ranked"]
        infeasible = [c for c in ranked if not c["is_feasible"]]
        assert len(infeasible) == stats["rejected_by_constraints"]
        for c in infeasible:
            assert c["is_compliant"] is False
            assert c["score"] is None
            assert len(c["violations"]) > 0
            for v in c["violations"]:
                assert "rule_id" in v
                assert "message" in v

    def test_feasible_configurations_and_metrics_integrity(self, tmp_path: Path):
        """Verify feasible configurations contain actual simulation metrics."""
        result = run_baseline_experiment(
            seed=42,
            total_occupants=20,
            output_dir=tmp_path,
            verbose=False,
        )

        summary = result["baseline_summary"]
        assert summary["has_feasible_solution"] is True
        assert summary["best_configuration_id"] is not None
        assert summary["best_exit_positions"] is not None
        assert summary["best_score"] is not None
        assert summary["best_score"] > 0

        best_cfg = summary["best_configuration"]
        assert best_cfg is not None
        assert best_cfg["evacuation_rate"] == 1.0
        assert best_cfg["evacuated_count"] == 20
        assert best_cfg["total_evacuation_time"] == summary["best_score"]
        assert best_cfg["mean_evacuation_time"] is not None
        assert best_cfg["mean_evacuation_time"] > 0

        # Check exit utilisation sums to total occupants
        util = best_cfg["exit_utilization"]
        assert sum(util.values()) == 20

    def test_ranking_order_rules(self, tmp_path: Path):
        """Verify ranking follows Stage 12/13 rules (feasible first, best score ascending)."""
        result = run_baseline_experiment(
            seed=42,
            total_occupants=20,
            output_dir=tmp_path,
            verbose=False,
        )

        ranked = result["all_configurations_ranked"]
        assert len(ranked) == 6

        # Feasible must appear before infeasible
        feasible_seen_done = False
        feasible_scores = []
        for c in ranked:
            if c["is_feasible"]:
                assert not feasible_seen_done, "Feasible configuration appeared after infeasible!"
                feasible_scores.append(c["score"])
            else:
                feasible_seen_done = True

        # Feasible scores must be sorted ascending
        assert feasible_scores == sorted(feasible_scores)

        # Summary scores check
        summary = result["baseline_summary"]
        assert summary["worst_feasible_score"] >= summary["best_score"]
        assert summary["tied_for_best_count"] >= 1
        assert len(summary["tied_configuration_ids"]) == summary["tied_for_best_count"]

    def test_json_export_and_readability(self, tmp_path: Path):
        """Verify results are written to JSON and can be read back identically."""
        result = run_baseline_experiment(
            seed=42,
            total_occupants=20,
            output_dir=tmp_path,
            verbose=False,
        )

        json_files = list(tmp_path.glob("baseline_seed42_*.json"))
        assert len(json_files) == 1
        with open(json_files[0], "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["experiment_type"] == result["experiment_type"]
        assert loaded["baseline_summary"]["best_score"] == result["baseline_summary"]["best_score"]
        assert loaded["search_statistics"]["total_generated"] == 6

    def test_multi_seed_suite(self, tmp_path: Path):
        """Verify running a baseline suite across multiple seeds."""
        seeds = [42, 20]
        suite_res = run_baseline_suite(
            seeds=seeds,
            total_occupants=20,
            output_dir=tmp_path,
            verbose=False,
        )

        assert suite_res["suite_type"] == "exhaustive_baseline_multiseed_suite"
        assert suite_res["total_seeds"] == 2
        assert len(suite_res["runs"]) == 2

        # Verify suite JSON exists
        suite_files = list(tmp_path.glob("baseline_suite_*.json"))
        assert len(suite_files) == 1

        # Both runs found feasible solutions
        assert suite_res["runs"][0]["baseline_summary"]["has_feasible_solution"] is True
        assert suite_res["runs"][1]["baseline_summary"]["has_feasible_solution"] is True

    def test_exhaustive_flag_and_limit_guard(self, tmp_path: Path):
        """Verify is_exhaustive is True and max_configurations guard raises on overflow."""
        # Exhaustive run
        res = run_baseline_experiment(
            seed=42,
            total_occupants=20,
            output_dir=tmp_path,
            verbose=False,
        )
        assert res["search_statistics"]["is_exhaustive"] is True

        # Limit exceeded should raise ValueError
        with pytest.raises(ValueError, match="exceeds max_configurations limit"):
            run_baseline_experiment(
                seed=42,
                total_occupants=20,
                max_configurations=3,  # limit is 3, but 6 combinations exist
                output_dir=tmp_path,
                verbose=False,
            )
