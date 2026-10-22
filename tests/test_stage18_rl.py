"""
test_stage18_rl.py
------------------
Stage 18: Tests for Masked Actor-Critic RL Training and Evaluation.

Verifies:
1. Feature extraction tensor shapes, normalization, and action-mask layout.
2. Action-masked policy output (0 probability for invalid candidate doors).
3. Checkpoint saving and loading (weight preservation and metadata serialization).
4. Short end-to-end training execution and log generation.
5. Deterministic greedy policy evaluation on held-out scenarios.
6. Action mask enforcement ensuring illegal actions are never chosen.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from evacuation_simulation.simulation.rl_environment import ExitPlacementEnv, RLObservation
from evacuation_simulation.simulation.rl_policy import MaskedActorCritic, extract_features
from evacuation_simulation.train_rl import make_training_scenario, train_rl_agent
from evacuation_simulation.evaluate_rl import (
    create_scenario,
    evaluate_policy_on_scenario,
    evaluate_random_baseline_on_scenario,
)


class TestStage18RL:
    """Test suite for Stage 18 RL components."""

    def test_feature_extraction_shape_and_values(self):
        """Verify feature extraction creates properly bounded tensors."""
        obs = RLObservation(
            grid_shape=(20, 30),
            candidate_exits=[
                {"position": [0, 7], "width": 1.5, "door_id": "D1", "is_selected": False},
                {"position": [10, 0], "width": 1.5, "door_id": "D2", "is_selected": False},
                {"position": [10, 29], "width": 1.5, "door_id": "D3", "is_selected": False},
                {"position": [19, 22], "width": 1.5, "door_id": "D4", "is_selected": False},
            ],
            selected_candidate_indices=[],
            action_mask=[True, True, True, True],
            exits_remaining=2,
            total_agents=20,
        )

        state, mask = extract_features(obs, max_candidates=8)

        # Expected dimensions: 4 global + 8 * 5 per-candidate = 44
        assert state.shape == (1, 44)
        assert mask.shape == (1, 8)
        assert mask[0, :4].all()  # first 4 legal
        assert not mask[0, 4:].any()  # padding slots masked out

        # Global features bounded in [0, 1]
        assert 0.0 <= state[0, 0].item() <= 1.0  # rows
        assert 0.0 <= state[0, 1].item() <= 1.0  # cols

    def test_masked_policy_strictly_zeroes_illegal_actions(self):
        """Verify that masked actions have exactly zero probability."""
        model = MaskedActorCritic(max_candidates=8)

        obs = RLObservation(
            grid_shape=(20, 30),
            candidate_exits=[
                {"position": [0, 7], "width": 1.5, "door_id": "D1", "is_selected": False},
                {"position": [10, 0], "width": 1.5, "door_id": "D2", "is_selected": True},
                {"position": [10, 29], "width": 1.5, "door_id": "D3", "is_selected": False},
                {"position": [19, 22], "width": 1.5, "door_id": "D4", "is_selected": False},
            ],
            selected_candidate_indices=[1],
            action_mask=[True, False, True, False],  # only 0 and 2 legal
            exits_remaining=1,
            total_agents=20,
        )

        state, mask = extract_features(obs, max_candidates=8)
        probs, val = model(state, mask)

        # Probabilities for masked actions must be exactly 0
        assert probs[0, 1].item() == 0.0
        assert probs[0, 3].item() == 0.0
        assert probs[0, 4:].sum().item() == 0.0
        # Valid actions sum to 1.0
        assert pytest.approx(probs[0, 0].item() + probs[0, 2].item(), abs=1e-5) == 1.0

        # Sampled action must be legal
        for _ in range(20):
            action, _, _ = model.get_action(obs, deterministic=False)
            assert action in (0, 2)

    def test_model_save_and_load(self, tmp_path: Path):
        """Verify checkpoint saving and identical restoration."""
        model = MaskedActorCritic(max_candidates=8)
        ckpt_path = tmp_path / "test_model.pt"

        metadata = {"test_run": True, "episodes": 100}
        model.save(ckpt_path, metadata=metadata)
        assert ckpt_path.exists()

        loaded_model, loaded_meta = MaskedActorCritic.load(ckpt_path)
        assert loaded_meta["test_run"] is True
        assert loaded_meta["episodes"] == 100

        # Verify output parity
        obs = RLObservation(
            grid_shape=(20, 30),
            candidate_exits=[
                {"position": [0, 7], "width": 1.5, "door_id": "D1", "is_selected": False},
                {"position": [10, 0], "width": 1.5, "door_id": "D2", "is_selected": False},
            ],
            selected_candidate_indices=[],
            action_mask=[True, True],
            exits_remaining=2,
            total_agents=20,
        )
        state, mask = extract_features(obs, max_candidates=8)
        p1, v1 = model(state, mask)
        p2, v2 = loaded_model(state, mask)
        assert torch.allclose(p1, p2)
        assert torch.allclose(v1, v2)

    def test_short_training_run_executes(self, tmp_path: Path):
        """Verify that training pipeline runs and saves artifacts."""
        res = train_rl_agent(
            episodes=6,
            train_seeds=[42, 20],
            total_occupants=20,
            batch_size=2,
            output_dir=tmp_path,
            verbose=False,
        )

        assert Path(res["model_path"]).exists()
        assert Path(res["history_path"]).exists()
        assert len(res["history"]) == 6
        assert res["runtime_seconds"] > 0

        # Check training history file validity
        with open(res["history_path"], "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["training_metadata"]["episodes"] == 6
        assert len(data["episodes"]) == 6

    def test_policy_evaluation_on_scenario(self, tmp_path: Path):
        """Verify policy evaluation generates valid metrics."""
        model = MaskedActorCritic(max_candidates=4)
        scenario = create_scenario(seed=42, total_occupants=20)

        res = evaluate_policy_on_scenario(model, scenario, exit_count=2, deterministic=True)
        assert res["method"] == "Trained_RL_Agent"
        assert len(res["selected_indices"]) == 2
        assert len(res["selected_exits"]) == 2
        assert isinstance(res["is_feasible"], bool)
        assert res["decision_time_seconds"] > 0

    def test_random_baseline_evaluation(self):
        """Verify random baseline on scenario."""
        scenario = create_scenario(seed=42, total_occupants=20)
        res = evaluate_random_baseline_on_scenario(scenario, exit_count=2, seed=42)
        assert res["method"] == "Random_Valid_Baseline"
        assert len(res["selected_indices"]) == 2
        assert isinstance(res["is_feasible"], bool)

    def test_illegal_action_penalised_in_env(self):
        """Verify environment penalises illegal action with PENALTY_INVALID_ACTION."""
        env = ExitPlacementEnv()
        scenario = create_scenario(seed=42, total_occupants=20)
        obs = env.reset(scenario, exit_count=2)

        # Illegal action index (out of range)
        _, reward, terminated, info = env.step(99)
        assert terminated is True
        assert reward == ExitPlacementEnv.PENALTY_INVALID_ACTION
        assert info["reason"] == "invalid_action"
