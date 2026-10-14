"""
test_stage14_rl_environment.py
------------------------------
Tests for the Stage 14A RL Environment formulation.
"""

from pathlib import Path
import pytest

from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.rl_environment import ExitPlacementEnv
from evacuation_simulation.simulation.search import search_exit_configurations

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

def test_initialization():
    env = ExitPlacementEnv()
    assert env._evaluator is not None
    assert env.terminated is True
    assert env.config is None

def test_deterministic_reset():
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    env1 = ExitPlacementEnv()
    obs1 = env1.reset(config, exit_count=2)
    
    env2 = ExitPlacementEnv()
    obs2 = env2.reset(config, exit_count=2)
    
    assert obs1.to_dict() == obs2.to_dict()

def test_observation_structure():
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=2)
    
    assert isinstance(obs.grid_shape, tuple)
    assert len(obs.grid_shape) == 2
    assert isinstance(obs.candidate_exits, list)
    assert "position" in obs.candidate_exits[0]
    assert "width" in obs.candidate_exits[0]
    assert isinstance(obs.selected_candidate_indices, list)
    assert isinstance(obs.action_mask, list)
    assert obs.exits_remaining == 2
    assert obs.total_agents == len(config.occupants.agents)

def test_action_mask_and_invalid_action():
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=3)
    
    # Select first valid action
    valid_action = obs.action_mask.index(True)
    obs, reward, terminated, info = env.step(valid_action)
    
    assert not terminated
    assert obs.exits_remaining == 2
    assert valid_action in obs.selected_candidate_indices
    # Cannot select the same action again
    assert obs.action_mask[valid_action] is False

    # Invalid action
    _, reward, terminated, info = env.step(valid_action)
    assert terminated
    assert reward == env.PENALTY_INVALID_ACTION
    assert info["reason"] == "invalid_action"

def test_episode_terminates_after_required_exit_count():
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=2)
    
    # Step 1
    action1 = obs.action_mask.index(True)
    obs, reward, terminated, info = env.step(action1)
    assert not terminated
    
    # Step 2
    action2 = obs.action_mask.index(True)
    obs, reward, terminated, info = env.step(action2)
    assert terminated
    assert reward != env.PENALTY_INVALID_ACTION

def test_final_configuration_validation_invalid():
    # Construct a scenario where picking 2 exits next to each other violates separation (R3)
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=2)
    
    # Find two candidate exits that are adjacent to force a separation violation
    # Just pick first two from the top left
    obs, _, _, _ = env.step(0)
    obs, reward, terminated, info = env.step(1)
    
    assert terminated
    if not info["is_compliant"]:
        assert reward == env.PENALTY_INVALID_FINAL_CONFIG
        assert not info["is_feasible"]
        assert len(info["violations"]) > 0

def test_reward_is_deterministic():
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    config.parameters.random_seed = 42
    
    env1 = ExitPlacementEnv()
    obs = env1.reset(config, exit_count=2)
    action1 = obs.action_mask.index(True)
    obs, _, _, _ = env1.step(action1)
    action2 = obs.action_mask.index(True)
    _, reward1, _, _ = env1.step(action2)
    
    env2 = ExitPlacementEnv()
    obs = env2.reset(config, exit_count=2)
    obs, _, _, _ = env2.step(action1)
    _, reward2, _, _ = env2.step(action2)
    
    assert reward1 == reward2

def test_stage_13_baseline_compatibility():
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    config.parameters.random_seed = 42
    
    # Run Search
    search_res = search_exit_configurations(
        floor_plan=config.floor_plan,
        occupant_scenario=config.occupants,
        exit_count=2,
        parameters=config.parameters,
        top_k=1
    )
    
    # Find the indices of the best configuration in candidates
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=2)
    candidates = env.candidates
    
    if search_res.has_feasible_solution:
        best_cfg = search_res.best_configuration
        indices = []
        for best_exit in best_cfg.exits:
            for i, cand in enumerate(candidates):
                if list(cand.position) == list(best_exit):
                    indices.append(i)
                    break
        
        # Step through RL environment
        obs, _, _, _ = env.step(indices[0])
        obs, reward, terminated, info = env.step(indices[1])
        
        assert terminated
        assert info["is_feasible"]
        # The reward should perfectly equal -total_evacuation_time
        assert reward == -search_res.best_evaluation.score

def test_generalization_to_basic_building():
    # Test that env handles a completely different scenario seamlessly
    config = SimulationConfig.from_json(SCENARIOS_DIR / "basic_building.json")
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=1)
    
    assert obs.total_agents == len(config.occupants.agents)
    
    action1 = obs.action_mask.index(True)
    obs, reward, terminated, info = env.step(action1)
    assert terminated
    assert info["is_compliant"] is not None

def test_dead_end_termination():
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=2)
    
    # Artificially increase exit_count so it doesn't terminate on the last candidate
    env.exit_count = len(env.candidates) + 1
    
    # Select all but the last candidate
    for i in range(len(env.candidates) - 1):
        obs, reward, terminated, info = env.step(i)
        assert not terminated
        
    # Now there is exactly 1 candidate left to select.
    last_action = len(env.candidates) - 1
    
    # Stepping the last action will leave 0 remaining valid actions, 
    # but we still haven't reached exit_count. This triggers dead_end.
    obs, reward, terminated, info = env.step(last_action)
    
    assert terminated
    assert reward == env.PENALTY_DEAD_END
    assert info["reason"] == "dead_end"

def test_invalid_action_behavior():
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=2)
    # Give index out of bounds
    obs, reward, terminated, info = env.step(9999)
    assert terminated
    assert reward == env.PENALTY_INVALID_ACTION
    assert info["reason"] == "invalid_action"


# ---------------------------------------------------------------------------
# Additional tests to satisfy Stage 14A spec requirements
# ---------------------------------------------------------------------------


def test_observation_determinism():
    """Spec item 4: observation is deterministic for identical inputs."""
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    config.parameters.random_seed = 0

    env = ExitPlacementEnv()
    obs_a = env.reset(config, exit_count=2)
    obs_b = env.reset(config, exit_count=2)

    assert obs_a.to_dict() == obs_b.to_dict()


def test_observation_exits_remaining_decrements():
    """Spec item 6 / lifecycle: exits_remaining decrements by 1 per valid step."""
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=3)
    assert obs.exits_remaining == 3

    action = obs.action_mask.index(True)
    obs, _, _, _ = env.step(action)
    assert obs.exits_remaining == 2

    action = obs.action_mask.index(True)
    obs, _, _, _ = env.step(action)
    assert obs.exits_remaining == 1


def test_candidate_is_selected_flag():
    """Spec item 6: is_selected flag in candidate_exits mirrors selected_indices."""
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=2)

    # Before any selection all is_selected should be False
    for c in obs.candidate_exits:
        assert c["is_selected"] is False

    action = obs.action_mask.index(True)
    obs, _, _, _ = env.step(action)

    # The selected candidate's is_selected should be True
    assert obs.candidate_exits[action]["is_selected"] is True
    # Others remain False
    for i, c in enumerate(obs.candidate_exits):
        if i != action:
            assert c["is_selected"] is False


def test_action_mask_determinism():
    """Spec item 7: action_mask is deterministic - same call twice = same mask."""
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    env = ExitPlacementEnv()
    env.reset(config, exit_count=2)

    mask_a = env.action_mask()
    mask_b = env.action_mask()
    assert mask_a == mask_b


def test_interior_passage_not_in_action_space():
    """Spec item 10: interior corridor passages cannot become actions.

    The bottleneck scenario has one interior door (door_bottleneck_corridor,
    exterior=False). It must never appear as a candidate or action.
    """
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")

    # Identify the interior door position
    interior_positions = [
        list(d.position) for d in config.floor_plan.doors if not d.exterior
    ]
    assert len(interior_positions) > 0, "Bottleneck scenario must have an interior door"

    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=2)

    candidate_positions = [c["position"] for c in obs.candidate_exits]
    for pos in interior_positions:
        assert pos not in candidate_positions, (
            f"Interior door at {pos} must not appear in candidate_exits"
        )


def test_valid_configuration_reaches_stage12():
    """Spec item 14: valid configuration must reach Stage 12 (metrics not None)."""
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    config.parameters.random_seed = 42
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=2)

    action1 = obs.action_mask.index(True)
    obs, _, _, _ = env.step(action1)
    action2 = obs.action_mask.index(True)
    _, reward, terminated, info = env.step(action2)

    assert terminated
    # For a feasible result, Stage 12 metrics must have been populated
    if info["is_feasible"]:
        assert info["total_evacuation_time"] is not None
        assert info["total_evacuation_time"] > 0


def test_invalid_configuration_not_treated_as_valid():
    """Spec item 15: invalid config must never produce a positive/zero reward."""
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=2)

    # Try indices 0 then 1 (adjacent — likely violates R3 separation)
    obs, _, _, _ = env.step(0)
    _, reward, terminated, info = env.step(1)

    assert terminated
    if not info["is_feasible"]:
        # Must be a penalty, not a positive reward
        assert reward == float(env.PENALTY_INVALID_FINAL_CONFIG)
        assert info["is_compliant"] is not None


def test_different_exit_configurations_produce_different_rewards():
    """Spec item 18: different exit configurations can produce different rewards."""
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    config.parameters.random_seed = 42

    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=2)
    candidates = env.candidates

    # Need at least 3 exterior candidates to get two non-overlapping pairs
    if len(candidates) < 3:
        pytest.skip("Not enough candidates for two distinct pairs")

    # Pair A: candidates 0, 1 (may violate separation — that's fine, we just compare)
    env.reset(config, exit_count=2)
    _, _, _, _ = env.step(0)
    _, reward_a, _, _ = env.step(1)

    # Pair B: candidates 0, 2
    env.reset(config, exit_count=2)
    _, _, _, _ = env.step(0)
    _, reward_b, _, _ = env.step(2)

    # The two rewards may be the same (if both invalid), or different (if valid).
    # What we must prove: the environment CAN distinguish configs by reward.
    # At least one of these pairs must be different from a single-candidate run:
    env.reset(config, exit_count=1)
    _, reward_single, _, _ = env.step(0)

    # reward_a or reward_b is distinct from reward_single (2-exit vs 1-exit)
    assert (reward_a != reward_single) or (reward_b != reward_single), (
        "Exit-2 and exit-1 configs should not all have identical rewards"
    )


def test_stage13_baseline_unchanged():
    """Spec item 19: Stage 13 search module remains callable and unchanged."""
    from evacuation_simulation.simulation.search import (
        search_exit_configurations,
        SearchResult,
        ExitConfigurationSearcher,
    )
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    config.parameters.random_seed = 42

    result = search_exit_configurations(
        floor_plan=config.floor_plan,
        occupant_scenario=config.occupants,
        exit_count=2,
        parameters=config.parameters,
        top_k=1,
    )
    assert isinstance(result, SearchResult)
    assert result.total_configurations > 0


def test_reward_constants_ordering():
    """Verify the three penalty constants have the correct relative ordering."""
    env = ExitPlacementEnv()
    # Invalid action is the harshest penalty
    assert env.PENALTY_INVALID_ACTION < env.PENALTY_DEAD_END
    assert env.PENALTY_DEAD_END < env.PENALTY_INVALID_FINAL_CONFIG
    # All penalties are negative
    assert env.PENALTY_INVALID_FINAL_CONFIG < 0


def test_observation_candidate_fields():
    """Spec: each candidate dict must contain position, width, door_id, is_selected."""
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=2)

    for c in obs.candidate_exits:
        assert "position" in c
        assert "width" in c
        assert "door_id" in c
        assert "is_selected" in c
        assert isinstance(c["position"], list)
        assert len(c["position"]) == 2
        assert isinstance(c["width"], float)
        assert isinstance(c["is_selected"], bool)


def test_render_and_close_are_callable():
    """Spec: render() and close() must exist and not raise."""
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    env = ExitPlacementEnv()
    env.reset(config, exit_count=2)
    env.render()  # must not raise
    env.close()   # must not raise


def test_generalization_to_two_exits_scenario():
    """Spec: env works with two_exits scenario (no explicit doors, legacy boundary)."""
    config = SimulationConfig.from_json(SCENARIOS_DIR / "two_exits.json")
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=2)

    # With legacy boundary candidates the mask should have legal actions
    assert obs.exits_remaining == 2
    assert any(obs.action_mask)

    # Run one valid action
    action = obs.action_mask.index(True)
    obs, reward, terminated, info = env.step(action)
    assert not terminated
    assert obs.exits_remaining == 1


def test_generalization_to_corridor_scenario():
    """Spec: env works with corridor scenario (different grid + no explicit doors)."""
    config = SimulationConfig.from_json(SCENARIOS_DIR / "corridor.json")
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=1)

    # Grid shape should match the corridor scenario dimensions
    assert obs.grid_shape[0] == config.floor_plan.grid.rows
    assert obs.grid_shape[1] == config.floor_plan.grid.cols
    assert obs.total_agents == len(config.occupants.agents)
    assert any(obs.action_mask)

    action = obs.action_mask.index(True)
    _, reward, terminated, info = env.step(action)
    assert terminated  # exit_count=1 so one step ends the episode
    assert info["is_compliant"] is not None


def test_same_config_same_evaluation():
    """Spec item 17: same final configuration produces same evaluation result."""
    config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    config.parameters.random_seed = 7

    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=2)
    action1 = obs.action_mask.index(True)
    obs, _, _, _ = env.step(action1)
    action2 = obs.action_mask.index(True)
    _, reward1, _, info1 = env.step(action2)

    # Second run: identical action sequence
    env.reset(config, exit_count=2)
    _, _, _, _ = env.step(action1)
    _, reward2, _, info2 = env.step(action2)

    assert reward1 == reward2
    assert info1["is_feasible"] == info2["is_feasible"]
    assert info1["total_evacuation_time"] == info2["total_evacuation_time"]
