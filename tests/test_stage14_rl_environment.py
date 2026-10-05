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
    assert reward == env.PENALTY_INVALID_ACTION
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
