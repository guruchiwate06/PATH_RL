"""
test_profile.py
---------------
Dedicated test suite for Stage 6: Extensible Heterogeneous Agent Profile System.

Covers:
- Test 1: Default profile (speed=1.0, reaction_delay=0)
- Test 2: Explicit profile loading
- Test 3: Heterogeneous profiles (different agents with different profiles)
- Test 4: Validation (bounds on speed and reaction_delay)
- Test 5: Backward compatibility (old scenarios match explicit baseline)
- Test 6: Speed 1.0 preserves baseline movement frequency
- Test 7: Speed 0.5 deterministic 1-in-2 step cadence
- Test 8: Speed 0.25 deterministic 1-in-4 step cadence
- Test 9: No multi-cell movement in a single timestep
- Test 10: Reaction delay exact timestep convention
- Test 11: Semantic distinction: reaction delay is NOT congestion waiting
- Test 12: Mixed population execution in a shared environment
- Test 13: Determinism under identical configuration and seed
- Test 14: Stage 5 experiment integration (parameter sweep with profiles)
- Test 15: Serialization round-trip and export preservation
- Test 16: Inspector metadata exposure
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from evacuation_simulation.simulation.agent import Agent, AgentState
from evacuation_simulation.simulation.config import (
    AgentConfig,
    AgentProfileConfig,
    SimulationConfig,
)
from evacuation_simulation.simulation.experiment import (
    SweepPoint,
    aggregate_results,
    export_csv,
    export_json,
    run_scenario,
    run_sweep,
)
from evacuation_simulation.simulation.profile import AgentProfile
from evacuation_simulation.simulation.simulation import Simulation
from evacuation_simulation.simulation.strategy import ShortestPathStrategy


def _make_config(
    agents: list[AgentConfig | dict],
    rows: int = 10,
    cols: int = 10,
    exits: list[tuple[int, int]] | None = None,
    walls: list[tuple[int, int]] | None = None,
    capacity: int = 1,
    seed: int = 42,
    max_steps: int = 100,
) -> SimulationConfig:
    """Helper to build a SimulationConfig with explicit agents."""
    agent_configs = []
    for a in agents:
        if isinstance(a, AgentConfig):
            agent_configs.append(a)
        else:
            agent_configs.append(AgentConfig(**a))

    return SimulationConfig(
        scenario_name="profile_test",
        grid={"rows": rows, "cols": cols},
        walls=[list(w) for w in (walls or [])],
        exits=[list(e) for e in (exits or [(rows - 1, cols - 1)])],
        agents=agent_configs,
        parameters={
            "max_timesteps": max_steps,
            "random_seed": seed,
            "default_cell_capacity": capacity,
        },
    )


# ---------------------------------------------------------------------------
# Test 1 & 2: Profile Construction & Defaults
# ---------------------------------------------------------------------------


class TestProfileBasics:
    def test_default_profile(self):
        """Test 1: Default profile has speed=1.0 and reaction_delay=0."""
        prof = AgentProfile()
        assert prof.speed == 1.0
        assert prof.reaction_delay == 0
        assert prof.input_attributes == {}
        assert prof.derived_parameters == {}

        # Default Agent has default profile
        agent = Agent("a1", row=0, col=0)
        assert agent.speed == 1.0
        assert agent.reaction_delay == 0
        assert agent.profile.speed == 1.0

    def test_explicit_profile(self):
        """Test 2: Explicit profile loading via dict and AgentProfileConfig."""
        prof_cfg = AgentProfileConfig(speed=0.5, reaction_delay=3, age=45)
        agent_cfg = AgentConfig(agent_id="a_exp", row=1, col=2, profile=prof_cfg)

        agent = Agent.from_config(agent_cfg)
        assert agent.speed == 0.5
        assert agent.reaction_delay == 3
        assert agent.profile.input_attributes.get("age") == 45

    def test_different_agents_different_profiles(self):
        """Test 3: Different agents can have different profiles simultaneously."""
        cfg = _make_config([
            {"agent_id": "fast", "row": 0, "col": 0, "profile": {"speed": 1.0, "reaction_delay": 0}},
            {"agent_id": "slow", "row": 0, "col": 1, "profile": {"speed": 0.5, "reaction_delay": 2}},
            {"agent_id": "default", "row": 0, "col": 2},
        ])
        sim = Simulation(cfg, ShortestPathStrategy())
        agents = {a.agent_id: a for a in sim.agents}

        assert agents["fast"].speed == 1.0
        assert agents["fast"].reaction_delay == 0
        assert agents["slow"].speed == 0.5
        assert agents["slow"].reaction_delay == 2
        assert agents["default"].speed == 1.0
        assert agents["default"].reaction_delay == 0


# ---------------------------------------------------------------------------
# Test 4: Validation
# ---------------------------------------------------------------------------


class TestProfileValidation:
    def test_speed_validation(self):
        """Test 4a: Invalid speed values are rejected."""
        with pytest.raises(ValueError):
            AgentProfile(speed=0.0)

        with pytest.raises(ValueError):
            AgentProfile(speed=-0.5)

        with pytest.raises(ValueError):
            AgentProfile(speed=1.5)

        # Via Pydantic config
        with pytest.raises(Exception):
            AgentProfileConfig(speed=0.0)

        with pytest.raises(Exception):
            AgentProfileConfig(speed=1.1)

    def test_reaction_delay_validation(self):
        """Test 4b: Invalid reaction delay is rejected."""
        with pytest.raises(ValueError):
            AgentProfile(reaction_delay=-1)

        with pytest.raises(Exception):
            AgentProfileConfig(reaction_delay=-1)


# ---------------------------------------------------------------------------
# Test 5 & 6: Backward Compatibility & Speed 1.0
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_backward_compatibility_execution(self):
        """Test 5 & 6: Omitting profile produces identical simulation behavior as explicit speed=1.0, delay=0."""
        # Config A: No profile specified (old format)
        cfg_old = _make_config([
            {"agent_id": "a0", "row": 0, "col": 0},
            {"agent_id": "a1", "row": 0, "col": 1},
        ], rows=5, cols=5, exits=[(4, 4)])

        # Config B: Explicit baseline profile
        cfg_explicit = _make_config([
            {"agent_id": "a0", "row": 0, "col": 0, "profile": {"speed": 1.0, "reaction_delay": 0}},
            {"agent_id": "a1", "row": 0, "col": 1, "profile": {"speed": 1.0, "reaction_delay": 0}},
        ], rows=5, cols=5, exits=[(4, 4)])

        sim_old = Simulation(cfg_old, ShortestPathStrategy())
        res_old = sim_old.run()

        sim_exp = Simulation(cfg_explicit, ShortestPathStrategy())
        res_exp = sim_exp.run()

        assert res_old.total_timesteps == res_exp.total_timesteps
        assert res_old.evacuated_count == res_exp.evacuated_count
        assert res_old.mean_evacuation_time == res_exp.mean_evacuation_time
        assert res_old.evacuation_times == res_exp.evacuation_times
        assert res_old.total_waiting_steps == res_exp.total_waiting_steps


# ---------------------------------------------------------------------------
# Test 7, 8, 9: Speed Semantics & Cadence
# ---------------------------------------------------------------------------


class TestSpeedSemantics:
    def test_speed_half_cadence(self):
        """Test 7: Speed 0.5 moves exactly every 2 steps in isolation."""
        # 1 agent, 2 cells away from exit at (0, 2). Starting at (0, 0).
        # Path length = 2 hops.
        # Speed 1.0: moves step 1 -> (0, 1), step 2 -> (0, 2) [evacuated at step 2].
        # Speed 0.5: step 1 -> stationary, step 2 -> (0, 1), step 3 -> stationary, step 4 -> (0, 2) [evacuated at step 4].
        cfg = _make_config([
            {"agent_id": "a_half", "row": 0, "col": 0, "profile": {"speed": 0.5, "reaction_delay": 0}}
        ], rows=3, cols=3, exits=[(0, 2)])

        sim = Simulation(cfg, ShortestPathStrategy())
        agent = sim.agents[0]

        # Step 1: not eligible, remains at (0, 0)
        cont = sim.step()
        assert cont is True
        assert agent.position == (0, 0)
        assert agent.state == AgentState.MOVING

        # Step 2: eligible, moves to (0, 1)
        cont = sim.step()
        assert cont is True
        assert agent.position == (0, 1)

        # Step 3: not eligible, remains at (0, 1)
        cont = sim.step()
        assert cont is True
        assert agent.position == (0, 1)

        # Step 4: eligible, moves to (0, 2) and evacuates
        cont = sim.step()
        assert cont is False  # simulation terminates because all agents evacuated
        assert agent.state == AgentState.EVACUATED
        assert agent.evacuation_timestep == 4

    def test_speed_quarter_cadence(self):
        """Test 8: Speed 0.25 moves exactly every 4 steps in isolation."""
        # 1 agent, 1 cell away from exit at (0, 1). Starting at (0, 0).
        cfg = _make_config([
            {"agent_id": "a_quarter", "row": 0, "col": 0, "profile": {"speed": 0.25, "reaction_delay": 0}}
        ], rows=3, cols=3, exits=[(0, 1)])

        sim = Simulation(cfg, ShortestPathStrategy())
        agent = sim.agents[0]

        # Steps 1, 2, 3: stationary
        for step_num in (1, 2, 3):
            sim.step()
            assert agent.position == (0, 0)

        # Step 4: moves to (0, 1) and evacuates
        sim.step()
        assert agent.state == AgentState.EVACUATED
        assert agent.evacuation_timestep == 4

    def test_no_multi_cell_movement(self):
        """Test 9: An agent never moves more than one Manhattan cell in one timestep."""
        cfg = _make_config([
            {"agent_id": "a0", "row": 0, "col": 0, "profile": {"speed": 1.0}},
            {"agent_id": "a1", "row": 1, "col": 0, "profile": {"speed": 0.75}},
            {"agent_id": "a2", "row": 2, "col": 0, "profile": {"speed": 0.5}},
        ], rows=6, cols=6, exits=[(5, 5)])

        sim = Simulation(cfg, ShortestPathStrategy())
        while not sim.state.is_terminated:
            prev_positions = {a.agent_id: a.position for a in sim.agents if a.is_active}
            sim.step()
            for a in sim.agents:
                if a.agent_id in prev_positions:
                    r0, c0 = prev_positions[a.agent_id]
                    r1, c1 = a.position
                    manhattan_dist = abs(r1 - r0) + abs(c1 - c0)
                    assert manhattan_dist <= 1, (
                        f"Agent {a.agent_id} jumped {manhattan_dist} cells in timestep {sim.state.timestep}!"
                    )


# ---------------------------------------------------------------------------
# Test 10 & 11: Reaction Delay Semantics & Independence from Waiting
# ---------------------------------------------------------------------------


class TestReactionDelaySemantics:
    def test_reaction_delay_exact_timesteps(self):
        """Test 10: reaction_delay=2 remains stationary at steps 1 & 2, begins moving at step 3."""
        # 1 agent, 1 cell away from exit at (0, 1). Starting at (0, 0).
        cfg = _make_config([
            {"agent_id": "a_del", "row": 0, "col": 0, "profile": {"speed": 1.0, "reaction_delay": 2}}
        ], rows=3, cols=3, exits=[(0, 1)])

        sim = Simulation(cfg, ShortestPathStrategy())
        agent = sim.agents[0]

        # Step 1: delayed
        sim.step()
        assert agent.position == (0, 0)
        assert sim.last_step_trace.agent_results[0].result == "REACTION_DELAY"

        # Step 2: delayed
        sim.step()
        assert agent.position == (0, 0)
        assert sim.last_step_trace.agent_results[0].result == "REACTION_DELAY"

        # Step 3: active, moves to (0, 1) and evacuates
        sim.step()
        assert agent.state == AgentState.EVACUATED
        assert agent.evacuation_timestep == 3

    def test_reaction_delay_is_not_congestion_waiting(self):
        """Test 11: Reaction delay does NOT increment total_waiting_steps."""
        cfg = _make_config([
            {"agent_id": "a_del", "row": 0, "col": 0, "profile": {"speed": 1.0, "reaction_delay": 5}}
        ], rows=3, cols=3, exits=[(0, 1)])

        sim = Simulation(cfg, ShortestPathStrategy())
        res = sim.run()

        assert res.evacuation_times["a_del"] == 6
        # Crucial semantic check: delay is NOT capacity waiting
        assert res.total_waiting_steps == 0
        assert res.per_agent_waiting_steps.get("a_del", 0) == 0


# ---------------------------------------------------------------------------
# Test 12 & 13: Mixed Population & Determinism
# ---------------------------------------------------------------------------


class TestMixedPopulationAndDeterminism:
    def test_mixed_population_execution(self):
        """Test 12: Multiple agents with varied speeds and delays operate concurrently."""
        # Setup: Two symmetrical independent corridors separated by a wall down column 2.
        # Lane A: col 0, exit at (4, 0). Agent fast_lane at (0, 0): speed=1.0, delay=0.
        # Lane B: col 4, exit at (4, 4). Agent slow_lane at (0, 4): speed=0.5, delay=2.
        cfg = _make_config(
            [
                {"agent_id": "fast_lane", "row": 0, "col": 0, "profile": {"speed": 1.0, "reaction_delay": 0}},
                {"agent_id": "slow_lane", "row": 0, "col": 4, "profile": {"speed": 0.5, "reaction_delay": 2}},
            ],
            rows=5,
            cols=5,
            exits=[(4, 0), (4, 4)],
            walls=[(r, 2) for r in range(5)],
        )

        sim = Simulation(cfg, ShortestPathStrategy())
        res = sim.run()

        assert res.evacuated_count == 2
        # Identical distance (4 hops): fast non-delayed agent must evacuate before slow delayed agent
        assert res.evacuation_times["fast_lane"] == 4
        assert res.evacuation_times["slow_lane"] == 10
        assert res.evacuation_times["fast_lane"] < res.evacuation_times["slow_lane"]

    def test_determinism_reproducibility(self):
        """Test 13: Identical heterogeneous scenario + same seed yields identical results."""
        cfg_a = _make_config([
            {"agent_id": "a1", "row": 0, "col": 0, "profile": {"speed": 0.5, "reaction_delay": 1}},
            {"agent_id": "a2", "row": 1, "col": 0, "profile": {"speed": 1.0, "reaction_delay": 2}},
        ], rows=4, cols=4, exits=[(3, 3)], seed=123)

        cfg_b = _make_config([
            {"agent_id": "a1", "row": 0, "col": 0, "profile": {"speed": 0.5, "reaction_delay": 1}},
            {"agent_id": "a2", "row": 1, "col": 0, "profile": {"speed": 1.0, "reaction_delay": 2}},
        ], rows=4, cols=4, exits=[(3, 3)], seed=123)

        res_a = Simulation(cfg_a, ShortestPathStrategy()).run()
        res_b = Simulation(cfg_b, ShortestPathStrategy()).run()

        assert res_a.total_timesteps == res_b.total_timesteps
        assert res_a.evacuation_times == res_b.evacuation_times
        assert res_a.total_waiting_steps == res_b.total_waiting_steps


# ---------------------------------------------------------------------------
# Test 14 & 15: Experiment Integration & Serialization
# ---------------------------------------------------------------------------


class TestExperimentIntegrationAndSerialization:
    def test_experiment_sweep_with_profiles(self, tmp_path: Path):
        """Test 14: Stage 5 sweep framework varies profile parameters."""
        base_agents = [{"agent_id": "a0", "row": 0, "col": 0}]

        # Create 2 sweep points varying speed
        cfg_fast = _make_config(base_agents, rows=4, cols=4, exits=[(3, 3)])
        cfg_fast.agents[0].profile = AgentProfileConfig(speed=1.0)

        cfg_slow = _make_config(base_agents, rows=4, cols=4, exits=[(3, 3)])
        cfg_slow.agents[0].profile = AgentProfileConfig(speed=0.5)

        points = [
            SweepPoint("speed_1.0", cfg_fast, {"speed": 1.0}),
            SweepPoint("speed_0.5", cfg_slow, {"speed": 0.5}),
        ]
        seeds = [10, 20]

        # 2 sweep points x 2 seeds = 4 runs
        exp = run_sweep(points, seeds, experiment_id="test_profile_sweep")
        assert exp.run_count == 4

        fast_runs = [r for r in exp.runs if r.params.get("speed") == 1.0]
        slow_runs = [r for r in exp.runs if r.params.get("speed") == 0.5]

        # Fast runs evacuate sooner than slow runs
        assert fast_runs[0].result.total_timesteps < slow_runs[0].result.total_timesteps

        # Test CSV export contains profile parameters
        csv_file = tmp_path / "profile_sweep.csv"
        export_csv(exp.runs, csv_file)
        assert csv_file.exists()
        content = csv_file.read_text(encoding="utf-8")
        assert "speed" in content

        # Test JSON export
        json_file = tmp_path / "profile_sweep.json"
        export_json(exp, json_file)
        assert json_file.exists()
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert data["run_count"] == 4

    def test_serialization_and_json_roundtrip(self, tmp_path: Path):
        """Test 15: Profile attributes survive JSON serialization and scenario re-loading."""
        raw_json = {
            "scenario_name": "serialized_profile_scenario",
            "grid": {"rows": 5, "cols": 5},
            "exits": [[4, 4]],
            "walls": [],
            "agents": [
                {
                    "agent_id": "a0",
                    "row": 0,
                    "col": 0,
                    "profile": {
                        "speed": 0.75,
                        "reaction_delay": 3,
                        "age": 62,
                        "input_attributes": {"group": "elderly"},
                    },
                }
            ],
            "parameters": {"max_timesteps": 100},
        }
        json_path = tmp_path / "test_scenario.json"
        json_path.write_text(json.dumps(raw_json), encoding="utf-8")

        loaded_cfg = SimulationConfig.from_json(json_path)
        agent = loaded_cfg.agents[0]
        assert agent.profile is not None
        assert agent.profile.speed == 0.75
        assert agent.profile.reaction_delay == 3
        assert agent.profile.age == 62

        sim_agent = Agent.from_config(agent)
        assert sim_agent.speed == 0.75
        assert sim_agent.reaction_delay == 3
        assert sim_agent.profile.input_attributes.get("age") == 62
        assert sim_agent.profile.input_attributes.get("group") == "elderly"

    def test_inspector_metadata_exposure(self):
        """Test 16: get_agent_info exposes speed and reaction_delay for inspector."""
        cfg = _make_config([
            {"agent_id": "inspector_agent", "row": 0, "col": 0, "profile": {"speed": 0.5, "reaction_delay": 2}}
        ], rows=3, cols=3, exits=[(2, 2)])

        sim = Simulation(cfg, ShortestPathStrategy())
        sim.step()

        info = sim.get_agent_info("inspector_agent")
        assert info is not None
        assert info["speed"] == 0.5
        assert info["reaction_delay"] == 2
        assert "profile" in info
        assert info["profile"]["speed"] == 0.5
        assert info["profile"]["reaction_delay"] == 2
