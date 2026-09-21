"""
test_validation.py
------------------
Stage 7: Internal Validation & Controlled Experiment Framework.

This module systematically tests:
1. Model invariants (A-J): position validity, capacity bounds, occupancy consistency,
   evacuation consistency, movement validity, determinism, reproducibility, monotonic
   evacuation count, valid timestamps, non-negative waiting.
2. Controlled homogeneous baselines (Scenarios A-E).
3. Heterogeneity effects (individual vs. system effects).
4. One-factor-at-a-time (OFAT) parameter sensitivity.
5. Capacity sensitivity in bottleneck environments.
6. Reaction delay vs. congestion waiting decoupling.
7. Speed discrete movement semantics.
8. Experiment framework (sweeps, aggregation, CSV, JSON) with heterogeneous profiles.
9. Analytical sanity checks (single-agent, unreachable exits, wall routing, empty grid, exit starts).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from evacuation_simulation.simulation.agent import Agent, AgentState
from evacuation_simulation.simulation.config import (
    AgentConfig,
    AgentProfileConfig,
    GridConfig,
    SimulationConfig,
    SimulationParameters,
)
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.experiment import (
    RunRecord,
    SweepPoint,
    aggregate_results,
    export_csv,
    export_json,
    run_repeated,
    run_scenario,
    run_sweep,
)
from evacuation_simulation.simulation.metrics import MetricsCollector, SimulationResult
from evacuation_simulation.simulation.movement import MovementRequest, RejectionReason
from evacuation_simulation.simulation.profile import AgentProfile
from evacuation_simulation.simulation.simulation import Simulation
from evacuation_simulation.simulation.strategy import ShortestPathStrategy


# ===========================================================================
# Step 1: Model Invariants (A - J)
# ===========================================================================


class TestSimulationInvariants:
    """Rigorous tests for core simulation invariants across step-by-step execution."""

    @pytest.fixture
    def bottleneck_config(self) -> SimulationConfig:
        scenario_path = Path(__file__).parent.parent / "scenarios" / "bottleneck.json"
        with scenario_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return SimulationConfig(**data)

    def test_invariants_across_bottleneck_simulation(self, bottleneck_config: SimulationConfig) -> None:
        """
        Run the bottleneck simulation step-by-step and assert invariants at every single timestep:
        - Invariant A: Position validity (within bounds, not on a wall)
        - Invariant B: Capacity invariant (cell occupancy <= capacity)
        - Invariant C: Occupancy consistency (occupancy map matches non-evacuated agent positions)
        - Invariant D: Evacuated-agent consistency (valid timestamp, zero cell occupancy)
        - Invariant E: Movement validity (adjacent, passable, valid source)
        - Invariant H: Monotonic evacuation count (evacuated(t+1) >= evacuated(t))
        - Invariant I: Evacuation time validity (timestep >= 1)
        - Invariant J: Non-negative waiting steps
        """
        sim = Simulation(bottleneck_config, ShortestPathStrategy())
        env = sim.environment
        capacity = sim.occupancy.default_capacity

        prev_evacuated_count = 0

        while sim.step():
            t = sim.state.timestep
            active_agents = [a for a in sim.agents if a.is_active]
            evacuated_agents = [a for a in sim.agents if a.state == AgentState.EVACUATED]

            # Invariant A: Position validity
            for a in active_agents:
                assert env.is_within_bounds(a.row, a.col), f"Agent {a.agent_id} out of bounds at t={t}: ({a.row}, {a.col})"
                assert not env.is_wall(a.row, a.col), f"Agent {a.agent_id} on wall at t={t}: ({a.row}, {a.col})"

            # Invariant B: Capacity invariant
            occ_cells = sim.occupancy.get_occupied_cells()
            for cell, count in occ_cells.items():
                assert count <= capacity, f"Capacity violated at cell {cell} on t={t}: count={count} > cap={capacity}"

            # Invariant C: Occupancy consistency
            expected_counts: dict[tuple[int, int], int] = {}
            for a in active_agents:
                pos = a.position
                expected_counts[pos] = expected_counts.get(pos, 0) + 1
            assert occ_cells == expected_counts, f"Occupancy map mismatch at t={t}: {occ_cells} != {expected_counts}"

            # Invariant D: Evacuated-agent consistency
            for a in evacuated_agents:
                assert a.evacuation_timestep is not None and a.evacuation_timestep >= 1
                assert a.position not in occ_cells or occ_cells[a.position] == expected_counts.get(a.position, 0)
                # Evacuated agent must not be counted in occupancy
                assert a not in active_agents

            # Invariant E: Movement validity for approved steps in trace
            trace = sim.last_step_trace
            assert trace is not None
            for step_res in trace.agent_results:
                if step_res.result == "APPROVED":
                    r_before, c_before = step_res.position_before
                    r_after, c_after = step_res.position_after
                    manhattan_dist = abs(r_after - r_before) + abs(c_after - c_before)
                    assert manhattan_dist == 1, f"Invalid jump for {step_res.agent_id} at t={t}: dist={manhattan_dist}"
                    assert not env.is_wall(r_after, c_after), f"Moved into wall: {step_res.position_after}"

            # Invariant H: Monotonic evacuation count
            curr_evacuated_count = len(evacuated_agents)
            assert curr_evacuated_count >= prev_evacuated_count, (
                f"Evac count decreased at t={t}: {curr_evacuated_count} < {prev_evacuated_count}"
            )
            prev_evacuated_count = curr_evacuated_count

        result = sim.metrics.compute_result(total_timesteps=sim.state.timestep)
        # Invariant I: Evacuation time validity
        for aid, et in result.evacuation_times.items():
            assert et >= 1, f"Evacuation time < 1 for agent {aid}: {et}"

        # Invariant J: Waiting validity
        assert result.total_waiting_steps >= 0
        assert result.mean_waiting_steps >= 0.0
        assert result.max_waiting_steps >= 0
        for aid, ws in result.per_agent_waiting_steps.items():
            assert ws >= 0, f"Negative waiting steps for agent {aid}: {ws}"

    def test_determinism_invariant(self, bottleneck_config: SimulationConfig) -> None:
        """Invariant F: Same scenario, config, profiles, and seed produce identical results."""
        sim1 = Simulation(bottleneck_config, ShortestPathStrategy())
        res1 = sim1.run()

        sim2 = Simulation(bottleneck_config, ShortestPathStrategy())
        res2 = sim2.run()

        assert res1.total_timesteps == res2.total_timesteps
        assert res1.evacuated_count == res2.evacuated_count
        assert res1.total_waiting_steps == res2.total_waiting_steps
        assert res1.evacuation_times == res2.evacuation_times
        assert res1.per_agent_waiting_steps == res2.per_agent_waiting_steps

    def test_reproducibility_invariant(self, bottleneck_config: SimulationConfig) -> None:
        """Invariant G: Repeated experiment runs with identical seed produce identical aggregates."""
        records1 = run_repeated(bottleneck_config, seeds=[42, 42, 42])
        records2 = run_repeated(bottleneck_config, seeds=[42, 42, 42])

        agg1 = aggregate_results(records1)
        agg2 = aggregate_results(records2)

        assert agg1.total_timesteps_mean == agg2.total_timesteps_mean
        assert agg1.total_waiting_steps_mean == agg2.total_waiting_steps_mean
        assert agg1.mean_evacuation_time_mean == agg2.mean_evacuation_time_mean


# ===========================================================================
# Step 2: Controlled Homogeneous Baseline Experiments (Scenarios A - E)
# ===========================================================================


class TestControlledBaselines:
    """Benchmark scenarios verifying predictable cause/effect."""

    def test_scenario_a_open_corridor(self) -> None:
        """
        Scenario A: Open corridor.
        Corridor of width 1, length 10. Exit at (1, 9).
        Agent 0 at (1, 0) -> distance = 9 hops.
        Agent 1 at (1, 1) -> distance = 8 hops.
        Agent 1 arrives at t=8. Because simultaneous vacating is not credited
        in advance, Agent 0 waits at t=1 for cell (1, 1) to clear, then moves
        one step behind Agent 1, arriving at t=10.
        """
        scenario_path = Path(__file__).parent.parent / "scenarios" / "corridor.json"
        with scenario_path.open("r", encoding="utf-8") as f:
            cfg = SimulationConfig(**json.load(f))

        sim = Simulation(cfg, ShortestPathStrategy())
        res = sim.run()

        assert res.evacuated_count == 2
        assert res.evacuation_times["agent_1"] == 8
        assert res.evacuation_times["agent_0"] == 10
        assert res.per_agent_waiting_steps["agent_0"] == 1
        assert res.per_agent_waiting_steps.get("agent_1", 0) == 0

    def test_scenario_b_two_exits(self) -> None:
        """
        Scenario B: Two exits.
        Exits at (2, 0) [West] and (2, 8) [East].
        Agents choose nearest exit based on shortest path.
        """
        scenario_path = Path(__file__).parent.parent / "scenarios" / "two_exits.json"
        with scenario_path.open("r", encoding="utf-8") as f:
            cfg = SimulationConfig(**json.load(f))

        sim = Simulation(cfg, ShortestPathStrategy())
        res = sim.run()

        assert res.evacuated_count == 4
        # agent_west started at (2, 2), dist to (2, 0) is 2
        assert res.evacuation_times["agent_west"] == 2
        # agent_east started at (2, 6), dist to (2, 8) is 2
        assert res.evacuation_times["agent_east"] == 2
        # agent_north_west at (1, 3): dist to (2, 0) is 1+3=4
        assert res.evacuation_times["agent_north_west"] == 4
        # agent_south_east at (3, 5): dist to (2, 8) is 1+3=4
        assert res.evacuation_times["agent_south_east"] == 4

    def test_scenario_c_bottleneck(self) -> None:
        """
        Scenario C: Bottleneck.
        12 agents must pass through a single-width gap at (4, 2) to exit at (9, 2).
        Verifies capacity congestion and waiting emergence.
        """
        scenario_path = Path(__file__).parent.parent / "scenarios" / "bottleneck.json"
        with scenario_path.open("r", encoding="utf-8") as f:
            cfg = SimulationConfig(**json.load(f))

        sim = Simulation(cfg, ShortestPathStrategy())
        res = sim.run()

        assert res.evacuated_count == 12
        assert res.total_waiting_steps > 0
        assert res.max_waiting_steps > 0
        assert res.congested_cell_steps > 0

    def test_scenario_d_mixed_speed_population(self) -> None:
        """
        Scenario D: Mixed-speed population.
        Same environment & positions.
        Exp 1: All speed = 1.0
        Exp 2: Agent 0 slowed down to speed = 0.5
        Exp 3: Agent 0 further slowed down to speed = 0.25
        """
        def make_config(sp_agent0: float) -> SimulationConfig:
            return SimulationConfig(
                scenario_name=f"mixed_speed_{sp_agent0}",
                grid={"rows": 3, "cols": 10},
                walls=[[0, c] for c in range(10)] + [[2, c] for c in range(10)],
                exits=[[1, 9]],
                agents=[
                    AgentConfig(
                        agent_id="agent_0",
                        row=1,
                        col=0,
                        profile=AgentProfileConfig(speed=sp_agent0),
                    ),
                    AgentConfig(
                        agent_id="agent_1",
                        row=1,
                        col=1,
                        profile=AgentProfileConfig(speed=1.0),
                    ),
                ],
                parameters={"max_timesteps": 100, "random_seed": 42},
            )

        # Exp 1: all 1.0 -> agent 1 evacuates at 8, agent 0 at 10
        res1 = Simulation(make_config(1.0), ShortestPathStrategy()).run()
        assert res1.evacuation_times["agent_0"] == 10
        assert res1.evacuation_times["agent_1"] == 8

        # Exp 2: agent 0 speed 0.5 -> moves every 2 steps.
        # Agent 1 is far ahead, so agent 0 has 0 waiting steps and takes 9*2 = 18 steps.
        res2 = Simulation(make_config(0.5), ShortestPathStrategy()).run()
        assert res2.evacuation_times["agent_0"] == 18

        # Exp 3: agent 0 speed 0.25 -> moves every 4 steps. 9*4 = 36 steps.
        res3 = Simulation(make_config(0.25), ShortestPathStrategy()).run()
        assert res3.evacuation_times["agent_0"] == 36

    def test_scenario_e_reaction_delay_isolation(self) -> None:
        """
        Scenario E: Reaction delay.
        Isolate reaction delay from congestion waiting in a single-agent scenario.
        """
        def run_with_delay(delay: int) -> SimulationResult:
            cfg = SimulationConfig(
                scenario_name=f"delay_{delay}",
                grid={"rows": 3, "cols": 10},
                walls=[],
                exits=[[1, 9]],
                agents=[
                    AgentConfig(
                        agent_id="a0",
                        row=1,
                        col=0,
                        profile=AgentProfileConfig(speed=1.0, reaction_delay=delay),
                    )
                ],
                parameters={"max_timesteps": 50, "random_seed": 42},
            )
            return Simulation(cfg, ShortestPathStrategy()).run()

        res_no_delay = run_with_delay(0)
        res_with_delay = run_with_delay(5)

        # Distance is 9 hops.
        # No delay -> evacuates at t=9
        assert res_no_delay.evacuation_times["a0"] == 9
        assert res_no_delay.total_waiting_steps == 0

        # Delay 5 -> evacuates at t = 5 + 9 = 14
        assert res_with_delay.evacuation_times["a0"] == 14
        # Crucial: Reaction delay is NOT congestion waiting!
        assert res_with_delay.total_waiting_steps == 0


# ===========================================================================
# Step 3: Heterogeneity Effect Experiments (Individual vs. System Effects)
# ===========================================================================


class TestHeterogeneityEffects:
    """Evaluate individual vs. system-level metrics when altering per-agent profiles."""

    def test_lead_agent_delay_in_single_file_queue(self) -> None:
        """
        In a single-file corridor, delaying the lead agent affects the entire queue
        (system effect: total completion time increases by the delay).
        """
        # Corridor with agent 1 ahead of agent 0
        def make_queue_cfg(lead_delay: int) -> SimulationConfig:
            return SimulationConfig(
                scenario_name="queue",
                grid={"rows": 3, "cols": 10},
                walls=[[0, c] for c in range(10)] + [[2, c] for c in range(10)],
                exits=[[1, 9]],
                agents=[
                    AgentConfig(
                        agent_id="a_trailing",
                        row=1,
                        col=0,
                        profile=AgentProfileConfig(speed=1.0, reaction_delay=0),
                    ),
                    AgentConfig(
                        agent_id="a_lead",
                        row=1,
                        col=1,
                        profile=AgentProfileConfig(speed=1.0, reaction_delay=lead_delay),
                    ),
                ],
                parameters={"max_timesteps": 50, "random_seed": 42, "default_cell_capacity": 1},
            )

        res_base = Simulation(make_queue_cfg(0), ShortestPathStrategy()).run()
        res_delayed = Simulation(make_queue_cfg(3), ShortestPathStrategy()).run()

        # In baseline: lead evacuates at 8. Trailing waits 1 step at t=1 for lead to clear, evacuates at 10.
        assert res_base.evacuation_times["a_lead"] == 8
        assert res_base.evacuation_times["a_trailing"] == 10
        assert res_base.total_timesteps == 10

        # When lead is delayed by 3:
        # a_lead starts at t=4, evacuates at 3 + 8 = 11.
        assert res_delayed.evacuation_times["a_lead"] == 11
        # a_trailing was blocked by a_lead behind it: trailing evacuates at 10 + 3 = 13!
        assert res_delayed.evacuation_times["a_trailing"] == 13
        # System completion increased by 3 (from 10 to 13)
        assert res_delayed.total_timesteps == 13
        # a_trailing experienced capacity waiting!
        assert res_delayed.per_agent_waiting_steps["a_trailing"] > 0
        assert res_delayed.per_agent_waiting_steps.get("a_lead", 0) == 0

    def test_trailing_agent_delay_does_not_affect_lead(self) -> None:
        """
        Delaying the trailing agent affects ONLY the trailing agent (individual effect),
        leaving the lead agent completely unaffected.
        """
        cfg = SimulationConfig(
            scenario_name="trailing_delay",
            grid={"rows": 3, "cols": 10},
            walls=[[0, c] for c in range(10)] + [[2, c] for c in range(10)],
            exits=[[1, 9]],
            agents=[
                AgentConfig(
                    agent_id="a_trailing",
                    row=1,
                    col=0,
                    profile=AgentProfileConfig(speed=1.0, reaction_delay=5),
                ),
                AgentConfig(
                    agent_id="a_lead",
                    row=1,
                    col=1,
                    profile=AgentProfileConfig(speed=1.0, reaction_delay=0),
                ),
            ],
            parameters={"max_timesteps": 50, "random_seed": 42, "default_cell_capacity": 1},
        )
        res = Simulation(cfg, ShortestPathStrategy()).run()

        # Lead agent unaffected
        assert res.evacuation_times["a_lead"] == 8
        # Trailing agent delayed
        assert res.evacuation_times["a_trailing"] == 5 + 9
        # Trailing agent experienced 0 capacity waiting because lead was already far ahead
        assert res.total_waiting_steps == 0


# ===========================================================================
# Step 4: One-Factor-At-A-Time (OFAT) Sensitivity Experiments
# ===========================================================================


class TestOFATSensitivity:
    """Vary exactly one parameter while holding all other factors constant."""

    @pytest.fixture
    def base_config(self) -> SimulationConfig:
        return SimulationConfig(
            scenario_name="ofat_base",
            grid={"rows": 5, "cols": 5},
            walls=[],
            exits=[[4, 4]],
            agents=[
                AgentConfig(agent_id="a1", row=0, col=0, profile=AgentProfileConfig(speed=1.0, reaction_delay=0)),
                AgentConfig(agent_id="a2", row=4, col=0, profile=AgentProfileConfig(speed=1.0, reaction_delay=0)),
            ],
            parameters={"max_timesteps": 100, "random_seed": 42, "default_cell_capacity": 1},
        )

    def test_ofat_speed_variation(self, base_config: SimulationConfig) -> None:
        """Vary only speed: 1.0 vs 0.5."""
        cfg1 = base_config.model_copy(deep=True)
        cfg2 = base_config.model_copy(deep=True)
        cfg2.agents[0].profile.speed = 0.5

        res1 = Simulation(cfg1, ShortestPathStrategy()).run()
        res2 = Simulation(cfg2, ShortestPathStrategy()).run()

        # Agent 1 takes longer with lower speed
        assert res2.evacuation_times["a1"] > res1.evacuation_times["a1"]
        # Agent 2 speed unchanged -> evacuation time identical
        assert res2.evacuation_times["a2"] == res1.evacuation_times["a2"]

    def test_ofat_reaction_delay_variation(self, base_config: SimulationConfig) -> None:
        """Vary only reaction delay: 0 vs 4."""
        cfg1 = base_config.model_copy(deep=True)
        cfg2 = base_config.model_copy(deep=True)
        cfg2.agents[0].profile.reaction_delay = 4

        res1 = Simulation(cfg1, ShortestPathStrategy()).run()
        res2 = Simulation(cfg2, ShortestPathStrategy()).run()

        assert res2.evacuation_times["a1"] == res1.evacuation_times["a1"] + 4
        assert res2.evacuation_times["a2"] == res1.evacuation_times["a2"]

    def test_ofat_capacity_variation(self) -> None:
        """Vary only capacity in a bottleneck: 1 vs 2."""
        scenario_path = Path(__file__).parent.parent / "scenarios" / "bottleneck.json"
        with scenario_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        cfg_cap1 = SimulationConfig(**data)
        data_cap2 = dict(data)
        data_cap2["parameters"]["default_cell_capacity"] = 2
        cfg_cap2 = SimulationConfig(**data_cap2)

        res_cap1 = Simulation(cfg_cap1, ShortestPathStrategy()).run()
        res_cap2 = Simulation(cfg_cap2, ShortestPathStrategy()).run()

        # Higher capacity should reduce total waiting steps and total timesteps
        assert res_cap2.total_waiting_steps < res_cap1.total_waiting_steps
        assert res_cap2.total_timesteps <= res_cap1.total_timesteps


# ===========================================================================
# Step 5: Capacity Sensitivity (Bottleneck C=1, 2, 3)
# ===========================================================================


class TestCapacitySensitivity:
    """Systematic evaluation of cell capacity scaling in bottleneck scenario."""

    def test_bottleneck_capacity_scaling_monotonic_waiting(self) -> None:
        """
        Comparing capacity = 1, 2, 3 on bottleneck.json.
        Verify:
        - total waiting steps strictly decreases from cap=1 to cap=2, and cap=2 to cap=3
        - maximum cell occupancy never exceeds configured capacity
        - total timesteps decreases or remains bounded
        """
        scenario_path = Path(__file__).parent.parent / "scenarios" / "bottleneck.json"
        with scenario_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        results: dict[int, SimulationResult] = {}
        for cap in (1, 2, 3):
            data = dict(raw)
            data["parameters"] = dict(raw["parameters"])
            data["parameters"]["default_cell_capacity"] = cap
            cfg = SimulationConfig(**data)
            sim = Simulation(cfg, ShortestPathStrategy())
            res = sim.run()
            results[cap] = res

            # Invariant: max occupancy <= capacity
            assert res.max_cell_occupancy <= cap

        # Waiting steps should decrease as capacity expands
        assert results[1].total_waiting_steps > results[2].total_waiting_steps
        assert results[2].total_waiting_steps > results[3].total_waiting_steps

        # Total completion time should not increase with larger capacity
        assert results[1].total_timesteps >= results[2].total_timesteps >= results[3].total_timesteps


# ===========================================================================
# Step 6: Reaction Delay Validation
# ===========================================================================


class TestReactionDelayValidation:
    """Verify that reaction_delay is conceptually separate from congestion waiting."""

    def test_reaction_delay_uncongested_zero_waiting(self) -> None:
        """Agent with reaction delay in empty corridor must have 0 waiting steps."""
        cfg = SimulationConfig(
            scenario_name="uncongested_delay",
            grid={"rows": 3, "cols": 10},
            walls=[],
            exits=[[1, 9]],
            agents=[
                AgentConfig(
                    agent_id="solitary",
                    row=1,
                    col=0,
                    profile=AgentProfileConfig(speed=1.0, reaction_delay=7),
                )
            ],
            parameters={"max_timesteps": 50, "random_seed": 42},
        )
        sim = Simulation(cfg, ShortestPathStrategy())
        res = sim.run()

        assert res.evacuation_times["solitary"] == 7 + 9
        assert res.total_waiting_steps == 0
        assert res.per_agent_waiting_steps.get("solitary", 0) == 0

    def test_reaction_delay_in_trace_distinguishable_from_capacity_rejection(self) -> None:
        """Inspect step trace to verify result strings are REACTION_DELAY, not REJECTED_CAPACITY."""
        cfg = SimulationConfig(
            scenario_name="trace_delay_check",
            grid={"rows": 3, "cols": 5},
            walls=[],
            exits=[[1, 4]],
            agents=[
                AgentConfig(
                    agent_id="delayed_agent",
                    row=1,
                    col=0,
                    profile=AgentProfileConfig(speed=1.0, reaction_delay=2),
                )
            ],
            parameters={"max_timesteps": 10, "random_seed": 42},
        )
        sim = Simulation(cfg, ShortestPathStrategy())

        # Step 1: reaction delayed
        sim.step()
        assert sim.last_step_trace is not None
        res_t1 = sim.last_step_trace.agent_results[0]
        assert res_t1.result == "REACTION_DELAY"
        assert res_t1.rejection_reason is None

        # Step 2: reaction delayed
        sim.step()
        assert sim.last_step_trace is not None
        res_t2 = sim.last_step_trace.agent_results[0]
        assert res_t2.result == "REACTION_DELAY"

        # Step 3: active movement approved
        sim.step()
        assert sim.last_step_trace is not None
        res_t3 = sim.last_step_trace.agent_results[0]
        assert res_t3.result == "APPROVED"


# ===========================================================================
# Step 7: Speed Semantics Validation
# ===========================================================================


class TestSpeedSemanticsValidation:
    """Verify Stage 6 discrete speed representation and bounds."""

    def test_speed_greater_than_1_is_rejected(self) -> None:
        """Speed > 1.0 is strictly prohibited in AgentProfile and AgentProfileConfig."""
        with pytest.raises(ValueError):
            AgentProfile(speed=1.5)

        with pytest.raises(ValueError):
            AgentProfileConfig(speed=2.0)

    def test_speed_non_positive_is_rejected(self) -> None:
        """Speed <= 0.0 is strictly prohibited."""
        with pytest.raises(ValueError):
            AgentProfile(speed=0.0)

        with pytest.raises(ValueError):
            AgentProfile(speed=-0.5)

    def test_no_multi_cell_jumps_under_any_speed(self) -> None:
        """Verify that an agent never moves more than 1 cell in any discrete step."""
        cfg = SimulationConfig(
            scenario_name="speed_step_limit",
            grid={"rows": 3, "cols": 10},
            walls=[],
            exits=[[1, 9]],
            agents=[
                AgentConfig(
                    agent_id="fast_agent",
                    row=1,
                    col=0,
                    profile=AgentProfileConfig(speed=1.0),
                )
            ],
            parameters={"max_timesteps": 20, "random_seed": 42},
        )
        sim = Simulation(cfg, ShortestPathStrategy())
        while sim.step():
            trace = sim.last_step_trace
            assert trace is not None
            for r in trace.agent_results:
                r1, c1 = r.position_before
                r2, c2 = r.position_after
                assert abs(r1 - r2) + abs(c1 - c2) <= 1


# ===========================================================================
# Step 8: Experiment Framework Validation
# ===========================================================================


class TestExperimentFrameworkValidation:
    """Verify run_repeated, run_sweep, aggregate_results, CSV, and JSON exports."""

    def test_experiment_with_heterogeneous_profiles(self, tmp_path: Path) -> None:
        """Test full experiment pipeline preserving per-agent profiles."""
        base_cfg = SimulationConfig(
            scenario_name="exp_hetero",
            grid={"rows": 3, "cols": 10},
            walls=[],
            exits=[[1, 9]],
            agents=[
                AgentConfig(
                    agent_id="a1",
                    row=1,
                    col=0,
                    profile=AgentProfileConfig(speed=1.0, reaction_delay=0),
                ),
                AgentConfig(
                    agent_id="a2",
                    row=1,
                    col=1,
                    profile=AgentProfileConfig(speed=0.5, reaction_delay=1),
                ),
            ],
            parameters={"max_timesteps": 50, "random_seed": 42},
        )

        # 1. run_repeated
        reps = run_repeated(base_cfg, seeds=[42, 43, 44])
        assert len(reps) == 3
        agg = aggregate_results(reps)
        assert agg.run_count == 3
        assert agg.evacuation_rate_mean == 1.0

        # 2. run_sweep
        pts = [
            SweepPoint(label="default", config=base_cfg, params={"delay": 0}),
            SweepPoint(
                label="more_delay",
                config=base_cfg.model_copy(
                    update={
                        "agents": [
                            base_cfg.agents[0],
                            base_cfg.agents[1].model_copy(
                                update={"profile": AgentProfileConfig(speed=0.5, reaction_delay=4)}
                            ),
                        ]
                    }
                ),
                params={"delay": 4},
            ),
        ]
        exp_res = run_sweep(pts, seeds=[42, 43])
        assert exp_res.run_count == 4

        # 3. export_csv
        csv_path = tmp_path / "test_out.csv"
        export_csv(exp_res.runs, csv_path)
        assert csv_path.exists()
        content = csv_path.read_text(encoding="utf-8")
        assert "delay" in content
        assert "mean_evacuation_time" in content

        # 4. export_json
        json_path = tmp_path / "test_out.json"
        export_json(exp_res, json_path)
        assert json_path.exists()
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["run_count"] == 4
        assert len(data["runs"]) == 4
        assert "aggregate" in data


# ===========================================================================
# Step 9: Analytical Sanity Checks
# ===========================================================================


class TestAnalyticalSanityChecks:
    """Exact analytical expectations for edge cases and simple topologies."""

    def test_single_agent_corridor_exact_timestep(self) -> None:
        """Single agent with N hops must evacuate at exactly timestep N."""
        hops = 7
        cfg = SimulationConfig(
            scenario_name="analytical_hops",
            grid={"rows": 2, "cols": 10},
            walls=[],
            exits=[[0, hops]],
            agents=[AgentConfig(agent_id="solitary", row=0, col=0)],
            parameters={"max_timesteps": 20, "random_seed": 42},
        )
        sim = Simulation(cfg, ShortestPathStrategy())
        res = sim.run()

        assert res.evacuated_count == 1
        assert res.evacuation_times["solitary"] == hops
        assert res.total_timesteps == hops

    def test_unreachable_exit_agent_stays_put(self) -> None:
        """Agent surrounded by walls cannot reach the exit; stays put, does not evacuate."""
        cfg = SimulationConfig(
            scenario_name="unreachable",
            grid={"rows": 5, "cols": 5},
            walls=[[0, 1], [1, 1], [2, 1], [2, 0]],  # seals (0, 0) and (1, 0)
            exits=[[4, 4]],
            agents=[AgentConfig(agent_id="trapped", row=0, col=0)],
            parameters={"max_timesteps": 10, "random_seed": 42},
        )
        sim = Simulation(cfg, ShortestPathStrategy())
        res = sim.run()

        assert res.evacuated_count == 0
        assert res.non_evacuated_count == 1
        assert "trapped" not in res.evacuation_times
        assert res.total_timesteps == 10  # ran to max_timesteps

    def test_wall_barrier_routing(self) -> None:
        """Agent must route around an intervening wall if an open path exists."""
        # Wall at row 1 from col 0 to 3. Open cell at (1, 4).
        cfg = SimulationConfig(
            scenario_name="wall_routing",
            grid={"rows": 3, "cols": 5},
            walls=[[1, 0], [1, 1], [1, 2], [1, 3]],
            exits=[[2, 0]],
            agents=[AgentConfig(agent_id="router", row=0, col=0)],
            parameters={"max_timesteps": 20, "random_seed": 42},
        )
        sim = Simulation(cfg, ShortestPathStrategy())
        res = sim.run()

        # Path: (0,0)->(0,1)->(0,2)->(0,3)->(0,4)->(1,4)->(2,4)->(2,3)->(2,2)->(2,1)->(2,0) = 10 steps
        assert res.evacuated_count == 1
        assert res.evacuation_times["router"] == 10

    def test_capacity_one_bottleneck_never_exceeded(self) -> None:
        """Multiple agents funneling into a cell of capacity 1 never violate capacity."""
        cfg = SimulationConfig(
            scenario_name="funnel",
            grid={"rows": 3, "cols": 3},
            walls=[],
            exits=[[2, 2]],
            agents=[
                AgentConfig(agent_id="a1", row=0, col=0),
                AgentConfig(agent_id="a2", row=0, col=1),
                AgentConfig(agent_id="a3", row=1, col=0),
            ],
            parameters={"max_timesteps": 20, "random_seed": 42, "default_cell_capacity": 1},
        )
        sim = Simulation(cfg, ShortestPathStrategy())
        while sim.step():
            assert sim.occupancy.get_capacity((1, 1)) == 1
            for _, count in sim.occupancy.get_occupied_cells().items():
                assert count <= 1

        assert sim.occupancy.default_capacity == 1
        res = sim.metrics.compute_result(total_timesteps=sim.state.timestep)
        assert res.evacuated_count == 3
        assert res.max_cell_occupancy == 1

    def test_empty_environment_valid_metrics(self) -> None:
        """An environment with 0 agents terminates cleanly and produces valid metrics."""
        cfg = SimulationConfig(
            scenario_name="empty",
            grid={"rows": 4, "cols": 4},
            walls=[],
            exits=[[3, 3]],
            agents=[],
            parameters={"max_timesteps": 10, "random_seed": 42},
        )
        sim = Simulation(cfg, ShortestPathStrategy())
        res = sim.run()

        assert res.total_agents == 0
        assert res.evacuated_count == 0
        assert res.non_evacuated_count == 0
        assert res.evacuation_rate == 0.0
        assert res.mean_evacuation_time is None
        assert res.min_evacuation_time is None
        assert res.max_evacuation_time is None
        assert res.total_waiting_steps == 0
        assert res.total_timesteps == 1  # 1 step executed to evaluate active agents and terminate cleanly

    def test_already_evacuated_agent_at_start(self) -> None:
        """An agent starting on an exit cell evacuates at timestep 1 without moving."""
        cfg = SimulationConfig(
            scenario_name="start_on_exit",
            grid={"rows": 3, "cols": 3},
            walls=[],
            exits=[[1, 1]],
            agents=[AgentConfig(agent_id="on_exit", row=1, col=1)],
            parameters={"max_timesteps": 10, "random_seed": 42},
        )
        sim = Simulation(cfg, ShortestPathStrategy())
        res = sim.run()

        assert res.evacuated_count == 1
        assert res.evacuation_times["on_exit"] == 1
        assert res.total_timesteps == 1
        assert res.total_waiting_steps == 0
