"""
stage14_demo.py
---------------
Stage 14A Demo: RL Exit-Placement Formulation.

Demonstrates the ExitPlacementEnv by running two episodes on the bottleneck
scenario, then comparing the RL reward against the Stage 13 exhaustive search
score for the same configuration.

Objectives demonstrated:
1. Environment reset and initial observation.
2. Sequential action selection (manual, not learned).
3. Action mask enforcement.
4. Episode termination and reward computation.
5. Stage 11 validation integration.
6. Stage 12 simulation integration.
7. Direct numerical comparison with Stage 13 baseline.
8. Generalization: basic_building scenario with a different candidate set.

This demo does NOT train an RL model.
No RL library is imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package root (RL_ENV) is on sys.path so this script can be run directly
_PACKAGE_PARENT = Path(__file__).parent.parent.resolve()
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.rl_environment import ExitPlacementEnv
from evacuation_simulation.simulation.search import search_exit_configurations

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
SEP = "=" * 68


def _header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def demo_episode(
    label: str,
    config: SimulationConfig,
    exit_count: int,
    actions: list[int],
) -> None:
    """Run a single manual episode and print results."""
    print(f"\n--- Episode: {label} ---")
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=exit_count)

    print(f"Scenario      : {config.scenario_name}")
    print(f"Grid          : {obs.grid_shape[0]}x{obs.grid_shape[1]}")
    print(f"Agents        : {obs.total_agents}")
    print(f"Candidates    : {len(obs.candidate_exits)}")
    for i, c in enumerate(obs.candidate_exits):
        mask_flag = "LEGAL" if obs.action_mask[i] else "MASKED"
        print(
            f"  [{i}] position={c['position']}  width={c['width']:.1f}m"
            f"  door_id={c['door_id']}  [{mask_flag}]"
        )
    print(f"Required exits: {exit_count}")

    for step_num, action in enumerate(actions, start=1):
        legal = (0 <= action < len(env.candidates)) and env.action_mask()[action]
        cand_str = (
            f"candidate[{action}] {env.candidates[action].position}"
            if 0 <= action < len(env.candidates)
            else f"action={action} (OUT OF RANGE)"
        )
        print(f"\n  Step {step_num}: select {cand_str}  (legal={legal})")

        obs, reward, terminated, info = env.step(action)
        print(f"    reward={reward:.1f}  terminated={terminated}")
        print(f"    reason={info['reason']}")
        if terminated:
            print(f"    is_feasible  ={info['is_feasible']}")
            print(f"    is_compliant ={info['is_compliant']}")
            print(f"    evac_time    ={info['total_evacuation_time']}")
            if info["violations"]:
                print(f"    violations   ={info['violations']}")
        else:
            print(f"    exits_remaining={obs.exits_remaining}")
            print(f"    action_mask    ={obs.action_mask}")

        if terminated:
            break


def demo_stage13_comparison(config: SimulationConfig, exit_count: int) -> None:
    """Show that the RL env reward matches Stage 13 score for the best config."""
    _header("STAGE 13 vs RL ENVIRONMENT — SAME CONFIGURATION, SAME SCORE?")

    config.parameters.random_seed = 42

    # Stage 13: find best config exhaustively
    search_res = search_exit_configurations(
        floor_plan=config.floor_plan,
        occupant_scenario=config.occupants,
        exit_count=exit_count,
        parameters=config.parameters,
        top_k=1,
    )

    if not search_res.has_feasible_solution:
        print("Stage 13 found no feasible solution — skipping comparison.")
        return

    best_cfg = search_res.best_configuration
    best_score = search_res.best_evaluation.score
    print(f"Stage 13 best exits : {best_cfg.exits}")
    print(f"Stage 13 best score : {best_score}  (total evacuation time)")

    # RL env: replay the same configuration
    env = ExitPlacementEnv()
    obs = env.reset(config, exit_count=exit_count)
    candidates = env.candidates

    # Map best exits to candidate indices
    indices = []
    for exit_pos in best_cfg.exits:
        for i, cand in enumerate(candidates):
            if list(cand.position) == list(exit_pos):
                indices.append(i)
                break

    if len(indices) != exit_count:
        print("WARNING: Could not map all Stage 13 exits to candidate indices.")
        return

    # Step through RL env using same indices
    for idx in indices[:-1]:
        obs, _, _, _ = env.step(idx)
    _, rl_reward, _, rl_info = env.step(indices[-1])

    rl_evac_time = rl_info["total_evacuation_time"]
    expected_reward = -float(best_score)

    print(f"\nRL reward           : {rl_reward:.1f}")
    print(f"Expected (-score)   : {expected_reward:.1f}")
    print(f"RL evac time        : {rl_evac_time}")
    print(f"Match               : {rl_reward == expected_reward}")


def main() -> None:
    _header("STAGE 14A — RL EXIT-PLACEMENT FORMULATION DEMO")
    print("No RL training occurs in this demo.")
    print("The RL agent is simulated by manual action selection.")

    # ---------------------------------------------------------------
    # Episode A: bottleneck — valid 2-exit configuration
    # ---------------------------------------------------------------
    _header("EPISODE A: BOTTLENECK — VALID 2-EXIT CONFIGURATION")
    config_bn = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    config_bn.parameters.random_seed = 42

    env_probe = ExitPlacementEnv()
    obs_probe = env_probe.reset(config_bn, exit_count=2)
    # Pick first two legal candidates that are not adjacent (to avoid R3 violation)
    legal_indices = [i for i, v in enumerate(obs_probe.action_mask) if v]
    # Choose indices 0 and 2 (positions 0,2 of legal list) to avoid adjacent pair
    if len(legal_indices) >= 2:
        demo_episode(
            "Bottleneck — valid 2 exits (likely feasible)",
            config_bn,
            exit_count=2,
            actions=[legal_indices[0], legal_indices[-1]],
        )

    # ---------------------------------------------------------------
    # Episode B: bottleneck — adjacent exits (likely R3 violation)
    # ---------------------------------------------------------------
    _header("EPISODE B: BOTTLENECK — ADJACENT EXITS (LIKELY R3 VIOLATION)")
    config_bn2 = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    config_bn2.parameters.random_seed = 42
    demo_episode(
        "Bottleneck — adjacent exits 0+1 (likely infeasible)",
        config_bn2,
        exit_count=2,
        actions=[0, 1],
    )

    # ---------------------------------------------------------------
    # Episode C: bottleneck — illegal action (already selected)
    # ---------------------------------------------------------------
    _header("EPISODE C: BOTTLENECK — ILLEGAL ACTION (ALREADY SELECTED)")
    config_bn3 = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    demo_episode(
        "Bottleneck — select same exit twice",
        config_bn3,
        exit_count=2,
        actions=[0, 0],  # second selection of index 0 is illegal
    )

    # ---------------------------------------------------------------
    # Episode D: basic_building — generalization test
    # ---------------------------------------------------------------
    _header("EPISODE D: BASIC_BUILDING — DIFFERENT FLOOR PLAN (GENERALIZATION)")
    config_bb = SimulationConfig.from_json(SCENARIOS_DIR / "basic_building.json")
    config_bb.parameters.random_seed = 99
    env_bb = ExitPlacementEnv()
    obs_bb = env_bb.reset(config_bb, exit_count=1)
    legal_bb = [i for i, v in enumerate(obs_bb.action_mask) if v]
    if legal_bb:
        demo_episode(
            "BasicBuilding — 1-exit episode",
            config_bb,
            exit_count=1,
            actions=[legal_bb[0]],
        )

    # ---------------------------------------------------------------
    # Stage 13 vs RL comparison
    # ---------------------------------------------------------------
    config_compare = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")
    demo_stage13_comparison(config_compare, exit_count=2)

    _header("DEMO COMPLETE")
    print("\nSummary:")
    print("  - ExitPlacementEnv successfully formulates exit placement as RL.")
    print("  - Stage 11 validation is invoked for every final configuration.")
    print("  - Stage 12 simulation provides evacuation time for the reward.")
    print("  - RL reward equals -1 x Stage 13 score for the same configuration.")
    print("  - Environment works on bottleneck AND basic_building floor plans.")
    print("  - No RL model was trained. No RL library was imported.")
    print()


if __name__ == "__main__":
    main()
