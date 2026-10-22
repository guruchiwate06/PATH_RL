"""
train_rl.py
-----------
Stage 18: Masked Actor-Critic RL Training for Exit Placement.

Trains an RL agent to select compliant, evacuation-minimising exit door configurations
on procedurally generated commercial building layouts using ExitPlacementEnv.

Usage:
    python evacuation_simulation/train_rl.py
    python evacuation_simulation/train_rl.py --episodes 200 --lr 0.001
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# Ensure import paths
_PACKAGE_PARENT = Path(__file__).parent.parent.resolve()
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))
_SIM_DIR = Path(__file__).parent.resolve()
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

from evacuation_simulation.simulation.config import SimulationConfig, SimulationParameters
from evacuation_simulation.simulation.generators.commercial import (
    CommercialGeneratorConfig,
    CommercialLayoutGenerator,
)
from evacuation_simulation.simulation.occupancy_seeder import (
    OccupantSeeder,
    OccupantSeederConfig,
)
from evacuation_simulation.simulation.rl_environment import ExitPlacementEnv
from evacuation_simulation.simulation.rl_policy import (
    MaskedActorCritic,
    extract_features,
)

# ---------------------------------------------------------------------------
# Default Configuration & Seeds
# ---------------------------------------------------------------------------
_RL_OUTPUT_DIR = Path(__file__).parent / "experiment_output" / "rl"

# Diverse training seeds (layout & occupant variations)
DEFAULT_TRAIN_SEEDS = [10, 19, 20, 24, 32, 36, 37, 38, 41, 42]
# Strictly held-out evaluation seeds for Stage 18 benchmark
DEFAULT_EVAL_SEEDS = [2, 5, 43, 49]


# ---------------------------------------------------------------------------
# Scenario Generator Helper
# ---------------------------------------------------------------------------


def make_training_scenario(
    seed: int,
    total_occupants: int = 20,
    grid_rows: int = 20,
    grid_cols: int = 30,
    num_candidate_exits: int = 4,
    max_timesteps: int = 500,
) -> SimulationConfig:
    """Generate a reproducible synthetic building layout and occupant scenario."""
    gen_config = CommercialGeneratorConfig(
        seed=seed,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        num_exterior_exits=num_candidate_exits,
    )
    generator = CommercialLayoutGenerator()
    layout = generator.generate(gen_config)

    seeder_config = OccupantSeederConfig(
        total_occupants=total_occupants,
        seed=seed,
        strategy="random_walkable",
        max_occupants_per_cell=1,
    )
    occupants = OccupantSeeder.seed(
        layout.floor_plan, layout.exit_configuration, seeder_config
    )

    return SimulationConfig(
        scenario_name=f"synthetic_commercial_seed_{seed}",
        floor_plan=layout.floor_plan,
        exit_configuration=layout.exit_configuration,
        occupants=occupants,
        parameters=SimulationParameters(max_timesteps=max_timesteps, random_seed=seed),
    )


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------


def train_rl_agent(
    episodes: int = 200,
    train_seeds: Optional[list[int]] = None,
    total_occupants: int = 20,
    grid_rows: int = 20,
    grid_cols: int = 30,
    num_candidate_exits: int = 4,
    exit_count: int = 2,
    lr: float = 0.001,
    gamma: float = 0.99,
    entropy_coef: float = 0.02,
    batch_size: int = 8,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Train a Masked Actor-Critic RL policy on synthetic building scenarios.

    Parameters
    ----------
    episodes:
        Total training episodes.
    train_seeds:
        List of synthetic layout seeds used during training.
    total_occupants:
        Number of building occupants per scenario.
    grid_rows:
        Building grid rows.
    grid_cols:
        Building grid cols.
    num_candidate_exits:
        Candidate door count (default 4).
    exit_count:
        Number of exits to select per episode (default 2).
    lr:
        Adam learning rate.
    gamma:
        Discount factor.
    entropy_coef:
        Exploration entropy bonus coefficient.
    batch_size:
        Number of episodes per gradient update.
    output_dir:
        Output folder for model weights and training logs.
    verbose:
        Whether to print periodic progress to stdout.

    Returns
    -------
    dict[str, Any]
        Training summary and history metrics.
    """
    start_time = time.perf_counter()
    out_dir = output_dir or _RL_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = train_seeds or DEFAULT_TRAIN_SEEDS

    # Pre-cache training scenarios for speed
    if verbose:
        print("=" * 72)
        print("  PATH_RL — Stage 18: Masked Actor-Critic RL Training Run")
        print("=" * 72)
        print(f"  Training Episodes   : {episodes}")
        print(f"  Training Seeds ({len(seeds)}): {seeds}")
        print(f"  Grid Size           : {grid_rows}x{grid_cols}")
        print(f"  Occupants           : {total_occupants}")
        print(f"  Candidate Exits (C) : {num_candidate_exits}")
        print(f"  Exits to Select (k) : {exit_count}")
        print(f"  Learning Rate (lr)  : {lr}")
        print("-" * 72)
        print("  Pre-generating training building scenarios ...")

    scenarios = [
        make_training_scenario(
            seed=s,
            total_occupants=total_occupants,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            num_candidate_exits=num_candidate_exits,
        )
        for s in seeds
    ]

    env = ExitPlacementEnv()
    model = MaskedActorCritic(max_candidates=num_candidate_exits)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: list[dict[str, Any]] = []
    best_feasible_reward = -float("inf")
    best_model_path = out_dir / "model_stage18.pt"

    # Batch collection buffers
    batch_states: list[torch.Tensor] = []
    batch_masks: list[torch.Tensor] = []
    batch_actions: list[int] = []
    batch_returns: list[float] = []

    for ep in range(1, episodes + 1):
        # Sample training scenario
        sc_idx = (ep - 1) % len(scenarios)
        scenario = scenarios[sc_idx]

        obs = env.reset(scenario, exit_count=exit_count)
        ep_states: list[torch.Tensor] = []
        ep_masks: list[torch.Tensor] = []
        ep_actions: list[int] = []
        ep_rewards: list[float] = []

        terminated = False
        info: dict[str, Any] = {}

        while not terminated:
            state, mask = extract_features(obs, max_candidates=num_candidate_exits)
            action, log_prob, val = model.get_action(obs, deterministic=False)

            ep_states.append(state)
            ep_masks.append(mask)
            ep_actions.append(action)

            obs, reward, terminated, info = env.step(action)
            ep_rewards.append(reward)

        # Compute discounted returns
        R = 0.0
        discounted_returns: list[float] = []
        for r in reversed(ep_rewards):
            R = r + gamma * R
            discounted_returns.insert(0, R)

        batch_states.extend(ep_states)
        batch_masks.extend(ep_masks)
        batch_actions.extend(ep_actions)
        batch_returns.extend(discounted_returns)

        # Perform gradient update every batch_size episodes
        if ep % batch_size == 0 or ep == episodes:
            states_tensor = torch.cat(batch_states, dim=0)
            masks_tensor = torch.cat(batch_masks, dim=0)
            actions_tensor = torch.tensor(batch_actions, dtype=torch.long)
            returns_tensor = torch.tensor(batch_returns, dtype=torch.float32)

            log_probs, entropy, values = model.evaluate_actions(
                states_tensor, masks_tensor, actions_tensor
            )

            # Advantage estimation and normalization
            advantages = returns_tensor - values.detach()
            if len(advantages) > 1:
                adv_std = advantages.std() + 1e-8
                advantages = (advantages - advantages.mean()) / adv_std

            # Actor & Critic losses
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = F.mse_loss(values, returns_tensor)
            entropy_loss = -entropy.mean()

            total_loss = actor_loss + 0.5 * critic_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Clear buffers
            batch_states.clear()
            batch_masks.clear()
            batch_actions.clear()
            batch_returns.clear()

        # Record episode metrics
        final_reward = ep_rewards[-1]
        is_feasible = bool(info.get("is_feasible", False))
        evac_time = info.get("total_evacuation_time")

        history.append({
            "episode": ep,
            "seed": scenario.parameters.random_seed,
            "reward": final_reward,
            "is_feasible": is_feasible,
            "is_compliant": bool(info.get("is_compliant", False)),
            "total_evacuation_time": evac_time,
            "selected_exits": list(env.selected_indices),
        })

        if is_feasible and final_reward > best_feasible_reward:
            best_feasible_reward = final_reward
            model.save(
                best_model_path,
                metadata={
                    "stage": 18,
                    "algorithm": "MaskedActorCritic",
                    "episodes": episodes,
                    "best_feasible_reward": best_feasible_reward,
                    "best_episode": ep,
                    "train_seeds": seeds,
                },
            )

        if verbose and (ep % 25 == 0 or ep == episodes or ep == 1):
            recent = history[-25:]
            recent_feas = [h["is_feasible"] for h in recent]
            feas_rate = sum(recent_feas) / len(recent_feas)
            feas_times = [h["total_evacuation_time"] for h in recent if h["total_evacuation_time"]]
            mean_time = (sum(feas_times) / len(feas_times)) if feas_times else None
            time_str = f"{mean_time:.1f} steps" if mean_time else "N/A"
            print(f"  Episode {ep:3d}/{episodes:3d} | "
                  f"Last Reward: {final_reward:7.1f} | "
                  f"Feasibility (last 25): {feas_rate:5.1%} | "
                  f"Mean Evac Time: {time_str}")

    runtime = time.perf_counter() - start_time

    # Save final model if not saved yet
    if not best_model_path.exists():
        model.save(
            best_model_path,
            metadata={
                "stage": 18,
                "algorithm": "MaskedActorCritic",
                "episodes": episodes,
                "train_seeds": seeds,
            },
        )

    # Save training history
    history_path = out_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_metadata": {
                    "stage": 18,
                    "algorithm": "MaskedActorCritic",
                    "episodes": episodes,
                    "runtime_seconds": round(runtime, 3),
                    "train_seeds": seeds,
                    "learning_rate": lr,
                    "model_path": str(best_model_path),
                },
                "episodes": history,
            },
            f,
            indent=2,
        )

    if verbose:
        total_feas = sum(1 for h in history if h["is_feasible"])
        print("-" * 72)
        print(f"  Training Complete in {runtime:.2f}s ({episodes / runtime:.1f} eps/sec)")
        print(f"  Total Feasible Episodes : {total_feas}/{episodes} ({total_feas / episodes:.1%})")
        print(f"  Saved Model Checkpoint  : {best_model_path}")
        print(f"  Saved Training Log      : {history_path}")
        print("=" * 72)

    return {
        "model_path": str(best_model_path),
        "history_path": str(history_path),
        "episodes": episodes,
        "runtime_seconds": runtime,
        "history": history,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PATH_RL Stage 18 — Masked Actor-Critic RL Training"
    )
    parser.add_argument("--episodes", type=int, default=200, help="Total training episodes")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated training seeds (default: 10 diverse seeds)",
    )
    parser.add_argument("--occupants", type=int, default=20, help="Occupants per building")
    parser.add_argument("--rows", type=int, default=20, help="Grid rows")
    parser.add_argument("--cols", type=int, default=30, help="Grid cols")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Quiet output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else None
    train_seeds = (
        [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else None
    )

    train_rl_agent(
        episodes=args.episodes,
        train_seeds=train_seeds,
        total_occupants=args.occupants,
        grid_rows=args.rows,
        grid_cols=args.cols,
        lr=args.lr,
        output_dir=out_dir,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
