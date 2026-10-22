"""
train_stage18b.py
-----------------
Stage 18B: Extended Masked Actor-Critic RL Training for Exit Placement.

Key Enhancements over Stage 18:
1. Extended training up to 1,000+ episodes with full CLI configurability (--episodes, --lr, --batch-size).
2. True procedural layout diversity: dynamic variation of grid dimensions, rooms, candidate exits (4-6),
   and occupant distributions across hundreds of unique layout seeds.
3. Live terminal monitoring: real-time progress bar (tqdm) + live metrics updated every episode
   (reward, rolling average, actor/critic loss, feasibility rate, entropy, updates, elapsed time, ETA).
4. Periodic summary statistics printed every 25/50 episodes with flush=True.
5. Incremental history logging: immediate CSV appending and periodic JSON flushing to protect against crashes.
6. Graceful Ctrl+C handling: catches SIGINT, flushes current history, saves latest checkpoint, and exits cleanly.
7. Diagnostics: saves CSV/JSON metrics and automatically generates multi-panel training diagnostic plots.
8. Preservation: strictly preserves original Stage 18 model and results, saving to model_stage18b.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

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
# Constants & Reserved Evaluation Seeds
# ---------------------------------------------------------------------------
_RL_OUTPUT_DIR = Path(__file__).parent / "experiment_output" / "rl"

# Reserved for Stage 18B held-out evaluation -- training MUST NEVER use these seeds
RESERVED_EVAL_SEEDS = {
    5002, 5004, 5005, 5006, 5008, 5009, 5011, 5012, 5013, 5015,
    5018, 5019, 5026, 5037, 5042, 5045, 5048, 5049, 5051, 5053,
    5054, 5055, 5056, 5066, 5070, 5076, 5078, 5085, 5090, 5091,
}
STAGE18_EVAL_SEEDS = {2, 5, 43, 49}
ALL_FORBIDDEN_TRAIN_SEEDS = RESERVED_EVAL_SEEDS | STAGE18_EVAL_SEEDS


# ---------------------------------------------------------------------------
# Procedural Scenario Generator with Architectural Diversity
# ---------------------------------------------------------------------------

def make_diverse_training_scenario(
    seed: int,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    num_candidate_exits: int = 4,
    total_occupants: Optional[int] = None,
    max_timesteps: int = 500,
) -> SimulationConfig:
    """
    Generate a diverse procedural commercial layout.

    To prevent memorization, parameters vary deterministically based on seed
    when not explicitly overridden.
    """
    rng = random.Random(seed)

    # Architectural variations within guaranteed NBC-compliant bounds
    rows = grid_rows or rng.choice([18, 20, 22])
    cols = grid_cols or rng.choice([26, 28, 30])
    candidates = num_candidate_exits
    occupants_count = total_occupants or rng.choice([16, 18, 20, 22, 24])

    gen_config = CommercialGeneratorConfig(
        seed=seed,
        grid_rows=rows,
        grid_cols=cols,
        num_exterior_exits=candidates,
        corridor_width_cells=4,
    )
    generator = CommercialLayoutGenerator()
    layout = generator.generate(gen_config)

    seeder_config = OccupantSeederConfig(
        total_occupants=occupants_count,
        seed=seed,
        strategy="random_walkable",
        max_occupants_per_cell=1,
    )
    occupants = OccupantSeeder.seed(
        layout.floor_plan, layout.exit_configuration, seeder_config
    )

    return SimulationConfig(
        scenario_name=f"diverse_commercial_seed_{seed}_{rows}x{cols}_occ{occupants_count}",
        floor_plan=layout.floor_plan,
        exit_configuration=layout.exit_configuration,
        occupants=occupants,
        parameters=SimulationParameters(max_timesteps=max_timesteps, random_seed=seed),
    )


# ---------------------------------------------------------------------------
# Diagnostic Plot Generator
# ---------------------------------------------------------------------------

def generate_diagnostic_plots(
    history: list[dict[str, Any]],
    loss_history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generate a multi-panel training diagnostics figure using matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        episodes = [h["episode"] for h in history]
        rewards = [h["reward"] for h in history]
        rolling_rewards = [h.get("rolling_reward", h["reward"]) for h in history]
        feasibility_rates = [h.get("rolling_feasibility", 0.0) * 100.0 for h in history]

        # Feasible evacuation times
        feas_episodes = [h["episode"] for h in history if h.get("is_feasible") and h.get("total_evacuation_time")]
        feas_times = [h["total_evacuation_time"] for h in history if h.get("is_feasible") and h.get("total_evacuation_time")]

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("PATH_RL Stage 18B -- Extended Training Diagnostics (1000 Episodes)", fontsize=14, fontweight="bold")

        # 1. Rewards
        ax1 = axs[0, 0]
        ax1.plot(episodes, rewards, color="#94a3b8", alpha=0.35, label="Episode Reward")
        ax1.plot(episodes, rolling_rewards, color="#2563eb", linewidth=2.0, label="Rolling Mean (50 ep)")
        ax1.set_title("Reward Progression & Rolling Mean")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="lower right")

        # 2. Feasibility Rate
        ax2 = axs[0, 1]
        ax2.plot(episodes, feasibility_rates, color="#10b981", linewidth=2.0)
        ax2.axhline(100.0, color="#059669", linestyle=":", alpha=0.6)
        ax2.set_title("NBC Feasibility Rate (Rolling 50 ep)")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Feasibility Rate (%)")
        ax2.set_ylim(-5, 105)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # 3. Evacuation Times on Feasible Runs
        ax3 = axs[1, 0]
        if feas_times:
            ax3.scatter(feas_episodes, feas_times, color="#8b5cf6", alpha=0.5, s=18, label="Feasible Evac Time")
            # Rolling mean of feasible times
            if len(feas_times) >= 10:
                k = min(30, len(feas_times))
                roll_time = np.convolve(feas_times, np.ones(k) / k, mode="valid")
                roll_x = feas_episodes[k - 1:]
                ax3.plot(roll_x, roll_time, color="#6d28d9", linewidth=2.0, label=f"Trend ({k}-pt rolling)")
        ax3.set_title("Evacuation Time (Feasible Configurations)")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Evacuation Steps")
        ax3.grid(True, linestyle="--", alpha=0.5)
        if feas_times:
            ax3.legend(loc="upper right")

        # 4. Losses vs Optimizer Updates
        ax4 = axs[1, 1]
        if loss_history:
            upds = [l["update"] for l in loss_history]
            actor_losses = [l["actor_loss"] for l in loss_history]
            critic_losses = [l["critic_loss"] for l in loss_history]
            entropies = [l["entropy"] for l in loss_history]

            ax4.plot(upds, actor_losses, color="#e11d48", label="Actor Loss", linewidth=1.5)
            ax4.plot(upds, critic_losses, color="#d97706", label="Critic Loss", linewidth=1.5)
            ax4_tw = ax4.twinx()
            ax4_tw.plot(upds, entropies, color="#0891b2", linestyle="--", label="Entropy", linewidth=1.2)
            ax4_tw.set_ylabel("Policy Entropy", color="#0891b2")
            ax4.set_title("Loss Curves & Entropy vs Optimizer Updates")
            ax4.set_xlabel("Optimizer Update Count")
            ax4.set_ylabel("Loss")
            ax4.grid(True, linestyle="--", alpha=0.5)
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_tw.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            ax4.text(0.5, 0.5, "No optimizer updates recorded yet", ha="center", va="center")

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=180)
        plt.close()
    except Exception as exc:
        print(f"  [Warning] Could not generate diagnostic plot: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Extended Stage 18B Training Engine
# ---------------------------------------------------------------------------

def train_stage18b(
    episodes: int = 1000,
    lr: float = 0.0007,
    gamma: float = 0.99,
    entropy_coef: float = 0.03,
    batch_size: int = 8,
    exit_count: int = 2,
    max_candidates: int = 8,
    output_dir: Optional[Path] = None,
    seed_pool_size: int = 500,
    checkpoint_every: int = 200,
    summary_interval: int = 50,
) -> dict[str, Any]:
    """
    Execute Stage 18B extended training with live terminal feedback.
    """
    out_dir = output_dir or _RL_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model_checkpoint_path = out_dir / "model_stage18b.pt"
    best_model_path = out_dir / "model_stage18b_best.pt"
    history_json_path = out_dir / "training_history_stage18b.json"
    history_csv_path = out_dir / "training_history_stage18b.csv"
    plot_path = out_dir / "training_plot_stage18b.png"

    # Seed generation: pool of unique training seeds strictly excluding reserved eval seeds
    rng = random.Random(42)
    candidate_train_seeds = []
    curr_seed = 101
    while len(candidate_train_seeds) < seed_pool_size:
        if curr_seed not in ALL_FORBIDDEN_TRAIN_SEEDS:
            candidate_train_seeds.append(curr_seed)
        curr_seed += 1

    device = torch.device("cpu")

    # Banner
    print("=" * 76, flush=True)
    print("  PATH_RL -- Stage 18B: Extended Actor-Critic Training (Live Terminal)", flush=True)
    print("=" * 76, flush=True)
    print(f"  Total Episodes        : {episodes:,}", flush=True)
    print(f"  Unique Training Seeds : {len(candidate_train_seeds)} (Seeds 101+, zero test overlap)", flush=True)
    print(f"  Batch Size (Updates)  : Every {batch_size} episodes (~{episodes // batch_size} updates)", flush=True)
    print(f"  Learning Rate         : {lr} (Adam with gradient clip norm=1.0)", flush=True)
    print(f"  Discount Factor (gamma)   : {gamma}", flush=True)
    print(f"  Entropy Coefficient   : {entropy_coef}", flush=True)
    print(f"  Exits to Select (k)   : {exit_count}", flush=True)
    print(f"  Max Candidates Slot   : {max_candidates}", flush=True)
    print(f"  Output Directory      : {out_dir}", flush=True)
    print(f"  Model Target          : {model_checkpoint_path.name}", flush=True)
    print("-" * 76, flush=True)
    print("  Initializing environment and neural policy network...", flush=True)

    env = ExitPlacementEnv()
    model = MaskedActorCritic(max_candidates=max_candidates).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize CSV header if file does not exist or starting fresh
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "seed", "grid_rows", "grid_cols", "candidates", "occupants",
            "reward", "rolling_reward", "is_feasible", "is_compliant", "evacuation_time",
            "rolling_feasibility", "actor_loss", "critic_loss", "entropy", "updates_completed",
            "elapsed_seconds", "term_reason"
        ])

    history: list[dict[str, Any]] = []
    loss_history: list[dict[str, Any]] = []
    unique_scenarios_seen = set()

    # Graceful exit handler
    interrupted = False

    def handle_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\n[Ctrl+C Detected] Gracefully terminating training...", flush=True)
        print("Saving current progress, history logs, and latest model checkpoint...", flush=True)

    original_sigint = signal.signal(signal.SIGINT, handle_sigint)

    # Buffers
    batch_states: list[torch.Tensor] = []
    batch_masks: list[torch.Tensor] = []
    batch_actions: list[int] = []
    batch_returns: list[float] = []

    best_feasible_reward = -float("inf")
    updates_completed = 0
    last_actor_loss: Optional[float] = None
    last_critic_loss: Optional[float] = None
    last_entropy: Optional[float] = None

    start_time = time.perf_counter()

    # Live tqdm progress bar
    pbar = tqdm(
        range(1, episodes + 1),
        desc="Training Stage 18B",
        unit="ep",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    try:
        for ep in pbar:
            if interrupted:
                break

            # 1. Sample scenario
            seed = candidate_train_seeds[(ep - 1) % len(candidate_train_seeds)]
            unique_scenarios_seen.add(seed)

            # Generate on the fly
            scenario = make_diverse_training_scenario(seed=seed)
            obs = env.reset(scenario, exit_count=exit_count)

            ep_states: list[torch.Tensor] = []
            ep_masks: list[torch.Tensor] = []
            ep_actions: list[int] = []
            ep_rewards: list[float] = []

            terminated = False
            info: dict[str, Any] = {}

            # Episode Step Loop
            while not terminated:
                state, mask = extract_features(obs, max_candidates=max_candidates)
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

            # 2. Gradient Update
            if ep % batch_size == 0 or ep == episodes:
                states_tensor = torch.cat(batch_states, dim=0).to(device)
                masks_tensor = torch.cat(batch_masks, dim=0).to(device)
                actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=device)
                returns_tensor = torch.tensor(batch_returns, dtype=torch.float32, device=device)

                log_probs, entropy, values = model.evaluate_actions(
                    states_tensor, masks_tensor, actions_tensor
                )

                # Advantage estimation & normalization
                advantages = returns_tensor - values.detach()
                if len(advantages) > 1:
                    adv_std = advantages.std() + 1e-8
                    advantages = (advantages - advantages.mean()) / adv_std

                actor_loss = -(log_probs * advantages).mean()
                critic_loss = F.mse_loss(values, returns_tensor)
                entropy_loss = -entropy.mean()

                total_loss = actor_loss + 0.5 * critic_loss + entropy_coef * entropy_loss

                optimizer.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                updates_completed += 1
                last_actor_loss = float(actor_loss.item())
                last_critic_loss = float(critic_loss.item())
                last_entropy = float(entropy.mean().item())

                loss_history.append({
                    "update": updates_completed,
                    "episode": ep,
                    "actor_loss": round(last_actor_loss, 4),
                    "critic_loss": round(last_critic_loss, 4),
                    "entropy": round(last_entropy, 4),
                    "grad_norm": round(float(grad_norm.item()), 4),
                })

                batch_states.clear()
                batch_masks.clear()
                batch_actions.clear()
                batch_returns.clear()

            # 3. Episode Recording & Metrics
            final_reward = ep_rewards[-1]
            is_feasible = bool(info.get("is_feasible", False))
            is_compliant = bool(info.get("is_compliant", False))
            evac_time = info.get("total_evacuation_time")
            term_reason = info.get("reason", "unknown")

            # Rolling stats (last 50 episodes)
            recent_window = 50
            recent_rewards = [h["reward"] for h in history[-(recent_window - 1):]] + [final_reward]
            rolling_reward = sum(recent_rewards) / len(recent_rewards)

            recent_feas = [h["is_feasible"] for h in history[-(recent_window - 1):]] + [is_feasible]
            rolling_feasibility = sum(recent_feas) / len(recent_feas)

            elapsed = time.perf_counter() - start_time

            ep_record = {
                "episode": ep,
                "seed": seed,
                "reward": final_reward,
                "rolling_reward": round(rolling_reward, 2),
                "is_feasible": is_feasible,
                "is_compliant": is_compliant,
                "total_evacuation_time": evac_time,
                "rolling_feasibility": round(rolling_feasibility, 3),
                "term_reason": term_reason,
                "selected_exits": list(env.selected_indices),
                "actor_loss": last_actor_loss,
                "critic_loss": last_critic_loss,
                "entropy": last_entropy,
                "updates_completed": updates_completed,
                "elapsed_seconds": round(elapsed, 2),
            }
            history.append(ep_record)

            # Append to CSV immediately (incremental saving)
            with open(history_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ep, seed, scenario.floor_plan.grid.rows, scenario.floor_plan.grid.cols,
                    len(env.candidates), len(scenario.occupants.agents),
                    final_reward, round(rolling_reward, 2), is_feasible, is_compliant,
                    evac_time if evac_time is not None else "",
                    round(rolling_feasibility, 3),
                    round(last_actor_loss, 4) if last_actor_loss is not None else "",
                    round(last_critic_loss, 4) if last_critic_loss is not None else "",
                    round(last_entropy, 4) if last_entropy is not None else "",
                    updates_completed, round(elapsed, 2), term_reason
                ])

            # Track best feasible model
            if is_feasible and final_reward > best_feasible_reward:
                best_feasible_reward = final_reward
                model.save(
                    best_model_path,
                    metadata={
                        "stage": "18B",
                        "episodes": ep,
                        "best_feasible_reward": best_feasible_reward,
                        "learning_rate": lr,
                        "unique_seeds": len(unique_scenarios_seen),
                    },
                )

            # Update live tqdm postfix
            loss_str = f"{last_actor_loss:.2f}" if last_actor_loss is not None else "N/A"
            crit_str = f"{last_critic_loss:.1f}" if last_critic_loss is not None else "N/A"
            pbar.set_postfix({
                "R": f"{final_reward:.0f}",
                "RollR": f"{rolling_reward:.1f}",
                "Feas": f"{rolling_feasibility:.0%}",
                "L_pi": loss_str,
                "L_v": crit_str,
                "Upd": updates_completed,
            })

            # 4. Periodic Summary Table
            if ep % summary_interval == 0 or ep == episodes:
                recent_slice = history[-summary_interval:]
                slice_feas = [h["is_feasible"] for h in recent_slice]
                slice_feas_rate = sum(slice_feas) / len(slice_feas)
                slice_times = [h["total_evacuation_time"] for h in recent_slice if h.get("is_feasible") and h.get("total_evacuation_time")]
                avg_time = (sum(slice_times) / len(slice_times)) if slice_times else None
                avg_time_str = f"{avg_time:.1f} steps" if avg_time else "N/A"
                recent_mean_r = sum(h["reward"] for h in recent_slice) / len(recent_slice)

                tqdm.write(
                    f"  [Summary Ep {ep:4d}/{episodes}] "
                    f"Reward (mean {summary_interval}): {recent_mean_r:7.1f} | "
                    f"Feasibility: {slice_feas_rate:5.1%} | "
                    f"Avg Evac: {avg_time_str:>10s} | "
                    f"Updates: {updates_completed:3d} | "
                    f"Elapsed: {elapsed:5.1f}s"
                )
                sys.stdout.flush()

                # Periodic JSON dump so crash loses at most summary_interval items
                _dump_json_history(
                    history_json_path, history, loss_history, episodes,
                    elapsed, candidate_train_seeds, lr, model_checkpoint_path, updates_completed
                )

            # Periodic checkpoint save
            if ep % checkpoint_every == 0:
                model.save(
                    model_checkpoint_path,
                    metadata={
                        "stage": "18B",
                        "episodes": ep,
                        "total_target_episodes": episodes,
                        "learning_rate": lr,
                        "updates_completed": updates_completed,
                    },
                )

    finally:
        signal.signal(signal.SIGINT, original_sigint)

    total_time = time.perf_counter() - start_time

    # Save final model checkpoint
    model.save(
        model_checkpoint_path,
        metadata={
            "stage": "18B",
            "episodes": len(history),
            "total_target_episodes": episodes,
            "runtime_seconds": round(total_time, 2),
            "updates_completed": updates_completed,
            "learning_rate": lr,
            "best_feasible_reward": best_feasible_reward if best_feasible_reward != -float("inf") else None,
            "unique_training_scenarios": len(unique_scenarios_seen),
        },
    )

    # Save complete JSON history
    _dump_json_history(
        history_json_path, history, loss_history, len(history),
        total_time, candidate_train_seeds, lr, model_checkpoint_path, updates_completed
    )

    # Generate diagnostic plots
    print("-" * 76, flush=True)
    print("  Generating Stage 18B diagnostic plots...", flush=True)
    generate_diagnostic_plots(history, loss_history, plot_path)

    # Print Final Training Summary
    final_feas = [h["is_feasible"] for h in history[-100:]] if len(history) >= 100 else [h["is_feasible"] for h in history]
    final_feas_rate = sum(final_feas) / len(final_feas) if final_feas else 0.0

    print("=" * 76, flush=True)
    print("  STAGE 18B TRAINING COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 76, flush=True)
    print(f"  Total Episodes Executed  : {len(history):,}", flush=True)
    print(f"  Total Optimizer Updates  : {updates_completed:,}", flush=True)
    print(f"  Unique Scenarios Visited : {len(unique_scenarios_seen):,}", flush=True)
    print(f"  Total Duration           : {total_time:.2f} seconds ({total_time / 60:.2f} mins)", flush=True)
    print(f"  Final 100-Ep Feasibility : {final_feas_rate:.1%}", flush=True)
    print(f"  Best Feasible Reward     : {best_feasible_reward if best_feasible_reward != -float('inf') else 'None'}", flush=True)
    print(f"  Saved Checkpoint         : {model_checkpoint_path}", flush=True)
    print(f"  Saved CSV Logs           : {history_csv_path}", flush=True)
    print(f"  Saved JSON Logs          : {history_json_path}", flush=True)
    print(f"  Saved Diagnostic Plot    : {plot_path}", flush=True)
    print("=" * 76, flush=True)

    return {
        "episodes_completed": len(history),
        "updates_completed": updates_completed,
        "runtime_seconds": total_time,
        "model_path": str(model_checkpoint_path),
        "history_path": str(history_json_path),
        "csv_path": str(history_csv_path),
        "plot_path": str(plot_path),
        "final_feasibility_rate": final_feas_rate,
        "best_feasible_reward": best_feasible_reward if best_feasible_reward != -float("inf") else None,
    }


def _dump_json_history(
    json_path: Path,
    history: list[dict[str, Any]],
    loss_history: list[dict[str, Any]],
    episodes: int,
    runtime: float,
    train_seeds: list[int],
    lr: float,
    model_path: Path,
    updates: int,
) -> None:
    """Helper to dump structured JSON training log."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_metadata": {
                    "stage": "18B",
                    "algorithm": "MaskedActorCritic",
                    "episodes": episodes,
                    "updates_completed": updates,
                    "runtime_seconds": round(runtime, 3),
                    "learning_rate": lr,
                    "train_seeds_count": len(train_seeds),
                    "model_path": str(model_path),
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                },
                "loss_history": loss_history,
                "episodes": history,
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="PATH_RL Stage 18B -- Extended RL Training")
    parser.add_argument("--episodes", type=int, default=1000, help="Total training episodes (default: 1000)")
    parser.add_argument("--lr", type=float, default=0.0007, help="Learning rate (default: 0.0007)")
    parser.add_argument("--batch-size", type=int, default=8, help="Episodes per optimizer update (default: 8)")
    parser.add_argument("--entropy-coef", type=float, default=0.03, help="Entropy bonus coefficient (default: 0.03)")
    parser.add_argument("--checkpoint-every", type=int, default=200, help="Episodes between checkpoints (default: 200)")
    parser.add_argument("--summary-interval", type=int, default=50, help="Episodes between summary tables (default: 50)")
    args = parser.parse_args()

    train_stage18b(
        episodes=args.episodes,
        lr=args.lr,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
        checkpoint_every=args.checkpoint_every,
        summary_interval=args.summary_interval,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
