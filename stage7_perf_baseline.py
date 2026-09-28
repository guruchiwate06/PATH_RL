"""
stage7_perf_baseline.py
-----------------------
Stage 7: Actual simulation performance measurement.

Measures the runtime of evacuation simulations across representative scenarios
to determine whether the simulator is fast enough to support future:

    many exit configurations × repeated simulations × optimization loops

IMPORTANT: All numbers reported here are MEASURED, not estimated or invented.
Run this script to obtain current baseline measurements for your hardware.

Usage:
    python stage7_perf_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

_PARENT = Path(__file__).parent.parent.resolve()
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from evacuation_simulation.simulation.config import (
    AgentConfig,
    AgentProfileConfig,
    SimulationConfig,
)
from evacuation_simulation.simulation.simulation import Simulation
from evacuation_simulation.simulation.strategy import ShortestPathStrategy

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


class PerfResult(NamedTuple):
    label: str
    agents: int
    grid_size: str
    n_runs: int
    total_s: float
    mean_ms: float
    runs_per_s: float


def _time_config(config: SimulationConfig, n_runs: int = 200) -> tuple[float, float, float]:
    """
    Run *config* exactly *n_runs* times and return (total_s, mean_ms, runs_per_s).

    Each run creates a fresh Simulation instance so there is zero shared state.
    """
    start = time.perf_counter()
    for _ in range(n_runs):
        sim = Simulation(config, ShortestPathStrategy())
        sim.run()
    total_s = time.perf_counter() - start
    mean_ms = (total_s / n_runs) * 1000.0
    runs_per_s = n_runs / total_s
    return total_s, mean_ms, runs_per_s


def measure_scenario(
    label: str,
    config: SimulationConfig,
    n_runs: int = 200,
) -> PerfResult:
    """Run performance measurement for a single scenario configuration."""
    n_agents = len(config.agents)
    grid_size = f"{config.grid.rows}×{config.grid.cols}"
    total_s, mean_ms, rps = _time_config(config, n_runs)
    return PerfResult(
        label=label,
        agents=n_agents,
        grid_size=grid_size,
        n_runs=n_runs,
        total_s=total_s,
        mean_ms=mean_ms,
        runs_per_s=rps,
    )


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


def load_scenario(name: str) -> SimulationConfig:
    path = SCENARIOS_DIR / f"{name}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return SimulationConfig(**data)


def make_heterogeneous_scenario() -> SimulationConfig:
    """Mixed-speed, mixed-delay population in a medium corridor."""
    return SimulationConfig(
        scenario_name="perf_heterogeneous",
        grid={"rows": 3, "cols": 20},
        walls=[[0, c] for c in range(20)] + [[2, c] for c in range(20)],
        exits=[[1, 19]],
        agents=[
            AgentConfig(
                agent_id=f"agent_{i}",
                row=1,
                col=i,
                profile=AgentProfileConfig(
                    speed=[1.0, 0.5, 0.25, 0.75][i % 4],
                    reaction_delay=i % 3,
                ),
            )
            for i in range(10)
        ],
        parameters={"max_timesteps": 500, "random_seed": 42, "default_cell_capacity": 1},
    )


def make_exit_config_scenario(exit_col: int) -> SimulationConfig:
    """Exit-config comparison scenario: same floor plan, exit position varies."""
    return SimulationConfig(
        scenario_name=f"perf_exit_col{exit_col}",
        grid={"rows": 3, "cols": 12},
        walls=[[0, c] for c in range(12)] + [[2, c] for c in range(12)],
        exits=[[1, exit_col]],
        agents=[
            AgentConfig(agent_id=f"a{i}", row=1, col=5 + i)
            for i in range(6)
        ],
        parameters={"max_timesteps": 100, "random_seed": 42, "default_cell_capacity": 1},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print(" STAGE 7 — SIMULATION PERFORMANCE BASELINE")
    print(" All measurements are ACTUAL runtimes on this machine.")
    print("=" * 72)
    print()

    N = 200  # runs per scenario
    results: list[PerfResult] = []

    # --- Standard scenarios ---
    for name, label in [
        ("basic_building", "Basic building (10×10, 6 agents)"),
        ("bottleneck", "Bottleneck corridor (10×5, 12 agents)"),
        ("corridor", "Open corridor (3×10, 2 agents)"),
        ("two_exits", "Two exits (5×9, 4 agents)"),
    ]:
        print(f"  Measuring: {label} ... ", end="", flush=True)
        cfg = load_scenario(name)
        r = measure_scenario(label, cfg, n_runs=N)
        results.append(r)
        print(f"done ({r.mean_ms:.3f} ms/run)")

    # --- Heterogeneous population ---
    print(f"  Measuring: Heterogeneous (3×20, 10 mixed-profile agents) ... ", end="", flush=True)
    cfg_hetero = make_heterogeneous_scenario()
    r_hetero = measure_scenario(
        "Heterogeneous pop. (3×20, 10 agents, mixed speed+delay)",
        cfg_hetero,
        n_runs=N,
    )
    results.append(r_hetero)
    print(f"done ({r_hetero.mean_ms:.3f} ms/run)")

    # --- Exit config sweep simulation ---
    print(f"  Measuring: Exit-config sweep (3 configs × {N} runs each) ... ", end="", flush=True)
    sweep_total_s = 0.0
    sweep_n = 0
    for exit_col in [0, 5, 11]:
        cfg_ec = make_exit_config_scenario(exit_col)
        t, _, _ = _time_config(cfg_ec, n_runs=N)
        sweep_total_s += t
        sweep_n += N
    r_sweep = PerfResult(
        label=f"Exit-config sweep (3 configs × {N} runs = {sweep_n} total)",
        agents=6,
        grid_size="3×12",
        n_runs=sweep_n,
        total_s=sweep_total_s,
        mean_ms=(sweep_total_s / sweep_n) * 1000.0,
        runs_per_s=sweep_n / sweep_total_s,
    )
    results.append(r_sweep)
    print(f"done ({r_sweep.mean_ms:.3f} ms/run)")

    # --- Print results table ---
    print()
    print("-" * 72)
    hdr = f"{'Scenario':<45} {'Agents':>6} {'ms/run':>8} {'runs/s':>10}"
    print(hdr)
    print("-" * 72)
    for r in results:
        print(
            f"{r.label:<45} {r.agents:>6} {r.mean_ms:>8.3f} {r.runs_per_s:>10.1f}"
        )
    print("-" * 72)

    # --- Capacity projection for optimization loop ---
    print()
    print("CAPACITY PROJECTION")
    print("-" * 72)
    ref_result = results[0]  # basic building as reference
    rps = ref_result.runs_per_s
    print(f"  Reference: {ref_result.label}")
    print(f"  Measured throughput: {rps:.1f} simulations/second")
    print()
    for n_configs in [10, 100, 1_000, 10_000]:
        secs = n_configs / rps
        print(
            f"  {n_configs:>6} exit configs x 1 seed  ->  "
            f"estimated {secs:.2f}s total"
        )
    print()
    print("  [NOTE] Projections assume single-threaded execution.")
    print("  [NOTE] Actual exit-config search will use repeated seeds.")
    print()
    print("=" * 72)
    print(" CONCLUSION")
    print("=" * 72)
    if rps >= 100:
        print(f"  Throughput ({rps:.0f} sim/s) is sufficient for small-to-medium")
        print("  exit configuration sweeps without optimization.")
        print("  Larger spaces will require batching or parallelism at Stage 13+.")
    else:
        print(f"  Throughput ({rps:.0f} sim/s) — consider profiling before Stage 13.")
    print()


if __name__ == "__main__":
    main()
