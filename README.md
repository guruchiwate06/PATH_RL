# 2D Multi-Agent Evacuation Simulation

A computational scenario-analysis system for modelling discrete-time evacuation
dynamics in 2D grid-based environments.

---

## 1. What Is This Project?

This is a Python framework for constructing, configuring, and executing
discrete-time 2D evacuation simulations.

Buildings are represented as grid-based environments.  Occupants are
represented as autonomous agents with explicit lifecycle states.  The system
advances time in discrete steps, evaluating agent positions against exit
conditions, and records outcome statistics for analysis.

**This is a computational simulation / scenario-analysis system.**
It does not predict real-world human evacuation behaviour and makes no
claims of empirical validity against observed crowd dynamics.

---

## 2. What Problem Does It Model?

The system is intended to support structured, reproducible computational
experiments around questions such as:

- How does building layout (corridors, exits, walls) affect evacuation time?
- How does the number and starting distribution of occupants affect outcomes?
- What movement strategies lead to faster or safer evacuations?
- Where do bottlenecks emerge under different agent population sizes?

---

## 3. Current Scope (Stage 5 — Experimental Framework)

The system currently supports discrete-time agent movement, shortest-path navigation,
capacity-constrained cell occupancy and congestion dynamics, an interactive visualizer,
and a non-invasive experimental framework for batch execution and parameter sweeps.

| Component | Status | Notes |
|---|---|---|
| `SimulationConfig` | ✅ Complete | Pydantic-validated JSON scenario loader |
| `Environment` | ✅ Complete | Grid with walls, exits, neighbour queries |
| `Agent` | ✅ Complete | Stateful entity with lifecycle states |
| `NavigationGraph` | ✅ Complete | NetworkX graph over traversable cells |
| `OccupancyMap` | ✅ Complete | Dynamic cell occupancy and capacity tracking (Stage 3) |
| `ShortestPathStrategy` | ✅ Complete | BFS-based evacuation movement (Stage 2/3) |
| `Simulation` | ✅ Complete | Discrete timestep loop, conflict resolution, trace generation |
| `MetricsCollector` | ✅ Complete | Per-agent event recording + comprehensive result computation |
| Visualizer | ✅ Complete | HTML/Canvas interactive visualizer + Matplotlib snapshot/GIF export (Stage 4) |
| `experiment.py` | ✅ Complete | Scenario runner, multi-seed repetition, parameter sweeps, export (Stage 5) |
| Test suite | ✅ Complete | 319 passing tests across all modules |

---

## 4. Current Limitations

- **No dynamic hazards.** Fire, smoke, and hazardous propagation are not yet modelled.
- **Homogeneous agents.** Heterogeneous agent capabilities (speed variations, reaction delay, panic) are not yet implemented.
- **Single-threaded.** Experiments run sequentially in memory (sufficient for current scenario sizes).
- **Rule-based.** Reinforcement learning and learned policies are not yet incorporated.

---

## 5. Project Architecture

```
Scenario Configuration  (config.py)
         │
         ▼
   Environment          (environment.py)   — grid, walls, exits
         │
         ▼
      Agents            (agent.py)         — position, state
         │
         ▼
  NavigationGraph       (pathfinding.py)   — traversable-cell graph
         │
         ▼
   OccupancyMap         (occupancy.py)     — cell capacity & congestion
         │
         ▼
    Simulation          (simulation.py)    — timestep loop + strategy protocol
         │
         ▼
 MetricsCollector       (metrics.py)       — event recording + outcome computation

Experiment Framework    (experiment.py)    — orchestrates Simulation & aggregates Metrics
```

**Design rules enforced:**
- Dependency direction is strictly top-down (no circular imports).
- `experiment.py` acts as a wrapper around `Simulation` and does NOT modify the simulation core.
- The simulation engine remains fully functional and testable without `experiment.py`.
- Visualization is entirely decoupled from simulation and experimental logic.

### Directory Structure

```
evacuation_simulation/
│
├── simulation/
│   ├── __init__.py
│   ├── config.py          ← Pydantic scenario models + JSON loader
│   ├── environment.py     ← NumPy-backed grid environment
│   ├── agent.py           ← Agent entity + AgentState IntEnum
│   ├── pathfinding.py     ← NetworkX navigation graph
│   ├── occupancy.py       ← OccupancyMap and capacity tracking
│   ├── movement.py        ← MovementRequest and conflict resolution
│   ├── strategy.py        ← ShortestPathStrategy protocol implementation
│   ├── simulation.py      ← Timestep loop + MovementStrategy protocol
│   ├── metrics.py         ← MetricsCollector + SimulationResult
│   └── experiment.py      ← Scenario runner, sweeps, aggregation, CSV/JSON export
│
├── scenarios/
│   ├── basic_building.json   ← 10×10 scenario (6 agents)
│   └── bottleneck.json       ← 10×5 bottleneck corridor scenario (12 agents)
│
├── tests/
│   ├── conftest.py
│   ├── test_environment.py
│   ├── test_agent.py
│   ├── test_pathfinding.py
│   ├── test_movement.py
│   ├── test_congestion.py
│   ├── test_simulation.py
│   ├── test_strategy.py
│   ├── test_strategy_integration.py
│   ├── test_metrics.py
│   ├── test_visualization.py
│   └── test_experiment.py
│
├── experiment_output/     ← Default directory for exported CSV & JSON runs
├── main.py                ← CLI entry point (single runs & --experiment mode)
├── visualize.py           ← Offline visualization and data generation
├── visualize.html         ← Interactive browser-based visualizer
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 6. Installation

**Requirements:** Python 3.11+

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 7. Running the Simulation

### Single Simulation Run

```bash
# Run default basic building scenario
python main.py

# Run a specific scenario
python main.py --scenario scenarios/bottleneck.json

# Run and visualize final state
python main.py --scenario scenarios/bottleneck.json --visualize
```

### Multi-Seed Experiment Mode (Stage 5)

Run repeated stochastic evaluations across multiple random seeds with automated result aggregation and export:

```bash
# Run 5 seeds on the bottleneck scenario
python main.py --experiment --scenario scenarios/bottleneck.json --seeds 1,2,3,4,5

# Specify custom output directory for CSV and JSON exports
python main.py --experiment --scenario scenarios/bottleneck.json --seeds 1,2,3,4,5 --output-dir experiment_output/
```

---

## 8. Experimental Framework (`simulation/experiment.py`)

Stage 5 introduces a non-invasive experimental framework that wraps `Simulation.run()` without modifying simulation core logic.

### Core Concepts and Distinctions

- **Individual Simulation (`run_scenario`)**: Executes one scenario configuration with an optional seed override. Returns a `RunRecord` containing the `SimulationResult` and execution metadata.
- **Repeated Experiment (`run_repeated`)**: Executes the exact same scenario configuration across an explicit list of seeds. Produces an independent simulation instance per seed with zero shared state.
- **Parameter Sweep (`run_sweep`)**: Evaluates a set of named `SweepPoint` configurations across multiple seeds ($N \times M$ total runs), tagging each run with its sweep parameter values.
- **Aggregated Result (`aggregate_results`)**: Computes summary statistics across collections of run records (mean, min, max, standard deviation, and median for numeric metrics like evacuation time, wait steps, and congestion). Handles `None` values gracefully without converting them to zero.

### Python API Example: Parameter Sweep & Export

```python
from pathlib import Path
from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.experiment import (
    SweepPoint,
    run_sweep,
    export_csv,
    export_json,
)

# 1. Load baseline scenarios
scenarios_dir = Path("scenarios")
basic_cfg = SimulationConfig.from_json(scenarios_dir / "basic_building.json")
bottleneck_cfg = SimulationConfig.from_json(scenarios_dir / "bottleneck.json")

# 2. Define sweep points
sweep_points = [
    SweepPoint("basic_6", basic_cfg, {"scenario": "basic_building", "agents": 6}),
    SweepPoint("bottleneck_12", bottleneck_cfg, {"scenario": "bottleneck", "agents": 12}),
]

# 3. Execute sweep over 5 seeds (2 x 5 = 10 independent runs)
seeds = [10, 20, 30, 40, 50]
experiment = run_sweep(sweep_points, seeds, experiment_id="density_sweep")

# 4. Access aggregate statistics
agg = experiment.aggregate
print(f"Total runs: {experiment.run_count}")
print(f"Evacuation rate: {agg.evacuation_rate_mean:.1%}")
print(f"Mean evacuation time: {agg.mean_evacuation_time_mean:.2f}s (std={agg.mean_evacuation_time_std:.4f})")

# 5. Export to CSV (one row per run) and JSON (full experiment metadata & runs)
output_dir = Path("experiment_output")
export_csv(experiment.runs, output_dir / "sweep_results.csv")
export_json(experiment, output_dir / "sweep_experiment.json")
```

---

## 9. Running the Test Suite

```bash
# Run all tests (all 319 passing)
python -m pytest

# Run only experimental framework tests
python -m pytest tests/test_experiment.py -v
```

---

## 10. Scenario Configuration Format


Scenario files are JSON documents following this schema:

```json
{
  "scenario_name": "my_building",
  "grid": { "rows": 10, "cols": 10 },
  "walls": [[2, 0], [2, 1]],
  "exits": [[9, 4], [9, 5]],
  "agents": [
    { "agent_id": "agent_0", "row": 0, "col": 0 }
  ],
  "parameters": {
    "max_timesteps": 500,
    "random_seed": 42
  }
}
```

Validation rules (enforced by Pydantic):
- At least one exit must be defined.
- Exits and walls must not overlap.
- All agent IDs must be unique.
- All agent starting positions must be inside the grid and not on walls.

---

## 11. Extending the System (Future Stages)

The architecture is designed to be extended without modifying core components.

| Stage | Planned Addition | Status |
|---|---|---|
| Stage 1 | Foundation, Pydantic configuration, environment, metrics | ✅ Complete |
| Stage 2 | Shortest-path movement strategy & agent baseline | ✅ Complete |
| Stage 3 | Cell capacity, occupancy map, and congestion dynamics | ✅ Complete |
| Stage 4 | Interactive web visualizer & Matplotlib snapshots | ✅ Complete |
| Stage 5 | Scenario runner, parameter sweeps, and export framework | ✅ Complete |
| Stage 6 | Heterogeneous agents (speed, mobility, response time) | Planned |
| Stage 7 | Dynamic hazards (fire, smoke propagation) | Planned |
| Stage 8 | Reinforcement learning integration | Planned |

New movement strategies implement the `MovementStrategy` protocol and are
injected into `Simulation(config, movement_strategy=...)` with no changes
to existing code.

---

## 12. Dependencies


| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.26 | Grid arrays, random generation |
| `networkx` | ≥3.2 | Navigation graph, path algorithms |
| `pydantic` | ≥2.5 | Scenario configuration validation |
| `pytest` | ≥8.0 | Test runner |
| `pytest-cov` | ≥4.1 | Test coverage reporting |
