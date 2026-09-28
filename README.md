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

## 3. Current Scope (Stage 9 — Candidate Exit Generation & Feasibility)

The system supports discrete-time agent movement, shortest-path navigation,
capacity-constrained cell occupancy and congestion dynamics, an interactive visualizer,
an experimental runner for batch execution and parameter sweeps, an extensible
heterogeneous agent profile architecture (`AgentProfile`) supporting individualized
speed and reaction delay, a comprehensive internal validation framework, an 
explicit separation of building geometry (`FloorPlan`) from exit routing (`ExitConfiguration`),
and an independent generator for geometrically possible `CandidateExit` locations.

| Component | Status | Notes |
|---|---|---|
| `SimulationConfig` | ✅ Complete | Nested Pydantic-validated JSON scenario loader (Stage 8) |
| `FloorPlan` | ✅ Complete | Decoupled static building geometry (Stage 8) |
| `ExitConfiguration` | ✅ Complete | Decoupled exit placement configuration (Stage 8) |
| `generation.py` | ✅ Complete | `CandidateExit` and config generators (`generate_candidate_exits`) (Stage 9) |
| `Environment` | ✅ Complete | Grid with walls, exits, neighbour queries |
| `AgentProfile` | ✅ Complete | Per-agent extensible profile (`speed`, `reaction_delay`, metadata) (Stage 6) |
| `Agent` | ✅ Complete | Stateful entity with lifecycle states and profile delegates |
| `NavigationGraph` | ✅ Complete | NetworkX graph over traversable cells |
| `OccupancyMap` | ✅ Complete | Dynamic cell occupancy and capacity tracking (Stage 3) |
| `ShortestPathStrategy` | ✅ Complete | BFS-based evacuation movement (Stage 2/3) |
| `Simulation` | ✅ Complete | Discrete timestep loop, eligibility filtering, conflict resolution |
| `MetricsCollector` | ✅ Complete | Per-agent event recording + comprehensive result computation |
| Visualizer | ✅ Complete | Web visualizer (`visualize.html`) + Matplotlib snapshot/GIF export (`visualize.py`) |
| `experiment.py` | ✅ Complete | Scenario runner, sweeps, aggregation, CSV/JSON exports (Stage 5) |
| Validation Suite | ✅ Complete | Invariants A-J, OFAT sweeps, benchmarks (`stage7_validation_demo.py`) (Stage 7) |
| Test suite | ✅ Complete | 411 passing tests across 15 modules |

---

## 4. Current Limitations

- **No dynamic hazards.** Fire, smoke, and hazardous propagation are not yet modelled.
- **No age-based derivation formulas.** The profile architecture supports descriptive metadata (`age`), but physiological/demographic parameter derivation is reserved for a future modeling stage.
- **Single-threaded.** Experiments run sequentially in memory (execution speed is ~0.005s/run).
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

## 9. Heterogeneous Agent Profiles (`simulation/profile.py`)

Stage 6 introduces an extensible per-agent profile architecture (`AgentProfile`) allowing each agent to possess individual operational parameters and descriptive metadata.

### Core Attributes & Semantics

- **`speed` ($0.0 < \text{speed} \le 1.0$)**: Movement frequency at cell/timestep resolution. Evaluated deterministically via rate accumulation:
  - `speed = 1.0` (default): Movement opportunity every timestep.
  - `speed = 0.5`: Movement opportunity every 2 timesteps ($t=2, 4, 6, \dots$).
  - `speed = 0.25`: Movement opportunity every 4 timesteps ($t=4, 8, 12, \dots$).
  - An agent never jumps more than 1 cell in a single timestep.
- **`reaction_delay` ($\ge 0$)**: Number of initial timesteps the agent remains stationary before beginning movement ($t \le \text{reaction\_delay}$ delayed).
- **Semantic Separation**: Reaction delay is strictly decoupled from congestion waiting (`waiting_steps`). Congestion waiting only records capacity-blocked attempts.
- **Extensible Architecture**: The profile supports descriptive metadata (e.g. `age`, `input_attributes`, `derived_parameters`) for future demographic models without requiring simulation core changes.

---

## 10. Running the Test Suite

```bash
# Run all tests (all 335 passing)
python -m pytest

# Run profile test suite
python -m pytest tests/test_profile.py -v
```

---

## 11. Scenario Configuration Format

Scenario files (since Stage 8) cleanly separate the building geometry (`floor_plan`), 
exit placement (`exit_configuration`), and occupants (`occupants`).

```json
{
  "scenario_name": "heterogeneous_building",
  "floor_plan": {
    "grid": { "rows": 10, "cols": 10 },
    "walls": [[2, 0], [2, 1]]
  },
  "exit_configuration": {
    "configuration_id": "reference",
    "exits": [[9, 4], [9, 5]]
  },
  "occupants": {
    "agents": [
      {
        "agent_id": "fast_agent",
        "row": 0,
        "col": 0,
        "profile": { "speed": 1.0, "reaction_delay": 0 }
      },
      {
        "agent_id": "slow_delayed_agent",
        "row": 1,
        "col": 1,
        "profile": { "speed": 0.5, "reaction_delay": 3, "age": 70 }
      },
      {
        "agent_id": "default_agent",
        "row": 2,
        "col": 2
      }
    ]
  },
  "parameters": {
    "max_timesteps": 500,
    "random_seed": 42
  }
}
```

Validation rules:
- Omitted `profile` defaults to baseline `speed=1.0, reaction_delay=0`.
- `speed` must satisfy $0.0 < \text{speed} \le 1.0$.
- `reaction_delay` must be $\ge 0$.

---

## 12. Stage 7 — Baseline Validation & Exit-Configuration Readiness

Stage 7 establishes the **internal correctness, reproducibility, and exit-configuration readiness** of the simulation. The key goal is to prove the simulator can evaluate different exit configurations under otherwise identical conditions — the foundational capability for Stage 8+ exit optimization.

### Software Verification vs. Real-World Validation
> [!IMPORTANT]
> **Boundary of Claims:**
> This framework performs **software verification and internal model validation**. It confirms that:
> 1. Implementation invariants and physical capacity bounds hold strictly.
> 2. The simulation responds deterministically and predictably to parameter changes.
> 3. Results are bitwise reproducible given identical initial conditions.
> 4. The same floor plan with different exit positions produces measurably different outcomes.
>
> **This does NOT validate real-world human evacuation behavior.** The simulation makes no claim of empirical fidelity to actual pedestrian or panic dynamics and is not calibrated against real-world evacuation datasets.

### Invariants Verified (Invariants A through J)
The validation suite ([test_validation.py](file:///f:/Projects/RL_ENV/evacuation_simulation/tests/test_validation.py)) verifies 10 structural invariants at every timestep:
- **Invariant A (Position Validity):** Agents never occupy walls, non-existent cells, or invalid coordinates.
- **Invariant B (Capacity Invariant):** $\forall t, c: \text{occupancy}(c) \le \text{capacity}(c)$.
- **Invariant C (Occupancy Consistency):** The dynamic `OccupancyMap` exactly agrees with active agent coordinates.
- **Invariant D (Evacuated-Agent Consistency):** Evacuated agents have valid timestamps ($t \ge 1$), zero occupancy footprint, and submit no further movement requests.
- **Invariant E (Movement Validity):** Approved movements originate from current cell, move to an adjacent 4-connected passable cell, and respect target cell capacity.
- **Invariant F (Determinism):** Identical configurations and seeds yield bitwise-identical simulation results.
- **Invariant G (Reproducibility):** Repeated experiment runner executions produce identical aggregate metrics.
- **Invariant H (Monotonic Evacuation Count):** $\text{evacuated\_count}(t+1) \ge \text{evacuated\_count}(t)$.
- **Invariant I (Evacuation Time Validity):** Evacuation timestamps are strictly positive integers.
- **Invariant J (Waiting Validity):** Cumulative and per-agent waiting steps are strictly non-negative.

### Controlled Benchmark Scenarios
Five standardized benchmark scenarios isolate cause and effect:
1. **Scenario A — Open Corridor ([corridor.json](file:///f:/Projects/RL_ENV/evacuation_simulation/scenarios/corridor.json)):** Predictable 1D travel time. Evaluates non-credited simultaneous vacating clearance.
2. **Scenario B — Dual Exits ([two_exits.json](file:///f:/Projects/RL_ENV/evacuation_simulation/scenarios/two_exits.json)):** Verifies shortest-path exit partitioning and routing.
3. **Scenario C — Bottleneck Capacity Scaling ([bottleneck.json](file:///f:/Projects/RL_ENV/evacuation_simulation/scenarios/bottleneck.json)):** Compares capacity constraints ($C \in \{1, 2, 3\}$).
4. **Scenario D — Mixed-Speed Populations:** Confirms speed=0.5 doubles travel time; speed=0.25 quadruples it.
5. **Scenario E — Reaction Delay Isolation:** Confirms reaction delay shifts movement activation without incorrectly contributing to capacity waiting metrics.

### Critical Stage 7 Experiment: Same Floor Plan, Different Exit Configuration

The most important Stage 7 deliverable: the simulator can evaluate different exit configurations while holding **everything else constant**.

Experiment setup:
- **Floor plan:** 3×12 corridor, walls at rows 0 and 2 (fixed)
- **Agents:** 6 agents at columns 5–10 (fixed)
- **Seed:** 42 (fixed)
- **Only varies:** exit positions

| Configuration | Exits | Total Timesteps | Mean Evac Time | Total Wait |
|---|---|---|---|---|
| reference (right exit) | `(1, 11)` | 11 | 6.0 | 15 |
| alternative_A (left exit) | `(1, 0)` | 15 | 10.0 | 15 |
| alternative_B (center exit) | `(1, 5)` | 10 | 5.2 | 15 |

**Key observation:** Same floor plan + same agents + same seed + **different exits** → measurably different total simulation time and mean evacuation time.

> [!NOTE]
> Reference exits are **NOT** claimed to be optimal. They are a reproducible comparison baseline. The future optimization stages will search for configurations that improve on this baseline.

### Per-Exit Utilization Tracking

As of Stage 7, `SimulationResult` includes `per_exit_utilization: dict[tuple[int, int], int]` — a count of how many agents evacuated via each exit cell. This enables exit-load analysis for future evaluation.

```python
result.per_exit_utilization  # e.g. {(1, 11): 6}  or  {(2, 0): 2, (2, 8): 4}
```

### Reference Exit Configuration — BenchmarkConfig

A lightweight `BenchmarkConfig` dataclass (`simulation/benchmark.py`) identifies named reference benchmarks:

```python
from evacuation_simulation.simulation.benchmark import BenchmarkConfig

bc = BenchmarkConfig(
    benchmark_id="BENCH_TWO_EXITS_V1",
    scenario_name="two_exits",
    reference_exits=[(2, 0), (2, 8)],
    seed=42,
    description="Two-exit open room — 4 agents, split routing baseline.",
)
```

`BenchmarkConfig` stores **known/reference exits only**. It makes no optimality claim.

### Heterogeneous Agent Validation

Stage 6 profiles are formally verified:
- `speed=0.5` agent takes exactly 2× the timesteps of a `speed=1.0` agent on the same path
- `reaction_delay=N` shifts evacuation time by exactly N timesteps without inflating capacity wait counters
- System-level cascade: delaying a leading agent propagates the delay to all trailing agents in a single-file queue

### Measured Performance Baseline

Actual throughput on development hardware (single-threaded, Python 3.13):

| Scenario | Agents | ms/run | runs/second |
|---|---|---|---|
| Basic building (10×10) | 6 | ~4 ms | ~250/s |
| Bottleneck (10×5) | 12 | ~5 ms | ~210/s |
| Two exits (5×9) | 4 | ~1 ms | ~1000/s |
| Open corridor (3×10) | 2 | ~0.4 ms | ~2500/s |
| Heterogeneous (3×20) | 10 | ~5–7 ms | ~150–200/s |

Projected capacity for exit-config evaluation loops (single-threaded):
- 100 exit configs × 1 seed → ~0.4 s
- 1,000 exit configs × 1 seed → ~4 s
- 10,000 exit configs × 1 seed → ~40 s

Run the actual measurement: `python stage7_perf_baseline.py`

## 13. Stage 9 — Candidate Exit Generation & Feasibility

Stage 9 introduces an independent mechanism for identifying potential exit locations (`CandidateExit`) on a `FloorPlan`, and safely combinatorially constructing `ExitConfiguration`s from them.

### Vocabulary
*   **Geometric Candidate**: An exit location that is physically possible (e.g., a non-wall cell on the boundary of the grid). Handled by Stage 9.
*   **Feasible Configuration**: A set of candidate exits that are generated together. Handled by Stage 9.
*   **Legally Compliant Configuration**: A configuration that satisfies building regulations (e.g., minimum distances between exits, occupant-load capacity). **NOT** handled by Stage 9.

> [!IMPORTANT]
> **Stage 9 does NOT determine legal compliance or exit optimality.** 
> It purely serves as the foundation for the search space by bounding the set of geometrically valid exit combinations.

### Scaling Behavior
The `generate_exit_configurations` function uses `itertools.combinations` to yield configurations deterministically. Because the output is an `Iterator`, generation is heavily memory-efficient (O(1) memory), but evaluating the *entire* generated list can grow exponentially. 
For example, generating 5 exits out of 100 candidates will result in over 75 million combinations. Downstream pipelines (Stages 11+) will require heuristic sampling or RL logic instead of exhaustive iteration for large `exit_count` numbers.

### Current Limitation

> [!IMPORTANT]
> **No Safety or Regulatory Constraints.**
>
> While Stage 9 cleanly decouples exit generation from the simulation, it does not apply legal bounding boxes to those configurations.
>
> **Stages 10+** will introduce:
> - Regulatory safety constraint checking (e.g. minimum travel distances).
> - Automated placement optimization workflows.

### Benchmark Runner & Dataset Generation

```bash
# Run all Stage 7 benchmarks + exit-config experiment
python stage7_validation_demo.py

# Run performance baseline measurement
python stage7_perf_baseline.py
```

Output artifacts:
- `experiment_output/stage7_baseline_benchmark.csv / .json` — Benchmarks A–E
- `experiment_output/stage7_exit_config_experiment.csv / .json` — Exit-config comparison

## 14. Stage 10 — Realistic Floor-Plan & Exit Representation

Stage 10 upgrades the floor-plan and exit representation so that candidate emergency exits are derived from explicit, physical door openings rather than treating every boundary non-wall cell as an exit.

### Why Boundary-Cell Candidates Were a Simplification (Stage 9)
In Stage 9, any non-wall cell on the outer boundary was treated as a candidate exit. This served to test combination scaling but does not reflect real-world architectural design: in real buildings, people cannot escape through solid exterior walls unless there is an actual door opening. Furthermore, treating all boundary cells as exits causes combinatorial explosion (e.g., 34 candidates yielded 561 combinations for 2 exits).

### Explicit Door Openings (Stage 10)
In Stage 10, the `FloorPlan` explicitly represents structural openings via the `DoorOpening` model. Candidate emergency exits are derived strictly from explicit openings that connect the building to the outside.

### Core Architectural Distinctions
*   **`DoorOpening`**: Represents a physical aperture or doorway in the floor plan geometry. Contains `position` $(r, c)$, physical `width` (in meters), and an `exterior` boolean flag (`True` for doors leading outside, `False` for interior doors connecting rooms to hallways). **A DoorOpening is not automatically an exit.**
*   **`CandidateExit`**: A candidate emergency exit location derived from an explicit exterior `DoorOpening` (`exterior=True`). Preserves opening properties including physical width and door identifier. Interior openings are strictly excluded.
*   **`ExitConfiguration`**: A concrete combination of candidate exits evaluated together in simulation. Carries both exit grid coordinates and corresponding physical opening widths (`cfg.widths` and `cfg.get_width(cell)`).

### Stored Physical Width
Each `DoorOpening` and `CandidateExit` stores its physical width (e.g. 1.0m, 1.2m, 1.8m).
*   **Physical Property Only**: Width is purely a stored geometric property at this stage.
*   **NOT a Legal Check**: Stage 10 does NOT check building codes (e.g., NBC 2016 minimum exit width of 1.0m/1.5m), does NOT calculate occupant load, and does NOT alter simulation flow rates.
*   **No Exit Optimization**: Stage 10 does NOT select, rank, or recommend "best" exits.

### Preparation for Future Regulatory & Optimization Layers
This explicit representation establishes the bridge between raw architecture and legal compliance:
```
FloorPlan
   ├── walls: GridCell[]
   └── doors: DoorOpening[] (interior & exterior)
         ↓
CandidateExit[] (strictly exterior DoorOpenings, preserves width)
         ↓
ExitConfiguration[] (combinatorial exit sets with widths)
         ↓
[Future] RegulationProfile & ConstraintValidator (NBC 2016 minimum width, travel distance, separation)
         ↓
[Future] Feasible Compliant Configurations
         ↓
Evacuation Simulation Engine
         ↓
[Future] RL / Search Exit Placement Optimization
```

### Backward Compatibility (Legacy Fallback)
For scenarios created prior to Stage 10 without explicit `doors`, the generator provides a documented legacy fallback: when `floor_plan.doors` is empty and `fallback_legacy_boundary=True` (default), it scans boundary non-wall cells with default width 1.0m, ensuring zero regressions across all historical tests.

---

## 15. Future Architecture (Stages 11–15)

The architecture is designed for progressive extension without modifying the core simulation engine.

| Stage | Description | Status |
|---|---|---|
| 1 | Foundation, Pydantic config, environment, metrics | ✅ Complete |
| 2 | Shortest-path movement strategy & agent baseline | ✅ Complete |
| 3 | Cell capacity, occupancy map, congestion dynamics | ✅ Complete |
| 4 | Interactive web visualizer & Matplotlib snapshots | ✅ Complete |
| 5 | Scenario runner, parameter sweeps, and export framework | ✅ Complete |
| 6 | Heterogeneous agents (speed, reaction delay, AgentProfile) | ✅ Complete |
| 7 | Baseline validation & exit-configuration readiness | ✅ Complete |
| 8 | Explicit FloorPlan + ExitConfiguration architecture | ✅ Complete |
| 9 | Candidate exit location generation | ✅ Complete |
| 10 | Realistic floor-plan & exit representation (DoorOpening) | ✅ Complete |
| 11 | Regulatory / placement constraint layer | Planned |
| 12 | Exit configuration evaluation framework | Planned |
| 13 | Reference / ground-truth benchmark system | Planned |
| 14 | Non-RL optimization / search baseline | Planned |
| 15 | RL-based exit-placement optimization & UI | Planned |

### Intended Future Pipeline

```
Floor Plan
    ↓
Candidate Exits (Stage 9)
    ↓
Candidate Configurations (Stage 9)
    ↓
Regulatory Constraint Filtering (Stage 10)
    ↓
Feasible Configurations
    ↓
Evacuation Simulation  ← existing engine (Stages 1–7)
    ↓
Metrics (evacuation time, waiting, per-exit utilization)
    ↓
Non-RL Search Baseline (Stage 13)
    ↓
RL Optimization (Stage 14)
    ↓
Recommended Exit Configuration
```

New movement strategies implement the `MovementStrategy` protocol and are injected into `Simulation(config, movement_strategy=...)` with no changes to existing code.

---

## 13. Dependencies



| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.26 | Grid arrays, random generation |
| `networkx` | ≥3.2 | Navigation graph, path algorithms |
| `pydantic` | ≥2.5 | Scenario configuration validation |
| `pytest` | ≥8.0 | Test runner |
| `pytest-cov` | ≥4.1 | Test coverage reporting |
