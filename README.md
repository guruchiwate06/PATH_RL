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

## 3. Current Scope (Stage 13 — Non-RL Exit Configuration Search Baseline)

The system supports discrete-time agent movement, shortest-path navigation,
capacity-constrained cell occupancy and congestion dynamics, an interactive visualizer,
an experimental runner for batch execution and parameter sweeps, an extensible
heterogeneous agent profile architecture (`AgentProfile`) supporting individualized
speed and reaction delay, a comprehensive internal validation framework, an
explicit separation of building geometry (`FloorPlan`) from exit routing (`ExitConfiguration`),
an independent generator for geometrically possible `CandidateExit` locations,
a **modular regulatory constraint layer** (`rules.py`, `validator.py`) that
encodes NBC 2016 Part 4 rules for evaluating exit-configuration compliance,
an **exit configuration evaluation layer** (`evaluation.py`) that simulates
feasible configurations and scores them by primary objective (minimize total evacuation time),
and a **non-RL exhaustive search baseline** (`search.py`) that systematically searches
feasible combinatorial exit configurations to identify the optimal configuration and
establish a reproducible reference benchmark for future RL algorithms.

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
| `DoorOpening` model | ✅ Complete | Physical door geometry (width, exterior flag, door_id) (Stage 10) |
| `rules.py` | ✅ Complete | Modular rule engine: R1–R5, RuleSet, Violation, ValidationResult (Stage 11) |
| `validator.py` | ✅ Complete | ExitValidator with `check()` and `mask()` APIs (Stage 11) |
| `config/rules_nbc_2016.yaml` | ✅ Complete | Regulatory config with verified/unverified flags (Stage 11) |
| `evaluation.py` | ✅ Complete | `ConfigurationEvaluator`, `EvaluationResult`, `rank_evaluations` (Stage 12) |
| `search.py` | ✅ Complete | `ExitConfigurationSearcher`, `SearchResult`, `search_exit_configurations` (Stage 13) |
| Test suite | ✅ Complete | 561 passing tests across 21 modules |

---

## 4. Current Limitations

- **No dynamic hazards.** Fire, smoke, and hazardous propagation are not yet modelled.
- **No age-based derivation formulas.** The profile architecture supports descriptive metadata (`age`), but physiological/demographic parameter derivation is reserved for a future modeling stage.
- **Single-threaded.** Experiments run sequentially in memory (execution speed is ~0.005s/run).
- **Rule-based.** Reinforcement learning and learned policies are not yet incorporated.

---

## 5. Project Architecture

```
FloorPlan               (config.py)        — grid, walls, DoorOpenings
         │
         ▼
generate_candidate_exits()                 — exterior DoorOpenings only
         │
         ▼
 ExitValidator.mask()   (validator.py)     — partial action mask [R1 only]
         │
         ▼
 ExitConfiguration      (config.py)        — selected exits + widths
         │
         ▼
 ExitValidator.check()  (validator.py)     — full compliance [R1-R5]
         │           (if compliant)
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

---

## Stage 11 — Regulatory & Exit-Placement Constraint Layer

### What the Rules Layer Does

The Stage 11 constraint layer answers:

> *Is this exit configuration acceptable under the encoded regulatory and placement constraints?*

It does NOT simulate evacuation performance — that is done by the Simulation engine.
It encodes structural and regulatory constraints from **NBC 2016 Part 4 (Fire and Life Safety)**
for research and design-assistance purposes.

**IMPORTANT DISCLAIMER:**
This is a research/design-assistance implementation only.
It is **NOT legal certification**.
Results mean *"compliant with the encoded constraints in the selected regulation profile"*,
not *"legally compliant"*.

---

### Key Architectural Distinctions

#### DoorOpening vs CandidateExit vs ExitConfiguration

| Concept | What It Is |
|---|---|
| `DoorOpening` | A physical door or opening in the building (width, exterior flag, position). Not automatically an exit. |
| `CandidateExit` | An exterior DoorOpening that is *eligible* to become an emergency exit. Derived via `generate_candidate_exits()`. Interior doors are excluded. |
| `ExitConfiguration` | A set of selected exits. Each exit maps to a cell and a physical width. |

#### Hard Constraints vs Evacuation Objectives

| Type | Description | Mechanism |
|---|---|---|
| Hard constraint | Regulatory rules (R1-R5): must be satisfied. Illegal configs are rejected. | `validator.check()` returns `is_compliant=False` |
| Evacuation objective | Minimising evacuation time, reducing congestion. Used in future search/RL. | Simulation metrics |

This is critical for future RL: **illegal configurations must be filtered, not penalized**.
The validator provides the feasibility boundary; RL/search operates inside it.

---

### Rules Implemented (R1-R5)

| Rule | Description | NBC Clause | Verified |
|---|---|---|---|
| `R1_VALID_EXIT_POSITION` | Only exterior DoorOpenings may be exits | PATH_RL domain geometric constraint | YES |
| `R2_MINIMUM_EXIT_COUNT` | At least 2 exits required | Part 4 4.4.2.4.3 | NO (conditional rule modelled as universal prototype) |
| `R3_EXIT_SEPARATION` | Min separation = 1/2 floor diagonal (non-sprinklered) | Part 4 egress placement | NO (IBC formulation prototype) |
| `R4_EXIT_WIDTH` | Min physical width 1.0m per exit | Part 4 Table 4 | NO (conditional rule modelled as universal prototype) |
| `R5_TRAVEL_DISTANCE` | Max travel distance to nearest exit (22.5m residential) | Part 4 Table 5 | YES |

**Unverified / configurable values:**
- `R4` width-per-person: intentionally `null` (NBC uses unit-of-width table, not linear value)
- `R5` `cell_size_m = 1.0m`: modelling assumption, not an NBC value
- `R5` DEFAULT occupancy limit: conservative engineering default only

---

### check() API

```python
from evacuation_simulation.simulation.validator import make_validator

validator = make_validator()  # loads NBC 2016 Part 4 rules

result = validator.check(floor_plan, exit_configuration, occupant_scenario)

result.is_compliant   # True / False
result.violations     # list[Violation]

for v in result.violations:
    print(v.rule_id)        # e.g. "R3_EXIT_SEPARATION"
    print(v.message)        # human-readable
    print(v.measured_value) # e.g. 4.2 (m)
    print(v.required_value) # e.g. 7.07 (m)
    print(v.unit)           # e.g. "m"
    print(v.clause)         # e.g. "NBC 2016 Part 4 ..."
    print(v.severity)       # "error" or "warning"
    print(v.relevant_ids)   # list of affected object IDs
```

---

### mask() API

```python
mask = validator.mask(floor_plan, occupant_scenario, partial_config, candidates)
# mask: dict[GridCell, bool]
# True  = candidate may be selected as next exit
# False = candidate is forbidden at this stage
```

**IMPORTANT LIMITATIONS of the partial mask:**
- Only `R1` (valid exterior door) is evaluated per-candidate.
- `R2/R3/R4/R5` require a **complete** configuration and are evaluated by `check()` only.
- A `True` mask value does NOT guarantee the final configuration will be compliant.
- Always call `check()` on the complete configuration.

---

### Regulatory Parameters Audit (NBC 2016 Part 4)

*PATH_RL implements a configurable research prototype of regulatory/placement constraints. Only parameters explicitly verified against the referenced regulation are described as verified regulatory requirements.*

| Value | Amount | Unit | Clause | Verified |
|---|---|---|---|---|
| Min exits | 2 | exits | 4.4.2.4.3 | NO (conditional rule modelled as universal prototype) |
| Exit separation (non-sprinklered) | 1/2 diagonal | fraction | egress placement | NO (IBC formulation prototype) |
| Exit separation (sprinklered) | 1/3 diagonal | fraction | egress placement | NO (IBC formulation prototype) |
| Min exit width | 1.0 | m | Table 4 | NO (conditional rule modelled as universal prototype) |
| Unit of exit width | 0.5 | m | 4.4.2.3 | YES |
| Travel distance RESIDENTIAL | 22.5 | m | Table 5 | YES |
| Travel distance EDUCATIONAL | 22.5 | m | Table 5 | YES |
| Travel distance BUSINESS | 30.0 | m | Table 5 | YES |
| Travel distance ASSEMBLY | 22.5 | m | Table 5 | NO (unverified placeholder for this occupancy) |
| Width per person | null | — | — | NO (intentionally unverified) |
| cell_size_m | 1.0 | m/cell | — | NO (configurable assumption) |
| DEFAULT travel limit | 22.5 | m | — | NO (engineering default only) |

---

### Running the Stage 11 Demo

```bash
python stage11_demo.py
```

This demonstrates:
1. Candidate exterior doors discovered from FloorPlan
2. Interior passages NOT treated as exit candidates
3. A valid exit configuration checked — PASS
4. An invalid configuration checked — structured violations produced
5. Action mask API
6. Machine-readable Pydantic output

---

## Stage 12 — Exit Configuration Evaluation & Objective Layer

### Why Evaluation is Separate from Regulation

Stage 11 (regulation) and Stage 12 (evaluation) address fundamentally
different questions:

| Layer | Question answered |
|---|---|
| **Stage 11** — `validator.py` | *"Can this configuration be considered feasible under our encoded regulatory constraints?"* |
| **Stage 12** — `evaluation.py` | *"How well does this feasible configuration perform when evacuation is actually simulated?"* |

A configuration that passes Stage 11 is **allowed**. Stage 12 determines
how **well** it performs. These two questions must not be conflated.

### What Makes a Configuration Feasible

A configuration is considered **fully feasible** in Stage 12 when two
conditions are both satisfied:

1. It passes all Stage 11 regulatory checks (`is_compliant = True`).
2. All occupants evacuate before the simulation time limit
   (`evacuation_complete = True`).

A configuration that passes regulation but traps agents is flagged
`is_feasible = False` and cannot outrank a configuration that achieves
complete evacuation.

### The Pipeline

```
FloorPlan
    |
generate_candidate_exits()
    |
CandidateExit pool
    |
generate_exit_configurations()
    |
ExitConfiguration[]
    |
ExitValidator.check()       <- Stage 11
    |
INVALID -> Reject (no simulation, no score)
VALID   -> Simulation.run()
               |
          SimulationResult
               |
         EvaluationMetrics
               |
          primary objective score
               |
         EvaluationResult
    |
rank_evaluations()
    |
Ranked list (best first)
```

### Primary Objective

**Minimize total evacuation time.**

```python
score = total_timesteps   # if all agents evacuated
score = None              # if evacuation was incomplete
```

Lower score = better configuration.
`None` score configurations always rank below scored configurations.

### Secondary Metrics

The following metrics are preserved in `EvaluationMetrics` for analysis,
even though they are not part of the primary score:

| Metric | Description |
|---|---|
| `mean_evacuation_time` | Mean timestep at which agents evacuated |
| `max_evacuation_time` | Latest evacuation timestep |
| `mean_waiting_steps` | Mean capacity-blocked waiting steps per agent |
| `max_waiting_steps` | Maximum waiting steps for any single agent |
| `total_waiting_steps` | Sum of all waiting steps |
| `congested_cell_steps` | (cell, timestep) pairs where cell was congested |
| `max_cell_occupancy` | Peak single-cell occupancy |
| `max_occupancy_ratio` | Peak occupancy ratio (occupancy / capacity) |
| `exit_utilization` | Agents evacuated per exit cell |
| `evacuation_rate` | Fraction of agents that evacuated |

These metrics are available for analysis. Future stages may investigate
whether a weighted multi-objective score is warranted.

### How Configurations are Compared

All configurations in a comparison sweep are evaluated against the
**same** FloorPlan, OccupantScenario, and SimulationParameters
(including `random_seed`). Only the `ExitConfiguration` changes.

```python
evaluator = ConfigurationEvaluator()
results = evaluator.evaluate_configurations(
    floor_plan,
    [config_A, config_B, config_C],
    occupant_scenario,
    parameters=SimulationParameters(random_seed=42),
)
ranked = rank_evaluations(results)
```

### Why the Same Occupant Scenario is Required

If different configurations used different agent starting positions or
different random seeds, a configuration might appear better simply
because its agents happened to start closer to exits -- not because
its exit placement is superior. Using identical conditions for every
configuration makes the comparison scientifically meaningful.

### Why Invalid Configurations are Rejected (Not Penalized)

An invalid configuration **does not receive a terrible score** and
remain in the optimization pool. It is **excluded entirely**.

Rationale:

- A numerical penalty is an arbitrary engineering choice that would
  require tuning and could distort search/RL behavior.
- An invalid configuration is not a candidate for real-world use;
  there is no meaningful "how well does it perform" answer.
- Keeping invalid configurations in a scored pool would make it
  impossible to distinguish "very bad valid" from "invalid".

The explicit `is_feasible = False` state is unambiguous and machine-readable.

### Why RL Has Not Been Implemented Yet

Stage 12 establishes the **foundation** for future search and RL work.
The pipeline is:

```
EVALUATE -> COMPARE -> RANK
```

Before introducing a search algorithm or RL, we need to objectively
evaluate and compare exit configurations. Stage 12 provides that capability.

Reinforcement learning requires:

- A defined action space (exit selection)
- A well-defined objective / reward signal
- Deterministic or controlled stochasticity

Stage 12 provides the objective signal. Stage 13 will introduce the
search baseline. Later stages will introduce RL.

No RL libraries, policy networks, or reward functions are introduced here.

### Running the Stage 12 Demo

```bash
python stage12_demo.py
```

Evaluates all combinations of 1-exit and 2-exit configurations on the
bottleneck scenario and prints a ranked comparison table with actual
simulation results.

### API Summary

```python
from evacuation_simulation.simulation.evaluation import (
    ConfigurationEvaluator, EvaluationResult, rank_evaluations,
)

evaluator = ConfigurationEvaluator()

result = evaluator.evaluate(floor_plan, exit_config, occupant_scenario)
result.is_feasible                    # bool
result.score                          # int or None (primary objective)
result.metrics.mean_evacuation_time   # float or None
result.metrics.exit_utilization       # dict[(row, col), int]
result.validation_result.is_compliant # bool (Stage 11 result)

results = evaluator.evaluate_configurations(
    floor_plan, [cfg_A, cfg_B, cfg_C], occupant_scenario,
    parameters=SimulationParameters(random_seed=42),
)
ranked = rank_evaluations(results)
# ranked[0] is the best configuration
```

---

## Stage 13 — Non-RL Exit Configuration Search Baseline

### Purpose of Stage 13

While Stage 11 verifies **"Is this configuration allowed?"** and Stage 12 answers **"How well does this configuration perform?"**, Stage 13 answers:

> **"Among all available feasible exit configurations, which one performs best?"**

Stage 13 implements a deterministic, exhaustive, combinatorial search baseline. It systematically:
1. Generates all possible combinatorial exit configurations of a specified exit count (`choose(N, k)`).
2. Runs Stage 11 regulatory validation (`ExitValidator.check`).
3. Discards/flags invalid configurations without simulating them.
4. Simulates all valid configurations using the Stage 12 evaluation engine (`ConfigurationEvaluator`).
5. Ranks configurations strictly by the Stage 12 primary objective (**minimize total evacuation time**).
6. Returns a structured `SearchResult` containing the optimal configuration, statistics, and top-K rankings.

### Why a Non-RL Baseline is Essential

Stage 13 serves as the **ground-truth reference benchmark** for all future Reinforcement Learning (RL) agents.

```
                Exit Configurations
                       │
                Stage 11 Rules
                       │
                Feasible Options
                       │
             ┌─────────┴─────────┐
             │                   │
       Stage 13 Search       Future RL
       (Non-RL Baseline)         │
             │                   │
        Best Config          RL Config
             │                   │
             └─────────┬─────────┘
                       │
              Same Stage 12
                 Evaluation
                       │
                 Compare Results
```

By establishing a deterministic baseline evaluated under the **exact same Stage 12 objective and conditions**, future RL performance can be quantitatively measured against the true combinatorial optimum.

### Scientific Requirement: Fair Comparison

Every configuration in a search run uses identical conditions:
- Same `FloorPlan` geometry (never mutated)
- Same `OccupantScenario` and agent profiles (never mutated)
- Same `SimulationParameters` (including `random_seed`)

Only the `ExitConfiguration` changes between evaluations.

### Search Space & Early Filtering Breakdown

The search explicitly tracks and reports:
- **Total Configurations**: Combinations generated (`choose(N, k)`)
- **Valid Configurations**: Configurations passing all regulatory rules (R1–R5)
- **Invalid Configurations**: Configurations rejected by the validator
- **Evaluated Configurations**: Feasible configurations actually simulated
- **Search Time & Throughput**: Wall-clock time and evaluations per second

Invalid configurations are rejected **before simulation** to avoid unnecessary computation.

### Search API

```python
from evacuation_simulation.simulation.search import (
    ExitConfigurationSearcher,
    SearchResult,
    search_exit_configurations,
)

# Object-oriented API
searcher = ExitConfigurationSearcher()
result: SearchResult = searcher.search(
    floor_plan=floor_plan,
    candidates=candidates,
    occupant_scenario=occupant_scenario,
    exit_count=2,
    parameters=SimulationParameters(random_seed=42),
    top_k=3,
)

# Structured result attributes
result.best_configuration       # ExitConfiguration (optimal winner)
result.best_evaluation          # EvaluationResult (winner metrics & score)
result.total_configurations     # Total generated combinations
result.valid_configurations     # Number of compliant configurations
result.invalid_configurations   # Number of rejected configurations
result.evaluated_configurations # Number of simulated configurations
result.search_time_seconds      # Total search execution time
result.evaluations_per_second   # Evaluation throughput
result.top_k_evaluations        # Top K ranked EvaluationResults
result.to_dict()                # JSON-serializable dictionary summary
```

### Running the Stage 13 Demo

```bash
python stage13_demo.py
```

Runs the non-RL exhaustive search baseline on `scenarios/bottleneck.json`, displays the search space breakdown, top 3 rankings, and full winning metrics.

---

## Stage 14: RL Exit-Placement Formulation

Stage 14 formulates the exit-placement problem as a sequential decision process suitable for Reinforcement Learning. 

### What the RL Agent Controls
The agent sequentially selects a subset of emergency exits from a pool of valid candidate exits on a given floor plan.

### What the RL Agent Does NOT Control
The agent is not responsible for routing or moving individual people. The existing shortest-path simulation (Stage 12) still governs occupant movement.

### State Representation
The state (observation) is a structured representation designed for generalization:
- `grid_shape`: (rows, cols)
- `candidate_exits`: Properties of available candidates (positions, widths).
- `selected_candidate_indices`: Which exits the agent has already chosen.
- `action_mask`: Which candidates are still legal to select.
- `exits_remaining`: How many more exits the agent must choose.
- `total_agents`: Occupant count.

### Action Representation
A discrete action representing the integer index of the candidate exit being selected.

### Action Masking
The environment masks out invalid actions (returns `False` in the action mask) if:
- The candidate exit has already been selected.
- Selecting it violates the partial regulatory constraints enforced by Stage 11 (e.g., minimum physical width requirements per door).

### Episode Termination
An episode terminates when:
1. **Success**: The required number of exits is reached.
2. **Dead-End**: No valid actions remain in the mask before reaching the required exit count.
3. **Invalid Action**: The agent selects an action that is masked or out of bounds.

### Reward Objective
The primary objective is to **minimize total evacuation time**.
- Valid configuration: Reward = `-total_evacuation_time`
- Invalid/Non-compliant configuration: Strong penalty (e.g., `-2000`)
- Dead-end or invalid action: Strongest penalty (e.g., `-3000`)

### Relationship with Stage 11
Stage 11 remains the authoritative source for regulatory compliance. It provides the partial `action_mask` during the episode and the final `is_compliant` feasibility check at the end. The RL environment never invents new rules and never falsely reports an invalid layout as valid.

### Relationship with Stage 12
Stage 12 is the simulator and evaluator. Once a valid configuration is fully selected, the environment calls Stage 12 to run the simulation and extract the evacuation metrics (which inform the reward).

### Relationship with Stage 13
The RL formulation is perfectly compatible with the Stage 13 deterministic exhaustive search baseline. Both evaluate configurations using the exact same metrics and random seed, meaning their outputs are directly comparable.

### Why Training is Not Included Yet
This stage focuses exclusively on the environment formulation, safety, and correctness of the problem definition. Setting up the state/action/reward structure rigorously ensures that when RL algorithms are applied later, they learn the correct objective without bypassing constraints or relying on leaky abstractions.
