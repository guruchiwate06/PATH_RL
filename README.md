# 2D Evacuation Simulation & Exit Placement RL Environment

A computational simulation framework and reinforcement learning environment for discrete-time 2D multi-agent evacuation dynamics and regulatory-constrained architectural exit placement.

---

## Overview

The project models building evacuation dynamics on 2D grids and provides an optimization testbed for exit door placement. It serves two interconnected purposes:
1. **Multi-Agent Simulation**: Simulates discrete-time agent navigation, congestion, reaction delays, and bottleneck formation toward building exits.
2. **Exit Placement Optimization**: Evaluates and learns optimal exterior exit door placements under National Building Code of India (NBC 2016 Part 4) regulatory constraints using combinatorial search and reinforcement learning.

---

## Technology Stack

- **Language**: Python 3.10+ (tested on Python 3.13)
- **Core Scientific & Graph Libraries**:
  - `numpy`: Grid array representation and numeric operations.
  - `networkx`: Navigation graph construction and shortest-path distance field generation.
  - `pydantic` (v2): Data modeling, schema validation, and JSON serialization.
- **Testing & Verification**: `pytest`, `pytest-cov`.
- **Visualization**:
  - Web Visualizer: Standalone HTML5 Canvas / Vanilla JavaScript (`visualize.html`).
  - Headless / Export: `matplotlib` snapshot and GIF generation (`visualize.py`).
- **Data Ingestion (CubiCasa5K Converters)**: `shapely`, `svgpathtools` (vector geometry parsing and rasterization).

---

## Architecture

The codebase separates static geometry, dynamic routing, regulatory rules, and agent logic:

- **Static Geometry (`FloorPlan`)**: Immutable building layout comprising grid dimensions, wall cells, space labels, and physical door openings (`DoorOpening`).
- **Dynamic Routing (`ExitConfiguration`)**: Specific assignment of active exterior exits selected from candidate doors.
- **Agents & Movement**:
  - `AgentProfile`: Heterogeneous occupant attributes (speed, reaction delay).
  - `OccupancyMap`: Enforces cell capacity constraints and tracks spatial density.
  - `NavigationGraph`: BFS shortest-path distance fields precomputed across traversable cells.
  - `MovementController`: Timestep progression, agent eligibility, priority ordering, and conflict resolution.
- **Regulatory Constraint Layer (`ExitValidator`, `rules.py`)**:
  - Implements NBC 2016 Part 4 rules: R1 (travel distance), R2 (minimum exit count), R3 (aggregate door width), R4 (exit separation), R5 (stair discharge).
  - Supports `.mask()` for invalid action pruning during search/RL and `.check()` for post-placement compliance auditing.
- **Evaluation & Search**:
  - `ConfigurationEvaluator`: Runs simulation on compliant configurations and records metrics.
  - `ExitConfigurationSearcher`: Exhaustive combinatorial baseline for exit selection.
- **RL Environment (`ExitPlacementEnv`)**:
  - Formulates exit placement as a sequential Markov Decision Process (MDP) with action masking and reward $R = -\text{total\_evacuation\_time}$.

---

## Project Status

- **Completed & Verified**:
  - Core 2D grid simulation engine, multi-agent movement, and metrics collection.
  - Pydantic v2 decoupled configuration schema (`FloorPlan` vs `ExitConfiguration`).
  - NBC 2016 Part 4 regulatory validation layer (Rules R1–R5) with action masking.
  - Exhaustive search baseline for optimal exit configuration.
  - Gymnasium-style sequential decision RL environment (`ExitPlacementEnv`).
  - CubiCasa5K architectural SVG floor-plan ingestion pipeline.
  - Interactive HTML5 canvas visualizer and command-line experiment runner.
  - **End-to-end synthetic evacuation pipeline** (`run_e2e_experiment.py`): connects layout generator → NBC validator → occupant seeder → BFS simulation → structured results.
  - **Exhaustive exit-configuration baseline** (`run_baseline_experiment.py`, Stage 17): evaluates all candidate exit combinations against fixed building and occupant scenario, enforcing hard NBC constraints and ranking by evacuation time. 633 tests passing.
- **In Progress / Planned**:
  - Policy training baseline using standard RL algorithms (PPO/DQN).
  - Multi-story 3D IFC/BIM extraction pipeline (`scratch/kit_ifc`).
  - Dynamic hazard propagation (smoke and fire spread).

---

## How to Run

### 1. Run a Standard Simulation
Run the default scenario (`basic_building.json`) from the repository root:
```bash
python main.py
```

Run a specific scenario:
```bash
python main.py --scenario scenarios/bottleneck.json
```

### 2. Interactive Web Visualizer
Launch the local visualizer server:
```bash
run_visualizer.bat
# Or manually:
python stage11_web.py
```
Open `http://localhost:8000/visualize.html` in your browser.

### 3. End-to-End Synthetic Evacuation Experiment
Run a complete pipeline from layout generation to evacuation results:
```bash
python run_e2e_experiment.py
# Custom parameters:
python run_e2e_experiment.py --seed 13 --occupants 20 --rows 20 --cols 30
```
Results are saved as JSON to `experiment_output/`.

> **Note**: The NBC 2016 R5 travel-distance rule (22.5m limit) constrains which building sizes and occupant placements are compliant. The default 20×30 grid (15m × 10m) with 2 exits and 20 occupants is verified compliant. Larger buildings require proportionally more exits.

### 4. Exhaustive Exit-Configuration Baseline (Stage 17)
Run the combinatorial baseline search over all feasible exit configurations:
```bash
python run_baseline_experiment.py --seed 42 --occupants 20
# Multi-seed baseline suite:
python run_baseline_experiment.py --seeds 42,20,32,43
```
Results are saved as structured JSON benchmarks to `experiment_output/baseline/`.

### 5. Multi-Seed Batch Experiments
Run parameter sweeps across multiple seeds with aggregate statistics:
```bash
python main.py --experiment --seeds 1,2,3,4,5 --output-dir experiment_output/
```

### 6. Exhaustive Search Baseline Demo
Run the Stage 13 scenario baseline optimizer:
```bash
python stage13_demo.py
```

### 7. Reinforcement Learning Environment Demo
Run the sequential exit-placement MDP demo:
```bash
python stage14_demo.py
```

### 7. Run Test Suite
```bash
python -m pytest tests
```
