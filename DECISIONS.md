# Architecture & Technical Decisions

Record of major architectural, algorithmic, and data-handling decisions for the evacuation simulation and RL environment.

---

## 1. Decoupling Geometry from Exit Configuration

- **Decision**: Separate static building geometry (`FloorPlan`) from mutable exit selections (`ExitConfiguration`).
- **Why**: Enables search algorithms, parameter sweeps, and RL agents to test multiple exit combinations on the same building without re-parsing, duplicating, or mutating static walls and doors.
- **Trade-off**: Requires a dynamic merge step and validation check before each simulation run.

---

## 2. Discrete 2D Grid Representation over Continuous Space

- **Decision**: Represent layouts as uniform 2D grids (default 0.5m x 0.5m cells) with discrete timestep execution.
- **Why**: Provides high computational throughput (~0.005s per evacuation episode), exact spatial cell capacity enforcement via `OccupancyMap`, and a natural discrete action/observation space for RL.
- **Trade-off**: Sacrifices micro-level continuous crowd dynamics, diagonal edge sliding, and sub-meter body collisions.

---

## 3. Pydantic v2 for Scenario Schemas and Serialization

- **Decision**: Use Pydantic v2 `BaseModel` for configuration classes (`SimulationConfig`, `FloorPlan`, `ExitConfiguration`, `DoorOpening`).
- **Why**: Guarantees strict type checking, robust nested validation, and straightforward JSON export/import for reproducible experiment configs.
- **Trade-off**: Slight schema validation overhead at load time (negligible for batch simulations).

---

## 4. BFS Distance Fields for Agent Pathfinding

- **Decision**: Compute BFS distance-to-exit fields on a `NavigationGraph` (NetworkX) at episode start rather than computing per-agent paths dynamically each step.
- **Why**: Allows $O(1)$ next-cell navigation lookup per agent during stepping. Extremely fast, fully deterministic, and reproducible across runs.
- **Trade-off**: Agents follow static shortest routes; they do not dynamically detour around transient human congestion or bottlenecks.

---

## 5. Two-Tier Regulatory Constraint Enforcement (NBC 2016 Part 4)

- **Decision**: Implement rules R1–R5 inside a standalone `ExitValidator` offering both `.mask()` (action filtering) and `.check()` (full post-configuration audit).
- **Why**: Rule R1 (maximum travel distance to exit) can be computed per candidate door to prune invalid placements during generation, while composite rules (R2 occupant load, R3 aggregate width, R4 exit separation, R5 stair discharge) require a complete configuration to evaluate.
- **Trade-off**: Increases validator complexity, requiring separate mask generation and final configuration auditing logic.

---

## 6. High-Level MDP for RL (Exit Placement vs Agent Navigation)

- **Decision**: Formulate the RL environment (`ExitPlacementEnv`) at the building design level: the RL agent selects candidate exit doors over $k$ sequential steps, while internal crowd evacuation is governed by the deterministic movement engine.
- **Why**: Directly targets the design optimization goal (optimal exit placement) rather than micro-navigation policy learning, keeping the RL action space small, discrete, and tractable.
- **Trade-off**: Environment step latency is dominated by the full evacuation simulation executed at terminal steps.

---

## 7. Zero-Dependency Web Visualizer (HTML5 Canvas)

- **Decision**: Implement an interactive browser-based visualizer using Vanilla HTML5 and Canvas (`visualize.html`) served via Python's standard `http.server`, accompanied by headless Matplotlib snapshot exports.
- **Why**: Avoids heavy GUI framework dependencies (PyQt, Tkinter, Pygame) and runs smoothly across any OS and headless dev environments.
- **Trade-off**: Requires exporting simulation trajectories to JSON format or streaming step frames over a local server.

---

## 8. Reproducible Exhaustive Baseline on Fixed Layout & Occupants (Stage 17)

- **Decision**: Benchmark exit placement optimization by evaluating all feasible combinatorial exit configurations against an identical synthetic layout and fixed occupant scenario, ranking by total evacuation time.
- **Why**: Provides an unambiguous, reproducible ground-truth reference benchmark to measure whether subsequent RL policies achieve or approximate optimal building evacuation safety.
- **Trade-off**: Combinatorial scaling $\binom{N}{k}$ is feasible for building candidate doors ($N \le 12, k \le 4$), but requires heuristics/RL when scaling to massive architectural spaces.

---

## 9. Action-Masked Policy Optimization for Regulatory Exit Placement (Stage 18)

- **Decision**: Formulate policy learning via a Masked Actor-Critic network where invalid/illegal actions are zeroed in the action distribution, trained across diverse procedural layouts with terminal rewards $R = -\text{total\_evacuation\_time}$ and non-compliance penalties ($-2000$).
- **Why**: Enforces hard safety constraints during action selection without trial-and-error exploration of illegal doors, focusing policy capacity on global multi-exit coordination and evacuation time minimization.
- **Trade-off**: Requires computing the per-step legal action mask `action_mask()` from the validator.


