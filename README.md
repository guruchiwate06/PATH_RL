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

## 3. Current Scope (Stage 1 — Foundation)

This stage establishes the software architecture only.  The following are
implemented:

| Component | Status | Notes |
|---|---|---|
| `SimulationConfig` | ✅ Complete | Pydantic-validated JSON scenario loader |
| `Environment` | ✅ Complete | Grid with walls, exits, neighbour queries |
| `Agent` | ✅ Complete | Stateful entity with lifecycle states |
| `NavigationGraph` | ✅ Complete | NetworkX graph over traversable cells |
| `Simulation` | ✅ Complete | Discrete timestep loop + strategy protocol |
| `MetricsCollector` | ✅ Complete | Per-agent event recording + result computation |
| `NullMovementStrategy` | ✅ Complete | No-op stub so the loop can run |
| Test suite | ✅ Complete | 50+ assertions across all modules |
| Basic scenario JSON | ✅ Complete | `scenarios/basic_building.json` |

---

## 4. Current Limitations

- **Agents do not move.** The `NullMovementStrategy` is the only strategy
  currently implemented.  Evacuation only occurs if an agent starts on an
  exit cell.
- **No pathfinding algorithm.** The `NavigationGraph.shortest_path` method
  uses NetworkX's built-in BFS.  A* with custom heuristics is planned.
- **No dynamic hazards.** Fire, smoke, and other environmental changes are
  not yet modelled.
- **No multi-agent interaction.** Congestion, collision avoidance, and social
  force models are not implemented.
- **No visualization.** Matplotlib rendering is not yet connected.
- **Single-threaded.** No parallelism.

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
    Simulation          (simulation.py)    — timestep loop + strategy
         │
         ▼
 MetricsCollector       (metrics.py)       — event recording + aggregation
```

**Design rules enforced:**
- Dependency direction is strictly top-down (no circular imports).
- `config.py` imports nothing from other simulation submodules.
- Visualization is entirely absent from core logic.
- All mutation of agent state happens through named methods.
- Random state is isolated in a seeded `numpy.random.Generator`.

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
│   ├── simulation.py      ← Timestep loop + MovementStrategy protocol
│   └── metrics.py         ← MetricsCollector + SimulationResult
│
├── scenarios/
│   └── basic_building.json   ← Example 10×10 scenario
│
├── tests/
│   ├── conftest.py
│   ├── test_environment.py
│   ├── test_agent.py
│   ├── test_pathfinding.py
│   ├── test_simulation.py
│   └── test_metrics.py
│
├── main.py
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

```bash
python main.py
```

To use a custom scenario file:

```bash
python main.py --scenario path/to/your_scenario.json
```

---

## 8. Running the Test Suite

```bash
# Run all tests with verbose output
pytest evacuation_simulation/tests/ -v

# Run with coverage report
pytest evacuation_simulation/tests/ -v --cov=evacuation_simulation/simulation --cov-report=term-missing

# Run a specific test module
pytest evacuation_simulation/tests/test_pathfinding.py -v
```

---

## 9. Scenario Configuration Format

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

## 10. Extending the System (Future Stages)

The architecture is designed to be extended without modifying core components.

| Stage | Planned Addition |
|---|---|
| Stage 2 | A* pathfinding strategy; `ShortestPathStrategy` |
| Stage 3 | Cardinal + diagonal movement; congestion cost |
| Stage 4 | Dynamic hazards (fire, smoke cells) |
| Stage 5 | Heterogeneous agents (speed, mobility, response time) |
| Stage 6 | Multi-scenario batch runner + statistical analysis |
| Stage 7 | Matplotlib grid visualizer (outside core) |
| Stage 8 | Reinforcement learning integration (optional) |

New movement strategies implement the `MovementStrategy` protocol and are
injected into `Simulation(config, movement_strategy=...)` with no changes
to existing code.

---

## 11. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.26 | Grid arrays, random generation |
| `networkx` | ≥3.2 | Navigation graph, path algorithms |
| `pydantic` | ≥2.5 | Scenario configuration validation |
| `pytest` | ≥8.0 | Test runner |
| `pytest-cov` | ≥4.1 | Test coverage reporting |
