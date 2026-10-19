# System Execution Flow

Summary of execution paths, function call sequences, and data movement across core subsystems.

---

## 1. Single Simulation Run Flow (`python main.py`)

Execution flow for running a standard scenario simulation from the command line:

```
main.py
  └─► evacuation_simulation.main.main()
        │
        ├─► SimulationConfig.from_json(scenario_path)             [config.py]
        │     └─► Parses FloorPlan, ExitConfiguration, OccupantScenario
        │
        ├─► Simulation.from_config(config, strategy_name)         [simulation.py]
        │     ├─► Environment(floor_plan, exit_config)            [environment.py]
        │     ├─► NavigationGraph(environment)                    [pathfinding.py]
        │     │     └─► Precomputes BFS distance & next-hop maps
        │     ├─► OccupancyMap(rows, cols, cell_capacity)         [occupancy.py]
        │     ├─► Seeds Agents with AgentProfile (speed, delay)   [agent.py]
        │     ├─► MovementController(env, graph, occupancy)       [movement.py]
        │     └─► MetricsCollector(total_agents)                  [metrics.py]
        │
        ├─► Simulation.run()                                      [simulation.py]
        │     └─► While active agents remain and step < max_steps:
        │           ├─► MovementController.step(active_agents)    [movement.py]
        │           │     ├─ Filter eligible agents (delay elapsed)
        │           │     ├─ Query target cell from NavigationGraph
        │           │     ├─ Resolve collisions via OccupancyMap
        │           │     └─ Update agent states (ACTIVE -> EVACUATED)
        │           └─► MetricsCollector.record_step(...)         [metrics.py]
        │
        └─► MetricsCollector.summary()                            [metrics.py]
              └─► Prints summary table (evacuation time, throughput, survival)
```

---

## 2. Exhaustive Search Baseline Flow (`stage13_demo.py`)

Execution flow for combinatorial evaluation of all candidate exit door combinations:

```
stage13_demo.py
  └─► search_exit_configurations(config, exit_count)              [search.py]
        │
        ├─► generate_candidate_exits(floor_plan)                 [generation.py]
        │     └─ Filters DoorOpenings where exterior == True
        │
        ├─► ExitValidator.mask(floor_plan, candidates)            [validator.py]
        │     └─ Filters candidates violating NBC Rule R1 (travel distance)
        │
        ├─► Iterate all valid k-combinations of candidate exits:
        │     ├─ Build candidate ExitConfiguration
        │     ├─ ExitValidator.check(floor_plan, exit_config)     [validator.py]
        │     │     └─ Validates rules R1–R5 -> ValidationResult
        │     │
        │     └─ If compliant:
        │           └─► ConfigurationEvaluator.evaluate(...)       [evaluation.py]
        │                 ├─ Assembles temporary SimulationConfig
        │                 ├─ Executes Simulation.run()
        │                 └─ Returns EvaluationResult(evac_time, score)
        │
        └─► rank_evaluations(results)                              [evaluation.py]
              └─► Returns SearchResult with optimal configuration & rankings
```

---

## 3. RL Environment Episode Flow (`ExitPlacementEnv`)

Execution flow for formulating exit placement as an MDP step loop:

```
ExitPlacementEnv.reset(config, exit_count)                        [rl_environment.py]
  ├─ Parses FloorPlan and generates CandidateExit list
  ├─ Computes initial action_mask via ExitValidator.mask()
  └─ Returns RLObservation (grid shape, candidate states, mask, exits remaining)

Agent step loop:
ExitPlacementEnv.step(action)                                     [rl_environment.py]
  ├─ Validates action against action_mask
  │     └─ If invalid: returns PENALTY_INVALID_ACTION, terminated=True
  ├─ Marks candidate door as active
  ├─ Updates action_mask (disables selected and unreachable doors)
  ├─ Decrements exits_remaining
  │
  └─ If exits_remaining == 0:
        ├─ Assembles final ExitConfiguration
        ├─ ExitValidator.check(...)                               [validator.py]
        │     └─ If violations: reward = PENALTY_INVALID_FINAL_CONFIG
        ├─ ConfigurationEvaluator.evaluate(...)                   [evaluation.py]
        │     └─ Runs evacuation simulation to completion
        ├─ Reward = -1 * total_evacuation_time
        └─ Terminated = True
```

---

## 4. End-to-End Synthetic Evacuation Pipeline (`run_e2e_experiment.py`)

Execution flow for generating a building, placing occupants, enforcing NBC constraints, and running evacuation:

```
run_e2e_experiment.py
  └─► run_synthetic_evacuation_experiment(...)
        │
        ├─► CommercialLayoutGenerator.generate(...)              [commercial_generator.py]
        │     └─ Generates FloorPlan with perimeter corridor, offices, candidate doors
        │
        ├─► ExitValidator.check(floor_plan, exit_config)         [validator.py]
        │     └─ Validates NBC 2016 Part 4 rules R1-R5
        │
        ├─► OccupantSeeder.seed_occupants(...)                   [occupant_seeder.py]
        │     └─ Populates room spaces with heterogeneous agents
        │
        ├─► ConfigurationEvaluator.evaluate(...)                 [evaluation.py]
        │     └─ Runs BFS-directed multi-agent evacuation simulation
        │
        └─► Formats structured JSON metrics (evacuation time, throughput, survival)
```

---

## 5. Architectural Floor-Plan Ingestion Flow (`importer.py`)

Execution flow for converting CubiCasa5K vector floor plans into simulation grids:

```
import_cubicasa_plan(svg_path, resolution)                       [importer.py]
  │
  ├─► CubiCasaParser.parse_file(svg_path)                         [cubicasa_parser.py]
  │     └─ Extracts SVG polygons: spaces, walls, columns, doors
  │
  ├─► ScaleEstimator.estimate_scale(svg_data)                     [scale_estimator.py]
  │     └─ Infers pixels-to-meters ratio from dimension lines
  │
  ├─► DoorClassifier.classify_doors(svg_data)                     [door_classifier.py]
  │     └─ Classifies: INTERIOR, BALCONY (ignored), POTENTIAL_EXTERIOR
  │
  ├─► GridDiscretizer.discretize(svg_data, scale)                 [grid_discretizer.py]
  │     ├─ Rasterizes wall polygons to 2D boolean array
  │     ├─ Carves door thresholds through walls to prevent blockage
  │     └─ Maps room spaces to grid coordinates
  │
  └─► Constructs and returns:
        ├─ FloorPlan (grid dimensions, walls, carved DoorOpenings)
        ├─ list[CandidateExit] (exterior openings)
        └─ ReviewManifest (audit log of classifications and scale)
```

---

## 6. Exhaustive Exit-Configuration Baseline Pipeline (`run_baseline_experiment.py`)

Execution flow for evaluating all candidate exit combinations on a fixed building and occupant scenario:

```
run_baseline_experiment.py
  └─► run_baseline_experiment(...)
        │
        ├─► CommercialLayoutGenerator.generate(...)              [commercial.py]
        │     └─ Generates FloorPlan with exterior candidate doors
        │
        ├─► generate_candidate_exits(floor_plan)                 [generation.py]
        │     └─ Extracts candidate exterior doors
        │
        ├─► OccupantSeeder.seed(...)                             [occupancy_seeder.py]
        │     └─ Generates ONE fixed OccupantScenario for fair comparison
        │
        ├─► ExitConfigurationSearcher.search(...)                [search.py]
        │     ├─ Enumerates all k-combinations (generate_exit_configurations)
        │     │
        │     ├─ For each configuration:
        │     │     ├─ ExitValidator.check(floor_plan, cfg, occupants) [validator.py]
        │     │     │     └─ If invalid: mark infeasible, record violations, skip simulation
        │     │     │
        │     │     └─ If compliant:
        │     │           └─ ConfigurationEvaluator.evaluate(...)     [evaluation.py]
        │     │                 └─ Executes simulation to completion
        │     │
        │     └─ rank_evaluations(results)                            [evaluation.py]
        │           └─ Sorts feasible before infeasible, ordered by total evacuation time
        │
        └─► Exports structured JSON benchmark to experiment_output/baseline/
```

