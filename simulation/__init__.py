"""
evacuation_simulation.simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Core simulation package for the 2D multi-agent evacuation system.

Submodules
----------
config       - Pydantic models for scenario configuration
environment  - Grid-based environment (cells, walls, exits)
agent        - Agent entity (position, state, identity)
pathfinding  - Navigation graph construction over traversable cells
occupancy    - Dynamic cell-occupancy map (Stage 3)
movement     - MovementRequest, validation, conflict resolution, application
strategy     - Concrete movement strategies (ShortestPathStrategy, ...)
simulation   - Discrete-timestep loop coordinating all components
metrics      - Outcome statistics collection and computation
experiment   - Scenario runner, parameter sweeps, and export (Stage 5)
profile      - Extensible heterogeneous agent profile system (Stage 6)
rules        - Modular rule engine: Rule, RuleSet, Violation, ValidationResult (Stage 11)
validator    - ExitValidator with check() and mask() APIs (Stage 11)
"""

from evacuation_simulation.simulation.agent import Agent, AgentState
from evacuation_simulation.simulation.config import (
    AgentConfig,
    AgentProfileConfig,
    DoorOpening,
    ExitConfiguration,
    FloorPlan,
    OccupantScenario,
    SimulationConfig,
)
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.generation import (
    CandidateExit,
    generate_candidate_exits,
    generate_exit_configurations,
)
from evacuation_simulation.simulation.profile import AgentProfile
from evacuation_simulation.simulation.experiment import (
    AggregateStats,
    ExperimentResult,
    RunRecord,
    SweepPoint,
    aggregate_results,
    export_csv,
    export_json,
    run_repeated,
    run_scenario,
    run_sweep,
)
from evacuation_simulation.simulation.metrics import MetricsCollector, SimulationResult
from evacuation_simulation.simulation.movement import (
    MovementRequest,
    RejectionReason,
    apply_movements,
    resolve_conflicts,
    validate_request,
)
from evacuation_simulation.simulation.occupancy import OccupancyMap
from evacuation_simulation.simulation.pathfinding import NavigationGraph
from evacuation_simulation.simulation.simulation import (
    AgentStepResult,
    MovementStrategy,
    NullMovementStrategy,
    Simulation,
    TimestepTrace,
)
from evacuation_simulation.simulation.strategy import ShortestPathStrategy
from evacuation_simulation.simulation.rules import (
    Rule,
    RuleSet,
    ValidationResult,
    Violation,
    build_nbc_2016_ruleset,
    load_regulatory_config,
    R1ValidExitPosition,
    R2MinimumExitCount,
    R3ExitSeparation,
    R4ExitWidth,
    R5TravelDistance,
)
from evacuation_simulation.simulation.validator import ExitValidator, make_validator
from evacuation_simulation.simulation.evaluation import (
    ConfigurationEvaluator,
    EvaluationMetrics,
    EvaluationResult,
    rank_evaluations,
)
from evacuation_simulation.simulation.search import (
    ExitConfigurationSearcher,
    SearchResult,
    search_exit_configurations,
)
from evacuation_simulation.simulation.rl_environment import (
    ExitPlacementEnv,
    RLObservation,
)

__all__ = [
    # config & geometry (Stage 8 & 10)
    "SimulationConfig",
    "AgentConfig",
    "AgentProfileConfig",
    "DoorOpening",
    "FloorPlan",
    "ExitConfiguration",
    "OccupantScenario",
    # candidate exit generation (Stage 9 & 10)
    "CandidateExit",
    "generate_candidate_exits",
    "generate_exit_configurations",
    # environment
    "Environment",
    # agent & profile
    "Agent",
    "AgentProfile",
    # agent state
    "AgentState",
    # pathfinding
    "NavigationGraph",
    # occupancy (Stage 3)
    "OccupancyMap",
    # movement
    "MovementRequest",
    "RejectionReason",
    "validate_request",
    "resolve_conflicts",
    "apply_movements",
    # strategy
    "ShortestPathStrategy",
    # simulation
    "AgentStepResult",
    "MovementStrategy",
    "NullMovementStrategy",
    "Simulation",
    "TimestepTrace",
    # metrics
    "MetricsCollector",
    "SimulationResult",
    # experiment (Stage 5)
    "AggregateStats",
    "ExperimentResult",
    "RunRecord",
    "SweepPoint",
    "aggregate_results",
    "export_csv",
    "export_json",
    "run_repeated",
    "run_scenario",
    "run_sweep",
    # rules & validator (Stage 11)
    "Rule",
    "RuleSet",
    "ValidationResult",
    "Violation",
    "build_nbc_2016_ruleset",
    "load_regulatory_config",
    "R1ValidExitPosition",
    "R2MinimumExitCount",
    "R3ExitSeparation",
    "R4ExitWidth",
    "R5TravelDistance",
    "ExitValidator",
    "make_validator",
    # evaluation & ranking (Stage 12)
    "ConfigurationEvaluator",
    "EvaluationMetrics",
    "EvaluationResult",
    "rank_evaluations",
    # non-rl search baseline (Stage 13)
    "ExitConfigurationSearcher",
    "SearchResult",
    "search_exit_configurations",
    # RL exit-placement formulation (Stage 14A)
    "ExitPlacementEnv",
    "RLObservation",
]

