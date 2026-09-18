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
movement     - MovementRequest, validation, conflict resolution, application
strategy     - Concrete movement strategies (ShortestPathStrategy, ...)
simulation   - Discrete-timestep loop coordinating all components
metrics      - Outcome statistics collection and computation
"""

from evacuation_simulation.simulation.agent import Agent, AgentState
from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.metrics import MetricsCollector, SimulationResult
from evacuation_simulation.simulation.movement import (
    MovementRequest,
    apply_movements,
    resolve_conflicts,
    validate_request,
)
from evacuation_simulation.simulation.pathfinding import NavigationGraph
from evacuation_simulation.simulation.simulation import (
    MovementStrategy,
    NullMovementStrategy,
    Simulation,
)
from evacuation_simulation.simulation.strategy import ShortestPathStrategy

__all__ = [
    # config
    "SimulationConfig",
    # environment
    "Environment",
    # agent
    "Agent",
    "AgentState",
    # pathfinding
    "NavigationGraph",
    # movement
    "MovementRequest",
    "validate_request",
    "resolve_conflicts",
    "apply_movements",
    # strategy
    "ShortestPathStrategy",
    # simulation
    "MovementStrategy",
    "NullMovementStrategy",
    "Simulation",
    # metrics
    "MetricsCollector",
    "SimulationResult",
]
