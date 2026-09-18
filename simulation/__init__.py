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
simulation   - Discrete-timestep loop coordinating environment and agents
metrics      - Outcome statistics collection and computation
"""

from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.agent import Agent, AgentState
from evacuation_simulation.simulation.pathfinding import NavigationGraph
from evacuation_simulation.simulation.simulation import Simulation
from evacuation_simulation.simulation.metrics import MetricsCollector

__all__ = [
    "SimulationConfig",
    "Environment",
    "Agent",
    "AgentState",
    "NavigationGraph",
    "Simulation",
    "MetricsCollector",
]
