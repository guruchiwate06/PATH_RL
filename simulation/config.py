"""
config.py
---------
Pydantic models that define the schema for scenario configuration files.

Scenario configuration is the authoritative source of truth for initial
conditions.  All other components receive a ``SimulationConfig`` instance
rather than reading files directly — this keeps I/O concerns isolated.

Design notes
~~~~~~~~~~~~
- All fields have sensible defaults so that unit tests can construct minimal
  configs without providing a JSON file.
- Validation is handled by Pydantic; downstream code can trust that values
  are already in the correct ranges and types.
- This module has **no imports** from any other simulation submodule, making
  it safe to import anywhere without creating circular dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Primitive coordinate type alias
# ---------------------------------------------------------------------------

# A grid cell is addressed as (row, col) — both zero-indexed integers.
GridCell = tuple[int, int]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class GridConfig(BaseModel):
    """Dimensions of the simulated grid."""

    rows: Annotated[int, Field(ge=2, le=1000, description="Number of rows (height)")] = 10
    cols: Annotated[int, Field(ge=2, le=1000, description="Number of columns (width)")] = 10

    @model_validator(mode="after")
    def grid_must_have_area(self) -> "GridConfig":
        if self.rows * self.cols < 4:
            raise ValueError("Grid must contain at least 4 cells.")
        return self


class AgentProfileConfig(BaseModel):
    """
    Configuration for an individual agent's profile.

    Attributes
    ----------
    speed:
        Movement frequency at cell/timestep resolution (0.0 < speed <= 1.0).
        Default is 1.0.
    reaction_delay:
        Initial timesteps to delay before beginning movement (>= 0).
        Default is 0.
    age:
        Optional descriptive age attribute for future derivation models.
    input_attributes:
        Optional dictionary of descriptive metadata for future models.
    """

    speed: Annotated[
        float,
        Field(
            gt=0.0,
            le=1.0,
            description="Movement frequency at cell/timestep resolution (0.0 < speed <= 1.0)",
        ),
    ] = 1.0

    reaction_delay: Annotated[
        int,
        Field(
            ge=0,
            description="Initial timesteps to remain stationary before starting movement",
        ),
    ] = 0

    age: Optional[
        Annotated[
            int,
            Field(ge=0, le=150, description="Optional descriptive age attribute"),
        ]
    ] = None

    input_attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional descriptive metadata for future parameterization models",
    )


class AgentConfig(BaseModel):
    """Initial placement and profile of a single agent."""

    agent_id: str = Field(..., description="Unique identifier for this agent")
    row: Annotated[int, Field(ge=0, description="Starting row (zero-indexed)")] = 0
    col: Annotated[int, Field(ge=0, description="Starting column (zero-indexed)")] = 0
    profile: Optional[AgentProfileConfig] = Field(
        default=None,
        description="Optional individual profile for heterogeneous agent capabilities",
    )


class SimulationParameters(BaseModel):
    """Tunable parameters controlling simulation execution."""

    max_timesteps: Annotated[
        int,
        Field(ge=1, le=100_000, description="Hard upper limit on simulation steps"),
    ] = 500

    random_seed: int | None = Field(
        default=None,
        description="Seed for reproducible runs. None means non-deterministic.",
    )

    default_cell_capacity: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Maximum number of active agents that may occupy a single passable "
                "cell simultaneously.  Applies uniformly to all cells in Stage 3. "
                "A value of 1 (the default) reproduces Stage 2 one-agent-per-cell "
                "behaviour while now making that constraint explicit and configurable."
            ),
        ),
    ] = 1


# ---------------------------------------------------------------------------
# Top-level scenario configuration
# ---------------------------------------------------------------------------


class DoorOpening(BaseModel):
    """
    An explicit physical opening or doorway in a building.

    A DoorOpening represents physical geometry (position, width, and whether
    it opens to the exterior). It is NOT an exit by default; candidate exits
    are derived from exterior DoorOpenings.
    """

    position: GridCell = Field(
        ...,
        description="The (row, col) coordinates of the opening.",
    )
    width: Annotated[
        float,
        Field(
            gt=0.0,
            description="Width of the door opening in meters (must be > 0.0).",
        ),
    ] = 1.0
    exterior: bool = Field(
        default=True,
        description="True if opening connects to the building exterior; False if interior.",
    )
    door_id: Optional[str] = Field(
        default=None,
        description="Optional unique identifier for the door opening.",
    )

    @model_validator(mode="after")
    def position_must_be_non_negative(self) -> "DoorOpening":
        r, c = self.position
        if r < 0 or c < 0:
            raise ValueError(f"Door opening position ({r}, {c}) must be non-negative.")
        return self


class FloorPlan(BaseModel):
    """
    Static building geometry (grid, walls, and explicit door openings).
    Does NOT contain exits, agents, or simulation parameters.
    """
    grid: GridConfig = Field(default_factory=GridConfig)
    walls: list[GridCell] = Field(
        default_factory=list,
        description="List of [row, col] pairs that are solid walls",
    )
    doors: list[DoorOpening] = Field(
        default_factory=list,
        description="List of explicit door openings in the building",
    )

    @model_validator(mode="after")
    def walls_must_be_within_grid(self) -> "FloorPlan":
        for r, c in self.walls:
            if not (0 <= r < self.grid.rows and 0 <= c < self.grid.cols):
                raise ValueError(
                    f"Wall cell ({r}, {c}) is outside the grid ({self.grid.rows}×{self.grid.cols})."
                )
        return self

    @model_validator(mode="after")
    def doors_must_be_within_grid(self) -> "FloorPlan":
        for door in self.doors:
            r, c = door.position
            if not (0 <= r < self.grid.rows and 0 <= c < self.grid.cols):
                raise ValueError(
                    f"Door opening ({r}, {c}) is outside the grid ({self.grid.rows}×{self.grid.cols})."
                )
        return self

    @model_validator(mode="after")
    def doors_and_walls_must_not_overlap(self) -> "FloorPlan":
        wall_set = set(map(tuple, self.walls))
        for door in self.doors:
            if tuple(door.position) in wall_set:
                raise ValueError(
                    f"Door opening at ({door.position[0]}, {door.position[1]}) cannot be placed on a solid wall cell."
                )
        return self

    @model_validator(mode="after")
    def door_positions_must_be_unique(self) -> "FloorPlan":
        seen: set[tuple[int, int]] = set()
        for door in self.doors:
            pos = (door.position[0], door.position[1])
            if pos in seen:
                raise ValueError(
                    f"Duplicate door opening defined at position ({pos[0]}, {pos[1]})."
                )
            seen.add(pos)
        return self


class ExitConfiguration(BaseModel):
    """
    Authoritative configuration for exit cells and physical opening properties.
    """
    configuration_id: str = Field(
        default="reference",
        description="Identifier for this configuration in experiments."
    )
    exits: list[GridCell] = Field(
        default_factory=list,
        description="List of [row, col] pairs that are exit cells"
    )
    widths: list[float] = Field(
        default_factory=list,
        description="List of opening widths in meters corresponding to each exit in exits."
    )

    def get_width(self, cell: GridCell) -> float:
        """Return the physical width for a given exit cell (default: 1.0m)."""
        pos = tuple(cell)
        for i, exit_cell in enumerate(self.exits):
            if tuple(exit_cell) == pos:
                if i < len(self.widths):
                    return self.widths[i]
                break
        return 1.0

    @property
    def exit_widths(self) -> dict[GridCell, float]:
        """Dictionary mapping each exit position to its physical width."""
        return {tuple(cell): self.get_width(cell) for cell in self.exits}


class OccupantScenario(BaseModel):
    """
    Initial placement and profiles of occupants.
    """
    agents: list[AgentConfig] = Field(
        default_factory=list,
        description="Initial placement of agents",
    )

    @model_validator(mode="after")
    def agent_ids_must_be_unique(self) -> "OccupantScenario":
        ids = [a.agent_id for a in self.agents]
        if len(ids) != len(set(ids)):
            duplicates = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"Duplicate agent IDs detected: {duplicates}")
        return self


class SimulationConfig(BaseModel):
    """
    Complete configuration for a single simulation scenario.

    This model is loaded from a JSON scenario file and passed to all
    simulation components during initialisation.

    Attributes
    ----------
    scenario_name:
        Human-readable label for the scenario (used in reports/logs).
    floor_plan:
        Static building geometry.
    exit_configuration:
        Authoritative source for exit locations.
    occupants:
        Initial occupant scenario.
    parameters:
        Simulation execution parameters.
    """

    scenario_name: str = Field(default="unnamed_scenario", description="Scenario label")
    floor_plan: FloorPlan = Field(default_factory=FloorPlan)
    exit_configuration: ExitConfiguration = Field(default_factory=ExitConfiguration)
    occupants: OccupantScenario = Field(default_factory=OccupantScenario)
    parameters: SimulationParameters = Field(default_factory=SimulationParameters)

    # ------------------------------------------------------------------
    # Backward compatibility properties
    # ------------------------------------------------------------------

    @property
    def grid(self) -> GridConfig:
        return self.floor_plan.grid

    @property
    def walls(self) -> list[GridCell]:
        return self.floor_plan.walls

    @property
    def doors(self) -> list[DoorOpening]:
        return self.floor_plan.doors

    @property
    def exits(self) -> list[GridCell]:
        return self.exit_configuration.exits

    @property
    def agents(self) -> list[AgentConfig]:
        return self.occupants.agents

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_flat_schema(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Gracefully migrate legacy flat schema to the new nested schema.
        This ensures backward compatibility for tests and old configs.
        """
        if not isinstance(data, dict):
            return data
            
        data = data.copy()
        
        # If the flat keys exist and the nested keys don't, migrate them.
        if "grid" in data or "walls" in data or "doors" in data:
            if "floor_plan" not in data:
                data["floor_plan"] = {
                    "grid": data.pop("grid", {"rows": 10, "cols": 10}),
                    "walls": data.pop("walls", []),
                    "doors": data.pop("doors", []),
                }
        
        if "exits" in data:
            if "exit_configuration" not in data:
                data["exit_configuration"] = {
                    "configuration_id": "reference",
                    "exits": data.pop("exits")
                }
                
        if "agents" in data:
            if "occupants" not in data:
                data["occupants"] = {
                    "agents": data.pop("agents")
                }

        return data

    @model_validator(mode="after")
    def exits_must_exist(self) -> "SimulationConfig":
        if len(self.exit_configuration.exits) == 0:
            raise ValueError("Scenario must define at least one exit cell.")
        return self

    @model_validator(mode="after")
    def exits_must_be_within_grid(self) -> "SimulationConfig":
        for r, c in self.exit_configuration.exits:
            if not (0 <= r < self.floor_plan.grid.rows and 0 <= c < self.floor_plan.grid.cols):
                raise ValueError(
                    f"Exit cell ({r}, {c}) is outside the grid ({self.floor_plan.grid.rows}×{self.floor_plan.grid.cols})."
                )
        return self

    @model_validator(mode="after")
    def exits_and_walls_must_not_overlap(self) -> "SimulationConfig":
        overlap = set(map(tuple, self.floor_plan.walls)) & set(map(tuple, self.exit_configuration.exits))
        if overlap:
            raise ValueError(f"Exit cells and wall cells overlap: {overlap}")
        return self

    @model_validator(mode="after")
    def agents_must_be_within_grid(self) -> "SimulationConfig":
        for agent in self.occupants.agents:
            if agent.row >= self.floor_plan.grid.rows or agent.col >= self.floor_plan.grid.cols:
                raise ValueError(
                    f"Agent '{agent.agent_id}' starting position "
                    f"({agent.row}, {agent.col}) is outside the grid "
                    f"({self.floor_plan.grid.rows}×{self.floor_plan.grid.cols})."
                )
        return self

    @model_validator(mode="after")
    def agents_must_not_start_on_walls(self) -> "SimulationConfig":
        wall_set = set(map(tuple, self.floor_plan.walls))
        for agent in self.occupants.agents:
            if (agent.row, agent.col) in wall_set:
                raise ValueError(
                    f"Agent '{agent.agent_id}' starts on a wall cell "
                    f"({agent.row}, {agent.col})."
                )
        return self

    # ------------------------------------------------------------------
    # Factory / I/O helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "SimulationConfig":
        """
        Load and validate a scenario configuration from a JSON file.

        Parameters
        ----------
        path:
            Filesystem path to the ``.json`` scenario file.

        Returns
        -------
        SimulationConfig
            A fully-validated configuration instance.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        pydantic.ValidationError
            If the JSON content fails schema validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        return cls.model_validate(raw)

    def cell_is_within_grid(self, row: int, col: int) -> bool:
        """Return True if (row, col) falls within the grid boundaries."""
        return 0 <= row < self.floor_plan.grid.rows and 0 <= col < self.floor_plan.grid.cols
