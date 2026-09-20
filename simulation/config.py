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


class SimulationConfig(BaseModel):
    """
    Complete configuration for a single simulation scenario.

    This model is loaded from a JSON scenario file and passed to all
    simulation components during initialisation.

    Attributes
    ----------
    scenario_name:
        Human-readable label for the scenario (used in reports/logs).
    grid:
        Grid dimensions.
    walls:
        List of (row, col) cells that are impassable.
    exits:
        List of (row, col) cells that are designated exit points.
        At least one exit must be defined.
    agents:
        List of agent initial-placement descriptors.
    parameters:
        Simulation execution parameters.
    """

    scenario_name: str = Field(default="unnamed_scenario", description="Scenario label")

    grid: GridConfig = Field(default_factory=GridConfig)

    walls: list[GridCell] = Field(
        default_factory=list,
        description="List of [row, col] pairs that are solid walls",
    )

    exits: list[GridCell] = Field(
        default_factory=list,
        description="List of [row, col] pairs that are exit cells",
    )

    agents: list[AgentConfig] = Field(
        default_factory=list,
        description="Initial placement of agents",
    )

    parameters: SimulationParameters = Field(default_factory=SimulationParameters)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def exits_must_exist(self) -> "SimulationConfig":
        if len(self.exits) == 0:
            raise ValueError("Scenario must define at least one exit cell.")
        return self

    @model_validator(mode="after")
    def exits_and_walls_must_not_overlap(self) -> "SimulationConfig":
        overlap = set(map(tuple, self.walls)) & set(map(tuple, self.exits))
        if overlap:
            raise ValueError(f"Exit cells and wall cells overlap: {overlap}")
        return self

    @model_validator(mode="after")
    def agent_ids_must_be_unique(self) -> "SimulationConfig":
        ids = [a.agent_id for a in self.agents]
        if len(ids) != len(set(ids)):
            duplicates = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"Duplicate agent IDs detected: {duplicates}")
        return self

    @model_validator(mode="after")
    def agents_must_be_within_grid(self) -> "SimulationConfig":
        for agent in self.agents:
            if agent.row >= self.grid.rows or agent.col >= self.grid.cols:
                raise ValueError(
                    f"Agent '{agent.agent_id}' starting position "
                    f"({agent.row}, {agent.col}) is outside the grid "
                    f"({self.grid.rows}×{self.grid.cols})."
                )
        return self

    @model_validator(mode="after")
    def agents_must_not_start_on_walls(self) -> "SimulationConfig":
        wall_set = set(map(tuple, self.walls))
        for agent in self.agents:
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
        return 0 <= row < self.grid.rows and 0 <= col < self.grid.cols
