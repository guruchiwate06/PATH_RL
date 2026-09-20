"""
profile.py
----------
Extensible per-agent profile architecture for the 2D multi-agent
evacuation simulation (Stage 6).

Responsibilities
~~~~~~~~~~~~~~~~
- Encapsulate individual agent characteristics (AgentProfile).
- Distinguish between:
    1. Input/descriptive attributes (e.g. age, mobility descriptors).
    2. Explicit simulation parameters (speed, reaction_delay).
    3. Derived parameters (values from future parameterization models).
- Provide deterministic movement eligibility scheduling for speed and delay.
- Maintain backward compatibility with baseline default values (speed=1.0, delay=0).

Non-responsibilities (intentionally excluded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- No age-to-speed formulas or demographic stereotyping (reserved for future stages).
- No stochastic movement (random.random() is strictly forbidden).
- No multi-cell movement (an agent can move at most 1 cell per timestep).
- No experiment or visualization logic inside the profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentProfile:
    """
    Characteristics and operational parameters of an individual agent.

    Attributes
    ----------
    speed:
        Movement frequency at cell/timestep resolution (0.0 < speed <= 1.0).
        A speed of 1.0 grants a movement opportunity every timestep.
        A speed of 0.5 grants a movement opportunity approximately every 2 timesteps.
        A speed of 0.25 grants a movement opportunity approximately every 4 timesteps.
        Default is 1.0 (baseline movement rate).
    reaction_delay:
        Number of initial timesteps before the agent begins movement.
        During timesteps t <= reaction_delay, the agent remains stationary and
        does not submit movement requests.
        Default is 0 (immediate evacuation start).
    input_attributes:
        Dictionary of raw descriptive metadata provided by the user or scenario
        (e.g., {"age": 65, "mobility_aid": False}). Used as input for future
        derivation models; not directly consumed by the movement engine.
    derived_parameters:
        Dictionary recording simulation parameters that were computed by future
        parameterization models rather than explicitly supplied.
    explicit_speed:
        Tracks whether speed was explicitly provided by the user/scenario
        (True) or defaulted/derived (False/None).
    explicit_reaction_delay:
        Tracks whether reaction_delay was explicitly provided by the user/scenario
        (True) or defaulted/derived (False/None).
    """

    speed: float = 1.0
    reaction_delay: int = 0
    input_attributes: dict[str, Any] = field(default_factory=dict)
    derived_parameters: dict[str, Any] = field(default_factory=dict)
    explicit_speed: Optional[float] = None
    explicit_reaction_delay: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate parameter ranges and initialize tracking fields."""
        if not (0.0 < self.speed <= 1.0):
            raise ValueError(
                f"AgentProfile speed must satisfy 0.0 < speed <= 1.0, got {self.speed}."
            )
        if self.reaction_delay < 0:
            raise ValueError(
                f"AgentProfile reaction_delay must be non-negative (>= 0), got {self.reaction_delay}."
            )

        if self.explicit_speed is None:
            self.explicit_speed = self.speed
        if self.explicit_reaction_delay is None:
            self.explicit_reaction_delay = self.reaction_delay

    # ------------------------------------------------------------------
    # Deterministic Movement Scheduling
    # ------------------------------------------------------------------

    def is_reaction_delayed(self, timestep: int) -> bool:
        """
        Return True if the agent is still in its initial reaction delay period.

        Timestep convention:
        - Timesteps are 1-indexed (1, 2, 3, ...).
        - If reaction_delay == 0: never delayed (returns False for all t >= 1).
        - If reaction_delay == 2: delayed at t=1, t=2; active at t >= 3.
        """
        return timestep <= self.reaction_delay

    def is_speed_eligible(self, timestep: int) -> bool:
        """
        Deterministically determine if step *timestep* grants a movement opportunity.

        Uses exact discrete rate accumulation over active elapsed steps:
            t_active = timestep - reaction_delay
            delta = floor(t_active * speed + eps) - floor((t_active - 1) * speed + eps)
            eligible if delta >= 1

        Guarantees:
        - 100% deterministic (no randomness).
        - For speed 1.0: True on every step.
        - For speed 0.5: True on steps 2, 4, 6, ...
        - For speed 0.25: True on steps 4, 8, 12, ...
        - An agent can never move more than 1 cell in a single timestep.
        """
        if self.is_reaction_delayed(timestep):
            return False

        if self.speed >= 1.0:
            return True

        t_active = timestep - self.reaction_delay
        if t_active <= 0:
            return False

        eps = 1e-9
        prev_count = math.floor((t_active - 1) * self.speed + eps)
        curr_count = math.floor(t_active * self.speed + eps)

        return (curr_count - prev_count) >= 1

    def is_movement_eligible(self, timestep: int) -> bool:
        """
        Return True if the agent is eligible to submit a movement request at *timestep*.

        An agent is eligible if and only if:
        1. It has surpassed its initial reaction delay (timestep > reaction_delay).
        2. Its speed schedule grants an opportunity on this active timestep.
        """
        if self.is_reaction_delayed(timestep):
            return False
        return self.is_speed_eligible(timestep)

    # ------------------------------------------------------------------
    # Serialization and Factory
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile attributes to a dictionary."""
        data: dict[str, Any] = {
            "speed": self.speed,
            "reaction_delay": self.reaction_delay,
        }
        if self.input_attributes:
            data["input_attributes"] = dict(self.input_attributes)
        if self.derived_parameters:
            data["derived_parameters"] = dict(self.derived_parameters)
        return data

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "AgentProfile":
        """Construct an AgentProfile from a dictionary, with default fallback."""
        if not data:
            return cls()

        speed = float(data.get("speed", 1.0))
        delay = int(data.get("reaction_delay", 0))
        input_attrs = dict(data.get("input_attributes", {}))

        # Support top-level optional descriptive fields like "age"
        if "age" in data and "age" not in input_attrs:
            input_attrs["age"] = data["age"]

        derived = dict(data.get("derived_parameters", {}))

        return cls(
            speed=speed,
            reaction_delay=delay,
            input_attributes=input_attrs,
            derived_parameters=derived,
            explicit_speed=data.get("speed"),
            explicit_reaction_delay=data.get("reaction_delay"),
        )
