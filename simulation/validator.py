"""
validator.py
------------
Stage 11: ExitValidator — top-level API for the regulatory constraint layer.

The ExitValidator wraps a RuleSet and exposes the two main interfaces:

    result = validator.check(floor_plan, exit_configuration, occupant_scenario)
    mask   = validator.mask(floor_plan, occupant_scenario, partial_config)

This module is the integration point for future RL/search code.
It does NOT couple to any RL library.

Architecture
~~~~~~~~~~~~
ExitValidator
  check(fp, exit_cfg, scenario) -> ValidationResult
  mask(fp, scenario, partial_cfg) -> dict[GridCell, bool]

A full pipeline looks like:

    FloorPlan
       |
    generate_candidate_exits()
       |
    CandidateExit[]
       |
    ExitValidator.mask()      <- partial action mask (R1 evaluated here)
       |
    Legal candidate pool
       |
    Build complete ExitConfiguration (external: RL/search/user)
       |
    ExitValidator.check()     <- full compliance check (R1-R5)
       |
    ValidationResult (is_compliant, violations[])
       |
    Evacuation Simulation (if compliant)

DISCLAIMER: research/design-assistance tool only. Not legal certification.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from evacuation_simulation.simulation.config import (
    ExitConfiguration,
    FloorPlan,
    GridCell,
    OccupantScenario,
)
from evacuation_simulation.simulation.generation import (
    CandidateExit,
    generate_candidate_exits,
)
from evacuation_simulation.simulation.rules import (
    RuleSet,
    ValidationResult,
    build_nbc_2016_ruleset,
    load_regulatory_config,
)


class ExitValidator:
    """
    Top-level exit-configuration validator.

    Wraps a RuleSet and provides the public API for compliance checking
    and action masking.

    Parameters
    ----------
    rule_set:
        The RuleSet to use for evaluation. Defaults to the NBC 2016 Part 4
        rule set loaded from config/rules_nbc_2016.yaml.

    Example
    -------
    >>> validator = ExitValidator()
    >>> result = validator.check(floor_plan, exit_config, occupant_scenario)
    >>> result.is_compliant
    True
    >>> mask = validator.mask(floor_plan, occupant_scenario, partial_config)
    >>> mask[(0, 4)]
    True
    """

    def __init__(self, rule_set: Optional[RuleSet] = None) -> None:
        self._rule_set: RuleSet = rule_set or build_nbc_2016_ruleset()

    @property
    def rule_set(self) -> RuleSet:
        """The underlying RuleSet."""
        return self._rule_set

    def check(
        self,
        floor_plan: FloorPlan,
        exit_config: ExitConfiguration,
        occupant_scenario: OccupantScenario,
    ) -> ValidationResult:
        """
        Evaluate all rules against a COMPLETE exit configuration.

        Does NOT mutate floor_plan, exit_config, or occupant_scenario.

        Parameters
        ----------
        floor_plan:
            Static building geometry (DoorOpenings, walls, grid).
        exit_config:
            Complete exit configuration to evaluate.
        occupant_scenario:
            Occupant placement and profiles.

        Returns
        -------
        ValidationResult
            is_compliant=True and empty violations list if all encoded
            hard constraints are satisfied.
            is_compliant=False and populated violations list otherwise.

        Note
        ----
        A True result means "compliant with the encoded constraints in the
        selected regulation profile" — NOT "legally compliant".
        """
        return self._rule_set.check(floor_plan, exit_config, occupant_scenario)

    def mask(
        self,
        floor_plan: FloorPlan,
        occupant_scenario: OccupantScenario,
        partial_config: ExitConfiguration,
        candidates: Optional[list[CandidateExit]] = None,
    ) -> dict[GridCell, bool]:
        """
        Compute a partial action mask for candidate exits.

        This method is designed for use by future RL/search code that
        builds exit configurations incrementally.

        Parameters
        ----------
        floor_plan:
            Static building geometry.
        occupant_scenario:
            Occupant placement and profiles.
        partial_config:
            Partially built ExitConfiguration (may be empty).
        candidates:
            Pool of candidate exits to mask. If None, derives candidates
            from floor_plan using generate_candidate_exits().

        Returns
        -------
        dict[GridCell, bool]
            True  = candidate may be legally selected as the next exit.
            False = candidate is clearly forbidden at this stage.

        IMPORTANT LIMITATIONS
        ~~~~~~~~~~~~~~~~~~~~~
        The partial mask can only evaluate per-candidate constraints (R1).
        The following constraints CANNOT be evaluated on a partial
        configuration and are NOT reflected in this mask:

        - R2 (minimum exit count): requires complete configuration
        - R3 (exit separation): requires all exits to be known
        - R4 (exit width): total-width constraints need all exits
        - R5 (travel distance): requires all exits to find optimal routes

        These constraints WILL be enforced by check() on the final
        complete configuration. Do NOT assume mask=True means the full
        configuration will be compliant — always call check() afterwards.
        """
        if candidates is None:
            candidates = generate_candidate_exits(floor_plan)
        return self._rule_set.mask(
            floor_plan, occupant_scenario, partial_config, candidates
        )

    def candidate_exits(self, floor_plan: FloorPlan) -> list[CandidateExit]:
        """
        Convenience: derive candidate exits from the floor plan.

        Returns exterior DoorOpening candidates (or legacy boundary cells
        if no explicit doors are defined).
        """
        return generate_candidate_exits(floor_plan)

    def regulatory_summary(self) -> dict:
        """
        Return a machine-readable summary of all rules in the rule set.

        Useful for logging, debugging, and RL metadata.
        """
        return {
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "description": rule.description,
                }
                for rule in self._rule_set.rules
            ]
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_validator(config_path: Optional[Path] = None) -> ExitValidator:
    """
    Create an ExitValidator using the NBC 2016 Part 4 rule set.

    Parameters
    ----------
    config_path:
        Optional path to a custom YAML config.
        Defaults to config/rules_nbc_2016.yaml.

    Returns
    -------
    ExitValidator
    """
    rule_set = build_nbc_2016_ruleset(config_path)
    return ExitValidator(rule_set=rule_set)
