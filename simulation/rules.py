"""
rules.py
--------
Stage 11: Regulatory & Exit-Placement Constraint Layer.

Modular rule engine for evaluating whether an ExitConfiguration satisfies
encoded regulatory and placement constraints derived from NBC 2016 Part 4.

REGULATORY DISCLAIMER
~~~~~~~~~~~~~~~~~~~~~
Rules encoded here are derived from NBC 2016 Part 4 (Fire and Life Safety)
for research and design-assistance purposes ONLY.
This implementation does NOT constitute legal certification.
Verified values are marked verified=True in config/rules_nbc_2016.yaml.
Unverified/configurable values are marked verified=False.
Use wording "compliant with encoded constraints" rather than "legally compliant".

Non-goals (Stage 11)
~~~~~~~~~~~~~~~~~~~~
No RL, no reward functions, no optimization, no multi-floor, no fire/smoke.
"""
from __future__ import annotations

import math
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import networkx as nx
from pydantic import BaseModel, Field

from evacuation_simulation.simulation.config import (
    ExitConfiguration,
    FloorPlan,
    GridCell,
    OccupantScenario,
)
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.generation import (
    CandidateExit,
    generate_candidate_exits,
)
from evacuation_simulation.simulation.pathfinding import NavigationGraph

_DEFAULT_RULES_CONFIG = Path(__file__).parent.parent / "config" / "rules_nbc_2016.yaml"


# ---------------------------------------------------------------------------
# Violation
# ---------------------------------------------------------------------------

class Violation(BaseModel):
    """Machine-readable constraint violation record."""
    rule_id: str = Field(..., description="Unique rule identifier")
    message: str = Field(..., description="Human-readable violation description")
    severity: str = Field(default="error", description="'error' (hard) or 'warning' (soft)")
    clause: str = Field(default="", description="Regulatory clause/reference")
    measured_value: Optional[float] = Field(default=None, description="Actual measured value")
    required_value: Optional[float] = Field(default=None, description="Required/threshold value")
    unit: str = Field(default="", description="Physical unit")
    relevant_ids: list[str] = Field(default_factory=list, description="Relevant object IDs")


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """
    Result of a full exit-configuration compliance check.

    is_compliant is True only if all encoded hard constraints are satisfied.
    This does NOT imply legal compliance — use wording
    'compliant with encoded constraints in the selected regulation profile'.
    """
    is_compliant: bool = Field(...)
    violations: list[Violation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Regulatory config loader
# ---------------------------------------------------------------------------

def load_regulatory_config(path: Optional[Path] = None) -> dict[str, Any]:
    """Load YAML regulatory rules configuration."""
    p = path or _DEFAULT_RULES_CONFIG
    if not p.exists():
        raise FileNotFoundError(f"Regulatory config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Abstract Rule
# ---------------------------------------------------------------------------

class Rule(ABC):
    """
    Abstract base class for compliance rules.

    Rules are stateless: they receive all required context as arguments.
    check() evaluates a complete configuration.
    mask() evaluates individual candidates at partial-selection time.
    Only R1 overrides mask(); other rules use the default (allow all).
    """

    @property
    @abstractmethod
    def rule_id(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def check(
        self,
        floor_plan: FloorPlan,
        exit_config: ExitConfiguration,
        occupant_scenario: OccupantScenario,
    ) -> list[Violation]: ...

    def mask(
        self,
        floor_plan: FloorPlan,
        occupant_scenario: OccupantScenario,
        partial_config: ExitConfiguration,
        candidates: list[CandidateExit],
    ) -> dict[GridCell, bool]:
        """
        Default mask: allow all candidates.

        IMPORTANT: Rules that require a COMPLETE configuration for evaluation
        (R2 min-exit-count, R3 separation, R4 width sum, R5 travel-distance)
        do NOT override this method. The mask API for those rules can only
        be evaluated during the final check(), not at partial-selection time.
        """
        return {(c.position[0], c.position[1]): True for c in candidates}


# ---------------------------------------------------------------------------
# R1 — Valid Exit Position
# ---------------------------------------------------------------------------

class R1ValidExitPosition(Rule):
    """
    R1_VALID_EXIT_POSITION

    Only exterior DoorOpening entities may be selected as emergency exits.
    Interior doors, interior passages, walls, and arbitrary boundary cells
    that have no corresponding exterior DoorOpening are ineligible.

    This rule IS evaluable at partial-selection time (per-candidate check).
    """

    def __init__(self, config_data: dict[str, Any]) -> None:
        cfg = config_data.get("rules", {}).get("R1_VALID_EXIT_POSITION", {})
        self._clause: str = cfg.get("clause", "NBC 2016 Part 4 (geometric rule)")
        self._severity: str = cfg.get("severity", "error")

    @property
    def rule_id(self) -> str:
        return "R1_VALID_EXIT_POSITION"

    @property
    def description(self) -> str:
        return (
            "Only exterior DoorOpening entities may be emergency exits. "
            "Interior doors, passages, walls, and non-exterior cells are ineligible."
        )

    def _valid_positions(self, floor_plan: FloorPlan) -> set[GridCell]:
        if floor_plan.doors:
            return {
                (c.position[0], c.position[1])
                for c in generate_candidate_exits(floor_plan, fallback_legacy_boundary=False)
            }
        # Legacy: no explicit doors — accept all non-wall boundary cells
        wall_set = set(map(tuple, floor_plan.walls))
        rows, cols = floor_plan.grid.rows, floor_plan.grid.cols
        return {
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1)
            and (r, c) not in wall_set
        }

    def check(
        self,
        floor_plan: FloorPlan,
        exit_config: ExitConfiguration,
        occupant_scenario: OccupantScenario,
    ) -> list[Violation]:
        violations: list[Violation] = []
        valid = self._valid_positions(floor_plan)

        # Detect duplicate exits
        seen: set[tuple] = set()
        for raw_pos in exit_config.exits:
            pos = (raw_pos[0], raw_pos[1])
            if pos in seen:
                violations.append(Violation(
                    rule_id=self.rule_id,
                    message="Duplicate exit at position " + str(pos) + ".",
                    severity=self._severity,
                    clause=self._clause,
                    relevant_ids=[str(pos)],
                ))
            seen.add(pos)

        # Check each selected exit is a valid exterior door position
        for raw_pos in exit_config.exits:
            pos = (raw_pos[0], raw_pos[1])
            if pos not in valid:
                violations.append(Violation(
                    rule_id=self.rule_id,
                    message=(
                        "Selected exit at " + str(pos) + " is not a valid exterior DoorOpening. "
                        "Interior doors, passages, walls, and non-exterior positions are ineligible."
                    ),
                    severity=self._severity,
                    clause=self._clause,
                    relevant_ids=[str(pos)],
                ))
        return violations

    def mask(
        self,
        floor_plan: FloorPlan,
        occupant_scenario: OccupantScenario,
        partial_config: ExitConfiguration,
        candidates: list[CandidateExit],
    ) -> dict[GridCell, bool]:
        """
        Block already-selected candidates; allow remaining exterior-door candidates.
        This can be evaluated at partial-selection time.
        """
        already_selected: set[tuple] = {(p[0], p[1]) for p in partial_config.exits}
        return {
            (c.position[0], c.position[1]): (c.position[0], c.position[1]) not in already_selected
            for c in candidates
        }


# ---------------------------------------------------------------------------
# R2 — Minimum Exit Count
# ---------------------------------------------------------------------------

class R2MinimumExitCount(Rule):
    """
    R2_MINIMUM_EXIT_COUNT

    NBC 2016 Part 4 Section 4.4.2.4.3: minimum two means of egress (unverified universal).
    Configurable per occupancy type via YAML.

    Cannot be evaluated on a partial configuration.
    """

    def __init__(self, config_data: dict[str, Any]) -> None:
        rule_cfg = config_data.get("rules", {}).get("R2_MINIMUM_EXIT_COUNT", {})
        abs_min = rule_cfg.get("min_exits_absolute", {})
        self._min_exits: int = abs_min.get("value", 2)
        self._clause: str = abs_min.get("clause", "NBC 2016 Part 4 Section 4.4.2.4.3 (Conditionally verified)")
        self._severity: str = rule_cfg.get("severity", "error")
        self._overrides: dict = rule_cfg.get("occupancy_overrides", {})

    @property
    def rule_id(self) -> str:
        return "R2_MINIMUM_EXIT_COUNT"

    @property
    def description(self) -> str:
        return "Exit configuration must contain the regulatory minimum number of exits."

    def _min_for_occupancy(self, occ_type: Optional[str]) -> tuple[int, str]:
        if occ_type and occ_type in self._overrides:
            ov = self._overrides[occ_type]
            return ov.get("value", self._min_exits), ov.get("clause", self._clause)
        return self._min_exits, self._clause

    def check(
        self,
        floor_plan: FloorPlan,
        exit_config: ExitConfiguration,
        occupant_scenario: OccupantScenario,
    ) -> list[Violation]:
        occ_type = getattr(occupant_scenario, "occupancy_type", None)
        required, clause = self._min_for_occupancy(occ_type)
        actual = len(exit_config.exits)
        if actual < required:
            return [Violation(
                rule_id=self.rule_id,
                message=(
                    "Exit configuration has " + str(actual) + " exit(s) but the "
                    "regulatory minimum is " + str(required) + "."
                ),
                severity=self._severity,
                clause=clause,
                measured_value=float(actual),
                required_value=float(required),
                unit="exits",
            )]
        return []


# ---------------------------------------------------------------------------
# R3 — Exit Separation
# ---------------------------------------------------------------------------

class R3ExitSeparation(Rule):
    """
    R3_EXIT_SEPARATION

    NBC 2016 Part 4 diagonal rule:
    - Non-sprinklered: >= 1/2 floor diagonal (unverified)
    - Sprinklered:     >= 1/3 floor diagonal (unverified)

    Cannot be evaluated on a partial configuration.
    """

    def __init__(self, config_data: dict[str, Any]) -> None:
        rule_cfg = config_data.get("rules", {}).get("R3_EXIT_SEPARATION", {})
        self._severity: str = rule_cfg.get("severity", "error")
        ns = rule_cfg.get("non_sprinklered", {})
        self._ns_frac: float = ns.get("fraction_of_diagonal", 0.5)
        self._ns_clause: str = ns.get("clause", "NBC 2016 Part 4 egress placement (Diagonal formulation)")
        sp = rule_cfg.get("sprinklered", {})
        self._sp_frac: float = sp.get("fraction_of_diagonal", 0.333)
        self._sp_clause: str = sp.get("clause", "NBC 2016 Part 4 egress placement (Diagonal formulation)")
        self._default_spr: bool = rule_cfg.get("default_sprinklered", False)
        r5 = config_data.get("rules", {}).get("R5_TRAVEL_DISTANCE", {})
        self._cell_m: float = r5.get("cell_size_m", {}).get("value", 1.0)

    @property
    def rule_id(self) -> str:
        return "R3_EXIT_SEPARATION"

    @property
    def description(self) -> str:
        return "Selected exits must satisfy minimum separation (fraction of floor diagonal, NBC 2016 Part 4)."

    def _diagonal_m(self, fp: FloorPlan) -> float:
        return math.sqrt((fp.grid.rows * self._cell_m) ** 2 + (fp.grid.cols * self._cell_m) ** 2)

    def _dist_m(self, a: GridCell, b: GridCell) -> float:
        return math.sqrt(((a[0] - b[0]) * self._cell_m) ** 2 + ((a[1] - b[1]) * self._cell_m) ** 2)

    def check(
        self,
        floor_plan: FloorPlan,
        exit_config: ExitConfiguration,
        occupant_scenario: OccupantScenario,
    ) -> list[Violation]:
        exits = exit_config.exits
        if len(exits) < 2:
            return []
        is_spr: bool = getattr(occupant_scenario, "sprinklered", self._default_spr)
        frac = self._sp_frac if is_spr else self._ns_frac
        clause = self._sp_clause if is_spr else self._ns_clause
        diag = self._diagonal_m(floor_plan)
        req = frac * diag
        violations: list[Violation] = []
        for i in range(len(exits)):
            for j in range(i + 1, len(exits)):
                a: GridCell = (exits[i][0], exits[i][1])
                b: GridCell = (exits[j][0], exits[j][1])
                dist = self._dist_m(a, b)
                if dist < req:
                    frac_str = "1/" + str(round(1 / frac)) if frac else "?"
                    violations.append(Violation(
                        rule_id=self.rule_id,
                        message=(
                            "Exits at " + str(a) + " and " + str(b) + " are separated by "
                            + str(round(dist, 2)) + "m, below required " + str(round(req, 2))
                            + "m (" + frac_str + " of diagonal " + str(round(diag, 2)) + "m)."
                        ),
                        severity=self._severity,
                        clause=clause,
                        measured_value=round(dist, 4),
                        required_value=round(req, 4),
                        unit="m",
                        relevant_ids=[str(a), str(b)],
                    ))
        return violations


# ---------------------------------------------------------------------------
# R4 — Exit Width / Capacity
# ---------------------------------------------------------------------------

class R4ExitWidth(Rule):
    """
    R4_EXIT_WIDTH

    NBC 2016 Part 4:
    - Absolute minimum per exit: 1.0m (unverified universal prototype default)
    - Unit of exit width: 0.5m (verified)
    - Width-per-person: intentionally null/unverified (NBC uses unit-of-width table)

    IMPORTANT: physical_width_m is SEPARATE from legal/occupant-load capacity.
    Do NOT replace DoorOpening.width with a derived capacity.
    """

    def __init__(self, config_data: dict[str, Any]) -> None:
        rule_cfg = config_data.get("rules", {}).get("R4_EXIT_WIDTH", {})
        self._severity: str = rule_cfg.get("severity", "error")
        abs_min = rule_cfg.get("absolute_minimum_width_per_exit", {})
        self._abs_min_m: float = abs_min.get("value", 1.0)
        self._abs_min_clause: str = abs_min.get("clause", "NBC 2016 Part 4 Table 4")
        # Width per person is intentionally null/unverified
        self._width_per_person: None = None

    @property
    def rule_id(self) -> str:
        return "R4_EXIT_WIDTH"

    @property
    def description(self) -> str:
        return (
            "Each exit must meet minimum physical width. "
            "Physical width (DoorOpening.width) is separate from occupant-load capacity."
        )

    def check(
        self,
        floor_plan: FloorPlan,
        exit_config: ExitConfiguration,
        occupant_scenario: OccupantScenario,
    ) -> list[Violation]:
        violations: list[Violation] = []
        for exit_pos in exit_config.exits:
            t: GridCell = (exit_pos[0], exit_pos[1])
            width_m: float = exit_config.get_width(exit_pos)
            if width_m < self._abs_min_m:
                violations.append(Violation(
                    rule_id=self.rule_id,
                    message=(
                        "Exit at " + str(t) + " has physical width "
                        + str(round(width_m, 2)) + "m, below minimum "
                        + str(self._abs_min_m) + "m."
                    ),
                    severity=self._severity,
                    clause=self._abs_min_clause,
                    measured_value=round(width_m, 4),
                    required_value=self._abs_min_m,
                    unit="m",
                    relevant_ids=[str(t)],
                ))
        return violations

    def mask(
        self,
        floor_plan: FloorPlan,
        occupant_scenario: OccupantScenario,
        partial_config: ExitConfiguration,
        candidates: list[CandidateExit],
    ) -> dict[GridCell, bool]:
        """
        Mask out candidates whose physical width is strictly below the
        regulatory absolute minimum exit width.
        """
        return {
            (c.position[0], c.position[1]): (c.width >= self._abs_min_m)
            for c in candidates
        }


# ---------------------------------------------------------------------------
# R5 — Travel Distance
# ---------------------------------------------------------------------------

class R5TravelDistance(Rule):
    """
    R5_TRAVEL_DISTANCE

    Two distinct sub-checks:

    A. REGULATORY: travel distance <= NBC 2016 Part 4 Table 5 limit
       (occupancy-dependent, verified for listed types).

    B. FEASIBILITY: agent cannot reach any exit (disconnected graph).
       NOT described as an NBC regulatory rule — reported separately.

    Reuses NavigationGraph/NetworkX BFS from existing pathfinding.py.
    No new pathfinding implementation.

    cell_size_m is a modelling assumption, NOT a verified NBC value.
    DEFAULT occupancy limit is conservative engineering default, NOT NBC.
    """

    def __init__(self, config_data: dict[str, Any]) -> None:
        rule_cfg = config_data.get("rules", {}).get("R5_TRAVEL_DISTANCE", {})
        self._severity: str = rule_cfg.get("severity", "error")
        self._cell_m: float = rule_cfg.get("cell_size_m", {}).get("value", 1.0)
        self._occ_limits: dict = rule_cfg.get("occupancy_limits", {})
        self._default_spr: bool = rule_cfg.get("default_sprinklered", False)
        self._default_occ: str = rule_cfg.get("default_occupancy", "DEFAULT")

    @property
    def rule_id(self) -> str:
        return "R5_TRAVEL_DISTANCE"

    @property
    def description(self) -> str:
        return (
            "Every occupant must reach an exit within the maximum travel distance "
            "(NBC 2016 Part 4 Table 5). Unreachable occupants → feasibility violation."
        )

    def _get_limit(self, occ_type: Optional[str], is_spr: bool) -> tuple[float, str]:
        occ = occ_type or self._default_occ
        if occ not in self._occ_limits:
            occ = self._default_occ
        cfg = self._occ_limits.get(occ, {})
        key = "sprinklered_m" if is_spr else "non_sprinklered_m"
        limit_m = float(cfg.get(key, 22.5))
        clause = cfg.get("clause", "NBC 2016 Part 4 Table 5")
        return limit_m, clause

    def check(
        self,
        floor_plan: FloorPlan,
        exit_config: ExitConfiguration,
        occupant_scenario: OccupantScenario,
    ) -> list[Violation]:
        if not exit_config.exits:
            return []
        env = Environment.from_floor_plan_and_exits(floor_plan, exit_config)
        nav = NavigationGraph.from_environment(env)
        is_spr: bool = getattr(occupant_scenario, "sprinklered", self._default_spr)
        occ_type: Optional[str] = getattr(occupant_scenario, "occupancy_type", None)
        max_travel_m, clause = self._get_limit(occ_type, is_spr)
        exit_positions: list[GridCell] = [(e[0], e[1]) for e in exit_config.exits]
        violations: list[Violation] = []
        for agent in occupant_scenario.agents:
            apos: GridCell = (agent.row, agent.col)
            min_len: Optional[int] = None
            best_exit: Optional[GridCell] = None
            for ep in exit_positions:
                if not nav._graph.has_node(apos):
                    break
                if not nav._graph.has_node(ep):
                    continue
                try:
                    path = nx.shortest_path(nav._graph, source=apos, target=ep, weight="weight")
                    plen = len(path) - 1
                    if min_len is None or plen < min_len:
                        min_len = plen
                        best_exit = ep
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            if min_len is None:
                violations.append(Violation(
                    rule_id=self.rule_id,
                    message=(
                        "Agent '" + agent.agent_id + "' at " + str(apos)
                        + " cannot reach any selected exit. Feasibility violation."
                    ),
                    severity="error",
                    clause="Feasibility (not a specific NBC clause)",
                    relevant_ids=[agent.agent_id],
                ))
            else:
                travel_m = min_len * self._cell_m
                if travel_m > max_travel_m:
                    violations.append(Violation(
                        rule_id=self.rule_id,
                        message=(
                            "Agent '" + agent.agent_id + "' at " + str(apos)
                            + " must travel " + str(round(travel_m, 1)) + "m to exit "
                            + str(best_exit) + ", exceeding limit " + str(max_travel_m) + "m."
                        ),
                        severity=self._severity,
                        clause=clause,
                        measured_value=round(travel_m, 4),
                        required_value=max_travel_m,
                        unit="m",
                        relevant_ids=[agent.agent_id],
                    ))
        return violations


# ---------------------------------------------------------------------------
# RuleSet
# ---------------------------------------------------------------------------

class RuleSet:
    """
    Ordered collection of Rule objects providing combined check() and mask().
    """

    def __init__(self, rules: list[Rule]) -> None:
        self._rules = list(rules)

    @property
    def rules(self) -> list[Rule]:
        return list(self._rules)

    def check(
        self,
        floor_plan: FloorPlan,
        exit_config: ExitConfiguration,
        occupant_scenario: OccupantScenario,
    ) -> ValidationResult:
        """Evaluate all rules. Does NOT mutate any input."""
        all_violations: list[Violation] = []
        for rule in self._rules:
            all_violations.extend(rule.check(floor_plan, exit_config, occupant_scenario))
        return ValidationResult(
            is_compliant=len(all_violations) == 0,
            violations=all_violations,
        )

    def mask(
        self,
        floor_plan: FloorPlan,
        occupant_scenario: OccupantScenario,
        partial_config: ExitConfiguration,
        candidates: list[CandidateExit],
    ) -> dict[GridCell, bool]:
        """
        Combined action mask.

        A candidate is allowed (True) only when ALL rules that can evaluate
        at partial-selection time agree. Currently only R1 contributes.

        R2, R3, R4, R5 cannot be evaluated on a partial configuration and
        do NOT affect the partial mask — they are evaluated in check() only.
        """
        combined: dict[GridCell, bool] = {
            (c.position[0], c.position[1]): True for c in candidates
        }
        for rule in self._rules:
            for pos, allowed in rule.mask(
                floor_plan, occupant_scenario, partial_config, candidates
            ).items():
                if not allowed:
                    combined[pos] = False
        return combined


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_nbc_2016_ruleset(config_path: Optional[Path] = None) -> RuleSet:
    """
    Build the default NBC 2016 Part 4 rule set from YAML configuration.

    Returns a RuleSet containing R1–R5.
    """
    cfg = load_regulatory_config(config_path)
    return RuleSet(rules=[
        R1ValidExitPosition(cfg),
        R2MinimumExitCount(cfg),
        R3ExitSeparation(cfg),
        R4ExitWidth(cfg),
        R5TravelDistance(cfg),
    ])
