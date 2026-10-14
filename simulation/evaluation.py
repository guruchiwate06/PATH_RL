"""
evaluation.py
-------------
Stage 12: Exit Configuration Evaluation and Objective Layer.

PURPOSE
~~~~~~~
Stage 11 answers: "Is this exit configuration allowed?"
Stage 12 answers: "How well does this allowed configuration perform?"

This module sits ABOVE the simulator and ABOVE the validator. It:

1. Accepts a FloorPlan, ExitConfiguration, OccupantScenario, and
   SimulationParameters.
2. Runs Stage 11 validation (ExitValidator.check).
3. Rejects invalid configurations before simulation (never scores them).
4. Runs the existing Simulation for valid configurations.
5. Extracts relevant metrics from SimulationResult.
6. Calculates the primary objective score.
7. Returns a structured EvaluationResult.

SEPARATION OF RESPONSIBILITIES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
REGULATORY CONSTRAINT  ->  validator.py / rules.py
SIMULATION METRIC      ->  metrics.py / simulation.py
OBJECTIVE              ->  evaluation.py  (this module)

PRIMARY OBJECTIVE
~~~~~~~~~~~~~~~~~
Minimize total evacuation time.

    score = total_timesteps  (if all agents evacuated)
    score = None             (if evacuation was incomplete)

Lower score = better configuration.

SCIENTIFIC REQUIREMENT: FAIR COMPARISON
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When comparing multiple configurations, the SAME FloorPlan,
OccupantScenario, and SimulationParameters (including random_seed)
MUST be used. Only the ExitConfiguration changes.
This is enforced by the API design: floor_plan and occupant_scenario
are passed once; each configuration is substituted independently.

INCOMPLETE EVACUATION
~~~~~~~~~~~~~~~~~~~~~
A configuration where not all agents evacuated is flagged:
    evacuation_complete = False
Such configurations are scored as None and ranked below any
configuration where evacuation_complete = True.
No numerical penalty is invented; the state is explicit.

NON-GOALS (Stage 12)
~~~~~~~~~~~~~~~~~~~~~
No RL, no PPO, no DQN, no neural networks, no genetic algorithms,
no optimization, no new regulatory rules, no multiprocessing.
Stage 12 is ONLY: EVALUATE -> COMPARE -> RANK.

FUTURE RL COMPATIBILITY
~~~~~~~~~~~~~~~~~~~~~~~~
The ConfigurationEvaluator API is intentionally simple so future
search or RL code can call:

    evaluator.evaluate(floor_plan, exit_config, occupant_scenario)

without needing to know internal details.
No RL library dependencies are introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from evacuation_simulation.simulation.config import (
    ExitConfiguration,
    FloorPlan,
    OccupantScenario,
    SimulationConfig,
    SimulationParameters,
)
from evacuation_simulation.simulation.metrics import SimulationResult
from evacuation_simulation.simulation.rules import ValidationResult
from evacuation_simulation.simulation.simulation import Simulation
from evacuation_simulation.simulation.strategy import ShortestPathStrategy
from evacuation_simulation.simulation.validator import ExitValidator


# ---------------------------------------------------------------------------
# EvaluationMetrics
# ---------------------------------------------------------------------------


@dataclass
class EvaluationMetrics:
    """
    Performance metrics extracted from a SimulationResult for one
    evaluated exit configuration.

    All fields map 1-to-1 to existing SimulationResult attributes.
    No metrics are invented here.

    Attributes
    ----------
    total_evacuation_time:
        Total timesteps the simulation ran. This is the PRIMARY OBJECTIVE
        metric. Lower is better.
        None if evacuation was incomplete (not all agents evacuated).
    mean_evacuation_time:
        Mean timestep at which evacuated agents reached an exit.
        None if no agents evacuated.
    min_evacuation_time:
        Earliest evacuation timestep. None if no agents evacuated.
    max_evacuation_time:
        Latest evacuation timestep. None if no agents evacuated.
    evacuation_rate:
        Fraction of agents that evacuated (0.0 to 1.0).
    total_agents:
        Total agent count for this run.
    evacuated_count:
        Number of agents that reached an exit.
    non_evacuated_count:
        Number of agents that did NOT evacuate by simulation end.
    mean_waiting_steps:
        Mean capacity-blocked waiting steps per agent.
    max_waiting_steps:
        Maximum capacity-blocked waiting steps for any single agent.
    total_waiting_steps:
        Sum of capacity-blocked waiting steps across all agents.
    congested_cell_steps:
        Cumulative (cell, timestep) pairs where occupancy >= capacity.
    max_cell_occupancy:
        Highest single-cell occupancy recorded.
    max_occupancy_ratio:
        Highest occupancy ratio (occupancy / capacity) recorded.
    exit_utilization:
        Number of agents that evacuated via each exit cell.
        Keys are (row, col) tuples.
    """

    total_evacuation_time: Optional[int]
    mean_evacuation_time: Optional[float]
    min_evacuation_time: Optional[int]
    max_evacuation_time: Optional[int]
    evacuation_rate: float
    total_agents: int
    evacuated_count: int
    non_evacuated_count: int
    mean_waiting_steps: float
    max_waiting_steps: int
    total_waiting_steps: int
    congested_cell_steps: int
    max_cell_occupancy: int
    max_occupancy_ratio: float
    exit_utilization: dict = field(default_factory=dict)

    @classmethod
    def from_simulation_result(
        cls,
        result: SimulationResult,
        *,
        evacuation_complete: bool,
    ) -> "EvaluationMetrics":
        """
        Build EvaluationMetrics from a SimulationResult.

        Parameters
        ----------
        result:
            The SimulationResult produced by Simulation.run().
        evacuation_complete:
            True if all agents evacuated. When False, total_evacuation_time
            is set to None to signal an incomplete evacuation.
        """
        total_time: Optional[int] = (
            result.total_timesteps if evacuation_complete else None
        )
        return cls(
            total_evacuation_time=total_time,
            mean_evacuation_time=result.mean_evacuation_time,
            min_evacuation_time=result.min_evacuation_time,
            max_evacuation_time=result.max_evacuation_time,
            evacuation_rate=result.evacuation_rate,
            total_agents=result.total_agents,
            evacuated_count=result.evacuated_count,
            non_evacuated_count=result.non_evacuated_count,
            mean_waiting_steps=result.mean_waiting_steps,
            max_waiting_steps=result.max_waiting_steps,
            total_waiting_steps=result.total_waiting_steps,
            congested_cell_steps=result.congested_cell_steps,
            max_cell_occupancy=result.max_cell_occupancy,
            max_occupancy_ratio=result.max_occupancy_ratio,
            exit_utilization=dict(result.per_exit_utilization),
        )


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """
    Structured result of evaluating one exit configuration.

    Attributes
    ----------
    configuration_id:
        Identifier of the ExitConfiguration that was evaluated.
    is_feasible:
        True if the configuration passed Stage 11 validation AND
        resulted in a complete evacuation (all agents evacuated).
        False if rejected by the validator OR evacuation was incomplete.
    evacuation_complete:
        True if all agents evacuated before the timestep limit.
        None if the configuration was invalid (simulation never ran).
    validation_result:
        The Stage 11 ValidationResult. Always populated.
    simulation_result:
        The raw SimulationResult. None if the configuration was invalid
        (validation failed; simulation was not run).
    metrics:
        Structured EvaluationMetrics. None if simulation never ran.
    score:
        Primary objective value: total_evacuation_time.
        Lower is better. None if the configuration is invalid or
        evacuation was incomplete (not all agents evacuated).

    NOTE ON FEASIBILITY:
        A configuration is considered fully feasible only when both:
        (a) it passes Stage 11 regulatory validation, and
        (b) the evacuation completes (all agents evacuate).
        A configuration that passes regulation but traps agents is
        still flagged is_feasible=False so it cannot outrank a
        configuration that achieves complete evacuation.

    NOTE ON RANKING:
        Invalid configurations (is_feasible=False) are ALWAYS ranked
        below any configuration with is_feasible=True, regardless of
        score values. Two infeasible configurations have undefined
        relative ordering (they are not scored).
    """

    configuration_id: str
    is_feasible: bool
    evacuation_complete: Optional[bool]
    validation_result: ValidationResult
    simulation_result: Optional[SimulationResult]
    metrics: Optional[EvaluationMetrics]
    score: Optional[int]

    def summary_line(self) -> str:
        """Return a compact one-line summary for display in tables."""
        if not self.validation_result.is_compliant:
            return (
                f"[{self.configuration_id}] INVALID (regulatory) | "
                f"score=- | time=- | rate=-"
            )
        if self.metrics is None:
            return f"[{self.configuration_id}] NO_METRICS"
        evac_str = (
            str(self.metrics.total_evacuation_time)
            if self.metrics.total_evacuation_time is not None
            else "INCOMPLETE"
        )
        return (
            f"[{self.configuration_id}] "
            f"feasible={self.is_feasible} | "
            f"score={self.score} | "
            f"time={evac_str} | "
            f"rate={self.metrics.evacuation_rate:.0%}"
        )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def rank_evaluations(
    evaluations: list[EvaluationResult],
) -> list[EvaluationResult]:
    """
    Rank a list of EvaluationResults by feasibility then primary objective.

    Ranking rules:
    1. Feasible configurations (is_feasible=True) always rank above
       infeasible ones (is_feasible=False).
    2. Among feasible configurations, lower score (lower total evacuation
       time) ranks higher.
    3. Infeasible configurations retain their original relative order
       (stable sort within the infeasible group).

    Parameters
    ----------
    evaluations:
        List of EvaluationResult objects to rank.

    Returns
    -------
    list[EvaluationResult]
        A new list sorted from best (rank 1) to worst.
    """

    def _sort_key(er: EvaluationResult) -> tuple:
        feasibility_key = 0 if er.is_feasible else 1
        score_key = er.score if er.score is not None else float("inf")
        return (feasibility_key, score_key)

    return sorted(evaluations, key=_sort_key)


# ---------------------------------------------------------------------------
# ConfigurationEvaluator
# ---------------------------------------------------------------------------


class ConfigurationEvaluator:
    """
    Evaluates exit configurations by running Stage 11 validation
    followed by evacuation simulation.

    PIPELINE (per configuration):

        ExitConfiguration
              |
        ExitValidator.check()      <- Stage 11
              |
        INVALID -> Reject (EvaluationResult, is_feasible=False)
        VALID   -> Simulation.run()
                        |
                   SimulationResult
                        |
                  EvaluationMetrics
                        |
                   primary objective score
                        |
                   EvaluationResult

    Parameters
    ----------
    validator:
        ExitValidator instance for Stage 11 checks.
        Defaults to the standard NBC 2016 validator.

    Example
    -------
    >>> evaluator = ConfigurationEvaluator()
    >>> result = evaluator.evaluate(floor_plan, exit_config, occupant_scenario)
    >>> result.score
    42
    >>> result.metrics.mean_evacuation_time
    18.3
    """

    def __init__(self, validator: Optional[ExitValidator] = None) -> None:
        self._validator: ExitValidator = validator or ExitValidator()

    # ------------------------------------------------------------------
    # Single evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        floor_plan: FloorPlan,
        exit_config: ExitConfiguration,
        occupant_scenario: OccupantScenario,
        parameters: Optional[SimulationParameters] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single exit configuration.

        Does NOT mutate floor_plan, exit_config, or occupant_scenario.

        Parameters
        ----------
        floor_plan:
            Static building geometry. Shared across calls when comparing
            configurations; never mutated.
        exit_config:
            The exit configuration to evaluate.
        occupant_scenario:
            Initial agent placement and profiles. Shared across calls
            when comparing configurations; never mutated.
        parameters:
            Simulation execution parameters (max_timesteps, random_seed,
            cell_capacity). None uses defaults. Supply the same parameters
            (especially random_seed) for all configurations in a comparison
            to ensure a fair, scientifically meaningful comparison.

        Returns
        -------
        EvaluationResult
            Structured result containing validation outcome, simulation
            metrics, and primary objective score.
        """
        params = parameters or SimulationParameters()

        # --- Step 1: Stage 11 validation ---
        validation_result: ValidationResult = self._validator.check(
            floor_plan, exit_config, occupant_scenario
        )

        if not validation_result.is_compliant:
            # Rejected: do not simulate. Return explicit infeasible result.
            return EvaluationResult(
                configuration_id=exit_config.configuration_id,
                is_feasible=False,
                evacuation_complete=None,
                validation_result=validation_result,
                simulation_result=None,
                metrics=None,
                score=None,
            )

        # --- Step 2: Assemble SimulationConfig (no mutation of inputs) ---
        sim_config = SimulationConfig(
            scenario_name=exit_config.configuration_id,
            floor_plan=floor_plan,
            exit_configuration=exit_config,
            occupants=occupant_scenario,
            parameters=params,
        )

        # --- Step 3: Run simulation (reuse existing engine) ---
        strategy = ShortestPathStrategy()
        sim = Simulation(config=sim_config, movement_strategy=strategy)
        sim_result: SimulationResult = sim.run()

        # --- Step 4: Determine evacuation completeness ---
        evacuation_complete: bool = sim_result.non_evacuated_count == 0

        # --- Step 5: Extract structured metrics ---
        metrics = EvaluationMetrics.from_simulation_result(
            sim_result, evacuation_complete=evacuation_complete
        )

        # --- Step 6: Compute primary objective score ---
        # PRIMARY OBJECTIVE: minimize total evacuation time.
        # Score is None if evacuation was incomplete.
        score: Optional[int] = (
            sim_result.total_timesteps if evacuation_complete else None
        )

        # is_feasible: regulatory compliance AND complete evacuation
        is_feasible: bool = evacuation_complete

        return EvaluationResult(
            configuration_id=exit_config.configuration_id,
            is_feasible=is_feasible,
            evacuation_complete=evacuation_complete,
            validation_result=validation_result,
            simulation_result=sim_result,
            metrics=metrics,
            score=score,
        )

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_configurations(
        self,
        floor_plan: FloorPlan,
        configurations: list[ExitConfiguration],
        occupant_scenario: OccupantScenario,
        parameters: Optional[SimulationParameters] = None,
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple exit configurations against the same scenario.

        SCIENTIFIC REQUIREMENT:
        The same floor_plan, occupant_scenario, and parameters (including
        random_seed) are used for every configuration. Only exit_config
        changes. This ensures results are directly comparable.

        Parameters
        ----------
        floor_plan:
            Static building geometry. Never mutated.
        configurations:
            List of ExitConfiguration objects to evaluate.
        occupant_scenario:
            Occupant placement and profiles. Never mutated.
        parameters:
            Shared simulation parameters for all evaluations.

        Returns
        -------
        list[EvaluationResult]
            One EvaluationResult per configuration, in the same order
            as the input list (NOT sorted by rank).
            Use rank_evaluations() to sort.
        """
        return [
            self.evaluate(floor_plan, cfg, occupant_scenario, parameters=parameters)
            for cfg in configurations
        ]
