"""
search.py
---------
Stage 13: Non-RL Exit Configuration Search Baseline.

PURPOSE
~~~~~~~
Stage 11 answers: "Is this exit configuration allowed?" (Validation)
Stage 12 answers: "How well does this allowed configuration perform?" (Evaluation)
Stage 13 answers: "Among all available feasible exit configurations, which one performs best?" (Search)

This module provides a traditional, deterministic, exhaustive search baseline
that evaluates all combinatorial exit configurations of a given count, rejects
invalid ones according to Stage 11 regulations, evaluates valid ones via
Stage 12 simulation, and identifies the best configuration minimizing total
evacuation time.

ARCHITECTURE & SEPARATION OF RESPONSIBILITIES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
generation.py -> What configurations are possible?
validator.py  -> Is a configuration allowed? (Stage 11)
evaluation.py -> How well does a configuration perform? (Stage 12)
search.py     -> Which configuration is best? (Stage 13 baseline)

SCIENTIFIC REQUIREMENT: FAIR COMPARISON
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All candidate configurations in a search run are evaluated against the EXACT SAME
FloorPlan, OccupantScenario, and SimulationParameters (including random seed).
Only the ExitConfiguration changes.

NON-GOALS (Stage 13)
~~~~~~~~~~~~~~~~~~~~
No reinforcement learning, no neural networks, no policy gradients, no PPO/DQN,
no gym environments, no heuristic approximations, no multiprocessing.
Stage 13 is strictly a deterministic, reproducible non-RL baseline to serve
as the benchmark for future AI/RL methods.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from evacuation_simulation.simulation.config import (
    ExitConfiguration,
    FloorPlan,
    OccupantScenario,
    SimulationConfig,
    SimulationParameters,
)
from evacuation_simulation.simulation.evaluation import (
    ConfigurationEvaluator,
    EvaluationResult,
    rank_evaluations,
)
from evacuation_simulation.simulation.generation import (
    CandidateExit,
    generate_candidate_exits,
    generate_exit_configurations,
)
from evacuation_simulation.simulation.validator import ExitValidator


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """
    Structured outcome of an exit configuration search.

    Attributes
    ----------
    best_configuration:
        The highest-ranking ExitConfiguration that achieved the minimum
        evacuation time among all feasible configurations. None if no valid/feasible
        configuration was found.
    best_evaluation:
        The EvaluationResult corresponding to best_configuration. None if no
        valid/feasible configuration was found.
    total_configurations:
        Total number of candidate exit configurations generated and considered.
    valid_configurations:
        Number of configurations that passed Stage 11 regulatory validation.
    invalid_configurations:
        Number of configurations rejected by Stage 11 regulatory validation.
    evaluated_configurations:
        Number of configurations that were actually simulated (passed validation).
    search_time_seconds:
        Total elapsed wall-clock time spent searching and evaluating.
    evaluations_per_second:
        Throughput metric: evaluated_configurations / search_time_seconds
        (0.0 if search_time_seconds is 0).
    top_k_evaluations:
        Top K ranked EvaluationResults (sorted best-to-worst according to
        Stage 12 primary objective).
    all_evaluations:
        Complete list of all EvaluationResults in evaluated order.
    exit_count:
        The requested exit count per configuration.
    scenario_name:
        Optional name of the evaluated scenario.
    """

    best_configuration: Optional[ExitConfiguration]
    best_evaluation: Optional[EvaluationResult]
    total_configurations: int
    valid_configurations: int
    invalid_configurations: int
    evaluated_configurations: int
    search_time_seconds: float
    evaluations_per_second: float
    top_k_evaluations: list[EvaluationResult] = field(default_factory=list)
    all_evaluations: list[EvaluationResult] = field(default_factory=list)
    exit_count: int = 1
    scenario_name: Optional[str] = None

    @property
    def has_feasible_solution(self) -> bool:
        """True if at least one valid and fully evacuated configuration was found."""
        return self.best_evaluation is not None and self.best_evaluation.is_feasible

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the search result to a plain JSON-compatible dictionary.
        """
        best_cfg_dict = None
        if self.best_configuration is not None:
            best_cfg_dict = {
                "configuration_id": self.best_configuration.configuration_id,
                "exits": [list(pos) for pos in self.best_configuration.exits],
                "widths": list(self.best_configuration.widths),
            }

        top_k_summaries = []
        for rank, er in enumerate(self.top_k_evaluations, start=1):
            top_k_summaries.append({
                "rank": rank,
                "configuration_id": er.configuration_id,
                "is_feasible": er.is_feasible,
                "score": er.score,
                "total_evacuation_time": (
                    er.metrics.total_evacuation_time if er.metrics else None
                ),
                "mean_evacuation_time": (
                    er.metrics.mean_evacuation_time if er.metrics else None
                ),
                "evacuation_rate": (
                    er.metrics.evacuation_rate if er.metrics else 0.0
                ),
                "is_compliant": er.validation_result.is_compliant,
            })

        return {
            "scenario_name": self.scenario_name,
            "exit_count": self.exit_count,
            "total_configurations": self.total_configurations,
            "valid_configurations": self.valid_configurations,
            "invalid_configurations": self.invalid_configurations,
            "evaluated_configurations": self.evaluated_configurations,
            "search_time_seconds": round(self.search_time_seconds, 4),
            "evaluations_per_second": round(self.evaluations_per_second, 2),
            "has_feasible_solution": self.has_feasible_solution,
            "best_configuration": best_cfg_dict,
            "best_score": self.best_evaluation.score if self.best_evaluation else None,
            "top_k": top_k_summaries,
        }


# ---------------------------------------------------------------------------
# ExitConfigurationSearcher
# ---------------------------------------------------------------------------


class ExitConfigurationSearcher:
    """
    Exhaustive Non-RL Exit Configuration Searcher.

    Systematically generates candidate exit configurations, validates them against
    Stage 11 building regulations, evaluates feasible configurations using the
    Stage 12 simulation evaluator, and ranks results by total evacuation time.

    Parameters
    ----------
    evaluator:
        ConfigurationEvaluator instance for Stage 11 validation + Stage 12 simulation.
        If None, creates a default ConfigurationEvaluator.
    validator:
        ExitValidator instance for Stage 11 validation. Used if evaluator is None.
    """

    def __init__(
        self,
        evaluator: Optional[ConfigurationEvaluator] = None,
        validator: Optional[ExitValidator] = None,
    ) -> None:
        if evaluator is not None:
            self._evaluator = evaluator
        else:
            self._evaluator = ConfigurationEvaluator(validator=validator)

    @property
    def evaluator(self) -> ConfigurationEvaluator:
        return self._evaluator

    def search(
        self,
        floor_plan: FloorPlan,
        candidates: Optional[list[CandidateExit]],
        occupant_scenario: OccupantScenario,
        exit_count: int,
        parameters: Optional[SimulationParameters] = None,
        top_k: int = 3,
        max_configurations: Optional[int] = None,
        scenario_name: Optional[str] = None,
    ) -> SearchResult:
        """
        Execute an exhaustive search over all exit combinations of size `exit_count`.

        Parameters
        ----------
        floor_plan:
            Static building geometry. Never mutated.
        candidates:
            Pool of candidate emergency exits. If None, generated automatically from
            `generate_candidate_exits(floor_plan)`.
        occupant_scenario:
            Initial agent placement and profiles. Never mutated.
        exit_count:
            Number of simultaneous exits per configuration.
        parameters:
            Simulation parameters (random seed, max timesteps, etc.). Shared
            identically across all configuration evaluations to ensure fair comparison.
        top_k:
            Number of top-performing configurations to include in top_k_evaluations.
        max_configurations:
            Optional safety limit on the number of generated configurations to evaluate.
            If total configurations exceeds this, raises ValueError.
        scenario_name:
            Optional identifier for reporting and serialization.

        Returns
        -------
        SearchResult
            Structured result containing winner, counts, metrics, and top-k rankings.
        """
        start_time = time.perf_counter()

        # Step 1: Candidate exits pool
        candidate_pool = (
            candidates
            if candidates is not None
            else generate_candidate_exits(floor_plan)
        )

        # Edge case: No candidates or exit_count out of bounds
        if not candidate_pool or exit_count < 1 or exit_count > len(candidate_pool):
            elapsed = time.perf_counter() - start_time
            return SearchResult(
                best_configuration=None,
                best_evaluation=None,
                total_configurations=0,
                valid_configurations=0,
                invalid_configurations=0,
                evaluated_configurations=0,
                search_time_seconds=elapsed,
                evaluations_per_second=0.0,
                top_k_evaluations=[],
                all_evaluations=[],
                exit_count=exit_count,
                scenario_name=scenario_name,
            )

        # Step 2: Generate all combinatorial configurations
        configurations: list[ExitConfiguration] = list(
            generate_exit_configurations(candidate_pool, exit_count=exit_count)
        )
        total_configs = len(configurations)

        if max_configurations is not None and total_configs > max_configurations:
            raise ValueError(
                f"Generated search space ({total_configs} configurations) exceeds "
                f"max_configurations limit ({max_configurations})."
            )

        # Step 3: Evaluate each configuration via Stage 12 evaluator
        # Note: ConfigurationEvaluator internally performs Stage 11 validation first,
        # rejecting non-compliant configurations without running the simulation.
        evaluations: list[EvaluationResult] = []
        config_map: dict[str, ExitConfiguration] = {}

        valid_count = 0
        invalid_count = 0
        evaluated_count = 0

        for cfg in configurations:
            config_map[cfg.configuration_id] = cfg
            eval_res = self._evaluator.evaluate(
                floor_plan=floor_plan,
                exit_config=cfg,
                occupant_scenario=occupant_scenario,
                parameters=parameters,
            )
            evaluations.append(eval_res)

            if eval_res.validation_result.is_compliant:
                valid_count += 1
                if eval_res.simulation_result is not None:
                    evaluated_count += 1
            else:
                invalid_count += 1

        # Step 4: Rank evaluations (Stage 12 primary objective: minimize total evacuation time)
        ranked = rank_evaluations(evaluations)

        # Step 5: Identify the best configuration
        best_eval: Optional[EvaluationResult] = None
        best_cfg: Optional[ExitConfiguration] = None

        if ranked and ranked[0].is_feasible:
            best_eval = ranked[0]
            best_cfg = config_map.get(best_eval.configuration_id)

        # Top-K evaluations (feasible first, then best score)
        k = max(1, top_k)
        top_k_list = ranked[:k] if ranked else []

        elapsed = time.perf_counter() - start_time
        rate = (evaluated_count / elapsed) if elapsed > 0 else 0.0

        return SearchResult(
            best_configuration=best_cfg,
            best_evaluation=best_eval,
            total_configurations=total_configs,
            valid_configurations=valid_count,
            invalid_configurations=invalid_count,
            evaluated_configurations=evaluated_count,
            search_time_seconds=elapsed,
            evaluations_per_second=rate,
            top_k_evaluations=top_k_list,
            all_evaluations=evaluations,
            exit_count=exit_count,
            scenario_name=scenario_name,
        )

    def search_from_scenario(
        self,
        config: SimulationConfig,
        exit_count: int,
        candidates: Optional[list[CandidateExit]] = None,
        top_k: int = 3,
        max_configurations: Optional[int] = None,
    ) -> SearchResult:
        """
        Convenience wrapper to run a search directly from a loaded SimulationConfig.
        """
        return self.search(
            floor_plan=config.floor_plan,
            candidates=candidates,
            occupant_scenario=config.occupants,
            exit_count=exit_count,
            parameters=config.parameters,
            top_k=top_k,
            max_configurations=max_configurations,
            scenario_name=config.scenario_name,
        )


# ---------------------------------------------------------------------------
# Functional convenience API
# ---------------------------------------------------------------------------


def search_exit_configurations(
    floor_plan: FloorPlan,
    occupant_scenario: OccupantScenario,
    exit_count: int,
    candidates: Optional[list[CandidateExit]] = None,
    parameters: Optional[SimulationParameters] = None,
    validator: Optional[ExitValidator] = None,
    evaluator: Optional[ConfigurationEvaluator] = None,
    top_k: int = 3,
    max_configurations: Optional[int] = None,
    scenario_name: Optional[str] = None,
) -> SearchResult:
    """
    Search across candidate exit configurations to find the best configuration
    minimizing total evacuation time.

    Parameters
    ----------
    floor_plan:
        Static building geometry.
    occupant_scenario:
        Occupant positions and profiles.
    exit_count:
        Number of exits per configuration.
    candidates:
        Candidate pool (auto-generated if None).
    parameters:
        SimulationParameters (shared across all runs).
    validator:
        Optional ExitValidator for Stage 11 rules.
    evaluator:
        Optional ConfigurationEvaluator.
    top_k:
        Number of top rankings to return.
    max_configurations:
        Max combination limit safeguard.
    scenario_name:
        Scenario label.

    Returns
    -------
    SearchResult
    """
    searcher = ExitConfigurationSearcher(evaluator=evaluator, validator=validator)
    return searcher.search(
        floor_plan=floor_plan,
        candidates=candidates,
        occupant_scenario=occupant_scenario,
        exit_count=exit_count,
        parameters=parameters,
        top_k=top_k,
        max_configurations=max_configurations,
        scenario_name=scenario_name,
    )
