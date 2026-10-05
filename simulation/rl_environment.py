"""
rl_environment.py
-----------------
Stage 14A: RL Exit-Placement Formulation.

This module provides the ExitPlacementEnv, a deterministic environment for
formulating the exit-selection problem as a sequential decision process.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

from evacuation_simulation.simulation.config import (
    ExitConfiguration,
    SimulationConfig,
    SimulationParameters,
    GridCell
)
from evacuation_simulation.simulation.evaluation import ConfigurationEvaluator
from evacuation_simulation.simulation.generation import (
    CandidateExit,
    generate_candidate_exits,
)
from evacuation_simulation.simulation.validator import ExitValidator


@dataclass
class RLObservation:
    """
    Structured, deterministic observation representation for the RL agent.

    This representation avoids hardcoding flat tensors so that future models
    (e.g., GNNs or variable-length sequence models) can generalize across
    different floor plans and candidate exit counts.
    """
    grid_shape: tuple[int, int]
    candidate_exits: list[dict[str, Any]]
    selected_candidate_indices: list[int]
    action_mask: list[bool]
    exits_remaining: int
    total_agents: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the observation to a standard Python dictionary."""
        return {
            "grid_shape": list(self.grid_shape),
            "candidate_exits": self.candidate_exits,
            "selected_candidate_indices": list(self.selected_candidate_indices),
            "action_mask": list(self.action_mask),
            "exits_remaining": self.exits_remaining,
            "total_agents": self.total_agents,
        }


class ExitPlacementEnv:
    """
    Reinforcement Learning Environment for sequential exit placement.
    
    The agent observes the building scenario and incrementally selects candidate
    exits until the required exit_count is met. The episode terminates when:
    1. The required number of exits is selected.
    2. The agent selects an invalid/masked action.
    3. No valid actions remain (dead-end).

    Reward formulation minimizes total evacuation time.
    """

    # Reward constants
    PENALTY_INVALID_FINAL_CONFIG = -2000
    PENALTY_INVALID_ACTION = -3000

    def __init__(
        self,
        evaluator: Optional[ConfigurationEvaluator] = None,
        validator: Optional[ExitValidator] = None,
    ) -> None:
        self._evaluator = evaluator or ConfigurationEvaluator(validator=validator)
        self._validator = self._evaluator._validator
        
        # State
        self.config: Optional[SimulationConfig] = None
        self.exit_count: int = 0
        self.candidates: list[CandidateExit] = []
        self.selected_indices: list[int] = []
        self.terminated: bool = True

    def reset(
        self, 
        config: SimulationConfig, 
        exit_count: int,
        candidates: Optional[list[CandidateExit]] = None
    ) -> RLObservation:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        config:
            The simulation configuration containing the floor plan and occupants.
        exit_count:
            The number of exits the agent must select.
        candidates:
            Optional pre-generated candidates. If None, they are generated.

        Returns
        -------
        RLObservation
            The initial state observation.
        """
        # Deep copy config to ensure environment state doesn't mutate original
        self.config = copy.deepcopy(config)
        self.exit_count = exit_count
        
        self.candidates = candidates if candidates is not None else generate_candidate_exits(self.config.floor_plan)
        self.selected_indices = []
        self.terminated = False

        if exit_count > len(self.candidates):
            raise ValueError(f"Requested {exit_count} exits but only {len(self.candidates)} candidates exist.")

        return self._get_obs()

    def _get_partial_config(self) -> ExitConfiguration:
        """Build the ExitConfiguration from currently selected indices."""
        exits = [self.candidates[idx].position for idx in self.selected_indices]
        widths = [self.candidates[idx].width for idx in self.selected_indices]
        return ExitConfiguration(
            configuration_id="partial_rl",
            exits=exits,
            widths=widths
        )

    def action_mask(self) -> list[bool]:
        """
        Compute the action mask for the current state.
        
        True = Action (candidate index) is legal.
        False = Action is illegal (already selected or violates Stage 11 partial rules).
        """
        if self.terminated or self.config is None:
            return [False] * len(self.candidates)

        partial_config = self._get_partial_config()
        
        # Get regulatory mask from validator (handles R1 per-candidate constraints)
        cell_mask = self._validator.mask(
            self.config.floor_plan, 
            self.config.occupants, 
            partial_config, 
            self.candidates
        )
        
        mask = []
        selected_set = set(self.selected_indices)
        for i, candidate in enumerate(self.candidates):
            if i in selected_set:
                mask.append(False)
            else:
                mask.append(cell_mask.get(tuple(candidate.position), True))
                
        return mask

    def _get_obs(self) -> RLObservation:
        """Construct the structured observation."""
        if self.config is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        candidate_dicts = [
            {"position": list(c.position), "width": c.width} 
            for c in self.candidates
        ]
        
        return RLObservation(
            grid_shape=(self.config.floor_plan.grid.rows, self.config.floor_plan.grid.cols),
            candidate_exits=candidate_dicts,
            selected_candidate_indices=list(self.selected_indices),
            action_mask=self.action_mask(),
            exits_remaining=self.exit_count - len(self.selected_indices),
            total_agents=len(self.config.occupants.agents)
        )

    def step(self, action: int) -> tuple[RLObservation, float, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Parameters
        ----------
        action:
            Index of the selected candidate exit.

        Returns
        -------
        tuple[RLObservation, float, bool, dict]
            (observation, reward, terminated, info)
        """
        if self.terminated or self.config is None:
            raise RuntimeError("Episode is already terminated or not reset.")

        mask = self.action_mask()
        
        # 1. Check if action is valid
        if action < 0 or action >= len(self.candidates) or not mask[action]:
            self.terminated = True
            return self._get_obs(), float(self.PENALTY_INVALID_ACTION), True, {"reason": "invalid_action"}

        # Apply action
        self.selected_indices.append(action)
        
        # 2. Check if required number of exits is reached
        if len(self.selected_indices) == self.exit_count:
            self.terminated = True
            reward, info = self._evaluate_final_config()
            return self._get_obs(), reward, True, info

        # 3. Check for dead-end (no legal actions remain before reaching exit_count)
        new_mask = self.action_mask()
        if not any(new_mask):
            self.terminated = True
            return self._get_obs(), float(self.PENALTY_INVALID_ACTION), True, {"reason": "dead_end"}

        # Step successful, episode continues
        return self._get_obs(), 0.0, False, {"reason": "step_success"}

    def _evaluate_final_config(self) -> tuple[float, dict[str, Any]]:
        """Evaluate the final ExitConfiguration using Stage 11 and Stage 12."""
        assert self.config is not None
        
        final_config = self._get_partial_config()
        final_config.configuration_id = f"rl_final_{len(self.selected_indices)}_exits"
        
        # Use evaluator which handles validation + simulation
        eval_result = self._evaluator.evaluate(
            floor_plan=self.config.floor_plan,
            exit_config=final_config,
            occupant_scenario=self.config.occupants,
            parameters=self.config.parameters
        )

        info = {
            "is_feasible": eval_result.is_feasible,
            "is_compliant": eval_result.validation_result.is_compliant,
            "total_evacuation_time": eval_result.metrics.total_evacuation_time if eval_result.metrics else None,
            "violations": [v.rule_id for v in eval_result.validation_result.violations]
        }

        if not eval_result.is_feasible or eval_result.score is None:
            return float(self.PENALTY_INVALID_FINAL_CONFIG), info
            
        # Reward is negative total evacuation time (minimize time -> maximize reward)
        reward = -float(eval_result.score)
        return reward, info
