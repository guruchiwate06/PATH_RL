"""
rl_environment.py
-----------------
Stage 14A: RL Exit-Placement Formulation.

This module provides the ExitPlacementEnv, a deterministic environment for
formulating the exit-selection problem as a sequential decision process.

WHAT THE RL AGENT CONTROLS
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The agent selects which candidate exit doors to activate. It operates at the
floor-plan / exit-configuration level, NOT at the individual-agent movement
level. Shortest-path evacuation movement (Stages 1-7) is unchanged.

WHAT THE RL AGENT DOES NOT CONTROL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- How people walk/navigate (unchanged movement engine).
- Whether a configuration is legally valid (Stage 11 remains authoritative).
- How evacuation performance is measured (Stage 12 remains authoritative).

EPISODE LIFECYCLE
~~~~~~~~~~~~~~~~~
1. reset(config, exit_count) -- supply floor plan, occupants, required exits.
2. Repeated step(action)    -- agent selects one candidate exit per step.
3. Termination occurs when:
   a. Required exits selected (normal end -> Stage 11 + 12 evaluation).
   b. Agent selects illegal action (PENALTY_INVALID_ACTION).
   c. No legal action remains before required count (PENALTY_DEAD_END).

OBSERVATION REPRESENTATION
~~~~~~~~~~~~~~~~~~~~~~~~~~~
RLObservation is a structured dataclass, NOT a fixed-length flat array.
Variable-length candidate lists allow the same env to be used across
buildings with different candidate exit counts without padding or truncation.

Fields:
  grid_shape              : (rows, cols) of the building grid.
  candidate_exits         : list of dicts, one per candidate:
                              position   : [row, col], raw grid ints.
                              width      : float, metres.
                              door_id    : str|None, from DoorOpening.
                              is_selected: bool, selected this episode.
  selected_candidate_indices: ordered list of 0-indexed selected candidates.
  action_mask             : list[bool] parallel to candidate_exits.
                              True  = legal next action.
                              False = already selected or R1-forbidden.
  exits_remaining         : int, how many more exits are needed.
  total_agents            : int, number of building occupants.

ACTION SPACE
~~~~~~~~~~~~
Action = 0-indexed position in the candidate_exits list.
Illegal actions (already selected, out of range, R1-forbidden) are
blocked by action_mask and result in episode termination with penalty.
Interior corridor passages never appear as candidates (filtered during
candidate generation by exterior=False), so they are never in the action space.

REWARD DESIGN
~~~~~~~~~~~~~
Primary objective: MINIMIZE total evacuation time (consistent with Stage 12/13).

  PENALTY_INVALID_ACTION       = -3000
    Illegal action (out of range or already selected). Strongest penalty.

  PENALTY_DEAD_END             = -2500
    No legal action remains before required exit count reached.
    Distinct constant to allow separate analysis of this failure mode.

  PENALTY_INVALID_FINAL_CONFIG = -2000
    Required exits selected but Stage 11 rejects configuration OR
    Stage 12 evacuation is incomplete.

  Normal reward = -total_evacuation_time  (negative integer)
    Lower time -> higher (less negative) reward -> better policy.
    Directly comparable with Stage 13 primary objective score.

STAGE 13 COMPATIBILITY
~~~~~~~~~~~~~~~~~~~~~~~
For the same FloorPlan, OccupantScenario, candidates, exit_count, parameters
and seed, the reward from this env for a given action sequence equals
-1 * Stage 13 best_evaluation.score for the same configuration.

NO NEW DEPENDENCY
~~~~~~~~~~~~~~~~~
No RL library (Gymnasium, Stable-Baselines3, PyTorch, TensorFlow) is imported.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

from evacuation_simulation.simulation.config import (
    ExitConfiguration,
    SimulationConfig,
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

    Attributes
    ----------
    grid_shape:
        (rows, cols) of the building floor-plan grid.
    candidate_exits:
        One dict per candidate: position [row, col], width (metres),
        door_id (str or None), is_selected (bool for this episode).
    selected_candidate_indices:
        Ordered list of 0-indexed candidates selected so far.
    action_mask:
        Parallel to candidate_exits. True = legal next action.
    exits_remaining:
        How many more exits must be selected to complete the episode.
    total_agents:
        Number of building occupants (reward context feature).
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

    # Reward constants (see module docstring for justification)
    PENALTY_INVALID_ACTION: int = -3000        # strongest -- illegal action
    PENALTY_DEAD_END: int = -2500              # mid -- episode stuck
    PENALTY_INVALID_FINAL_CONFIG: int = -2000  # weakest failure -- count reached

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
        if exit_count < 1:
            raise ValueError(f"Requested exit_count must be at least 1, got {exit_count}.")

        if exit_count > len(self.candidates):
            raise ValueError(f"Requested {exit_count} exits but only {len(self.candidates)} candidates exist.")

        initial_mask = self.action_mask()
        valid_count = sum(1 for m in initial_mask if m)
        if valid_count < exit_count:
            invalid_details = [
                f"{c.door_id or c.position} (width={c.width}m)"
                for c, m in zip(self.candidates, initial_mask)
                if not m
            ]
            raise ValueError(
                f"Cannot satisfy requested exit_count={exit_count}: only {valid_count} of {len(self.candidates)} "
                f"candidate(s) meet regulatory position (R1) and minimum width thresholds (R4). "
                f"Ineligible candidates: {', '.join(invalid_details)}."
            )

        return self._get_obs()

    def _build_partial_config(self) -> ExitConfiguration:
        """Build the ExitConfiguration from currently selected indices."""
        exits = [self.candidates[idx].position for idx in self.selected_indices]
        widths = [self.candidates[idx].width for idx in self.selected_indices]
        return ExitConfiguration(
            configuration_id="partial_rl",
            exits=exits,
            widths=widths,
        )

    # Backward-compatible alias so that any callers using the old name still work
    _get_partial_config = _build_partial_config

    def action_mask(self) -> list[bool]:
        """
        Compute the action mask for the current state.

        True  = candidate may be selected as the next exit.
        False = candidate is illegal (already selected, or violates Stage 11 R1).

        Notes
        -----
        Only R1 (per-candidate) constraints from Stage 11 are enforceable
        on a partial configuration. R2-R5 require the complete configuration
        and are checked at episode end via check().

        Interior corridor passages (exterior=False) never appear in the
        candidate list and therefore never appear in the action space.

        The mask is deterministic: identical state -> identical mask.
        When terminated or uninitialised, all entries are False.
        """
        if self.terminated or self.config is None:
            return [False] * len(self.candidates)

        partial_config = self._build_partial_config()

        # Stage 11 per-candidate regulatory mask
        cell_mask = self._validator.mask(
            self.config.floor_plan,
            self.config.occupants,
            partial_config,
            self.candidates,
        )

        mask = []
        selected_set = set(self.selected_indices)
        for i, candidate in enumerate(self.candidates):
            if i in selected_set:
                mask.append(False)
            else:
                pos_key = (int(candidate.position[0]), int(candidate.position[1]))
                mask.append(cell_mask.get(pos_key, True))

        return mask

    def _get_obs(self) -> RLObservation:
        """Construct the structured observation from current episode state."""
        if self.config is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        selected_set = set(self.selected_indices)
        candidate_dicts = [
            {
                "position": list(c.position),
                "width": c.width,
                "door_id": c.door_id,
                "is_selected": (i in selected_set),
            }
            for i, c in enumerate(self.candidates)
        ]

        return RLObservation(
            grid_shape=(self.config.floor_plan.grid.rows, self.config.floor_plan.grid.cols),
            candidate_exits=candidate_dicts,
            selected_candidate_indices=list(self.selected_indices),
            action_mask=self.action_mask(),
            exits_remaining=self.exit_count - len(self.selected_indices),
            total_agents=len(self.config.occupants.agents),
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
            return (
                self._get_obs(),
                float(self.PENALTY_INVALID_ACTION),
                True,
                {"reason": "invalid_action", "is_feasible": None,
                 "is_compliant": None, "total_evacuation_time": None,
                 "violations": []},
            )

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
            return (
                self._get_obs(),
                float(self.PENALTY_DEAD_END),
                True,
                {"reason": "dead_end", "is_feasible": None,
                 "is_compliant": None, "total_evacuation_time": None,
                 "violations": []},
            )

        # Step successful, episode continues
        return (
            self._get_obs(),
            0.0,
            False,
            {"reason": "step_success", "is_feasible": None,
             "is_compliant": None, "total_evacuation_time": None,
             "violations": []},
        )

    def _evaluate_final_config(self) -> tuple[float, dict[str, Any]]:
        """Evaluate the final ExitConfiguration using Stage 11 and Stage 12."""
        assert self.config is not None

        final_config = self._build_partial_config()
        final_config.configuration_id = f"rl_final_{len(self.selected_indices)}_exits"

        # Stage 11 validation + Stage 12 simulation via existing evaluator
        eval_result = self._evaluator.evaluate(
            floor_plan=self.config.floor_plan,
            exit_config=final_config,
            occupant_scenario=self.config.occupants,
            parameters=self.config.parameters,
        )

        info: dict[str, Any] = {
            "reason": "episode_complete",
            "is_feasible": eval_result.is_feasible,
            "is_compliant": eval_result.validation_result.is_compliant,
            "total_evacuation_time": (
                eval_result.metrics.total_evacuation_time
                if eval_result.metrics is not None
                else None
            ),
            "violations": [
                v.rule_id for v in eval_result.validation_result.violations
            ],
        }

        if not eval_result.is_feasible or eval_result.score is None:
            return float(self.PENALTY_INVALID_FINAL_CONFIG), info

        # Primary reward: negative total evacuation time
        return -float(eval_result.score), info

    def render(self) -> None:
        """No-op stub. Reserved for future ASCII/HTML visualisation."""
        pass

    def close(self) -> None:
        """No-op stub. Reserved for resource cleanup in future wrappers."""
        pass
