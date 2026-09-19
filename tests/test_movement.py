"""
test_movement.py
----------------
Tests for the movement layer:
  - MovementRequest creation
  - validate_request (bounds, wall, distance rules)
  - resolve_conflicts (no conflict, conflict, determinism)
  - apply_movements (positions updated)
"""

from __future__ import annotations

import pytest

from evacuation_simulation.simulation.agent import Agent
from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.movement import (
    MovementRequest,
    apply_movements,
    resolve_conflicts,
    validate_request,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def open_5x5_env() -> Environment:
    """5×5 fully-open grid with one exit at (4,4)."""
    config = SimulationConfig(
        scenario_name="open5",
        grid={"rows": 5, "cols": 5},
        exits=[[4, 4]],
    )
    return Environment.from_config(config)


@pytest.fixture()
def walled_5x5_env() -> Environment:
    """5×5 grid with a wall at (1,0)."""
    config = SimulationConfig(
        scenario_name="walled5",
        grid={"rows": 5, "cols": 5},
        walls=[[1, 0]],
        exits=[[4, 4]],
    )
    return Environment.from_config(config)


# ---------------------------------------------------------------------------
# MovementRequest construction
# ---------------------------------------------------------------------------


class TestMovementRequest:
    def test_is_frozen(self) -> None:
        req = MovementRequest(agent_id="a1", from_cell=(0, 0), to_cell=(0, 1))
        with pytest.raises((AttributeError, TypeError)):
            req.agent_id = "other"  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        req = MovementRequest(agent_id="x", from_cell=(1, 2), to_cell=(1, 3))
        assert req.agent_id == "x"
        assert req.from_cell == (1, 2)
        assert req.to_cell == (1, 3)


# ---------------------------------------------------------------------------
# validate_request
# ---------------------------------------------------------------------------


class TestValidateRequest:
    def test_valid_move_right(self, open_5x5_env: Environment) -> None:
        req = MovementRequest("a", (0, 0), (0, 1))
        assert validate_request(req, open_5x5_env) is True

    def test_valid_move_down(self, open_5x5_env: Environment) -> None:
        req = MovementRequest("a", (0, 0), (1, 0))
        assert validate_request(req, open_5x5_env) is True

    def test_valid_move_up(self, open_5x5_env: Environment) -> None:
        req = MovementRequest("a", (1, 0), (0, 0))
        assert validate_request(req, open_5x5_env) is True

    def test_valid_move_left(self, open_5x5_env: Environment) -> None:
        req = MovementRequest("a", (0, 1), (0, 0))
        assert validate_request(req, open_5x5_env) is True

    def test_valid_move_to_exit(self, open_5x5_env: Environment) -> None:
        req = MovementRequest("a", (3, 4), (4, 4))
        assert validate_request(req, open_5x5_env) is True

    def test_wall_blocked(self, walled_5x5_env: Environment) -> None:
        req = MovementRequest("a", (0, 0), (1, 0))  # (1,0) is a wall
        assert validate_request(req, walled_5x5_env) is False

    def test_out_of_bounds_negative(self, open_5x5_env: Environment) -> None:
        req = MovementRequest("a", (0, 0), (-1, 0))
        assert validate_request(req, open_5x5_env) is False

    def test_out_of_bounds_over_edge(self, open_5x5_env: Environment) -> None:
        req = MovementRequest("a", (0, 4), (0, 5))
        assert validate_request(req, open_5x5_env) is False

    def test_diagonal_move_rejected(self, open_5x5_env: Environment) -> None:
        req = MovementRequest("a", (0, 0), (1, 1))  # diagonal — delta = 2
        assert validate_request(req, open_5x5_env) is False

    def test_two_cell_jump_rejected(self, open_5x5_env: Environment) -> None:
        req = MovementRequest("a", (0, 0), (0, 2))  # two cells right
        assert validate_request(req, open_5x5_env) is False

    def test_staying_in_place_rejected(self, open_5x5_env: Environment) -> None:
        req = MovementRequest("a", (1, 1), (1, 1))  # delta = 0
        assert validate_request(req, open_5x5_env) is False


# ---------------------------------------------------------------------------
# resolve_conflicts
# ---------------------------------------------------------------------------


class TestResolveConflicts:
    def test_no_conflict_both_approved(self) -> None:
        req_a = MovementRequest("a0", (0, 0), (0, 1))
        req_b = MovementRequest("a1", (1, 0), (1, 1))
        approved, rejected, _ = resolve_conflicts([req_a, req_b])
        assert len(approved) == 2
        assert len(rejected) == 0

    def test_single_request_always_approved(self) -> None:
        req = MovementRequest("a0", (0, 0), (0, 1))
        approved, rejected, _ = resolve_conflicts([req])
        assert len(approved) == 1
        assert len(rejected) == 0

    def test_empty_input(self) -> None:
        approved, rejected, reasons = resolve_conflicts([])
        assert approved == []
        assert rejected == []
        assert reasons == {}

    def test_conflict_one_wins_one_rejected(self) -> None:
        req_a = MovementRequest("a0", (0, 0), (0, 1))
        req_b = MovementRequest("a1", (0, 2), (0, 1))  # same destination!
        approved, rejected, _ = resolve_conflicts([req_a, req_b])
        assert len(approved) == 1
        assert len(rejected) == 1

    def test_conflict_winner_is_lex_first(self) -> None:
        """Lexicographically smallest agent_id wins."""
        req_a0 = MovementRequest("a0", (0, 0), (0, 1))
        req_a1 = MovementRequest("a1", (0, 2), (0, 1))
        approved, rejected, _ = resolve_conflicts([req_a0, req_a1])
        assert approved[0].agent_id == "a0"
        assert rejected[0].agent_id == "a1"

    def test_conflict_winner_is_deterministic_regardless_of_input_order(
        self,
    ) -> None:
        """Winner must be the same whether a0 or a1 is first in input."""
        req_a0 = MovementRequest("a0", (0, 0), (0, 1))
        req_a1 = MovementRequest("a1", (0, 2), (0, 1))

        approved1, _, _ = resolve_conflicts([req_a0, req_a1])
        approved2, _, _ = resolve_conflicts([req_a1, req_a0])

        assert approved1[0].agent_id == approved2[0].agent_id == "a0"

    def test_three_way_conflict_one_wins(self) -> None:
        """Among a0, a1, a2 all wanting the same cell, a0 wins."""
        reqs = [
            MovementRequest("a2", (0, 2), (0, 1)),
            MovementRequest("a0", (0, 0), (0, 1)),
            MovementRequest("a1", (1, 1), (0, 1)),
        ]
        approved, rejected, _ = resolve_conflicts(reqs)
        assert len(approved) == 1
        assert len(rejected) == 2
        assert approved[0].agent_id == "a0"

    def test_partial_conflict_some_approved_some_rejected(self) -> None:
        """Two agents conflict, a third goes to a different cell — it's approved."""
        req_a0 = MovementRequest("a0", (0, 0), (0, 1))
        req_a1 = MovementRequest("a1", (0, 2), (0, 1))  # conflict with a0
        req_a2 = MovementRequest("a2", (2, 0), (2, 1))  # no conflict
        approved, rejected, _ = resolve_conflicts([req_a0, req_a1, req_a2])
        assert len(approved) == 2
        assert len(rejected) == 1
        approved_ids = {r.agent_id for r in approved}
        assert "a0" in approved_ids
        assert "a2" in approved_ids


# ---------------------------------------------------------------------------
# apply_movements
# ---------------------------------------------------------------------------


class TestApplyMovements:
    def test_approved_request_moves_agent(self) -> None:
        agent = Agent(agent_id="a0", row=0, col=0)
        agents_by_id = {"a0": agent}
        req = MovementRequest("a0", (0, 0), (0, 1))
        apply_movements([req], agents_by_id)
        assert agent.position == (0, 1)

    def test_multiple_agents_all_moved(self) -> None:
        a0 = Agent(agent_id="a0", row=0, col=0)
        a1 = Agent(agent_id="a1", row=1, col=0)
        agents_by_id = {"a0": a0, "a1": a1}
        reqs = [
            MovementRequest("a0", (0, 0), (0, 1)),
            MovementRequest("a1", (1, 0), (1, 1)),
        ]
        apply_movements(reqs, agents_by_id)
        assert a0.position == (0, 1)
        assert a1.position == (1, 1)

    def test_evacuated_agent_not_moved(self) -> None:
        agent = Agent(agent_id="a0", row=0, col=0)
        agent.mark_evacuated(timestep=1)
        agents_by_id = {"a0": agent}
        req = MovementRequest("a0", (0, 0), (0, 1))
        apply_movements([req], agents_by_id)
        # Position should not change — mark_evacuated freezes movement
        assert agent.position == (0, 0)

    def test_unknown_agent_id_silently_skipped(self) -> None:
        agents_by_id: dict[str, Agent] = {}
        req = MovementRequest("ghost", (0, 0), (0, 1))
        # Should not raise
        apply_movements([req], agents_by_id)

    def test_empty_approved_list_no_changes(self) -> None:
        agent = Agent(agent_id="a0", row=2, col=2)
        agents_by_id = {"a0": agent}
        apply_movements([], agents_by_id)
        assert agent.position == (2, 2)
