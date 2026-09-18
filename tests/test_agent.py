"""
test_agent.py
-------------
Tests for the Agent entity and its lifecycle state machine.
"""

from __future__ import annotations

import pytest

from evacuation_simulation.simulation.agent import Agent, AgentState
from evacuation_simulation.simulation.config import AgentConfig


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestAgentConstruction:
    def test_basic_construction(self) -> None:
        agent = Agent(agent_id="a1", row=0, col=0)
        assert agent.agent_id == "a1"
        assert agent.row == 0
        assert agent.col == 0

    def test_default_state_is_moving(self) -> None:
        agent = Agent(agent_id="a1", row=0, col=0)
        assert agent.state == AgentState.MOVING

    def test_custom_initial_state(self) -> None:
        agent = Agent(agent_id="a1", row=0, col=0, initial_state=AgentState.WAITING)
        assert agent.state == AgentState.WAITING

    def test_empty_id_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Agent(agent_id="", row=0, col=0)

    def test_negative_row_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Agent(agent_id="a1", row=-1, col=0)

    def test_negative_col_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Agent(agent_id="a1", row=0, col=-1)

    def test_from_config_factory(self) -> None:
        cfg = AgentConfig(agent_id="a2", row=3, col=5)
        agent = Agent.from_config(cfg)
        assert agent.agent_id == "a2"
        assert agent.row == 3
        assert agent.col == 5


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


class TestAgentPosition:
    def test_position_returns_tuple(self) -> None:
        agent = Agent(agent_id="a1", row=2, col=7)
        assert agent.position == (2, 7)

    def test_set_position_updates_coordinates(self) -> None:
        agent = Agent(agent_id="a1", row=0, col=0)
        agent.set_position(3, 4)
        assert agent.row == 3
        assert agent.col == 4
        assert agent.position == (3, 4)

    def test_set_position_negative_raises(self) -> None:
        agent = Agent(agent_id="a1", row=0, col=0)
        with pytest.raises(ValueError, match="non-negative"):
            agent.set_position(-1, 0)

    def test_set_position_on_evacuated_agent_raises(self) -> None:
        agent = Agent(agent_id="a1", row=0, col=0)
        agent.mark_evacuated(timestep=5)
        with pytest.raises(ValueError, match="evacuated"):
            agent.set_position(1, 1)


# ---------------------------------------------------------------------------
# Evacuation state transition
# ---------------------------------------------------------------------------


class TestAgentEvacuation:
    def test_mark_evacuated_changes_state(self) -> None:
        agent = Agent(agent_id="a1", row=0, col=0)
        agent.mark_evacuated(timestep=10)
        assert agent.state == AgentState.EVACUATED

    def test_mark_evacuated_records_timestep(self) -> None:
        agent = Agent(agent_id="a1", row=0, col=0)
        agent.mark_evacuated(timestep=42)
        assert agent.evacuation_timestep == 42

    def test_double_evacuation_raises(self) -> None:
        agent = Agent(agent_id="a1", row=0, col=0)
        agent.mark_evacuated(timestep=5)
        with pytest.raises(ValueError, match="already evacuated"):
            agent.mark_evacuated(timestep=10)

    def test_is_active_false_after_evacuation(self) -> None:
        agent = Agent(agent_id="a1", row=0, col=0)
        assert agent.is_active is True
        agent.mark_evacuated(timestep=1)
        assert agent.is_active is False

    def test_evacuation_timestep_none_before_evacuation(self) -> None:
        agent = Agent(agent_id="a1", row=0, col=0)
        assert agent.evacuation_timestep is None


# ---------------------------------------------------------------------------
# Equality and hashing
# ---------------------------------------------------------------------------


class TestAgentEquality:
    def test_same_id_agents_are_equal(self) -> None:
        a1 = Agent(agent_id="x", row=0, col=0)
        a2 = Agent(agent_id="x", row=3, col=3)
        assert a1 == a2

    def test_different_id_agents_are_not_equal(self) -> None:
        a1 = Agent(agent_id="x", row=0, col=0)
        a2 = Agent(agent_id="y", row=0, col=0)
        assert a1 != a2

    def test_agent_is_hashable(self) -> None:
        agent = Agent(agent_id="a1", row=0, col=0)
        agent_set = {agent}
        assert agent in agent_set

    def test_agents_usable_as_dict_keys(self) -> None:
        agent = Agent(agent_id="a1", row=0, col=0)
        d = {agent: "value"}
        assert d[agent] == "value"
