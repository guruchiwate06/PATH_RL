"""
test_stage13_search.py
-----------------------
Stage 13: Non-RL Exit Configuration Search Baseline Tests.

Tests verifying:
A. Search can evaluate multiple configurations.
B. Search uses Stage 11 validation.
C. Invalid configurations are not simulated.
D. Valid configurations are evaluated.
E. Best configuration minimizes total evacuation time.
F. Search result is structured (SearchResult).
G. Total configuration count is correct.
H. Invalid count is correct.
I. Valid count is correct.
J. Evaluated count is correct.
K. Top-K results are correctly ordered.
L. Same inputs produce same result (Determinism).
M. Search does not mutate FloorPlan.
N. Search does not mutate OccupantScenario.
O. Search does not mutate candidate exits.
P. Search uses the Stage 12 evaluator.
Q. Search does not duplicate simulation logic.
R. Search respects requested exit count.
S. Search handles zero valid configurations gracefully.
T. Search handles a single valid configuration.
U. Search reports search runtime.
V. Search result contains best EvaluationResult.
W. Search result contains best ExitConfiguration.
X. Stage 11 validation result is preserved in results.
Y. SearchResult serialization (to_dict).
Z. Search safety limits (max_configurations).
"""

from __future__ import annotations

import copy
import pytest
from pathlib import Path

from evacuation_simulation.simulation.config import (
    AgentConfig,
    AgentProfileConfig,
    DoorOpening,
    ExitConfiguration,
    FloorPlan,
    GridConfig,
    OccupantScenario,
    SimulationConfig,
    SimulationParameters,
)
from evacuation_simulation.simulation.evaluation import (
    ConfigurationEvaluator,
    EvaluationMetrics,
    EvaluationResult,
    rank_evaluations,
)
from evacuation_simulation.simulation.generation import (
    CandidateExit,
    generate_candidate_exits,
    generate_exit_configurations,
)
from evacuation_simulation.simulation.rules import (
    RuleSet,
    ValidationResult,
    Violation,
)
from evacuation_simulation.simulation.search import (
    ExitConfigurationSearcher,
    SearchResult,
    search_exit_configurations,
)
from evacuation_simulation.simulation.validator import ExitValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_floor_plan() -> FloorPlan:
    """10x10 empty room with 4 exterior door candidates on boundary walls."""
    # (0, 5), (9, 5), (5, 0), (5, 9)
    doors = [
        DoorOpening(position=(0, 5), width=1.0, exterior=True, door_id="north_exit"),
        DoorOpening(position=(9, 5), width=1.0, exterior=True, door_id="south_exit"),
        DoorOpening(position=(5, 0), width=1.0, exterior=True, door_id="west_exit"),
        DoorOpening(position=(5, 9), width=1.0, exterior=True, door_id="east_exit"),
    ]
    return FloorPlan(
        grid=GridConfig(rows=10, cols=10),
        walls=[],
        exits=[],
        doors=doors,
    )


@pytest.fixture
def basic_occupant_scenario() -> OccupantScenario:
    """4 agents placed in interior cells."""
    agents = [
        AgentConfig(agent_id="agent_1", row=3, col=3),
        AgentConfig(agent_id="agent_2", row=3, col=6),
        AgentConfig(agent_id="agent_3", row=6, col=3),
        AgentConfig(agent_id="agent_4", row=6, col=6),
    ]
    return OccupantScenario(agents=agents)


@pytest.fixture
def bottleneck_scenario_config() -> SimulationConfig:
    """Load actual bottleneck scenario."""
    scenario_path = Path("scenarios/bottleneck.json")
    if not scenario_path.exists():
        scenario_path = Path(__file__).parent.parent / "scenarios" / "bottleneck.json"
    return SimulationConfig.from_json(scenario_path)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_a_search_evaluates_multiple_configurations(basic_floor_plan, basic_occupant_scenario):
    """A: Search successfully evaluates multiple exit configurations."""
    searcher = ExitConfigurationSearcher()
    result = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=None,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    # 4 candidates choose 2 = 6 configurations
    assert result.total_configurations == 6
    assert len(result.all_evaluations) == 6


def test_b_search_uses_stage11_validation(basic_floor_plan, basic_occupant_scenario):
    """B: Search invokes Stage 11 validation on every configuration."""
    searcher = ExitConfigurationSearcher()
    result = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=None,
        occupant_scenario=basic_occupant_scenario,
        exit_count=1,  # R2 minimum exit count requires 2 exits for NBC 2016
    )
    # Single-exit configs should fail R2 (Minimum 2 exits)
    assert result.total_configurations == 4
    assert result.invalid_configurations == 4
    assert result.valid_configurations == 0
    assert result.evaluated_configurations == 0
    for eval_res in result.all_evaluations:
        assert not eval_res.validation_result.is_compliant
        assert any(v.rule_id.startswith("R2") for v in eval_res.validation_result.violations)


def test_c_invalid_configurations_not_simulated(basic_floor_plan, basic_occupant_scenario):
    """C: Invalid configurations bypass the simulation engine (simulation_result is None)."""
    searcher = ExitConfigurationSearcher()
    result = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=None,
        occupant_scenario=basic_occupant_scenario,
        exit_count=1,
    )
    for eval_res in result.all_evaluations:
        assert eval_res.simulation_result is None
        assert eval_res.metrics is None
        assert eval_res.score is None
        assert eval_res.is_feasible is False


def test_d_valid_configurations_are_evaluated(basic_floor_plan, basic_occupant_scenario):
    """D: Valid configurations are simulated and receive complete evaluation metrics."""
    searcher = ExitConfigurationSearcher()
    result = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=None,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert result.valid_configurations > 0
    assert result.evaluated_configurations == result.valid_configurations
    valid_evals = [er for er in result.all_evaluations if er.validation_result.is_compliant]
    for eval_res in valid_evals:
        assert eval_res.simulation_result is not None
        assert eval_res.metrics is not None
        assert eval_res.score is not None


def test_e_best_configuration_minimizes_total_evacuation_time(basic_floor_plan, basic_occupant_scenario):
    """E: The identified best configuration has the minimal total evacuation time among valid."""
    searcher = ExitConfigurationSearcher()
    result = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=None,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert result.has_feasible_solution
    assert result.best_evaluation is not None
    assert result.best_configuration is not None
    
    valid_scores = [
        er.score for er in result.all_evaluations
        if er.is_feasible and er.score is not None
    ]
    assert result.best_evaluation.score == min(valid_scores)


def test_f_search_result_structured(basic_floor_plan, basic_occupant_scenario):
    """F: Search result is a structured SearchResult with all expected attributes."""
    searcher = ExitConfigurationSearcher()
    result = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=None,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
        scenario_name="test_f",
    )
    assert isinstance(result, SearchResult)
    assert hasattr(result, "best_configuration")
    assert hasattr(result, "best_evaluation")
    assert hasattr(result, "total_configurations")
    assert hasattr(result, "valid_configurations")
    assert hasattr(result, "invalid_configurations")
    assert hasattr(result, "evaluated_configurations")
    assert hasattr(result, "search_time_seconds")
    assert hasattr(result, "evaluations_per_second")
    assert hasattr(result, "top_k_evaluations")
    assert hasattr(result, "all_evaluations")
    assert result.scenario_name == "test_f"


def test_g_total_configuration_count_is_correct(basic_floor_plan, basic_occupant_scenario):
    """G: Total configurations matches combinatorial choose(N, k)."""
    searcher = ExitConfigurationSearcher()
    # 4 candidates choose 2 = 6
    result = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=None,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert result.total_configurations == 6
    # 4 candidates choose 3 = 4
    result3 = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=None,
        occupant_scenario=basic_occupant_scenario,
        exit_count=3,
    )
    assert result3.total_configurations == 4


def test_h_i_j_counts_consistency(basic_floor_plan, basic_occupant_scenario):
    """H, I, J: total == valid + invalid, and evaluated == valid (when all valid are simulated)."""
    searcher = ExitConfigurationSearcher()
    result = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=None,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert result.total_configurations == result.valid_configurations + result.invalid_configurations
    assert result.evaluated_configurations == result.valid_configurations


def test_k_top_k_results_correctly_ordered(basic_floor_plan, basic_occupant_scenario):
    """K: top_k_evaluations is correctly ordered by feasibility then ascending score."""
    searcher = ExitConfigurationSearcher()
    result = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=None,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
        top_k=3,
    )
    assert len(result.top_k_evaluations) <= 3
    scores = [er.score for er in result.top_k_evaluations if er.score is not None]
    assert scores == sorted(scores)


def test_l_determinism_same_inputs_same_result(basic_floor_plan, basic_occupant_scenario):
    """L: Given the same inputs and seed, search produces identical results."""
    params = SimulationParameters(random_seed=12345)
    res1 = search_exit_configurations(
        floor_plan=basic_floor_plan,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
        parameters=params,
    )
    res2 = search_exit_configurations(
        floor_plan=basic_floor_plan,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
        parameters=params,
    )
    assert res1.total_configurations == res2.total_configurations
    assert res1.valid_configurations == res2.valid_configurations
    assert res1.invalid_configurations == res2.invalid_configurations
    assert res1.best_configuration.exits == res2.best_configuration.exits
    assert res1.best_evaluation.score == res2.best_evaluation.score


def test_m_search_does_not_mutate_floor_plan(basic_floor_plan, basic_occupant_scenario):
    """M: Search does not mutate the input FloorPlan."""
    fp_copy = copy.deepcopy(basic_floor_plan)
    search_exit_configurations(
        floor_plan=basic_floor_plan,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert basic_floor_plan.grid == fp_copy.grid
    assert basic_floor_plan.walls == fp_copy.walls
    assert basic_floor_plan.doors == fp_copy.doors


def test_n_search_does_not_mutate_occupant_scenario(basic_floor_plan, basic_occupant_scenario):
    """N: Search does not mutate the input OccupantScenario."""
    occ_copy = copy.deepcopy(basic_occupant_scenario)
    search_exit_configurations(
        floor_plan=basic_floor_plan,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert len(basic_occupant_scenario.agents) == len(occ_copy.agents)
    for a1, a2 in zip(basic_occupant_scenario.agents, occ_copy.agents):
        assert a1.agent_id == a2.agent_id
        assert (a1.row, a1.col) == (a2.row, a2.col)


def test_o_search_does_not_mutate_candidate_exits(basic_floor_plan, basic_occupant_scenario):
    """O: Search does not mutate the input CandidateExit list."""
    candidates = generate_candidate_exits(basic_floor_plan)
    cand_copy = copy.deepcopy(candidates)
    search_exit_configurations(
        floor_plan=basic_floor_plan,
        candidates=candidates,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert len(candidates) == len(cand_copy)
    for c1, c2 in zip(candidates, cand_copy):
        assert c1.position == c2.position
        assert c1.width == c2.width
        assert c1.door_id == c2.door_id


def test_p_search_uses_stage12_evaluator(basic_floor_plan, basic_occupant_scenario):
    """P: Search delegates configuration evaluation to ConfigurationEvaluator."""
    evaluator = ConfigurationEvaluator()
    searcher = ExitConfigurationSearcher(evaluator=evaluator)
    assert searcher.evaluator is evaluator
    res = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=None,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert res.has_feasible_solution


def test_q_search_does_not_create_duplicate_sim_engine(basic_floor_plan, basic_occupant_scenario):
    """Q: Search calls evaluator which runs standard Simulation."""
    searcher = ExitConfigurationSearcher()
    res = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=None,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert res.best_evaluation.simulation_result is not None
    # SimulationResult should have standard timesteps
    assert res.best_evaluation.simulation_result.total_timesteps > 0


def test_r_search_respects_requested_exit_count():
    """R: Configurations tested each have exactly the requested exit_count."""
    # 20x20 floor plan with 3 corner doors separated by >= 10m
    doors = [
        DoorOpening(position=(0, 0), width=1.0, exterior=True, door_id="nw"),
        DoorOpening(position=(0, 19), width=1.0, exterior=True, door_id="ne"),
        DoorOpening(position=(19, 0), width=1.0, exterior=True, door_id="sw"),
    ]
    fp = FloorPlan(
        grid=GridConfig(rows=20, cols=20),
        walls=[],
        doors=doors,
    )
    occ = OccupantScenario(agents=[AgentConfig(agent_id="a1", row=10, col=10)])
    searcher = ExitConfigurationSearcher()
    res = searcher.search(
        floor_plan=fp,
        candidates=None,
        occupant_scenario=occ,
        exit_count=3,
    )
    assert res.best_configuration is not None
    assert len(res.best_configuration.exits) == 3
    assert res.exit_count == 3



def test_s_search_handles_zero_valid_configurations_gracefully(basic_floor_plan, basic_occupant_scenario):
    """S: When zero valid configurations exist, returns structured SearchResult with best=None."""
    searcher = ExitConfigurationSearcher()
    # exit_count=1 violates R2 (NBC 2016 min 2 exits)
    res = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=None,
        occupant_scenario=basic_occupant_scenario,
        exit_count=1,
    )
    assert res.valid_configurations == 0
    assert res.invalid_configurations == 4
    assert res.best_configuration is None
    assert res.best_evaluation is None
    assert res.has_feasible_solution is False


def test_t_search_handles_single_valid_configuration(basic_floor_plan, basic_occupant_scenario):
    """T: Handles search spaces with a single valid configuration cleanly."""
    # 2 candidates with exit_count=2 -> exactly 1 configuration
    cand = [
        CandidateExit(position=(0, 5), width=1.0, door_id="n"),
        CandidateExit(position=(9, 5), width=1.0, door_id="s"),
    ]
    res = search_exit_configurations(
        floor_plan=basic_floor_plan,
        candidates=cand,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert res.total_configurations == 1
    assert res.valid_configurations == 1
    assert res.invalid_configurations == 0
    assert res.best_configuration is not None
    assert len(res.top_k_evaluations) == 1


def test_u_search_reports_runtime_and_throughput(basic_floor_plan, basic_occupant_scenario):
    """U: SearchResult records search_time_seconds >= 0 and valid throughput."""
    res = search_exit_configurations(
        floor_plan=basic_floor_plan,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert res.search_time_seconds >= 0.0
    assert res.evaluations_per_second >= 0.0


def test_v_w_search_result_contains_best_eval_and_config(basic_floor_plan, basic_occupant_scenario):
    """V & W: SearchResult contains matching best_configuration and best_evaluation."""
    res = search_exit_configurations(
        floor_plan=basic_floor_plan,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert isinstance(res.best_configuration, ExitConfiguration)
    assert isinstance(res.best_evaluation, EvaluationResult)
    assert res.best_configuration.configuration_id == res.best_evaluation.configuration_id


def test_x_stage11_validation_result_preserved(basic_floor_plan, basic_occupant_scenario):
    """X: ValidationResult is preserved on all EvaluationResults in the search result."""
    res = search_exit_configurations(
        floor_plan=basic_floor_plan,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    for er in res.all_evaluations:
        assert isinstance(er.validation_result, ValidationResult)
        assert er.validation_result.is_compliant in (True, False)


def test_y_to_dict_serialization(basic_floor_plan, basic_occupant_scenario):
    """Y: to_dict() returns a JSON-serializable dictionary with expected keys."""
    res = search_exit_configurations(
        floor_plan=basic_floor_plan,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
        scenario_name="serial_test",
    )
    d = res.to_dict()
    assert d["scenario_name"] == "serial_test"
    assert d["exit_count"] == 2
    assert d["total_configurations"] == 6
    assert "best_configuration" in d
    assert "best_score" in d
    assert "top_k" in d
    assert len(d["top_k"]) == 3


def test_z_max_configurations_safeguard(basic_floor_plan, basic_occupant_scenario):
    """Z: Exceeding max_configurations raises ValueError."""
    searcher = ExitConfigurationSearcher()
    with pytest.raises(ValueError, match="exceeds max_configurations"):
        searcher.search(
            floor_plan=basic_floor_plan,
            candidates=None,
            occupant_scenario=basic_occupant_scenario,
            exit_count=2,
            max_configurations=3,  # Generated will be 6 > 3
        )


def test_bottleneck_scenario_search(bottleneck_scenario_config):
    """End-to-end exhaustive search test on realistic bottleneck scenario."""
    searcher = ExitConfigurationSearcher()
    res = searcher.search_from_scenario(
        config=bottleneck_scenario_config,
        exit_count=2,
        top_k=3,
    )
    assert res.total_configurations > 0
    assert res.has_feasible_solution
    assert res.best_configuration is not None
    assert res.best_evaluation.score is not None
    assert len(res.top_k_evaluations) >= 1


def test_search_zero_candidates_pool(basic_floor_plan, basic_occupant_scenario):
    """Passing empty candidate list returns clean SearchResult with zero counts."""
    searcher = ExitConfigurationSearcher()
    res = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=[],
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert res.total_configurations == 0
    assert res.valid_configurations == 0
    assert res.best_configuration is None
    assert res.best_evaluation is None
    assert res.has_feasible_solution is False


def test_search_exit_count_larger_than_candidates(basic_floor_plan, basic_occupant_scenario):
    """Requesting exit_count > available candidates returns clean SearchResult without crashing."""
    searcher = ExitConfigurationSearcher()
    res = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=[CandidateExit(position=(0, 5), width=1.0)],
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert res.total_configurations == 0
    assert res.valid_configurations == 0
    assert res.best_configuration is None


def test_search_with_custom_validator(basic_floor_plan, basic_occupant_scenario):
    """Custom ExitValidator injected into ExitConfigurationSearcher is utilized."""
    validator = ExitValidator()
    searcher = ExitConfigurationSearcher(validator=validator)
    res = searcher.search(
        floor_plan=basic_floor_plan,
        candidates=None,
        occupant_scenario=basic_occupant_scenario,
        exit_count=2,
    )
    assert res.total_configurations == 6
    assert res.valid_configurations == 3



def test_basic_building_scenario_search():
    """Search baseline runs on basic_building scenario."""
    scenario_path = Path("scenarios/basic_building.json")
    if not scenario_path.exists():
        scenario_path = Path(__file__).parent.parent / "scenarios" / "basic_building.json"
    cfg = SimulationConfig.from_json(scenario_path)
    searcher = ExitConfigurationSearcher()
    res = searcher.search_from_scenario(
        config=cfg,
        exit_count=2,
        top_k=3,
    )
    assert res.total_configurations > 0
    assert res.search_time_seconds >= 0.0

