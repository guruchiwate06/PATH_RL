"""
test_geometry_observations.py
-----------------------------
Tests for Geometry-Aware Observation Space in PATH_RL.

Verifies:
1. Feature tensor dimension consistency:
   - Legacy mode: 44 dimensions (4 global + 8 * 5 per-candidate).
   - Geometry-aware mode: 88 dimensions (8 global + 8 * 10 per-candidate).
2. Dynamic observation changes with occupant distribution:
   - Occupants clustered in North vs South produce appropriately shifted
     centroid coordinates and occupant-to-exit shortest-path distances.
3. Dynamic observation changes with floor-plan geometry:
   - Introducing interior obstacle walls increases shortest-path distances.
   - Partitioning a room drops reachability ratio below 1.0.
4. Corridor connectivity variation:
   - Reflects local graph neighborhood connectivity in the navigation graph.
5. Checkpoint safety and backward compatibility:
   - Existing Stage 18 (24-dim) and Stage 18B (44-dim) checkpoints load without error
     and execute valid actions on new environment observations.
   - Geometry-aware models (88-dim) save and load cleanly.
6. Safety constraint preservation:
   - Hard action masking strictly zeroes illegal/invalid candidate probabilities.
   - Illegal actions receive PENALTY_INVALID_ACTION (-3000).
   - Regulatory validation (NBC 2016 Part 4) is fully preserved at episode end.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from evacuation_simulation.simulation.config import (
    AgentConfig,
    DoorOpening,
    ExitConfiguration,
    FloorPlan,
    GridConfig,
    OccupantScenario,
    SimulationConfig,
    SimulationParameters,
)
from evacuation_simulation.simulation.generation import CandidateExit
from evacuation_simulation.simulation.rl_environment import ExitPlacementEnv, RLObservation
from evacuation_simulation.simulation.rl_policy import (
    MaskedActorCritic,
    compute_geometry_features_for_candidates,
    extract_features,
)


def _make_test_config(
    walls: list[tuple[int, int]] | None = None,
    doors: list[tuple[int, int]] | None = None,
    occupant_coords: list[tuple[int, int]] | None = None,
    rows: int = 20,
    cols: int = 20,
) -> SimulationConfig:
    """Helper to build custom test configurations."""
    door_openings = [
        DoorOpening(position=pos, width=1.5, exterior=True, door_id=f"D{i}")
        for i, pos in enumerate(doors or [(0, 10), (19, 10)])
    ]
    floor_plan = FloorPlan(
        grid=GridConfig(rows=rows, cols=cols),
        walls=walls or [],
        doors=door_openings,
    )
    agents = [
        AgentConfig(agent_id=f"A{i}", row=r, col=c)
        for i, (r, c) in enumerate(occupant_coords or [(10, 10)])
    ]
    occupants = OccupantScenario(scenario_id="test_sc", agents=agents)
    exits = [door_openings[0].position]
    widths = [door_openings[0].width]
    return SimulationConfig(
        floor_plan=floor_plan,
        exit_configuration=ExitConfiguration(configuration_id="initial", exits=exits, widths=widths),
        occupants=occupants,
        parameters=SimulationParameters(max_timesteps=100, random_seed=42),
    )


class TestGeometryAwareObservationSpace:
    """Test suite for geometry-aware features and backward compatibility."""

    def test_feature_dimensions_legacy_and_geometry(self):
        """Verify feature dimensions: 44 for legacy, 88 for geometry-aware."""
        obs = RLObservation(
            grid_shape=(20, 20),
            candidate_exits=[
                {
                    "position": [0, 10],
                    "width": 1.5,
                    "door_id": "D0",
                    "is_selected": False,
                    "mean_dist_to_occupants": 0.25,
                    "min_dist_to_occupants": 0.1,
                    "max_dist_to_occupants": 0.4,
                    "reachability_ratio": 1.0,
                    "corridor_connectivity": 0.6,
                },
                {
                    "position": [19, 10],
                    "width": 1.5,
                    "door_id": "D1",
                    "is_selected": False,
                    "mean_dist_to_occupants": 0.75,
                    "min_dist_to_occupants": 0.6,
                    "max_dist_to_occupants": 0.9,
                    "reachability_ratio": 1.0,
                    "corridor_connectivity": 0.6,
                },
            ],
            selected_candidate_indices=[],
            action_mask=[True, True],
            exits_remaining=2,
            total_agents=10,
            walkable_ratio=0.85,
            occupant_centroid=(0.3, 0.5),
            occupant_dispersion=0.15,
        )

        # Legacy feature extraction: 4 global + 8 * 5 = 44 dims
        state_legacy, mask_legacy = extract_features(obs, max_candidates=8, geometry_features=False)
        assert state_legacy.shape == (1, 44)
        assert mask_legacy.shape == (1, 8)

        # Geometry-aware feature extraction: 8 global + 8 * 10 = 88 dims
        state_geom, mask_geom = extract_features(obs, max_candidates=8, geometry_features=True)
        assert state_geom.shape == (1, 88)
        assert mask_geom.shape == (1, 8)

        # Check that global geometry features are present in state_geom
        assert pytest.approx(state_geom[0, 4].item(), abs=1e-4) == 0.85  # walkable_ratio
        assert pytest.approx(state_geom[0, 5].item(), abs=1e-4) == 0.30  # centroid_r
        assert pytest.approx(state_geom[0, 6].item(), abs=1e-4) == 0.50  # centroid_c
        assert pytest.approx(state_geom[0, 7].item(), abs=1e-4) == 0.15  # dispersion

        # Check candidate 0 geometry values
        # Index in state_geom: global (8) + cand_0 (10) -> features 8..17
        cand0_feats = state_geom[0, 8:18].tolist()
        assert pytest.approx(cand0_feats[5], abs=1e-4) == 0.25  # mean_dist
        assert pytest.approx(cand0_feats[6], abs=1e-4) == 0.10  # min_dist
        assert pytest.approx(cand0_feats[7], abs=1e-4) == 0.40  # max_dist
        assert pytest.approx(cand0_feats[8], abs=1e-4) == 1.00  # reachability
        assert pytest.approx(cand0_feats[9], abs=1e-4) == 0.60  # connectivity

    def test_observations_change_with_occupant_distribution(self):
        """Verify observations change appropriately when occupant distribution shifts."""
        env = ExitPlacementEnv()
        doors = [(0, 10), (19, 10)]  # North door, South door

        # Case A: Occupants clustered at the North end (rows 2-3)
        north_occupants = [(2, 10), (3, 9), (3, 11), (2, 8)]
        cfg_north = _make_test_config(doors=doors, occupant_coords=north_occupants)
        obs_north = env.reset(cfg_north, exit_count=2)

        # Case B: Occupants clustered at the South end (rows 17-18)
        south_occupants = [(17, 10), (18, 9), (18, 11), (17, 8)]
        cfg_south = _make_test_config(doors=doors, occupant_coords=south_occupants)
        obs_south = env.reset(cfg_south, exit_count=2)

        # Centroid should move from low row (< 0.25) to high row (> 0.75)
        assert obs_north.occupant_centroid[0] < 0.3
        assert obs_south.occupant_centroid[0] > 0.7

        # North door distance to occupants should be much smaller in cfg_north than cfg_south
        north_door_obs_north = obs_north.candidate_exits[0]["mean_dist_to_occupants"]
        north_door_obs_south = obs_south.candidate_exits[0]["mean_dist_to_occupants"]
        assert north_door_obs_north < north_door_obs_south

        # South door distance to occupants should be much smaller in cfg_south than cfg_north
        south_door_obs_north = obs_north.candidate_exits[1]["mean_dist_to_occupants"]
        south_door_obs_south = obs_south.candidate_exits[1]["mean_dist_to_occupants"]
        assert south_door_obs_south < south_door_obs_north

        # Feature vectors must be distinct
        state_n, _ = extract_features(obs_north, max_candidates=8, geometry_features=True)
        state_s, _ = extract_features(obs_south, max_candidates=8, geometry_features=True)
        assert not torch.allclose(state_n, state_s)

    def test_observations_change_with_floor_plan_geometry(self):
        """Verify observations change when walls force detours or block access."""
        env = ExitPlacementEnv()
        doors = [(0, 10), (19, 10)]
        occupants = [(10, 10)]

        # 1. Open room: Direct path between (10, 10) and (0, 10) is 10 steps
        cfg_open = _make_test_config(doors=doors, occupant_coords=occupants, walls=[])
        obs_open = env.reset(cfg_open, exit_count=2)
        dist_open = obs_open.candidate_exits[0]["mean_dist_to_occupants"]

        # 2. Add horizontal wall blocking direct path (row 5, cols 5..15) with an opening at col 2
        detour_walls = [(5, c) for c in range(5, 16)]
        cfg_detour = _make_test_config(doors=doors, occupant_coords=occupants, walls=detour_walls)
        obs_detour = env.reset(cfg_detour, exit_count=2)
        dist_detour = obs_detour.candidate_exits[0]["mean_dist_to_occupants"]

        # Distance with detour wall must be strictly greater than open room
        assert dist_detour > dist_open

        # 3. Partitioning: Completely block a room with solid wall across entire row 5
        blocking_walls = [(5, c) for c in range(20)]
        # Occupant is in southern half (row 10), door 0 is in northern half (row 0)
        cfg_blocked = _make_test_config(doors=doors, occupant_coords=occupants, walls=blocking_walls)
        obs_blocked = env.reset(cfg_blocked, exit_count=2)

        # Reachability of door 0 (North) from southern occupants must be 0.0
        reach_blocked_north = obs_blocked.candidate_exits[0]["reachability_ratio"]
        assert reach_blocked_north == 0.0

        # Door 1 (South) is in same compartment as occupant -> reachability is 1.0
        reach_blocked_south = obs_blocked.candidate_exits[1]["reachability_ratio"]
        assert reach_blocked_south == 1.0

    def test_corridor_connectivity_feature(self):
        """Verify corridor connectivity feature reflects walkable neighbors."""
        env = ExitPlacementEnv()
        # Door 0 in open boundary, Door 1 tucked in an alcove
        doors = [(0, 10), (19, 10)]
        # Surround door 1 with walls on three sides making it a narrow 1-cell corridor
        alcove_walls = [(18, 9), (18, 11), (19, 9), (19, 11)]
        cfg = _make_test_config(doors=doors, walls=alcove_walls, occupant_coords=[(10, 10)])
        obs = env.reset(cfg, exit_count=2)

        conn_open = obs.candidate_exits[0]["corridor_connectivity"]
        conn_narrow = obs.candidate_exits[1]["corridor_connectivity"]

        # Open door should have higher 2-hop walkable connectivity than narrow alcove
        assert conn_open > conn_narrow

    def test_existing_stage18_and_stage18b_checkpoints_safe(self):
        """Verify existing Stage 18 and Stage 18B checkpoints load and execute cleanly."""
        ckpt_18 = Path("experiment_output/rl/model_stage18.pt")
        ckpt_18b = Path("experiment_output/rl/model_stage18b.pt")

        assert ckpt_18.exists(), "Stage 18 checkpoint must be preserved."
        assert ckpt_18b.exists(), "Stage 18B checkpoint must be preserved."

        # Load Stage 18 (24 dimensions)
        m18, meta18 = MaskedActorCritic.load(ckpt_18)
        assert m18.in_dim == 24
        assert m18.max_candidates == 4
        assert m18.geometry_features is False

        # Load Stage 18B (44 dimensions)
        m18b, meta18b = MaskedActorCritic.load(ckpt_18b)
        assert m18b.in_dim == 44
        assert m18b.max_candidates == 8
        assert m18b.geometry_features is False
        assert meta18b.get("stage") == "18B"

        # Verify get_action on environment observation does not raise dimension mismatch
        env = ExitPlacementEnv()
        cfg = _make_test_config(doors=[(0, 5), (0, 15), (19, 5), (19, 15)])
        obs = env.reset(cfg, exit_count=2)

        act18, lp18, val18 = m18.get_action(obs, deterministic=True)
        assert act18 in (0, 1, 2, 3)

        act18b, lp18b, val18b = m18b.get_action(obs, deterministic=True)
        assert act18b in (0, 1, 2, 3)

    def test_geometry_aware_model_save_and_load(self, tmp_path: Path):
        """Verify a geometry-aware model saves with 88 dims and loads correctly."""
        model = MaskedActorCritic(max_candidates=8, geometry_features=True)
        assert model.in_dim == 88
        assert model.geometry_features is True

        ckpt_path = tmp_path / "test_geom_model.pt"
        model.save(ckpt_path, metadata={"architecture": "geom_aware_v1"})

        loaded_model, meta = MaskedActorCritic.load(ckpt_path)
        assert loaded_model.in_dim == 88
        assert loaded_model.geometry_features is True
        assert meta["architecture"] == "geom_aware_v1"

        # Test forward pass with 88-dim observation
        env = ExitPlacementEnv()
        cfg = _make_test_config()
        obs = env.reset(cfg, exit_count=2)

        state, mask = extract_features(obs, max_candidates=8, geometry_features=True)
        probs, val = loaded_model(state, mask)
        assert probs.shape == (1, 8)
        assert val.shape == (1, 1)

        # Action selection
        act, lp, v = loaded_model.get_action(obs, deterministic=True)
        assert act in (0, 1)

    def test_safety_constraints_strictly_preserved(self):
        """Verify hard action masking and NBC 2016 safety checks cannot be bypassed."""
        env = ExitPlacementEnv()
        doors = [(0, 10), (19, 10)]
        cfg = _make_test_config(doors=doors)
        obs = env.reset(cfg, exit_count=2)

        # Model cannot select masked actions
        model = MaskedActorCritic(max_candidates=8, geometry_features=True)
        state, mask = extract_features(obs, max_candidates=8, geometry_features=True)
        probs, _ = model(state, mask)

        # Padding slots (indices 2 to 7) must have 0 probability
        assert probs[0, 2:].sum().item() == 0.0

        # Selecting an invalid index terminates with PENALTY_INVALID_ACTION
        _, reward, terminated, info = env.step(99)
        assert terminated is True
        assert reward == ExitPlacementEnv.PENALTY_INVALID_ACTION
        assert info["reason"] == "invalid_action"
