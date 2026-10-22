"""
rl_policy.py
------------
Stage 18: Masked Actor-Critic Policy for Exit Placement RL.

Provides:
1. Feature extraction from structured RLObservation into fixed-dimension tensors.
2. Action-masked Actor-Critic neural network architecture (PyTorch).
3. Action selection with rigorous action-masking (zero probability for invalid exits).
4. Deterministic greedy policy execution for evaluation.
5. Model checkpointing (saving & loading weights + metadata).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evacuation_simulation.simulation.rl_environment import RLObservation


# ---------------------------------------------------------------------------
# BFS / Graph Geometry Feature Computation
# ---------------------------------------------------------------------------


def compute_geometry_features_for_candidates(
    floor_plan: Any,
    occupants: Any,
    candidates: list[Any],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """
    Compute geometry-aware features using NetworkX BFS shortest paths.

    Reuses Environment and NavigationGraph logic to derive:
    - Global building geometry stats: walkable ratio, occupant centroid, dispersion.
    - Per-candidate door metrics: mean/min/max shortest-path distances to occupants,
      exit reachability ratio, and corridor/local connectivity.

    Parameters
    ----------
    floor_plan:
        Static building floor plan geometry.
    occupants:
        Scenario occupant configuration with agent coordinates.
    candidates:
        Candidate exit objects to analyze.

    Returns
    -------
    tuple[dict[str, float], list[dict[str, float]]]
        - global_features: dict with walkable_ratio, occupant_centroid_r, occupant_centroid_c, occupant_dispersion
        - candidate_features: list of dicts with mean_dist_to_occupants, min_dist_to_occupants, max_dist_to_occupants, reachability_ratio, corridor_connectivity
    """
    rows = floor_plan.grid.rows
    cols = floor_plan.grid.cols
    max_dim = float(max(1, max(rows, cols)))

    from evacuation_simulation.simulation.config import ExitConfiguration
    from evacuation_simulation.simulation.environment import Environment
    from evacuation_simulation.simulation.pathfinding import NavigationGraph

    probe_config = ExitConfiguration(
        configuration_id="probe_geometry",
        exits=[c.position for c in candidates],
        widths=[c.width for c in candidates],
    )
    env = Environment.from_floor_plan_and_exits(floor_plan, probe_config)
    nav = NavigationGraph.from_environment(env)

    # 1. Global features
    walkable_count = nav.node_count
    walkable_ratio = float(walkable_count) / float(max(1, rows * cols))

    agents = occupants.agents
    if agents:
        r_coords = [a.row for a in agents]
        c_coords = [a.col for a in agents]
        mean_r = float(np.mean(r_coords))
        mean_c = float(np.mean(c_coords))
        centroid_r = mean_r / float(rows)
        centroid_c = mean_c / float(cols)
        dists_to_centroid = [
            float(np.hypot(a.row - mean_r, a.col - mean_c)) for a in agents
        ]
        dispersion = float(np.mean(dists_to_centroid)) / max_dim
    else:
        centroid_r = 0.5
        centroid_c = 0.5
        dispersion = 0.0

    global_feats = {
        "walkable_ratio": walkable_ratio,
        "occupant_centroid_r": centroid_r,
        "occupant_centroid_c": centroid_c,
        "occupant_dispersion": dispersion,
    }

    # 2. Per-candidate metrics via BFS
    candidate_feats: list[dict[str, float]] = []
    for cand in candidates:
        cand_pos = (int(cand.position[0]), int(cand.position[1]))
        if nav.has_node(cand_pos):
            path_lengths = nx.single_source_shortest_path_length(nav._graph, cand_pos)

            occ_dists = [
                path_lengths[(a.row, a.col)]
                for a in agents
                if (a.row, a.col) in path_lengths
            ]
            reach_ratio = (float(len(occ_dists)) / float(len(agents))) if agents else 1.0

            if occ_dists:
                mean_d = float(np.mean(occ_dists)) / max_dim
                min_d = float(min(occ_dists)) / max_dim
                max_d = float(max(occ_dists)) / max_dim
            else:
                mean_d = 1.0
                min_d = 1.0
                max_d = 1.0

            # 2-hop walkable neighborhood (max 12 in 4-connected grid)
            two_hop_nodes = len(nx.single_source_shortest_path_length(nav._graph, cand_pos, cutoff=2)) - 1
            connectivity = min(1.0, float(two_hop_nodes) / 8.0)
        else:
            reach_ratio = 0.0
            mean_d = 1.0
            min_d = 1.0
            max_d = 1.0
            connectivity = 0.0

        candidate_feats.append({
            "mean_dist_to_occupants": mean_d,
            "min_dist_to_occupants": min_d,
            "max_dist_to_occupants": max_d,
            "reachability_ratio": reach_ratio,
            "corridor_connectivity": connectivity,
        })

    return global_feats, candidate_feats


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------


def extract_features(
    obs: RLObservation,
    max_candidates: int = 8,
    geometry_features: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert an RLObservation into fixed-dimension tensors for neural network input.

    Feature Layouts:
    ----------------
    When geometry_features=False (Legacy 44-dim mode for max_candidates=8):
    - Global features (4):
        [rows/100, cols/100, exits_remaining/max(1, exits_remaining + selected), total_agents/100]
    - Per-candidate features (5 per candidate, up to max_candidates):
        [pos_r/rows, pos_c/cols, width/2.0, is_selected, legal_mask]
    Total feature dimension = 4 + (max_candidates * 5) = 44 dimensions.

    When geometry_features=True (Geometry-aware 88-dim mode for max_candidates=8):
    - Global features (8):
        [rows/100, cols/100, exits_ratio, total_agents/100,
         walkable_ratio, occupant_centroid_r, occupant_centroid_c, occupant_dispersion]
    - Per-candidate features (10 per candidate, up to max_candidates):
        [pos_r/rows, pos_c/cols, width/2.0, is_selected, legal_mask,
         mean_dist_to_occupants, min_dist_to_occupants, max_dist_to_occupants,
         reachability_ratio, corridor_connectivity]
    Total feature dimension = 8 + (max_candidates * 10) = 88 dimensions.

    Parameters
    ----------
    obs:
        The environment observation.
    max_candidates:
        Maximum candidate capacity supported by the network.
    geometry_features:
        If True, extract the enriched 88-dimensional geometry features.
        If False, extract the legacy 44-dimensional features.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - state_tensor: shape (1, in_dim)
        - mask_tensor: shape (1, max_candidates) boolean mask
    """
    rows, cols = obs.grid_shape
    total_selected = len(obs.selected_candidate_indices)
    required_total = obs.exits_remaining + total_selected
    exits_ratio = (obs.exits_remaining / required_total) if required_total > 0 else 0.0

    if not geometry_features:
        global_feats = [
            rows / 100.0,
            cols / 100.0,
            exits_ratio,
            obs.total_agents / 100.0,
        ]

        candidate_feats: list[float] = []
        mask_list: list[bool] = []

        for i in range(max_candidates):
            if i < len(obs.candidate_exits):
                c = obs.candidate_exits[i]
                pos = c["position"]
                is_legal = bool(obs.action_mask[i])
                candidate_feats.extend([
                    pos[0] / max(1.0, float(rows)),
                    pos[1] / max(1.0, float(cols)),
                    float(c.get("width", 1.0)) / 2.0,
                    1.0 if c.get("is_selected", False) else 0.0,
                    1.0 if is_legal else 0.0,
                ])
                mask_list.append(is_legal)
            else:
                # Padding slot
                candidate_feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                mask_list.append(False)
    else:
        centroid = getattr(obs, "occupant_centroid", (0.5, 0.5))
        global_feats = [
            rows / 100.0,
            cols / 100.0,
            exits_ratio,
            obs.total_agents / 100.0,
            float(getattr(obs, "walkable_ratio", 0.0)),
            float(centroid[0]),
            float(centroid[1]),
            float(getattr(obs, "occupant_dispersion", 0.0)),
        ]

        candidate_feats = []
        mask_list = []

        for i in range(max_candidates):
            if i < len(obs.candidate_exits):
                c = obs.candidate_exits[i]
                pos = c["position"]
                is_legal = bool(obs.action_mask[i])
                candidate_feats.extend([
                    pos[0] / max(1.0, float(rows)),
                    pos[1] / max(1.0, float(cols)),
                    float(c.get("width", 1.0)) / 2.0,
                    1.0 if c.get("is_selected", False) else 0.0,
                    1.0 if is_legal else 0.0,
                    float(c.get("mean_dist_to_occupants", 0.5)),
                    float(c.get("min_dist_to_occupants", 0.5)),
                    float(c.get("max_dist_to_occupants", 0.5)),
                    float(c.get("reachability_ratio", 1.0 if is_legal else 0.0)),
                    float(c.get("corridor_connectivity", 0.5)),
                ])
                mask_list.append(is_legal)
            else:
                # Padding slot (10 per candidate)
                candidate_feats.extend([0.0] * 10)
                mask_list.append(False)

    feature_vec = global_feats + candidate_feats
    state_tensor = torch.tensor([feature_vec], dtype=torch.float32)
    mask_tensor = torch.tensor([mask_list], dtype=torch.bool)
    return state_tensor, mask_tensor


# ---------------------------------------------------------------------------
# Masked Actor-Critic Neural Network
# ---------------------------------------------------------------------------


class MaskedActorCritic(nn.Module):
    """
    Two-headed Actor-Critic Network with Action Masking.

    Actor outputs raw logits over candidate exit indices.
    Action mask enforces invalid/illegal candidate exits have probability 0.
    Critic outputs scalar state value V(s).
    """

    def __init__(
        self,
        max_candidates: int = 8,
        hidden_dim: int = 64,
        geometry_features: bool = False,
        in_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.max_candidates = max_candidates
        self.geometry_features = geometry_features

        if in_dim is not None:
            self.in_dim = in_dim
            if self.in_dim > 4 + (max_candidates * 5):
                self.geometry_features = True
        else:
            if geometry_features:
                self.in_dim = 8 + (max_candidates * 10)
            else:
                self.in_dim = 4 + (max_candidates * 5)

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy logits)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_candidates),
        )

        # Critic head (state value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute masked action probabilities and state value.

        Parameters
        ----------
        state:
            Tensor of shape (B, in_dim).
        mask:
            Boolean tensor of shape (B, max_candidates), True = legal.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - probs: (B, max_candidates) probability distribution
            - value: (B, 1) scalar value estimate
        """
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features)

        # Apply action mask: masked positions receive large negative value
        masked_logits = logits.masked_fill(~mask, -1e8)
        probs = F.softmax(masked_logits, dim=-1)
        return probs, value

    def get_action(
        self,
        obs: RLObservation,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select an action given an environment observation.

        Parameters
        ----------
        obs:
            RLObservation from the environment.
        deterministic:
            If True, selects argmax (greedy evaluation).
            If False, samples from masked categorical distribution (training).

        Returns
        -------
        tuple[int, torch.Tensor, torch.Tensor]
            - action: integer index of candidate exit.
            - log_prob: log probability of selected action.
            - value: scalar state value tensor.
        """
        use_geom = self.geometry_features or (self.in_dim > 4 + (self.max_candidates * 5))
        state, mask = extract_features(
            obs,
            max_candidates=self.max_candidates,
            geometry_features=use_geom,
        )
        probs, value = self.forward(state, mask)

        # Ensure at least one valid action exists
        if not mask.any():
            # Fallback for unexpected edge case
            return 0, torch.tensor(0.0), value

        dist = torch.distributions.Categorical(probs)
        if deterministic:
            action = int(torch.argmax(probs, dim=-1).item())
        else:
            action = int(dist.sample().item())

        log_prob = dist.log_prob(torch.tensor(action))
        return action, log_prob, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probs, entropy, and values for a batch of transitions.
        """
        probs, values = self.forward(states, masks)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze(-1)

    # -----------------------------------------------------------------------
    # Save & Load Checkpoints
    # -----------------------------------------------------------------------

    def save(self, filepath: Union[str, Path], metadata: Optional[dict] = None) -> None:
        """Save model weights and architectural metadata."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "max_candidates": self.max_candidates,
            "in_dim": self.in_dim,
            "geometry_features": self.geometry_features,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, p)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> tuple[MaskedActorCritic, dict]:
        """Load a saved checkpoint and instantiate the model."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {p}")
        checkpoint = torch.load(p, map_location="cpu", weights_only=False)
        max_cands = checkpoint.get("max_candidates", 8)
        in_dim = checkpoint.get("in_dim", 4 + (max_cands * 5))
        geom = checkpoint.get(
            "geometry_features",
            (in_dim is not None and in_dim > 4 + (max_cands * 5)),
        )
        model = cls(
            max_candidates=max_cands,
            geometry_features=geom,
            in_dim=in_dim,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, checkpoint.get("metadata", {})

