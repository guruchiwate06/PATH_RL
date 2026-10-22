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
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from evacuation_simulation.simulation.rl_environment import RLObservation


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------


def extract_features(
    obs: RLObservation,
    max_candidates: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert an RLObservation into fixed-dimension tensors for neural network input.

    Feature Layout:
    - Global features (4):
        [rows/100, cols/100, exits_remaining/max(1, exits_remaining + selected), total_agents/100]
    - Per-candidate features (5 per candidate, up to max_candidates):
        [pos_r/rows, pos_c/cols, width/2.0, is_selected, legal_mask]

    Total feature dimension = 4 + (max_candidates * 5) = 44 dimensions for max_candidates=8.

    Parameters
    ----------
    obs:
        The environment observation.
    max_candidates:
        Maximum candidate capacity supported by the network.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - state_tensor: shape (1, 44)
        - mask_tensor: shape (1, max_candidates) boolean mask
    """
    rows, cols = obs.grid_shape
    total_selected = len(obs.selected_candidate_indices)
    required_total = obs.exits_remaining + total_selected
    exits_ratio = (obs.exits_remaining / required_total) if required_total > 0 else 0.0

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
    ) -> None:
        super().__init__()
        self.max_candidates = max_candidates
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
        state, mask = extract_features(obs, max_candidates=self.max_candidates)
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
        model = cls(max_candidates=max_cands)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, checkpoint.get("metadata", {})
