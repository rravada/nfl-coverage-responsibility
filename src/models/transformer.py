"""
Factorized attention transformer for per-frame, per-agent coverage matchup
classification (AWS/NGS architecture, arxiv:2603.25901).

The model consumes the padded per-play tensors emitted by NFLCoverageDataset
(features (B, T, A, 5), agent_mask (B, T, A) bool) and produces per-agent
class logits (B, A, num_classes) after temporal mean-pooling.  Attention is
factorized into a temporal pass (across frames, per agent) and an agent pass
(across agents, per frame), so the cost is O(T^2 + A^2) per layer instead of
O((T*A)^2).

Input feature dimension is 5 (x, y, o_rad, dir_rad, frames_since_snap/10) —
raw kinematics plus a time-since-snap channel; see §3.3 of the paper for the
kinematic features.  The attention mechanism learns spatial relationships
from raw coordinates directly; no Voronoi / cushion / leverage features.

Agents include BOTH coverage-position defenders and possession-team
offensive skill players (WR/TE/RB/FB/QB) — the model needs to see receiver
positions to learn matchup assignments; offensive agents are never a
coverage-responsibility target themselves (their label is -1, ignored by the
training loss).

All identity/role information is injected at the INPUT, before any attention
block: a fixed sinusoidal temporal positional encoding (broadcast over
agents) plus per-agent position/team/slot embeddings (broadcast over time).
This lets the attention layers actually distinguish frame order and player
role while attending; nothing is added after pooling.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.config import (
    MAX_FRAMES,
    MAX_RECEIVERS,
    NAN_FILL_VALUE,
    NUM_POSITION_CLASSES,
    NUM_TEAM_CLASSES,
    TRANSFORMER_INPUT_DIM,
)


def _sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Standard fixed sinusoidal transformer positional encoding (Vaswani et al. 2017).

    Returns a (max_len, d_model) tensor; row t is the encoding for frame index t.
    """
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)          # (max_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class FactorizedAttentionBlock(nn.Module):
    """One factorized block: temporal attention → agent attention → FFN.

    Post-norm ordering throughout: norm(x + dropout(out)). This is intentional.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.agent_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        B, T, A, d_model = x.shape

        # Rows whose keys are ALL padded (padding agent slots in the temporal
        # pass; frames beyond the play's length in the agent pass) would softmax
        # over all -inf and emit NaN. The NaN itself is masked out downstream,
        # but its BACKWARD pass poisons parameter gradients (NaN * 0 = NaN in the
        # attention matmuls), which then get zeroed wholesale — silently freezing
        # input_proj and the attention projections at their random init. Unmask
        # such rows entirely instead: they produce finite garbage that never
        # reaches the loss (excluded at pooling / labelled -1), and gradients
        # through them are exactly zero.

        # --- temporal attention: attend across frames, independently per agent ---
        x_t = x.permute(0, 2, 1, 3).reshape(B * A, T, d_model)
        kpm_t = (~agent_mask).permute(0, 2, 1).reshape(B * A, T)
        kpm_t = kpm_t & ~kpm_t.all(dim=1, keepdim=True)
        out, _ = self.temporal_attn(x_t, x_t, x_t, key_padding_mask=kpm_t)
        out = out.reshape(B, A, T, d_model).permute(0, 2, 1, 3)
        x = self.norm1(x + self.dropout(out))

        # --- agent attention: attend across agents, independently per frame ---
        x_a = x.reshape(B * T, A, d_model)
        kpm_a = (~agent_mask).reshape(B * T, A)
        kpm_a = kpm_a & ~kpm_a.all(dim=1, keepdim=True)
        out, _ = self.agent_attn(x_a, x_a, x_a, key_padding_mask=kpm_a)
        out = out.reshape(B, T, A, d_model)
        x = self.norm2(x + self.dropout(out))

        # --- position-wise FFN ---
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class CoverageMatchupTransformer(nn.Module):
    """Factorized attention transformer producing per-frame, per-agent logits."""

    def __init__(
        self,
        num_classes: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(TRANSFORMER_INPUT_DIM, d_model)  # 5 → d_model
        self.blocks = nn.ModuleList(
            [
                FactorizedAttentionBlock(d_model, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.position_embedding = nn.Embedding(NUM_POSITION_CLASSES, d_model)
        self.team_embedding = nn.Embedding(NUM_TEAM_CLASSES, d_model)
        self.slot_embedding = nn.Embedding(MAX_RECEIVERS + 1, d_model)  # + NO_SLOT_ID
        # Fixed (non-learned) sinusoidal temporal positional encoding, broadcast
        # over agents. persistent=False: deterministic, no need to checkpoint.
        self.register_buffer(
            "temporal_pe", _sinusoidal_positional_encoding(MAX_FRAMES, d_model), persistent=False
        )
        self.output_head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        features: torch.Tensor,      # (B, T, A, 5)
        agent_mask: torch.Tensor,    # (B, T, A) bool
        position_ids: torch.Tensor,  # (B, A) int64
        team_ids: torch.Tensor,      # (B, A) int64
        slot_ids: torch.Tensor,      # (B, A) int64
    ) -> torch.Tensor:               # (B, A, num_classes)
        # Drop NAN_FILL_VALUE sentinel at padded positions before projection.
        features = features.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        x = self.input_proj(features)  # (B, T, A, d_model)

        # --- inject identity/role information BEFORE any attention block ---
        T = x.size(1)
        x = x + self.temporal_pe[:T].unsqueeze(0).unsqueeze(2)  # (1, T, 1, d_model)

        agent_embed = (
            self.position_embedding(position_ids)
            + self.team_embedding(team_ids)
            + self.slot_embedding(slot_ids)
        )  # (B, A, d_model)
        x = x + agent_embed.unsqueeze(1)  # broadcast over T → (B, T, A, d_model)

        for block in self.blocks:
            x = block(x, agent_mask)

        # Temporal mean-pool over valid frames per agent.
        mask_float = agent_mask.float()                                    # (B, T, A)
        valid_counts = mask_float.sum(dim=1).clamp(min=1).unsqueeze(-1)   # (B, A, 1)
        pooled = (x * mask_float.unsqueeze(-1)).sum(dim=1) / valid_counts
        # pooled: (B, A, d_model)

        return self.output_head(pooled)  # (B, A, num_classes)
