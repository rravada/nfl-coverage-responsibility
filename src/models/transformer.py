"""
Factorized attention transformer for per-frame, per-agent coverage matchup
classification (AWS/NGS architecture, arxiv:2603.25901).

The model consumes the padded per-play tensors emitted by NFLCoverageDataset
(features (B, T, A, 15), agent_mask (B, T, A) bool) and produces frame-by-frame
class logits (B, T, A, num_classes). Attention is factorized into a temporal pass
(across frames, per agent) and an agent pass (across agents, per frame), so the
cost is O(T^2 + A^2) per layer instead of O((T*A)^2).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import FEATURE_COLUMNS, NAN_FILL_VALUE, NUM_POSITION_CLASSES, NUM_TEAM_CLASSES  # noqa: F401  (sentinel intent)


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

        # --- temporal attention: attend across frames, independently per agent ---
        x_t = x.permute(0, 2, 1, 3).reshape(B * A, T, d_model)
        kpm_t = (~agent_mask).permute(0, 2, 1).reshape(B * A, T)
        out, _ = self.temporal_attn(x_t, x_t, x_t, key_padding_mask=kpm_t)
        # Fully padded sequences (padding agent slots) yield NaN; zero them — these
        # positions are masked / labelled -1 downstream, so this is inert.
        out = torch.nan_to_num(out, nan=0.0)
        out = out.reshape(B, A, T, d_model).permute(0, 2, 1, 3)
        x = self.norm1(x + self.dropout(out))

        # --- agent attention: attend across agents, independently per frame ---
        x_a = x.reshape(B * T, A, d_model)
        kpm_a = (~agent_mask).reshape(B * T, A)
        out, _ = self.agent_attn(x_a, x_a, x_a, key_padding_mask=kpm_a)
        out = torch.nan_to_num(out, nan=0.0)
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
        self.input_proj = nn.Linear(len(FEATURE_COLUMNS), d_model)
        self.blocks = nn.ModuleList(
            [
                FactorizedAttentionBlock(d_model, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.position_embedding = nn.Embedding(NUM_POSITION_CLASSES, d_model)
        self.team_embedding = nn.Embedding(NUM_TEAM_CLASSES, d_model)
        self.output_head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        features: torch.Tensor,      # (B, T, A, 15)
        agent_mask: torch.Tensor,    # (B, T, A) bool
        position_ids: torch.Tensor,  # (B, A) int64
        team_ids: torch.Tensor,      # (B, A) int64
    ) -> torch.Tensor:               # (B, A, num_classes)
        # Drop NAN_FILL_VALUE sentinel at padded positions before projection.
        features = features.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        x = self.input_proj(features)
        for block in self.blocks:
            x = block(x, agent_mask)

        # Temporal mean-pool over valid frames per agent.
        mask_float = agent_mask.float()                                    # (B, T, A)
        valid_counts = mask_float.sum(dim=1).clamp(min=1).unsqueeze(-1)   # (B, A, 1)
        pooled = (x * mask_float.unsqueeze(-1)).sum(dim=1) / valid_counts
        # pooled: (B, A, d_model)

        fused = pooled + self.position_embedding(position_ids) + self.team_embedding(team_ids)
        return self.output_head(fused)  # (B, A, num_classes)
