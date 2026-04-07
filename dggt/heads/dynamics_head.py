"""
DynamicsHead — predicts per-slot motion scalars for each frame.

For slot p at frame t:
  [slot_features_p || timestamp_t] → Linear → GELU → Linear → Tanh
  → S_{p,t} ∈ [-1, 1]

The scalar S_{p,t} is then mapped to a physical motion amount in rigid_transform.py:
  revolute:  angle = S_{p,t} × max_angle
  prismatic: displacement = S_{p,t} × max_translation

Slot 0 (static) has its scalars forced to 0 in training loss (handled externally).
"""

import torch
import torch.nn as nn


class DynamicsHead(nn.Module):
    """
    Predicts motion scalars S_{p,t} ∈ [-1, 1] for all (slot, frame) pairs.

    Args:
        dim_in:       slot feature dimension (D_slot)
        hidden_dim:   MLP hidden dimension
    """

    def __init__(self, dim_in: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim_in + 1),   # +1 for scalar timestamp
            nn.Linear(dim_in + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        slot_features: torch.Tensor,   # [B, P, D]
        timestamps: torch.Tensor,      # [B, S]  normalised to [0, 1]
    ) -> torch.Tensor:                 # [B, P, S]
        """
        For each (slot, frame) pair, predict a motion scalar.

        Args:
            slot_features: [B, P, D]   from PartSlotRouter
            timestamps:    [B, S]      frame timestamps normalised to [0, 1]
        Returns:
            scalars: [B, P, S]  motion scalars ∈ [-1, 1]
        """
        B, P, D = slot_features.shape
        S = timestamps.shape[1]

        # Expand for cross-product of (slot, frame)
        # slot_features: [B, P, D] → [B, P, S, D]
        feats = slot_features.unsqueeze(2).expand(-1, -1, S, -1)

        # timestamps: [B, S] → [B, 1, S, 1] → [B, P, S, 1]
        ts = timestamps.unsqueeze(1).unsqueeze(-1).expand(B, P, S, 1)

        # Concatenate along feature dim
        inp = torch.cat([feats, ts], dim=-1)     # [B, P, S, D+1]

        # Apply MLP (flatten batch dims for efficiency)
        inp_flat = inp.reshape(B * P * S, D + 1)
        out_flat = self.mlp(inp_flat)             # [B*P*S, 1]
        scalars  = out_flat.reshape(B, P, S)      # [B, P, S]

        return scalars
