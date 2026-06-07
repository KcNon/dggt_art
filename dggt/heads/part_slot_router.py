"""
PartSlotRouter — 按论文架构重新设计。

Token 组装:
  image_token = proj(cat([agg_token, dino_token, plucker_ray, stage_emb]))
  其中 dino_token 来自冻结 DINOv2 最后一层，提供精细语义线索。

Transformer 结构 (8层: 6 cross + 2 self, 比例 75/25):
  层顺序: [cross, cross, cross, self, cross, cross, cross, self]

  Cross-Attention 层:
    Q = image_tokens,  K = V = slot_tokens
    → 更新 image_tokens; attention weights 即分配图
    → 每个 image patch 学习"我属于哪个 part"

  Self-Attention 层:
    all_tokens = cat([image_tokens, slot_tokens])
    → 同时更新两者; slot 间互相抑制, 图像获得全局上下文

输出:
  slot_features: 经 self-attn 更新后的 slot_tokens  [B, P, D]
  assign_maps:   最后一层 cross-attn 的 attention weights
                 [B, P, H_p, W_p]  (在帧维度取平均)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layer blocks
# ---------------------------------------------------------------------------

class _FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden = dim * mlp_ratio
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CrossAttentionLayer(nn.Module):
    """
    Image tokens (Q) attend to slot tokens (K, V).
    Only image_tokens are updated.
    Returns (updated_image_tokens, attn_weights [B, N, P] or None).
    使用 F.scaled_dot_product_attention 以触发 Flash Attention。
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj  = nn.Linear(dim, dim, bias=False)
        self.k_proj  = nn.Linear(dim, dim, bias=False)
        self.v_proj  = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.ffn = _FFN(dim, mlp_ratio)

    def forward(
        self,
        image_tokens: torch.Tensor,      # [B, N, D]
        slot_tokens:  torch.Tensor,      # [B, P, D]
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, D = image_tokens.shape
        P = slot_tokens.shape[1]
        H = self.num_heads

        q  = self.q_proj(self.norm_q(image_tokens))        # [B, N, D]
        k  = self.k_proj(self.norm_kv(slot_tokens))        # [B, P, D]
        v  = self.v_proj(slot_tokens)                       # [B, P, D]  (no norm on V)

        # Reshape to multi-head
        q = q.reshape(B, N, H, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        k = k.reshape(B, P, H, self.head_dim).transpose(1, 2)  # [B, H, P, d]
        v = v.reshape(B, P, H, self.head_dim).transpose(1, 2)  # [B, H, P, d]

        if return_weights:
            # Need explicit weights → compute manually (last layer only, P=8 so cheap)
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, P]
            attn = F.softmax(attn, dim=-1)
            out  = torch.matmul(attn, v)                               # [B, H, N, d]
            weights = attn.mean(dim=1)                                 # [B, N, P]
        else:
            # Flash Attention (no explicit weight materialisation)
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            weights = None

        out = out.transpose(1, 2).reshape(B, N, D)         # [B, N, D]
        out = self.out_proj(out)
        image_tokens = image_tokens + out
        image_tokens = self.ffn(image_tokens)
        return image_tokens, weights


class SelfAttentionLayer(nn.Module):
    """
    Joint self-attention over cat([image_tokens, slot_tokens]).
    Both streams are updated. Flash Attention via F.sdpa.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.norm    = nn.LayerNorm(dim)
        self.qkv     = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.ffn_img  = _FFN(dim, mlp_ratio)
        self.ffn_slot = _FFN(dim, mlp_ratio)

    def forward(
        self,
        image_tokens: torch.Tensor,   # [B, N, D]
        slot_tokens:  torch.Tensor,   # [B, P, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = image_tokens.shape
        P = slot_tokens.shape[1]
        H = self.num_heads

        all_tokens = torch.cat([image_tokens, slot_tokens], dim=1)  # [B, N+P, D]
        x   = self.norm(all_tokens)
        qkv = self.qkv(x).reshape(B, N + P, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                            # each [B, H, N+P, d]

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, N + P, D)
        out = self.out_proj(out)
        all_tokens = all_tokens + out

        img_out  = self.ffn_img(all_tokens[:, :N])
        slot_out = self.ffn_slot(all_tokens[:, N:])
        return img_out, slot_out


# ---------------------------------------------------------------------------
# PartSlotRouter
# ---------------------------------------------------------------------------

class PartSlotRouter(nn.Module):
    """
    Args:
        dim_agg:        Aggregator output dim (= 2 × embed_dim, default 2048)
        dim_dino:       DINOv2 token dim      (= embed_dim,     default 1024)
        dim_slot:       Slot / projection dim (= embed_dim,     default 1024)
        num_slots:      Number of part slots P (default 8)
        num_layers:     Total transformer layers (default 8: 6 cross + 2 self)
        num_heads:      Attention heads (default 16)
        mlp_ratio:      FFN expansion ratio (default 4)
        state_embed_dim: Timestamp MLP output dim (default 64)
        patch_start_idx: Special tokens to skip in aggregator output (default 5)
        patch_size:     Patch size in pixels (default 14, matches DINOv2)
    """

    def __init__(
        self,
        dim_agg:         int = 2048,
        dim_dino:        int = 1024,
        dim_slot:        int = 1024,
        num_slots:       int = 8,
        num_layers:      int = 8,
        num_heads:       int = 16,
        mlp_ratio:       int = 4,
        state_embed_dim: int = 64,
        patch_start_idx: int = 5,
        patch_size:      int = 14,
    ):
        super().__init__()
        self.num_slots       = num_slots
        self.dim_slot        = dim_slot
        self.patch_start_idx = patch_start_idx
        self.patch_size      = patch_size

        # ── Timestamp → stage embedding ─────────────────────────────────
        self.state_embed = nn.Sequential(
            nn.Linear(1, state_embed_dim),
            nn.GELU(),
            nn.Linear(state_embed_dim, state_embed_dim),
        )

        # ── Input projection ────────────────────────────────────────────
        # agg(2D) + dino(D) + plucker(6) + stage_emb(D_s) → dim_slot
        dim_in = dim_agg + dim_dino + 6 + state_embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(dim_in, dim_slot),
            nn.LayerNorm(dim_slot),
        )

        # slot_tokens have been moved into Aggregator.
        # PartSlotRouter receives slot states from Aggregator as slot_init.
        # A fallback self.slot_tokens is kept for standalone use only.
        self._has_fallback_slots = False  # set True only if Aggregator has no slots

        # ── Transformer: [cross, cross, cross, self] × (num_layers // 4) ──
        # Layer schedule: every 4th layer (0-indexed: 3, 7, ...) is self-attn
        self.layers     : nn.ModuleList = nn.ModuleList()
        self.layer_types: list[str]     = []
        for i in range(num_layers):
            if (i + 1) % 4 == 0:           # indices 3, 7, 11, ...
                self.layers.append(SelfAttentionLayer(dim_slot, num_heads, mlp_ratio))
                self.layer_types.append("self")
            else:
                self.layers.append(CrossAttentionLayer(dim_slot, num_heads, mlp_ratio))
                self.layer_types.append("cross")

        self.slot_norm = nn.LayerNorm(dim_slot)

        # ── Background sink slot ──────────────────────────────────────────
        # A learnable token appended to the P part slots ONLY for the
        # attention/assignment. It absorbs background patches so the P part
        # slots (slot 0 = base part, paper-aligned) are not forced to claim
        # background. Excluded from slot_features (no part head decodes it).
        self.bg_token = nn.Parameter(torch.zeros(1, 1, dim_slot))

        self._init_weights()
        nn.init.normal_(self.bg_token, std=0.02)

    # ------------------------------------------------------------------ #

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        agg_tokens:   torch.Tensor,              # [B, S, P_total, dim_agg]
        dino_tokens:  torch.Tensor,              # [B, S, P_total, dim_dino]
        plucker_rays: torch.Tensor,              # [B, S, N_patches, 6]
        timestamps:   torch.Tensor,              # [B, S]
        img_hw:       tuple[int, int] | None = None,
        slot_init:    torch.Tensor | None = None,  # [B, P, dim_slot] from Aggregator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            slot_features : [B, P, dim_slot]
            assign_maps   : [B, P, H_p, W_p]  (P part slots; softmax over P+1 incl. bg, so sums to <1)
            bg_map        : [B, 1, H_p, W_p]  (background sink channel)
        """
        B, S, P_total, _ = agg_tokens.shape
        N_patches = P_total - self.patch_start_idx

        # Compute H_p, W_p
        if img_hw is not None:
            H_img, W_img = img_hw
            H_p = H_img // self.patch_size
            W_p = W_img // self.patch_size
            assert H_p * W_p == N_patches, (
                f"N_patches={N_patches} != H_p({H_p})×W_p({W_p}); "
                f"img_hw={img_hw}, patch_size={self.patch_size}"
            )
        else:
            H_p = W_p = int(math.isqrt(N_patches))
            assert H_p * W_p == N_patches, (
                f"N_patches={N_patches} not a perfect square; pass img_hw for non-square input"
            )

        # ── 1. Extract patch tokens (skip register/camera special tokens) ──
        agg_patch  = agg_tokens[:, :, self.patch_start_idx:, :]   # [B, S, N, dim_agg]
        dino_patch = dino_tokens[:, :, self.patch_start_idx:, :]  # [B, S, N, dim_dino]

        # ── 2. Stage embeddings ──────────────────────────────────────────
        # timestamps [B, S] → [B, S, N, D_state]
        state_emb = self.state_embed(timestamps.unsqueeze(-1))          # [B, S, D_state]
        state_emb = state_emb.unsqueeze(2).expand(-1, -1, N_patches, -1)

        # ── 3. Token composition & projection ───────────────────────────
        enriched = torch.cat([agg_patch, dino_patch, plucker_rays, state_emb], dim=-1)
        # [B, S, N, dim_agg + dim_dino + 6 + D_state]

        enriched = enriched.reshape(B, S * N_patches, -1)          # [B, S*N, total_in]
        image_tokens = self.input_proj(enriched)                    # [B, S*N, dim_slot]

        # ── 4. Transformer layers ────────────────────────────────────────
        # Use Aggregator-provided slot states as initial slot representations.
        # slot_init comes from Aggregator's global_blocks, already enriched with
        # multi-frame, multi-view context. PSR further refines them with plucker
        # rays, timestamps, and the projected image tokens.
        assert slot_init is not None, (
            "slot_init must be provided from Aggregator. "
            "Ensure Aggregator is constructed with num_slots > 0."
        )
        # Append the background sink slot → [B, P+1, dim_slot]. It participates
        # in all attention layers but is sliced off before returning slot_features.
        slots = torch.cat(
            [slot_init, self.bg_token.expand(B, -1, -1)], dim=1
        )                                                           # [B, P+1, dim_slot]
        last_attn_weights: torch.Tensor | None = None

        for i, (layer, ltype) in enumerate(zip(self.layers, self.layer_types)):
            is_last_cross = (
                ltype == "cross"
                and all(t == "self" for t in self.layer_types[i + 1:])
                    or i == len(self.layers) - 1 and ltype == "cross"
            )
            # We need weights only from the last cross-attn layer
            want_weights = (
                ltype == "cross"
                and not any(t == "cross" for t in self.layer_types[i + 1:])
            )

            if ltype == "cross":
                image_tokens, weights = layer(
                    image_tokens, slots, return_weights=want_weights
                )
                if want_weights and weights is not None:
                    last_attn_weights = weights   # [B, S*N, P]
            else:  # self
                image_tokens, slots = layer(image_tokens, slots)

        slot_features = self.slot_norm(slots)[:, :self.num_slots]   # [B, P, dim_slot] (drop bg)

        # ── 5. Assignment maps from last cross-attn weights ───────────────
        # Use frame 0 (canonical rest state) only.
        # Frame 0 is always the rest pose (all joints closed/at minimum), so the
        # GT part masks at frame 0 are unambiguous single-position masks.
        # Mean-over-frames was incorrect: for a moving part, averaging attention
        # across S frames gives diffuse maps that don't match the single-frame GT,
        # capping warmup IoU at ~0.62 regardless of LR.
        Pp1 = self.num_slots + 1   # P part slots + 1 background sink
        if last_attn_weights is not None:
            # [B, S*N, P+1] → reshape to [B, S, N, P+1] → take frame 0 → [B, N, P+1]
            assign = last_attn_weights.reshape(B, S, N_patches, Pp1)
            assign = assign[:, 0, :, :]                             # [B, N, P+1]
            assign = assign.permute(0, 2, 1).reshape(B, Pp1, H_p, W_p)
        else:
            # Fallback: dot-product (should not be reached in normal operation)
            temp = 1.0 / math.sqrt(self.dim_slot)
            all_slots = self.slot_norm(slots)                      # [B, P+1, dim_slot]
            logits = torch.bmm(image_tokens, all_slots.transpose(1, 2)) * temp
            assign = F.softmax(logits, dim=-1)                      # [B, S*N, P+1]
            assign = assign.reshape(B, S, N_patches, Pp1)[:, 0, :, :]
            assign = assign.permute(0, 2, 1).reshape(B, Pp1, H_p, W_p)

        # Split: P part-slot maps (sum to <1; bg mass removed) + bg sink map.
        assign_maps = assign[:, :self.num_slots]                   # [B, P, H_p, W_p]
        bg_map      = assign[:, self.num_slots:]                   # [B, 1, H_p, W_p]

        return slot_features, assign_maps, bg_map
