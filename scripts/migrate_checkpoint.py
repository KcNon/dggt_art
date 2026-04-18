"""
migrate_checkpoint.py

Migrate an ArtVGGT checkpoint trained with global_blocks (self-attention) to the
new state_blocks (cross-attention) architecture.

Mapping for each layer i (0..23):
  global_blocks[i].attn.qkv.weight  [3D, D]  → split into:
      state_blocks[i].state_attn.attn.q_proj.weight  [D, D]  (rows 0:D)
      state_blocks[i].state_attn.attn.k_proj.weight  [D, D]  (rows D:2D)
      state_blocks[i].state_attn.attn.v_proj.weight  [D, D]  (rows 2D:3D)
  global_blocks[i].attn.qkv.bias   [3D]      → split similarly
  global_blocks[i].attn.q_norm.*            → state_attn.attn.q_norm.*
  global_blocks[i].attn.k_norm.*            → state_attn.attn.k_norm.*
  global_blocks[i].attn.proj.*              → state_attn.attn.out_proj.*
  global_blocks[i].norm1.*                  → state_attn.norm_q.*
  global_blocks[i].norm2.*                  → state_attn.norm2.*
  global_blocks[i].ls1.gamma                → state_attn.ls1  (Parameter)
  global_blocks[i].ls2.gamma                → state_attn.ls2  (Parameter)
  global_blocks[i].mlp.fc1.*               → state_attn.mlp.0.*
  global_blocks[i].mlp.fc2.*               → state_attn.mlp.2.*

  frame_attn is initialised as an EXACT copy of state_attn weights above.
  This gives both sub-blocks a sensible warm start; they will diverge during
  Phase A fine-tuning.

  aggregator.state_tokens  — NOT migrated, kept as random init from new model.
  All other keys copied verbatim (frame_blocks, heads, etc.).
"""

import argparse
import torch
from collections import OrderedDict


def migrate(src_ckpt_path: str, dst_ckpt_path: str):
    print(f"Loading source checkpoint: {src_ckpt_path}")
    ckpt = torch.load(src_ckpt_path, map_location="cpu", weights_only=False)
    old_state = ckpt["model"]

    # Detect embedding dim from first global block
    d = old_state["aggregator.global_blocks.0.attn.qkv.weight"].shape[1]
    num_layers = sum(
        1 for k in old_state if k.startswith("aggregator.global_blocks.")
           and k.endswith(".attn.qkv.weight")
    )
    print(f"  embed_dim={d}, num_global_layers={num_layers}")

    new_state = OrderedDict()

    # ── 1. Copy all non-global_block keys verbatim ─────────────────────────
    for k, v in old_state.items():
        if not k.startswith("aggregator.global_blocks."):
            new_state[k] = v.clone()

    # ── 2. Migrate each global_block → state_block ─────────────────────────
    for i in range(num_layers):
        src_pre = f"aggregator.global_blocks.{i}"
        dst_pre = f"aggregator.state_blocks.{i}"

        # Helper: grab with fallback
        def get(suffix):
            return old_state[f"{src_pre}.{suffix}"].clone()

        # --- QKV split ---
        qkv_w = get("attn.qkv.weight")   # [3D, D]
        q_w, k_w, v_w = qkv_w.chunk(3, dim=0)

        has_qkv_bias = f"{src_pre}.attn.qkv.bias" in old_state
        if has_qkv_bias:
            qkv_b = get("attn.qkv.bias")  # [3D]
            q_b, k_b, v_b = qkv_b.chunk(3, dim=0)

        # --- Out proj, norms, layer scales, mlp ---
        out_proj_w = get("attn.proj.weight")
        out_proj_b = get("attn.proj.bias")

        norm_q_w   = get("norm1.weight")
        norm_q_b   = get("norm1.bias")
        norm_kv_w  = norm_q_w.clone()    # reuse norm1 for norm_kv (same init)
        norm_kv_b  = norm_q_b.clone()

        norm2_w    = get("norm2.weight")
        norm2_b    = get("norm2.bias")

        ls1_gamma  = get("ls1.gamma")
        ls2_gamma  = get("ls2.gamma")

        mlp_fc1_w  = get("mlp.fc1.weight")
        mlp_fc1_b  = get("mlp.fc1.bias")
        mlp_fc2_w  = get("mlp.fc2.weight")
        mlp_fc2_b  = get("mlp.fc2.bias")

        # Optional qk_norm params
        has_qnorm = f"{src_pre}.attn.q_norm.weight" in old_state
        if has_qnorm:
            q_norm_w = get("attn.q_norm.weight")
            q_norm_b = get("attn.q_norm.bias")
            k_norm_w = get("attn.k_norm.weight")
            k_norm_b = get("attn.k_norm.bias")

        def write_sub(sub: str):
            """Write migrated weights into dst_pre.{sub}.*"""
            p = f"{dst_pre}.{sub}"

            new_state[f"{p}.attn.q_proj.weight"] = q_w
            new_state[f"{p}.attn.k_proj.weight"] = k_w
            new_state[f"{p}.attn.v_proj.weight"] = v_w
            if has_qkv_bias:
                new_state[f"{p}.attn.q_proj.bias"] = q_b
                new_state[f"{p}.attn.k_proj.bias"] = k_b
                new_state[f"{p}.attn.v_proj.bias"] = v_b

            new_state[f"{p}.attn.out_proj.weight"] = out_proj_w
            new_state[f"{p}.attn.out_proj.bias"]   = out_proj_b

            if has_qnorm:
                new_state[f"{p}.attn.q_norm.weight"] = q_norm_w
                new_state[f"{p}.attn.q_norm.bias"]   = q_norm_b
                new_state[f"{p}.attn.k_norm.weight"] = k_norm_w
                new_state[f"{p}.attn.k_norm.bias"]   = k_norm_b

            new_state[f"{p}.norm_q.weight"]  = norm_q_w
            new_state[f"{p}.norm_q.bias"]    = norm_q_b
            new_state[f"{p}.norm_kv.weight"] = norm_kv_w
            new_state[f"{p}.norm_kv.bias"]   = norm_kv_b
            new_state[f"{p}.norm2.weight"]   = norm2_w
            new_state[f"{p}.norm2.bias"]     = norm2_b

            new_state[f"{p}.ls1.gamma"] = ls1_gamma
            new_state[f"{p}.ls2.gamma"] = ls2_gamma

            new_state[f"{p}.mlp.0.weight"] = mlp_fc1_w
            new_state[f"{p}.mlp.0.bias"]   = mlp_fc1_b
            new_state[f"{p}.mlp.2.weight"] = mlp_fc2_w
            new_state[f"{p}.mlp.2.bias"]   = mlp_fc2_b

        # Write to both sub-blocks (identical warm start)
        write_sub("state_attn")
        write_sub("frame_attn")

        if (i + 1) % 6 == 0:
            print(f"  Migrated layers 0..{i}")

    # ── 3. state_tokens stays as random init; we skip it here so the new
    #      model's __init__ value is used on load (strict=False). ───────────
    print("  state_tokens: kept as random init in new model (not in migrated dict)")

    # ── 4. Save ──────────────────────────────────────────────────────────
    new_ckpt = {
        "step":         0,   # reset so Phase A training loop starts from step 0
        "model":        new_state,
        # drop optimizer/scheduler — Phase A starts fresh optimiser
        "warmup_done":  ckpt.get("warmup_done", True),
        "cfg":          ckpt.get("cfg", {}),
    }
    torch.save(new_ckpt, dst_ckpt_path)
    print(f"\nSaved migrated checkpoint → {dst_ckpt_path}")
    print(f"  Keys in new model state: {len(new_state)}")
    print(f"  Keys in old model state: {len(old_state)}")

    # ── 5. Sanity: list any new model keys not in migrated dict ──────────
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from dggt.models.art_vggt import ArtVGGT
        cfg = ckpt.get("cfg", {})
        model = ArtVGGT(
            img_size=cfg.get("img_size", 518),
            n_gaussians=cfg.get("n_gaussians", 256),
            scene_radius=cfg.get("scene_radius", 1.0),
            state_size=256,
        )
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        missing_non_state = [k for k in missing if "state_tokens" not in k]
        print(f"\nSanity check (strict=False):")
        print(f"  Missing keys (excl. state_tokens): {len(missing_non_state)}")
        if missing_non_state:
            for k in missing_non_state[:10]:
                print(f"    {k}")
        print(f"  Unexpected keys: {len(unexpected)}")
        if unexpected:
            for k in unexpected[:5]:
                print(f"    {k}")
        if len(missing_non_state) == 0 and len(unexpected) == 0:
            print("  ✓ Migration clean — all non-state_tokens keys matched.")
    except Exception as e:
        print(f"  Sanity check skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",  required=True,  help="Source checkpoint path (global_blocks)")
    parser.add_argument("--dst",  required=True,  help="Destination checkpoint path (state_blocks)")
    args = parser.parse_args()
    migrate(args.src, args.dst)
