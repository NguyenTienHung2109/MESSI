#!/usr/bin/env python3
"""
Smoke test for Table 13 (computational cost) — DomainNet, DeiT-S/16.

Reports Params (M), Activated MACs (G), and Latency (ms, bs=1) for:
    GMoE, OMoE, MESSI L_MMD_ssi, MESSI L_OT_ssi.

No training, no checkpoints, random init. Run with the `gmoe` conda env so that
Tutel is available for the GMoE row:

    /home/hungnt/anaconda3/envs/gmoe/bin/python scripts/measure_compute_cost.py
    # or:  conda activate gmoe && python scripts/measure_compute_cost.py

Activated MACs are computed analytically (paper convention, matches
sweep/compute_flops.py): 2*L*active_params + 4*L^2*d*num_layers, with
active_params accounting for top-k sparse routing on MoE layers. A forward-
hook pass cross-checks the dense (non-MoE) portion and warns on >5% drift.
"""

import argparse
import functools
import sys
import os
import warnings
from contextlib import contextmanager

warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting .* in registry.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Importing from .* is deprecated.*")

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "domainbed"))   # vit_helpers etc.
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))     # measure_latency
import domainbed.tutel_patch  # noqa: F401  matches train.py setup
from domainbed import vision_transformer as vit_module
from domainbed.algorithms import get_algorithm_class
from domainbed.hparams_registry import default_hparams
from measure_latency import count_parameters, measure_latency


# ----------------------------------------------------------------------------
# Methods table
# ----------------------------------------------------------------------------

ALGOS = [
    # (label,             algo_class,    hparams_base, extra_hp)
    # gate_k=2 for GMoE/OMoE so Gram-Schmidt is non-trivial (k=1 → no-op).
    ("GMoE",              "GMOE",        "GMOE",  {"pretrained": False, "gate_k": 2}),
    ("OMoE",              "GMoEOMoE",    "GMOE",  {"use_omoe": True, "pretrained": False, "gate_k": 2}),
    # ERM hparams default 'model' to 'resnet50' — must override for MESSI.
    ("MESSI L_MMD_ssi",   "GMOE_InvMMD", "ERM",   {"pretrained": False, "model": "deit_small_patch16_224"}),
    ("MESSI L_OT_ssi",    "GMOE_InvOT",  "ERM",   {"pretrained": False, "model": "deit_small_patch16_224"}),
]

INPUT_SHAPE = (3, 224, 224)
NUM_CLASSES = 345          # DomainNet
NUM_DOMAINS = 6


# ----------------------------------------------------------------------------
# Model construction (with pretrained download disabled)
# ----------------------------------------------------------------------------

@contextmanager
def no_pretrained_deit():
    """Monkey-patch deit_*_patch16_224 factories on every module path that
    might own them, forcing pretrained=False. GMOE/GMoEOMoE hardcode
    pretrained=True at algorithms.py:318, but algorithms.py imports
    `vision_transformer` as a TOP-LEVEL module (line 20), not as
    `domainbed.vision_transformer`, so the two are distinct module objects
    when both are on sys.path. Patch both."""
    targets = []
    for mod_name in ("domainbed.vision_transformer", "vision_transformer"):
        m = sys.modules.get(mod_name)
        if m is not None:
            targets.append(m)

    def _wrap(orig):
        @functools.wraps(orig)
        def patched(*args, **kwargs):
            kwargs["pretrained"] = False
            return orig(*args, **kwargs)
        return patched

    originals = []   # list of (module, name, original_fn)
    for m in targets:
        for name in ("deit_small_patch16_224", "deit_base_patch16_224",
                     "deit_tiny_patch16_224"):
            fn = getattr(m, name, None)
            if fn is None:
                continue
            originals.append((m, name, fn))
            setattr(m, name, _wrap(fn))
    try:
        yield
    finally:
        for m, name, fn in originals:
            setattr(m, name, fn)


def build(algo_name, hparams_base, extra_hp):
    hp = default_hparams(hparams_base, "DomainNet")
    hp.update(extra_hp)
    cls = get_algorithm_class(algo_name)
    with no_pretrained_deit():
        model = cls(INPUT_SHAPE, NUM_CLASSES, NUM_DOMAINS, hp)
    return model.eval(), hp


# ----------------------------------------------------------------------------
# Architecture introspection
# ----------------------------------------------------------------------------

def infer_arch(algo_label, model):
    """Returns (L_tokens, embed_dim, num_layers, vit) for whichever code path."""
    # MESSI variants: model.featurizer.vit
    if hasattr(model, "featurizer") and hasattr(model.featurizer, "vit"):
        vit = model.featurizer.vit
    else:
        vit = model.model      # GMOE / GMoEOMoE
    L = vit.pos_embed.shape[1]            # tokens including cls (+ dist)
    d = vit.embed_dim
    n_layers = len(vit.blocks)
    return L, d, n_layers, vit


# ----------------------------------------------------------------------------
# Analytical activated MACs (paper convention)
# ----------------------------------------------------------------------------

def _is_tutel_moe(mod):
    return mod.__class__.__name__ == "MOELayer" and hasattr(mod, "experts") \
        and hasattr(mod.experts, "batched_fc1_w")


def _is_deep_moe(mod):
    return mod.__class__.__name__ == "DeepMoELayer"


def _is_explicit_moe_head(mod):
    return mod.__class__.__name__ == "ExplicitMoEHead"


def _params_in(mod):
    return sum(p.numel() for p in mod.parameters())


def active_params(model):
    """Walk module tree and compute activated parameters per inference.

    Sparse MoE FFN layers (Tutel MOELayer or DeepMoELayer) contribute only
    (gate_k / num_experts) of their expert weight count. The router/gate
    runs on every token, so its params count fully.

    ExplicitMoEHead (MESSI) uses soft routing → all experts active.
    """
    seen_ids = set()
    total = 0

    def add_mod(m):
        for p in m.parameters():
            if id(p) not in seen_ids:
                seen_ids.add(id(p))
                nonlocal_total[0] += p.numel()

    nonlocal_total = [0]

    for m in model.modules():
        if _is_tutel_moe(m):
            E = m.experts
            num_experts = E.batched_fc1_w.shape[0]
            try:
                gate_k = m.top_k
            except AttributeError:
                gate_k = m.gates[0].top_k if hasattr(m.gates[0], "top_k") else 1
            # Expert weights
            expert_param_ids = set()
            for p in E.parameters():
                expert_param_ids.add(id(p))
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    nonlocal_total[0] += int(round(p.numel() * gate_k / num_experts))
            # Gate (router) — full
            for p in m.gates.parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    nonlocal_total[0] += p.numel()

        elif _is_deep_moe(m):
            num_experts = m.num_experts
            gate_k = m.gate_k
            for p in m.experts.parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    nonlocal_total[0] += int(round(p.numel() * gate_k / num_experts))
            for p in m.gate_proj.parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    nonlocal_total[0] += p.numel()

    # Add everything else not yet counted (dense layers, attention, classifier,
    # ExplicitMoEHead — soft routing means full)
    for p in model.parameters():
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            nonlocal_total[0] += p.numel()

    return nonlocal_total[0]


def count_macs_analytical(model, L, d, num_layers):
    """Activated MACs (multiply-accumulates), per image:
        linear/conv MACs   = L * active_params         (1 MAC per output per weight)
        attention matmuls  = 2 * L^2 * d * num_layers  (QK^T + Attn*V, summed over heads)
    Note: sweep/compute_flops.py uses 2x these numbers because it reports FLOPs
    (mul + add). The paper Table 13 reports MACs."""
    ap = active_params(model)
    linear_macs = L * ap
    attn_macs = 2 * (L ** 2) * d * num_layers
    return linear_macs + attn_macs, ap


# ----------------------------------------------------------------------------
# Dense-portion hook cross-check (skips Tutel/DeepMoE internals)
# ----------------------------------------------------------------------------

def count_macs_dense_hooks(model, x):
    """Counts MACs only on the dense (non-MoE-FFN) portion via forward hooks.

    Skipped subtrees: any Tutel MOELayer or DeepMoELayer (their experts are
    counted analytically). Everything else — patch embed conv, attention QKV/
    proj linears, dense MLP linears, the ExplicitMoEHead linears, classifier
    — is hooked.
    """
    macs = {"linear": 0, "conv": 0, "attn_softmax": 0}

    def linear_hook(m, inputs, output):
        # inputs is a tuple; take first
        x_in = inputs[0]
        # tokens = product of all dims except the last (in_features)
        tokens = x_in.numel() // x_in.shape[-1]
        macs["linear"] += tokens * m.in_features * m.out_features

    def conv_hook(m, inputs, output):
        # output shape: (B, C_out, H, W)
        out = output
        out_elements = out.numel() // out.shape[1]   # B * H * W
        kH, kW = m.kernel_size
        macs["conv"] += out_elements * m.in_channels * m.out_channels * kH * kW

    # Identify which modules to skip (everything inside Tutel/DeepMoE)
    skip_ids = set()
    for m in model.modules():
        if _is_tutel_moe(m) or _is_deep_moe(m):
            for sub in m.modules():
                skip_ids.add(id(sub))

    handles = []
    for m in model.modules():
        if id(m) in skip_ids:
            continue
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))

    # Attention matmul portion (QK^T and Attn·V) — analytical MACs, per block.
    # MACs per block per head: L * L * d_head (QK^T) + L * L * d_head (Attn*V)
    #                       = 2 * L^2 * d_head; summed over heads = 2 * L^2 * d.
    L, d, n_layers, _ = infer_arch(None, model)
    macs["attn_softmax"] = 2 * (L ** 2) * d * n_layers

    try:
        with torch.no_grad():
            _ = model.predict(x)
    finally:
        for h in handles:
            h.remove()

    return macs["linear"] + macs["conv"] + macs["attn_softmax"], macs


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def fmt_mac(mac):
    return f"{mac / 1e9:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(1)
    device = torch.device(args.device)

    try:
        import tutel  # noqa: F401
        tutel_ok = True
    except ImportError:
        tutel_ok = False

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}    tutel: {'OK' if tutel_ok else 'MISSING (GMoE will fail — switch to gmoe env)'}")
    print(f"Backbone: DeiT-S/16    Input: {INPUT_SHAPE}    Classes: {NUM_CLASSES}    Domains: {NUM_DOMAINS}")
    print()

    rows = []
    for label, algo, base, extra in ALGOS:
        print(f"[BUILD] {label:<18s}  algo={algo:<12s} base_hparams={base}")
        try:
            model, hp = build(algo, base, extra)
            model.to(device)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            rows.append((label, None))
            continue

        total_params, trainable = count_parameters(model)
        L, d, n_layers, _ = infer_arch(label, model)

        macs_ana, ap = count_macs_analytical(model, L, d, n_layers)

        x = torch.randn(1, *INPUT_SHAPE, device=device)
        try:
            macs_hook_dense, hook_breakdown = count_macs_dense_hooks(model, x)
        except Exception as e:
            print(f"  HOOK FAILED: {type(e).__name__}: {e}")
            macs_hook_dense, hook_breakdown = None, None

        # Latency
        lat = measure_latency(model, INPUT_SHAPE, batch_size=1, device=device,
                              warmup=args.warmup, repeats=args.repeats)

        # Diagnostic: dense portion of analytical = MACs minus the activated
        # expert-FFN portion. We approximate by recomputing analytical with
        # (gate_k / num_experts) replaced by 1 — i.e. "all experts active" —
        # and subtracting; but a simpler check: compare the *total* analytical
        # to hook_dense + activated expert-FFN MACs. Implemented below.
        rows.append({
            "label": label,
            "params_M": total_params / 1e6,
            "trainable_M": trainable / 1e6,
            "macs_ana_G": macs_ana / 1e9,
            "macs_hook_dense_G": (macs_hook_dense / 1e9) if macs_hook_dense else None,
            "lat_ms": lat["per_image_ms"],
            "lat_std_ms": lat["total_std_ms"],
            "L": L, "d": d, "n_layers": n_layers, "active_params_M": ap / 1e6,
        })

        # Cross-check: compute analytical MACs assuming dense-only portion
        # (no MoE FFN at all) and compare hook to it within 5%.
        # Dense-only active params = total params minus MoE-expert params.
        dense_only_params = ap_dense_only(model)
        macs_dense_ana = L * dense_only_params + 2 * (L ** 2) * d * n_layers
        if macs_hook_dense is not None:
            drift = abs(macs_hook_dense - macs_dense_ana) / max(macs_dense_ana, 1)
            tag = "OK" if drift <= 0.05 else "WARN"
            print(f"  [{tag}] hook(dense)={macs_hook_dense/1e9:.3f}G  "
                  f"analytical(dense)={macs_dense_ana/1e9:.3f}G  drift={drift*100:.1f}%")

        del model
        torch.cuda.empty_cache()
        print()

    # ------------------------------------------------------------------------
    # Print final table
    # ------------------------------------------------------------------------
    print("=" * 86)
    print(f"{'Method':<22s} {'Params(M)':>10s} {'Train(M)':>10s} "
          f"{'ActMACs(G)':>12s} {'Latency bs=1 (ms)':>20s}")
    print("-" * 86)
    for r in rows:
        if isinstance(r, tuple):  # error case
            print(f"{r[0]:<22s}  (failed)")
            continue
        print(f"{r['label']:<22s} {r['params_M']:>10.2f} {r['trainable_M']:>10.2f} "
              f"{r['macs_ana_G']:>12.2f} {r['lat_ms']:>15.2f} ± {r['lat_std_ms']:>4.2f}")
    print("=" * 86)
    print()
    print("Activated MACs reported analytically (paper convention).")
    print("Dense-portion cross-check (hook vs analytical) printed above each row.")
    if not tutel_ok:
        print("WARNING: Tutel missing — GMoE row above is invalid. Use the gmoe conda env.")


def ap_dense_only(model):
    """Active params excluding all MoE-expert weights (used for hook cross-check)."""
    skip_ids = set()
    for m in model.modules():
        if _is_tutel_moe(m):
            for p in m.experts.parameters():
                skip_ids.add(id(p))
        elif _is_deep_moe(m):
            for p in m.experts.parameters():
                skip_ids.add(id(p))
    return sum(p.numel() for p in model.parameters() if id(p) not in skip_ids)


if __name__ == "__main__":
    main()
