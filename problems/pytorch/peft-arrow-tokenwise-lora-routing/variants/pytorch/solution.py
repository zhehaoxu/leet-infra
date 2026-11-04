"""PEFT: Arrow token-wise LoRA library routing (official PEFT arrow.py,
src/peft/tuners/lora/arrow.py, PR #2644; paper arXiv:2405.11157 NeurIPS 2024).

Official account:
  1. Prototype per adapter WITHOUT forming dW = B @ A: power iteration on the
     operator v -> A^T (B^T B) (A v) (the dominant right singular vector of
     dW). v0 = randn(in, generator seeded with rng_seed) normalized with eps;
     iters=15 steps of w = A^T (B^T B) (A v); v = w / (|w| + eps); all in
     float32, returned fp32 (cast to the LoRA dtype only for the buffer).
  2. Routing forward: tok = x.view(-1, F_in); sim = ABS(tok @ prototypes^T)
     (sign-agnostic); top-k per token over sim, non-selected entries set to
     -inf, coeff = softmax(full_score / temperature).
  3. Expert outputs: z = einsum("tf,erf->ter", tok, A_stack);
     y = einsum("ter,eor->teo", z, B_stack); y *= per-adapter scaling
     (task_adapter_names order); delta_flat = einsum("te,teo->to", coeff, y);
     dropout then reshape to (B, *rest, out).
  4. Library compatibility: every adapter must share rank r, identical A/B
     shapes and identical target_modules sets, else ValueError.
"""
import torch
import torch.nn.functional as F


def top_right_singular_vec(A, B, iters=15, seed=None, eps=1e-8):
    """Official power iteration for the top right singular vector of B@A
    without forming B@A: v <- A^T (B^T B) (A v), float32."""
    A32 = A.to(torch.float32)
    B32 = B.to(torch.float32)
    C = B32.t() @ B32
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
    v = torch.randn(A32.size(1), dtype=A32.dtype, generator=gen)
    v = v / (v.norm() + eps)
    for _ in range(iters):
        w = A32.t() @ (C @ (A32 @ v))
        v = w / (w.norm() + eps)
    return v


def build_prototypes(pairs, iters=15, seed=None, eps=1e-8):
    """Official build_prototypes: one top right singular vector per adapter,
    stacked to (E, in). The fp32 iteration result is cast back to A.dtype
    (official proto32.to(dtype=A.dtype)), so the buffer follows the LoRA dtype."""
    protos = [top_right_singular_vec(A, B, iters=iters, seed=seed, eps=eps).to(A.dtype)
              for A, B in pairs]
    return torch.stack(protos, dim=0)


def route_coeffs(tok, prototypes, top_k, temperature):
    """Official steps 1-2: sim = abs(tok @ protos^T), top-k, softmax(score/T)
    with non-selected entries at -inf (coefficient exactly 0)."""
    sim = torch.abs(tok @ prototypes.t())
    top_v, idx = torch.topk(sim, top_k, dim=1)
    full_score = tok.new_full((tok.size(0), prototypes.size(0)), float("-inf"))
    full_score = full_score.scatter(1, idx, top_v)
    return torch.softmax(full_score / temperature, dim=1), idx


def expert_outputs(tok, A_stack, B_stack, scales):
    """Official steps 3-6: per-expert z/y lifts, per-adapter scaling."""
    z = torch.einsum("t f, e r f -> t e r", tok, A_stack)
    y = torch.einsum("t e r, e o r -> t e o", z, B_stack)
    return y * scales.view(1, -1, 1)


def arrow_delta(x, prototypes, A_stack, B_stack, scales, top_k, temperature):
    """Official ArrowLoraLinearLayer.forward (dropout identity): token-wise
    top-k MoE over the adapter library."""
    B_, *rest, f_in = x.shape
    tok = x.reshape(-1, f_in)
    coeff, _ = route_coeffs(tok, prototypes, top_k, temperature)
    y = expert_outputs(tok, A_stack, B_stack, scales)
    delta_flat = torch.einsum("t e, t e o -> t o", coeff, y)
    out_dim = delta_flat.size(-1)
    return delta_flat.view(B_, *rest, out_dim)


def check_library_compat(ranks, shapes, module_sets):
    """Official check_loaded_lora_compatibility_arrow: shared rank, identical
    shapes, identical target_modules sets."""
    reference = None
    for r, shp, mods in zip(ranks, shapes, module_sets):
        if reference is None:
            reference = (r, shp, set(mods))
            continue
        if r != reference[0]:
            raise ValueError(f"rank mismatch: {r} != {reference[0]}")
        if shp != reference[1]:
            raise ValueError(f"shape mismatch: {shp} != {reference[1]}")
        if set(mods) != reference[2]:
            raise ValueError(
                f"target_modules mismatch:\n"
                f"  this adapter -> {sorted(set(mods))}\n"
                f"  reference   -> {sorted(reference[2])}"
            )
    return sorted(reference[2]), int(reference[0])


def solve(mode=None, **kw):
    if mode == "singular_vec":
        return top_right_singular_vec(kw["a"], kw["b"], iters=kw.get("iters", 15),
                                      seed=kw.get("seed"), eps=kw.get("eps", 1e-8))
    if mode == "prototypes":
        return build_prototypes(kw["pairs"], iters=kw.get("iters", 15),
                                seed=kw.get("seed"), eps=kw.get("eps", 1e-8))
    if mode == "route_coeffs":
        return route_coeffs(kw["tok"], kw["prototypes"], kw["top_k"],
                            kw["temperature"])
    if mode == "expert_outputs":
        return expert_outputs(kw["tok"], kw["a_stack"], kw["b_stack"], kw["scales"])
    if mode == "arrow_delta":
        return arrow_delta(kw["x"], kw["prototypes"], kw["a_stack"], kw["b_stack"],
                           kw["scales"], kw["top_k"], kw["temperature"])
    if mode == "check_compat":
        return check_library_compat(kw["ranks"], kw["shapes"], kw["module_sets"])
    raise ValueError(f"unknown mode {mode!r}")
