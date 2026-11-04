# PEFT: Arrow Token-wise Routing over a LoRA Library

## Problem Description

**Arrow** (arXiv:2405.11157, NeurIPS 2024, merged into HF PEFT as `ArrowConfig`/`create_arrow_model`, PR #2644) reuses a library of pre-trained LoRA adapters with a **gradient-free, token-wise mixture-of-experts router**: each adapter contributes a **prototype** (the top right singular vector of \(\Delta W = B@A\), found by **power iteration** on \(v \mapsto A^\top (B^\top B)(Av)\) without materializing \(\Delta W\)); every token is compared to the prototypes with a **sign-agnostic** similarity, routed to its top-k most relevant adapters, and the expert outputs are linearly combined after softmax temperature normalization.

This problem replicates the deterministic account of the official PEFT implementation (`arrow.py`'s `ArrowLoraLinearLayer` + `check_loaded_lora_compatibility_arrow`):

- **Power-iteration prototype**: v₀ = randn(in, anchored by `rng_seed`) divided by its norm (eps=1e-8); `iters=15` steps of `w = A^T(B^T B)(Av)`, `v = w/(|w|+eps)`; all in float32 (official numerical-stability choice), returned fp32.
- **Prototype library**: one vector per adapter; the official `build_prototypes` casts the fp32 result back to \(A\)'s dtype before stacking into (E, in).
- **Routing coefficients**: `sim = |tok @ protos^T|` (**absolute value** — sign-agnostic); top-k per token; non-selected entries filled with \(-\infty\) before `softmax(score / temperature)` — **non-top-k coefficients are exactly 0** and the selected ones sum to 1.
- **Expert outputs and combination**: \(z = \) einsum("tf,erf→ter"), \(y = \) einsum("ter,eor→teo"); multiply by the **per-adapter** scaling (task_adapter_names order); `delta = einsum("te,teo→to", coeff, y)` then reshape back to (B, *rest, out).
- **Library compatibility**: every adapter must share rank r, identical A/B shapes and an identical target_modules set, else ValueError.

## Task

- `mode="singular_vec"`: inputs a, b ((r,in) / (out,r)), iters, seed, eps; return the fp32 unit vector.
- `mode="prototypes"`: inputs pairs ((A,B) list), iters, seed; return the (E, in) library (dtype follows A).
- `mode="route_coeffs"`: inputs tok, prototypes, top_k, temperature; return (coeff, idx); non-selected coefficients are exactly 0.
- `mode="expert_outputs"`: inputs tok, a_stack, b_stack, scales; return the (t, E, out) scaled expert outputs.
- `mode="arrow_delta"`: inputs x (any leading dims), prototypes, a_stack, b_stack, scales, top_k, temperature; return the delta with the last dim replaced by out.
- `mode="check_compat"`: inputs ranks, shapes, module_sets; return (sorted modules, r); inconsistencies raise ValueError.

## Example

seeds 0/1/2/4, rng_seed=42, float64 pipeline (fp32 inside the power iteration):

```
singular_vec(A (3,8), B (6,3), seed=42) -> v (8,), v[0] = 0.6975467205047607
prototypes(pairs×3, seed=42) -> (3,8), protos[2,3] = -0.170495867729187
route_coeffs(tok (5,8), protos (4,8), top_k=2, T=1) -> coeff (5,4)
  coeff[0, idx[0,0]] = 0.7688820194964578; non-selected = 0; T→0 approaches one-hot
arrow_delta(x (2,3,8), ...) -> (2,3,6), delta[0,0] = 2.7192971987121317
check_compat([4,4], same shapes, [{"q","v"},{"v","q"}]) -> (["q","v"], 4)
```

## Constraints

- The similarity takes the **absolute value** (sign-agnostic); the singular vector itself is ±-ambiguous, so downstream code must be sign-invariant.
- Too few power iterations deviate significantly from the dominant direction (iters=2 vs 15 differ by ~0.26).
- Non-selected routing coefficients are **exactly 0** (-inf underflows in softmax), not small numbers.
- temperature divides BEFORE the softmax; T→0 degenerates to argmax one-hot.
- scaling acts on the expert outputs per adapter, before the weighted sum; different top_k/temperature change the routing.
- The prototype library dtype follows A (the fp32 iteration result is cast back); the routing runs without further dtype casts.
