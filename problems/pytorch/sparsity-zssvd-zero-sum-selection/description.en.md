# Zero-Sum SVD Global Singular-Component Selection

## Problem

Low-rank LLM compression must pick a rank for every layer. Independent per-layer decisions (trimming every module to the same ratio) ignore the fact that **truncation-loss sensitivity varies wildly across modules** — a global allocation can slash total loss under the same parameter budget. Zero-Sum SVD (ICML 2026) gets its name from the core selection policy: each accepted singular-component truncation accumulates a predicted loss change δ, and the selector always prefers positive-δ candidates ("recoveries"), squeezing the running Σδ near zero — **assuming a loss increase on one module can be offset by a gain on another**.

You are given a set of modules (each with `weight` (M, N), `grad` (M, N), and a whitening Cholesky factor `L` (N, N) or None), plus the global prune ratio `global_prune_ratio`, the rank-floor ratio `keep_rank_ratio`, and the δ-estimation config (`beta_mixing` / `delta_order`).

## The pipeline to implement

### 1. Whitened-domain spectrum + gradient projection

```
A = W @ L                        # A = W when L is None
U, Σ, Vᵀ = svd(A)                # full_matrices=False, sorted by Σ descending
H = G @ L⁻ᵀ                      # gradient whitening (triangular solve)
grad_σ = diag(Uᵀ H V)            # first-order loss sensitivity, one per singular component
```

### 2. Truncation-loss estimate δ (tail candidates start from the smallest σ)

```
first_order:  δ = −σ · grad_σ
second_order: δ = −σ · grad_σ + 0.5 · σ² · grad_σ²
beta_mixing:  δ = −sign(g) · |σ|^β · |g|^(1−β)     (β ∈ [0,1])
```

### 3. Break-even rank constraint

```
r* = break_even_rank = max(1, (M·N − 1) // (M + N))
min_rank = max(1, ceil(keep_rank_ratio · r*))
each module may drop at most min(M, N) − min_rank singular components
```

r* is the break-even point where the low-rank factor cost (M+N)·r exactly matches the dense cost M·N: above r*, the low-rank factorization saves no parameters.

### 4. Zero-sum sign-balanced selection (core loop)

```
while dense_removed < stage_budget:
    prefer_positive = (running_sum ≤ 0)
    find the module candidate with the smallest |δ| inside the preferred
    sign heap (δ matching the preferred sign); if that heap is empty,
    fall through to the opposite heap (official pop_prefer_then_any);
    accept: consumed += 1 (drop that module's current smallest σ),
            running_sum += δ, truncated += σ²
```

Ties on `|δ|` break by ascending module name (determinism). Stop when all components are exhausted even if the budget is unmet.

### 5. Parameter accounting and termination

```
effective_dense_delta(r_now):
    r_now > r*:  0                       # above break-even, no savings
    r_now == r*: M·N − r*·(M+N)          # exactly at break-even, settle all savings
    r_now < r*:  M+N                     # below break-even, each rank drop saves M+N
stage_budget = total initial params − int((1 − ratio) · total initial params)
```

Return `dict(ranks, keep_masks, truncated_sigma_sq, running_sum, dense_removed, stage_budget, selections)`.

## Input constraints

- `modules` is a non-empty dict with non-empty string keys; each value has exactly the keys `weight`/`grad`/`L`; `weight`/`grad` are same-shape 2-D floating tensors; `L` is None or a (N, N) 2-D floating tensor;
- `global_prune_ratio` ∈ [0, 1), `keep_rank_ratio` ∈ [0, 1], `beta_mixing` ∈ [0, 1] or None, `delta_order` ∈ {first_order, second_order} (bools do not count).

## Hints

- When the preferred heap is empty you **must fall through to the opposite heap** rather than terminate — that is the official `pop_prefer_then_any` semantics;
- The sensitivity `grad_σ = diag(UᵀHV)` uses the V from the **whitened-domain** factorization of A = W@L, not of W itself;
- H = G@L⁻ᵀ is a triangular solve against L, transposed: implement as `solve(L, Gᵀ).t()`.
