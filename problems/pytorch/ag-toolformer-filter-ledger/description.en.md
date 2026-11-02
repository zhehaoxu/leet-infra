# Toolformer Call Filtering Ledger (ag-toolformer-filter-ledger)

**Difficulty**: Medium ｜ **Collection**: agentic-systems ｜ **Order**: 727

A stateless rebuild of Toolformer's (arXiv **2302.04761**; no official code, pinned to the **v5 paper**; dedup/implementation details from [lucidrains/toolformer-pytorch](https://github.com/lucidrains/toolformer-pytorch)) three-stage candidate-call pipeline: position sampling (τ_s gate + top-k + per-position cap m) → weighted cross-entropy filtering (w̃_t weights, L⁻ min of two controls, τ_f gate) → text-level dedup (replace_all_but_first). Per-token log probabilities are explicit inputs.

## What you implement

`solve(text_with_calls, positions, cands, tau_s=0.05, tau_f=1.0, k=5, m=5)`: return the three-way sampling ledger, per-candidate loss/score ledger, and the dedup text ledger.

## Ledger points (verified against the paper and the lucidrains implementation)

1. **Position sampling**: `I = {i | p_i > tau_s}` — **strictly greater** (equality dropped; App A default τ_s=0.05); more than k → top-k by p **descending stable** (ties keep original order; default k=5).
2. **Per-position cap m**: each kept position evaluates only its **first m** candidates (input order; default m=5; calculator/MT officially use τ_s=0, k=20, m=10, τ_f=0.5 — all explicit inputs here).
3. **Weighting function (Sec 4.1)**: `w̃_t = max(0, 1 − 0.2·t)` (t is the suffix step; t≥5 weight 0 but still occupies the list), **globally normalized** `w_t = w̃_t / Σ_{s∈ℕ} w̃_s = w̃_t / 3.0` (Σ = 1+0.8+0.6+0.4+0.2 = 3 constant, **not re-normalized per candidate suffix** — a 2-token suffix of all -1 gives L = 1.8/3 = 0.6, not 1).
4. **Weighted CE**: `L(z) = −Σ_t w_t · lp_z[t]` (lp lists suffix-aligned: index t=0 is the token at the call position).
5. **Two controls, take the min**: `L⁻ = min(L(ε), L(e(c, ε)))` — no call at all vs call without response; `score = L⁻ − L⁺`.
6. **Filter gate**: keep iff **score ≥ tau_f (inclusive)** (App A default τ_f=1.0). **Float quirk**: a 5-token suffix of all -1 gives score = 0.9999999999999999 < 1.0 — the exact boundary is eaten by `(w̃/3)·lp` rounding and dropped at τ_f=1.0; the inclusive semantics are pinned by the "identical tables → score exactly 0.0" anchor.
7. **Text-level dedup (lucidrains)**: the regex `' [tool(args)]'` has three quirks — a **leading space is required** (calls at line start don't match), tool name `\w+` (empty name ` [(2)]` doesn't match), args `[^)]*` truncated at `)` (calls containing `)` in args don't match at all); the first full match is kept, later ones **replaced with ''**; removed records the full matched strings; `has_calls` = whether the pre-dedup text contains any call (the data pipeline drops call-less pairs based on this).
8. **Domain constraints**: missing candidate keys / unequal lp table lengths / pos not in positions → ValueError (**validated upfront for ALL candidates**, even those that fail sampling).

## Example

```python
positions = [(0, 0.05), (1, 0.6), (2, 0.7)]
cands = [{"pos": 1, "call": "qa1",
          "lp_plus": [0, 0, -5, 0, 0],        # t=2 weight 0.2 → L⁺=1.0 exactly
          "lp_none": [0, 0, -10, 0, 0],       # L=2.0
          "lp_call_only": [0, 0, -10, 0, 0]}]
led = solve("a [calc(1)] b [calc(2)] c", positions, cands,
            tau_s=0.05, tau_f=0.9, k=2, m=1)
# tau_dropped == [0] (0.05 equal → strict > drops)
# kept_positions == [1, 2]; m=1 takes 1 candidate per position
# score = 2.0 − 1.0 = 1.0 ≥ 0.9 → kept; kept_calls == ["qa1"]
# removed == [" [calc(2)]"] (dedup keeps the first)
```

## Scoring

8 checkpoints: the strict τ_s gate and stable top-k order, the m cap and skips, the weight table and global normalization, t≥5 zero weights, the two-control min, the inclusive gate and the float exact-boundary quirk, the three dedup quirks (leading space / empty name / parenthesized args), and 120 randomized pipeline cross-checks.
