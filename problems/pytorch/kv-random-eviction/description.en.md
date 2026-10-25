# Random KV Eviction Baseline: Sink, Window and Seeded Draw

## Problem

Alongside all the fancy presses, kvpress keeps the simplest one — `RandomPress` (`kvpress/presses/random_press.py`): give every KV pair a uniform random score and top-k it. It is the sanity floor for every other method — **a scorer that cannot beat dice has no signal**. This problem combines the random baseline with the StreamingLLM head/tail guarantee (order 145) into the standard evaluation shape: sink positions and the recent window are **always kept**; only the middle is sampled by lottery. It also asks for the **attention coverage** (share of the last query's attention mass landing on kept positions) so the answer is not just index bookkeeping but a direct measure of information loss.

The judge forbids hidden randomness: **the per-position uniform draws are handed in as an input tensor**, making the whole "random" procedure deterministic, reproducible and debuggable.

## Rules

Inputs: `draws` of shape \((L,)\) with values in \([0,1)\) — one uniform random number per position; `attn_weights` of shape \((L,)\) — non-negative attention of the last query on each key; integers `n_sink`, `window_size`; scalar `compression_ratio` \(r\in[0,1)\).

1. Forced set \(F\) = first \(\min(n\_sink,L)\) positions \(\cup\) last \(\min(window\_size,L)\) positions (union, de-duplicated when the two ranges overlap).
2. Total budget \(n\_kept=\max(1,\lfloor L(1-r)\rfloor)\); lottery quota \(q=\max(0,\ n\_kept-|F|)\).
3. Middle \(M=\{0,\dots,L-1\}\setminus F\): take the \(q\) largest `draws` (ties go to the smaller position); if \(|M|<q\), take all of \(M\).
4. `kept` = \(F\cup\) winners, output **ascending** (int64). Note: when \(|F|>n\_kept\) the kept set may exceed the budget — head/tail protection outranks the budget.
5. Coverage \(\text{coverage}=\sum_{j\in kept} a_j \big/ \sum_j a_j\); if the denominator is 0, fall back to \(|kept|/L\).

Return a pair: `kept` (1-D int64 tensor, ascending) and `coverage` (a Python float).

## Example

```
L=8, n_sink=2, window_size=2, r=0.375 → n_kept = max(1,⌊8×0.625⌋) = 5
draws        = [0.6, 0.1, 0.8, 0.3, 0.9, 0.4, 0.05, 0.7]
attn_weights = [0.30, 0.20, 0.10, 0.05, 0.15, 0.05, 0.08, 0.07]

Forced F = {0,1} ∪ {6,7}, |F|=4, quota q = 5-4 = 1
Middle M = {2,3,4,5}; the largest draw is position 4 (0.9) → winner {4}
kept     = [0, 1, 4, 6, 7]
coverage = (0.30+0.20+0.15+0.08+0.07) / 1.00 = 0.8
```

## Constraints

- CPU float32; `draws` \(\in[0,1)\); non-negative `attn_weights`; \(0\le r<1\); \(L\ge1\).
- When \(n\_sink+window\_size\ge L\) the middle is empty and everything is kept; coverage = 1 (provided \(\sum a_j>0\)).
- Ties in `draws` always break toward the smaller position — fully-equal draws may appear in tests.
- `kept` is ascending; `coverage` is returned as a float.
