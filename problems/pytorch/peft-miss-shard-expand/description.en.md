# PEFT: MiSS Shard-Sharing Expand, Forward and Merge

## Problem

**MiSS** (Matrix Shard Sharing, arXiv:2409.15371, ICLR 2026; official repo `Joluck/MiSS`, merged into HuggingFace PEFT as `tuners/miss` in 2024.11) attacks the root cause of LoRA's slow convergence: jointly optimizing the two matrices \(B\) and \(A\). MiSS abandons the two-matrix structure and keeps **a single zero-initialized small matrix** `miss_block`, expanding it into a low-rank update through a shard-sharing replication mechanism — parameter count \(O(d\cdot r)\), \(\Delta W=0\) at the start of training, and zero initialization cost.

This problem replicates the deterministic accounts of the three `init_weights` forms of PEFT `MissLayer`:

- **miss (default expand)**: `miss_block (r, out)`. The forward pass splits the input into shards of size \(r\), sums across shards into an \(r\)-dim aggregate (right zero-pad when not divisible), then multiplies `miss_block`; at merge time every input-shard position of \(\Delta W\) carries the full `miss_block`ᵀ — the two forms are mathematically equivalent (\(\Delta W\cdot x = B\cdot\sum_\text{blocks}x\)).
- **mini**: `miss_block (r, mini_r)`, first repeated `out//mini_r` times along the output axis into \((r, out)\), then the same account as miss.
- **bat**: `miss_block (out//r, r, r)`; the base weight is blocked as \((out/r, r, in/r, r)\) and the \(j\)-th output block computes \(W_\text{block}\,@\,B_j + B_j\) — the additive term breaks the collinear shard updates of the Bone method.

## Rules

- `mode="split_dims"`: returns `{"n_blocks", "pad_size", "miss_block", "zero_init"}`; \(r\le0\) or \(r>in\) raises `ValueError`; bat requires both \(in\) and \(out\) divisible by \(r\), mini requires \(out\) divisible by `mini_r` (violations raise `ValueError`).
- `mode="shard_sum"`: right zero-pad to `n_blocks*r`, reshape to \((\dots,n_\text{blocks},r)\) and **sum across shards** (`sum(dim=-2)`), output \((\dots,r)\).
- `mode="forward_delta"`: mini expands first via repeat, then `shard_sum(x) @ miss_block`; mini without `out_features` raises `ValueError`.
- `mode="expand_delta"`: the merge \(\Delta W\) account — \(\Delta W = \text{tile}(B^\top,(1,in//r))\), shape \((out,in)\).
- `mode="bat_delta"`: block \((j,i)\) of \(\Delta W\) equals \(W_{0}^{(j,i)}\,B_j+B_j\).
- Everything in float64; dropout=0 (the Identity branch carries no parameters).

## Example

\(in=8, out=6, r=4\) (2 input shards):

```
split_dims -> {n_blocks: 2, pad_size: 0, miss_block: (4, 6), zero_init: True}
shard_sum(x)  # (2,3,8) -> (2,3,4)
forward_delta = shard_sum(x) @ B   # (2,3,6)
expand_delta  = tile(B.T, (1, 2))  # (6, 8), and x @ dW.T == forward_delta
```

With \(in=7, r=4\) the input is padded to 8 (`pad_size=1`) before aggregation.

## Constraints

- `x` supports arbitrary leading dims with the last dim \(d_\text{in}\); the bat \(\Delta W\) needs the full base `w0` as input.
- mini's `forward_delta`/`expand_delta` require an explicit `out_features`.
- `expand_delta` must be mathematically equivalent to `forward_delta` (cross-checked in the tests).
