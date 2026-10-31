# Gated Linear Attention: Chunked-Parallel Form vs Naive Recurrence Identity

## Problem

Gated Linear Attention (2312.06635, implemented in fla-org/flash-linear-attention) adds a per-key-channel log-decay gate `g` to linear attention: the naive recurrence is `h_t = h_{t-1}·diag(exp(g_t)) + k_t ⊗ v_t`, `o_t = (q_t·scale)·h_t` with default `scale = K^-0.5` (the official `naive_recurrent_gla` formulation). This problem requires implementing the same recurrence in its **chunked-parallel form**: split `[0, T)` into left-closed right-open chunks of `chunk_size`; per chunk, split the contribution of the pre-chunk state and the intra-chunk pairwise terms using the inclusive in-chunk cumsum `S`, and report the state after every chunk. The chunked result must match the naive recurrence bit-for-bit within tolerance.

## Solution

For each chunk `[c0, c1)` of length l:

```
S = cumsum(g[c0:c1], dim=0)                  # inclusive in-chunk cumsum, [l, K]
# inter-chunk: read the pre-chunk state h_prev with decay exp(S_t)
o[c0:c1] += (q·scale ⊙ exp(S)) @ h_prev
# intra-chunk: A[t,j] = Σ_i q_t,i·k_j,i·exp(S_t,i − S_j,i), lower-triangular incl. diagonal
A = (q·scale ⊙ exp(S)) @ (k ⊙ exp(−S)).T
o[c0:c1] += tril(A) @ v[c0:c1]
# state update: end-of-chunk decay (row-wise per key channel) + decayed k accumulation
h = exp(S_end)·h_prev (row-wise) + (k ⊙ exp(S_end − S)).T @ v
```

Key structural points:

1. **`g` is `[T, K]`, per key channel**: cumsum and every exp are channel-wise; there is no scalar decay.
2. **End-of-chunk decay scales rows**: `exp(S_end)[:, None] * h` — a bare `[K] * [K,V]` is broadcast along the trailing dim by torch (column-wise scaling), silently wrong when K=V (an observed pitfall in this very problem).
3. **The diagonal lives in the intra term**: `A[t,t] = q_t·k_t` since `S_t − S_t = 0`; the mask is an inclusive lower triangle.
4. **Uneven tail chunk**: when the last chunk is short, l takes the actual length and `S_end = gc[-1]` keeps its meaning.
5. Everything is computed in float64 (pinned by the problem statement) so the comparison with the naive recurrence holds at 1e-8.
