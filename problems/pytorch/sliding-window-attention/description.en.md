# Sliding-Window (Local) Attention

## Problem

In standard causal attention, position $i$ can look back over the **entire** history of tokens; sliding-window attention restricts the lookback range to the most recent
`window` tokens, reducing attention complexity from $O(L^2)$ to $O(L\cdot\text{window})$, a key technique used by Mistral,
Longformer, and others for handling long sequences.

## Rules

- Implement `solve(q, k, v, window)`, single-head: `q`, `k`, `v` all have shape `(B, L, d)`, return `(B, L, d)`.
- The scaling factor is $1/\sqrt{d}$.
- The mask enforces **causality and the window at once**: position $i$ may attend only to
  keys/values with $\max(0,\, i-\text{window}+1) \le j \le i$ (i.e. $j \le i$ **and** $i-j < \text{window}$).
- The diagonal is always visible (each token can at least see itself), so no row degenerates to all $-\infty$.
- When `window >= L` it should degenerate to pure causal attention.

## Examples

```
Input: q=k=v=[[[1,0],[0,1],[1,1]]], window=2
Output: [[[1.000000, 0.000000],
        [0.330238, 0.669762],
        [0.669762, 1.000000]]]
(token 0 attends only to itself, output=v0)
```

## Constraints

- float32 inputs; $B \ge 1$, $L, d \ge 1$, `window >= 1`
- $q, k, v$ have equal sequence lengths ($L_k = L_q = L$)
