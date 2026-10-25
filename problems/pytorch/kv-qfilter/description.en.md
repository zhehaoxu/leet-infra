# Q-Filter: Learned Filter Negative-Dot Scoring

## Problem

Q-Filter (Zang et al., 2025, [arXiv:2503.02812](https://arxiv.org/abs/2503.02812); `QFilterPress` in `kvpress/presses/qfilter_press.py` of NVIDIA/kvpress) abandons all heuristics: for **every (layer, kv head) a filter vector** \(f\) of length head_dim is pre-trained, and a token's score is

\[ \text{score}_t = -\, f \cdot k_t \]

i.e. keys projecting *lower* along the filter (more negatively aligned) score higher and survive. The filters come neither from attention nor from norms — they are end-to-end learned "importance directions" shipped per model (kvpress loads `nthngdy/<model>_qfilt` from the HF hub); here they are given directly as an input tensor.

## Rules

Inputs: `q_filters` of shape \((N, H, D)\) — one filter per layer per kv head; `keys` of shape \((N, H, L, D)\) — RoPE-applied cached keys; scalar `compression_ratio` \(r\in[0,1)\).

1. Per \((n, h, t)\): \(\text{score} = -\sum_{d} \text{q\_filters}[n,h,d]\cdot \text{keys}[n,h,t,d]\) (negated dot product, no \(\sqrt{D}\) scaling).
2. \(n\_kept=\max(1,\lfloor L(1-r)\rfloor)\).
3. Per \((n,h)\) row take the top-\(n\_kept\) scores (larger first, **ties go to the smaller position**), then output the kept positions ascending.

Return two tensors: `scores` (float32, \((N, H, L)\)) and `kept` (int64, \((N, H, n\_kept)\)).

## Example

```
N=2, H=1, D=2, L=4, r=0.5 → n_kept = 2

layer0: f = [1.0, -2.0]
        keys = [[0.5, 1.0], [0.0, -1.0], [2.0, 0.5], [-1.0, -0.5]]
        dot  = [-1.5, 2.0, 1.0, 0.0]      → scores = [1.5, -2.0, -1.0, -0.0]
        top-2 → kept = [0, 3]
layer1: f = [-0.5, 0.25]
        keys = [[1.0, 1.0], [2.0, 0.0], [0.0, 4.0], [-1.0, 2.0]]
        dot  = [-0.25, -1.0, 1.0, 1.0]   → scores = [0.25, 1.0, -1.0, -1.0]
        top-2 → kept = [0, 1]
```

## Constraints

- CPU float32; \(N,H,D,L\ge1\); \(0\le r<1\).
- The score is the **negated** dot product: keys aligned with the filter score low and are evicted, anti-aligned keys survive — the sign is the algorithm itself; dropping the minus yields the opposite choice.
- If a filter is the zero vector, all scores of that (layer, head) are 0 and the tie rule keeps the leftmost \(n\_kept\) positions.
- All ties break toward the smaller position; output is sorted ascending.
