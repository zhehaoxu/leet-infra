# KV Cache: TPA Tensor-Product Attention Factor Contraction and Masking

## Problem

TPA (Tensor Product Attention, arXiv:2501.06425) factors each token's Q/K/V into a tensor product of two low-rank factors: a **per-head** coefficient and a **per-position, head-shared** basis. For keys,

\[
K_{m,h,d}=\sum_{s=1}^{S}\sigma(a^{k}_{m,h,s})\,b^{k}_{m,d,s},
\]

where \(a^k\) has shape `(M, H, S)` and \(b^k\) has shape `(M, D, S)`—note that \(b^k\) carries no head axis. The KV cache therefore stores only \(H\cdot S + D\cdot S\) numbers per token for keys (and \(H\cdot S_V + E\cdot S_V\) for values) instead of the dense $H\cdot(D+E)$; with `S=4, D=128` the saving is substantial. Decode also never materializes per-head K: it first computes `bq @ bk` into an \((R,S)\) tile, contracts \(a^q\) over \(R\) and \(a^k\) over the head axis, and only then applies softmax.

This problem implements the CPU reference path of `tpa_decode_torch` in upstream `decode/tpa_flash_kernel_prefill_mask.py`: sigmoid gating, four scale factors, and causal/additive/boolean masking.

## Task

Implement `solve(aq, ak, av, bq, bk, bv, is_causal=False, attn_mask=None, scale=None, scale_q=None, scale_k=None, scale_v=None)`:

- `aq` `(B, N, H, R)`, `ak` `(B, M, H, S)`, `av` `(B, M, H, S_V)`: per-head gated coefficients;
- `bq` `(B, N, R, D)`, `bk` `(B, M, D, S)`, `bv` `(B, M, E, S_V)`: per-position bases shared across heads;
- apply `sigmoid` first, then contract the last axis:

\[
q = \sigma(a^q)\cdot b^q\cdot s_q,\quad
k = \sigma(a^k)\cdot b^k\cdot s_k,\quad
v = \sigma(a^v)\cdot b^v\cdot s_v
\]

- Defaults are \(s_q=1/R\), \(s_k=1/S\), \(s_v=1/S_V\), \(s=D^{-1/2}\); explicit values override them;
- Scores \(S_{bhnm} = q\cdot k \cdot s + \text{bias}\), softmax over the `M` axis;
- `is_causal=True`: use a **top-left aligned** `tril(N, M)` mask (row `i` may only see `j <= i`), and `attn_mask` must be `None`;
- `attn_mask`: `(N, M)` bool (`True` means keep) or an additive mask with the input dtype, broadcast over batch/heads;
- Return a `dict`:
  - `out` `(B, N, H, E)`: attention output;
  - `prob` `(B, H, N, M)`: the softmax probabilities actually used (exactly 0 on masked positions).

Compute internally in float64 and cast `out`/`prob` back to the input dtype; do not modify any input. Raise `ValueError` for wrong dimensionality, mixed dtypes, wrong `attn_mask` shape/type, non-bool `is_causal`, `is_causal` together with `attn_mask`, or non-finite scale values.

## Example

```
aq = [[[[0]], [[0]]]]                     # (B,N,H,R) = (1,2,1,1)
bq = [[[[1, 0]], [[0, 1]]]]               # (B,N,R,D) = (1,2,1,2)
ak = [[[[0]], [[0]], [[0]]]]              # (B,M,H,S) = (1,3,1,1)
bk = [[[[1], [0]], [[0], [1]], [[1], [1]]]]        # (B,M,D,S) = (1,3,2,1)
av = [[[[0]], [[0]], [[0]]]]              # (B,M,H,S_V) = (1,3,1,1)
bv = [[[[1], [0]], [[0], [1]], [[2], [2]]]]        # (B,M,E,S_V) = (1,3,2,1)
is_causal = True

# sigmoid(0)=0.5, s_q=s_k=s_v=1, s=1/sqrt(2)
prob[0,0] = [[1, 0, 0],
             [0.45592056, 0.54407944, 0]]
out[0]    = [[[0.5, 0]],
             [[0.22796028, 0.27203972]]]
```

Row 0 may only see key 0, so its probability is `[1,0,0]` and the output is `v0 = 0.5*[1,0]`. Row 1 may see keys 0/1, whose scores are 0 and \(0.25/\sqrt{2}\); softmax yields `[0.4559, 0.5441]`. Key 2 also has a nonzero score but is pushed to `-inf` by the causal mask, so its probability is exactly 0.

## Constraints

- `B, N, M, H, R, S, S_V, D, E` are all ≥ 1; every input shares one dtype (float32/float64).
- The tensor-product convention is fixed: apply `sig(a)` first, then multiply by the basis; `aq`/`bq` contract over `R`, while `ak`/`av` contract with their bases over `S`/`S_V`.
- Scale placement: \(1/R\), \(1/S\), \(1/S_V\) multiply q, k, v respectively, and $D^{-1/2}$ multiplies the dot product; all four are real-valued overrides.
- Boolean `attn_mask` uses **True=keep** (upstream is `masked_fill_(mask.logical_not(), -inf)`); an additive mask is added to the scores directly.
- Every query must have at least one visible key (an all-`-inf` row makes softmax undefined); tests only build such inputs.
- Varlen is out of scope (upstream's torch path ignores `cu_seqlens`), as are Triton online softmax/LSE outputs; this is also not the sigmoid-free single-rank `flashtpa_decode_torch.py` variant—the masked `tpa_decode_torch` path is authoritative.
