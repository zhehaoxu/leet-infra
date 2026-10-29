# FlashAttention Backward: Recomputing Gradients from O, LSE and dO

## Problem Description

FlashAttention's central promise is to never materialize the \(n\times n\) probability matrix: the forward keeps only the output \(O\) and the per-row log-sum-exp \(LSE\). The backward therefore has to **recompute** \(P=e^{S+mask-LSE}\) and substitute a per-row scalar \(D=\text{rowsum}(dO\circ O)\) for the normalization information that used to live inside \(P\). Given `Q, K, V, O, LSE, dO` (and an optional `mask`), return `(dQ, dK, dV)`.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- Recompute with \(P=\exp(S+mask-LSE)\) — subtract the **LSE, not the row maximum**; masked entries then become zero automatically.
- \(dV=P^T dO\); \(dP=dO\,V^T\); \(D=\text{rowsum}(dO\circ O)\); \(dS=P\circ(dP-D)\cdot scale\); \(dQ=dS\,K\), \(dK=dS^T Q\).
- Dropping \(D\) is the same as forgetting the mean-subtraction of the softmax backward: \(dQ,dK\) go wrong while \(dV\) stays right.
- `LSE` must be `(B, H, Sq)` and `mask` must broadcast to `(B, H, Sq, Sk)`; any mismatch raises ValueError. Compute in float64, return dtypes matching Q/K/V.

## Example

One head, \(S_q=1\), \(S_k=2\), \(D=2\), `scale=1.0`; numbers come from the EXAMPLE_* constants in test_cases.py:

```
Q = [[[0, 1]]]  K = [[[1, 0], [0, 0]]]  V = [[[0, 0], [1, 1]]]
S = [0, 0] ⇒ P = [0.5, 0.5], LSE = log 2, O = P@V = [0.5, 0.5], dO = [1, 0]
dP = dO@Vᵀ = [0, 1],  D = dO·O = 0.5,  dS = P∘(dP−D) = [−0.25, 0.25]
dQ = [[[−0.25, 0]]]   dK = [[[0, −0.25], [0, 0.25]]]   dV = [[[0.5, 0], [0.5, 0]]]
```

## Constraints

- Cross-checked entry by entry against a naive `softmax(QKᵀ·scale+mask) @ V` plus autograd, atol=1e-9 in float64
- Property cases: adding a constant \(c\) to `LSE` alone scales all three gradients by \(e^{-c}\); adding \(c\) to both `mask` and `LSE` leaves them unchanged; shifting \(O\) is a no-op whenever the rows of \(dO\) sum to zero
