# Attention Backward w.r.t. Q/K/V: Masked Zeros and GQA Head Folding

## Problem Description

The attention backward is the most expensive stretch of Transformer training and the line that separates "has written a fused kernel" from "has only called the API". Three gradients are coupled: \(dV\) needs a single multiply with \(P^T\), while \(dQ/dK\) must pass through softmax — and the \(\sum_k P_k\,dP_k\) factor there is routinely misremembered as \(\sum_k dP_k\). On top of that, GQA lets several Q heads share one K/V head, so \(dK,dV\) must **sum** across the group and return `Hkv` rather than `H_q` heads. Given `Q, K, V, P, dO`, return `(dQ, dK, dV)`.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- \(dP=dO\,V_{head}^T\); apply the softmax backward along the last axis of \(P\): \(dS=P\circ(dP-\text{rowsum}(P\circ dP))\), then multiply by `scale`.
- \(dQ=dS\,K_{head}\) keeps the shape of `Q`; \(dK=dS^T Q\) and \(dV=P^T dO\) must accumulate the `groups` Q heads of each group and return tensors whose head count is `Hkv`.
- At masked positions \(P=0\) forces a zero gradient; no extra mask or constant may be added.
- `Hq % Hkv != 0`, unequal K/V head counts and any shape mismatch raise ValueError; the return dtype matches `Q` (accumulated in float64).

## Example

1 head, \(S_q=1\), \(S_k=2\), \(D=1\), `scale=0.5`; numbers come from the EXAMPLE_* constants in test_cases.py:

```
Q = [[[4.0]]]   K = [[[1.0], [2.0]]]   V = [[[1.0], [3.0]]]
P = [[[0.5, 0.5]]]   dO = [[[2.0]]]
dP = [2.0, 6.0],  rowsum(P*dP) = 4.0,  dS = P*(dP-4)*scale = [-0.5, 0.5]
dQ = [[[0.5]]]   dK = [[[-2.0], [2.0]]]   dV = [[[1.0], [1.0]]]
```

\(dQ=-0.5\cdot1+0.5\cdot2=0.5\); \(dK=dS^T Q=[-0.5\cdot4,\,0.5\cdot4]\).

## Constraints

- `P` is the forward probability matrix (rows sum to 1), not logits; accumulate in float64 then cast to `Q.dtype`
- Cross-checked against an oracle built with autograd as `softmax(QK^T*scale) @ V`, atol=1e-9 in float64; GQA cases use `Hq = 2·Hkv`
