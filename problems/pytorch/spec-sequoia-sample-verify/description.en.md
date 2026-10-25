# Sequoia: Multi-Branch Sampling and Verification (Algorithm 2)

## Problem

Speculative decoding must verify several candidate branches proposed by the draft **at the same position**. Classic rejection sampling flattens the draft into a single sequence and cannot exploit the multi-branch information at one position. The sampling/verification algorithm of **Sequoia** (Chen et al., 2024, [arXiv:2402.12374](https://arxiv.org/abs/2402.12374)) (its Algorithm 2, also related to SpecInfer [arXiv:2305.09781](https://arxiv.org/abs/2305.09781)) draws $k$ draft branches at one position in turn and decides acceptance for each using the residual distribution $R$ and the draft distribution $D$; a rejected token is removed from $D$ and $R$ is shrunk to $\mathrm{norm}(\max(R-D,0))$, which guarantees that the finally emitted token strictly follows the target distribution while also providing robustness properties such as "optimal transport".

This problem asks you to implement this per-position multi-branch verification loop. All randomness (sampling the branch, the acceptance coin, the fallback sample) is injected as explicit uniform numbers, keeping everything deterministic.

## Rules

Inputs:
- `P`: target distribution, length $V$.
- `Q`: draft distribution, length $V$.
- `k`: maximum number of branches verified at this position.
- `v_list`: $k$ uniform numbers; the $i$-th is used to invert the current $D$ and sample $s_i$.
- `r_list`: $k$ uniform numbers; the $i$-th is the acceptance coin of branch $s_i$.
- `u_final`: the uniform number used to sample from $R$ in the fallback.

Initialize $R=\mathrm{norm}(P)$, $D=\mathrm{norm}(Q)$, rejected set $S=\varnothing$. For $i=0,1,\dots,k-1$:

1. $s_i=\mathrm{cdf}(D, v_i)$ (the smallest $j$ with $\sum_{t\le j}D[t] > v_i$).
2. **Acceptance test**: if $D[s_i] > 0$ and $r_i < R[s_i]/D[s_i]$, **accept** $s_i$ and return $(s_i, \text{accepted})$.
3. **Otherwise reject** and update:
   - $R \leftarrow \mathrm{norm}(\max(R-D, 0))$ (elementwise non-negative, then normalize);
   - $D[s_i] \leftarrow 0$, $S \leftarrow S \cup \{s_i\}$;
   - if at this point $\sum D = 0$ (all draft mass has been rejected), set $D[t]=0\ (t\in S)$ and $1$ otherwise (reset to uniform over the non-rejected set);
   - $D \leftarrow \mathrm{norm}(D)$.

Here $\mathrm{norm}(x)$ divides by the sum of the elements; if the sum is $0$ it returns the uniform distribution.

After all $k$ branches have been rejected, return the fallback token $x=\mathrm{cdf}(R, u_{\text{final}})$, marked as residual.

Return a `torch.long` tensor `[token, reason]`, where `reason`: `0`=accepted, `1`=fallback.

## Examples

```
P=[0.1,0.2,0.3,0.4], Q=[0.4,0.3,0.2,0.1], k=2
v_list=[0.5,0.9], r_list=[0.9,0.5], u_final=0.3

i=0: s0 = cdf(Q,0.5)=1 (cum .4→.7);  r0=0.9 >= R[1]/D[1]=0.2/0.3=0.667  -> reject
     R = norm([0,0,0.1,0.3]) = [0,0,0.25,0.75]
     D[1]=0 -> norm([0.4,0,0.2,0.1]) = [0.571,0,0.286,0.143];  S={1}
i=1: s1 = cdf(D,0.9)=3;  r1=0.5 < R[3]/D[3]=0.75/0.143=5.25  -> accept s1=3

output = [3, 0]   # accepted, token=3
```

## Constraints

- CPU float32; return a `torch.long` of shape `(2,)`, compared with `torch.equal`.
- Sampling and acceptance follow exactly the order and boundary conditions above (strict $<$), so that the result matches the standalone oracle step by step.
- $\mathrm{cdf}$ takes the **smallest** $j$ with cumulative sum $> u$; $\mathrm{norm}$ degenerates to uniform for the zero vector.
