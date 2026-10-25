# Medusa: Typical-Acceptance Verification of a Multi-Head Candidate Tree

## Problem

**Medusa** (Cai et al., 2024, [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)) does not give the model a separate draft model. Instead it adds $K$ decoding heads on top of the backbone: head $k$ directly predicts several candidate tokens for the future position $k$. Combining the top candidates produced by these heads yields a **candidate tree** (multiple continuation paths of different lengths). In the verification stage, parallel tree attention computes the target distribution at every position in one shot, **typical acceptance** then decides layer by layer which candidates can be accepted, and finally the longest accepted path plus one bonus token is emitted.

This problem focuses on the **verification / selection** step: given an already assembled set of candidate paths and the target distributions, select the longest accepted path under the typical-acceptance rule and sample the bonus token.

## Rules

Inputs:
- `cands`: list of candidate paths `list[list[int]]`, each being a token-id sequence starting from the first speculative token (combinations of the Medusa per-head candidates; path lengths may differ).
- `P`: target probabilities of shape $(T, V)$; `P[t]` is the distribution at **position $t$** (`t=0` corresponds to the verification position of the first speculative token).
- `eps`: typical-acceptance threshold.
- `u_bonus`: a uniform number in $[0,1)$ used to sample the bonus from `P[D]`.

1. **Candidate-tree nodes** = **all prefixes** of all candidate paths (the paths themselves included), deduplicated; the empty prefix is the root.
2. **Layer-by-layer acceptance**: a node (a path of length $d$) is accepted at layer $i$ ($1\le i\le d$, token $\text{path}[i-1]$) if and only if $P[i-1][\text{path}[i-1]] \ge \text{eps}$. A whole path is accepted if and only if **every layer** is accepted (i.e. the entire parent chain is accepted).
3. **Path selection**: among all accepted nodes take the **deepest** one; on ties take the one with the **lexicographically smallest** token sequence. Let the selected length be $D$ ($D=0$ when only the root is accepted and no token is emitted).
4. **Bonus**: invert the CDF of `P[D]` at `u_bonus` (take the smallest $k$ such that $\mathrm{cumsum}_k > u$, i.e. $\text{bonus}=\min\{k:\sum_{j\le k}P[D][j] > u\}$).

Return a 1-D `torch.long` tensor: the tokens of the selected path followed by the bonus.

- Unlike per-token rejection sampling over a linear draft sequence ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)), Medusa verifies **a whole tree** at once and lets the parent-child chain jointly decide acceptance, so a deeper path can be emitted in a single step.
- Typical acceptance looks only at the target probability itself ($P\ge\varepsilon$); it needs neither the draft distribution $Q$ nor residual-normalized resampling.

## Examples

```
cands = [[1,2], [1,3], [0], [1,2,0]],  eps = 0.2,  u_bonus = 0.4

P[0] = [0.5, 0.3, 0.1, 0.1]
P[1] = [0.1, 0.3, 0.4, 0.2]
P[2] = [0.3, 0.1, 0.2, 0.4]
P[3] = [0.1, 0.5, 0.3, 0.1]

accepted nodes (every layer of the parent chain has P>=0.2):
  [0]      0.5>=0.2                    -> accept (depth 1)
  [1]      0.3>=0.2                    -> accept (depth 1)
  [1,2]    0.3,0.4                     -> accept (depth 2)
  [1,3]    0.3,0.2                     -> accept (depth 2)
  [1,2,0]  0.3,0.4,0.3                 -> accept (depth 3)  <- deepest
bonus = CDF(P[3], 0.4): cum=[0.3,0.4,0.6,..] -> index 1

output = [1, 2, 0, 1]
```

## Constraints

- CPU float32; return a 1-D `torch.long` tensor compared with `torch.equal`.
- Position alignment: the token at layer $i$ of a path is judged with `P[i-1]`; the bonus uses `P[D]`.
- Ties at the same depth are resolved by the **lexicographically smallest** token sequence, guaranteeing uniqueness.
- It is guaranteed that `len(P) >= D+1` ($D$ is the maximum candidate length).
