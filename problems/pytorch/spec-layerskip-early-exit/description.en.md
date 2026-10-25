# LayerSkip: Scheduling Early-Exit Drafts and Remaining-Layer Verification

## Problem

**LayerSkip** (Elhoushi & Shrivastava et al., 2024, [arXiv:2404.16710](https://arxiv.org/abs/2404.16710), code: [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)) is **self-speculative decoding**: no separate draft model. The same model drafts by exiting early and verifies by running the layers that were skipped. On the training side, layer dropout plus an early-exit loss turns one checkpoint into an ensemble of sub-models of different depths that all share a single LM head; on the inference side the draft stage runs only the first $E$ transformer blocks and unembeds directly, while the verification stage — thanks to the KVQ cache (the KV cache plus the query vector of the exit layer) — **runs only the remaining $L-E$ blocks**, so drafting and verifying share compute. The acceptance test in the paper is refreshingly plain: compare the token the full model predicts for each draft position against the drafted token, keep everything before the first disagreement, and append the target token at that point (or the extra token when everything agrees).

This problem turns that schedule into a **deterministic simulation**: you are given the toy model's output distribution at every draft position and every candidate exit depth, together with the full model's target distribution, and you must decide where each draft position exits, how many blocks verification costs, and then report **how many tokens the round produced, how many blocks were executed, and the resulting speedup**. The draft depth is **adaptive per token** (LayerSkip exits at different layers depending on how hard a token is; we formalize that with a confidence threshold), while verification is a single parallel forward that can only reuse the shallow cache prefix **shared by all** draft positions.

## Rules

Inputs:
- `Q`: float $(R, \gamma, D, V)$. `Q[r][j][l]` is the distribution that request $r$, draft position $j$ produces after running block $l+1$ and unembedding through the shared LM head (each row sums to 1, $D \le L$).
- `P`: float $(R, \gamma, V)$. `P[r][j]` is the target distribution of the full model (all $L$ blocks) at the same position.
- `tau`: float, the early-exit confidence threshold.
- `L`: int, total number of blocks ($1 \le D \le L$).

Simulate one round of "$\gamma$ draft positions + one parallel verification" per request, independently:

1. **Draft by early exit**: draft position $j$ starts at block 1 and goes deeper, taking the **first** layer $l$ with $\max_v \texttt{Q[r][j][l][v]} \ge \texttt{tau}$; the exit depth is $e_j = l + 1$, or $e_j = D$ if no layer qualifies. The draft token is $d_j = \arg\max_v \texttt{Q[r][j][}e_j\texttt{-1][}v]$ (ties go to the smallest index). Draft cost $=\sum_j e_j$ blocks.
2. **Verify with the remaining layers**: the parallel forward only runs the layers the draft cache did not cover, costing $\max(L-\min_j e_j,\,1)$ blocks ($\min_j e_j$ is the deepest prefix every draft position has executed; if the draft already ran the last block, the closing token still needs one LM-head pass, counted as 1 block).
3. **Greedy agreement test**: let $t_j = \arg\max_v \texttt{P[r][j][}v]$ (ties to the smallest index). Starting at $j=0$, accept the draft position while $d_j = t_j$; the first disagreement stops the scan and that position is filled by the target token $t_j$.
4. **Round yield**: tokens $=$ number of accepted draft positions $a$ plus $1$ (for $a < \gamma$ the extra token is the target token at the disagreement; for $a = \gamma$ it is the bonus token the full model produced for free).

Return the 4-tuple `(toks, lays, depth, speedup)`:
- `toks`: long $(R,)$ — tokens produced per request, $a+1$;
- `lays`: long $(R,)$ — blocks actually executed per request, $\sum_j e_j + \max(L-\min_j e_j, 1)$;
- `depth`: long $(R, \gamma)$ — the exit depth $e_j$ of every draft position;
- `speedup`: float — overall speedup $=\texttt{sum(toks)} \cdot L / \texttt{sum(lays)}$.

$\gamma = 0$ (`Q` of shape $(R,0,D,V)$) means no speculation this round, plain autoregressive full-model decoding: all `toks` are $1$, all `lays` are $L$, `depth` has shape $(R,0)$.

## Examples

```
R=2, gamma=3, D=3, L=5, V=4, tau=0.625 (every probability below is a multiple of 1/16)

request 0, per-draft-position layer maxima:
 pos0: [0.3125, 0.75, 0.5]      -> layer 2 qualifies -> e0 = 2, d0 = argmax of layer 2 = 1
 pos1: [0.5, 0.5, 0.375]        -> none qualifies    -> e1 = 3, d1 = argmax of layer 3 = 1
 pos2: [0.8125, 0.3125, 0.25]   -> layer 1 qualifies -> e2 = 1, d2 = argmax of layer 1 = 2
 draft cost = 2+3+1 = 6; shared cache prefix min e = 1 -> verify cost = max(5-1, 1) = 4
 target argmaxes: P[0] = [1, 1, 1]
 j=0: 1==1 accept; j=1: 1==1 accept; j=2: 1!=2 stop -> a=2
 toks[0] = 3, lays[0] = 6 + 4 = 10

request 1:
 pos0: [0.3125, 0.3125, 0.5625] -> e0 = 3, d0 = 2
 pos1: [0.75, 0.375, 0.25]      -> e1 = 1, d1 = 1
 pos2: [0.375, 0.625, 0.375]    -> e2 = 2, d2 = 2   (0.625 >= 0.625 sits exactly on the threshold)
 draft cost = 3+1+2 = 6; min e = 1 -> verify cost = 4
 target argmaxes: P[1] = [2, 3, 0]
 j=0: 2==2 accept; j=1: 3!=1 stop -> a=1
 toks[1] = 2, lays[1] = 6 + 4 = 10

batch: sum(toks)=5, sum(lays)=20 -> speedup = 5*5/20 = 1.25
output toks=[3, 2]  lays=[10, 10]  depth=[[2,3,1],[3,1,2]]  speedup=1.25

the same schedule on a shallower model, L=3 (D still 3): verify cost max(3-1,1)=2,
lays=[8, 8] -> speedup = 5*3/16 = 0.9375 < 1 (too few blocks left for verification to pay off)
```

## Constraints

- CPU float32; $R \ge 1$, $\gamma \ge 0$, $1 \le D \le L \le 64$, $V \ge 2$; every row of `Q` and of `P` sums to 1, `tau` $\in [0,1]$.
- Both argmaxes (draft token and target token) break ties by taking the **smallest index**; the early-exit test is $\ge$ (sitting exactly on the threshold qualifies).
- `toks`, `lays` and `depth` are `long` tensors compared with `torch.equal`; `speedup` is a python float compared with `allclose`.
- `speedup` uses the **batch-level, token-weighted** definition $\texttt{sum(toks)} \cdot L / \texttt{sum(lays)}$; it is **not** the arithmetic mean of the per-request values $\texttt{toks}[r] \cdot L / \texttt{lays}[r]$.
- The acceptance test is LayerSkip's own **draft equals target argmax**: unlike `inf-spec-decoding-rejection` (155, rejection sampling with acceptance probability $\min(1,p/q)$ and residual resampling) and unlike `spec-typical-acceptance` (178, the $p \ge \varepsilon$ threshold), this simulation is fully deterministic and needs no random numbers at all.
- This problem is only about the **layer schedule and its cost accounting**, not about tree drafts: optimal draft-tree construction and expected yield live in `spec-sequoia-optimal-tree` (177) and `spec-sequoia-depth-bounded-dp` (283), node-wise tree verification in `spec-specinfer-token-tree-verify` (285).
