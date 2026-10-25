# Solution Notes · Sequoia Multi-Branch Sampling and Verification

## Why verify several branches at one position

Single-sequence rejection sampling ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)) compares against only one draft token per position, whereas tree-based speculation (SpecInfer, Sequoia) hangs $k$ candidate branches at the same position. Sequoia's Algorithm 2 "consumes" these $k$ branches serially: each time it samples a candidate $s_i$ from the **current draft distribution $D$**, decides acceptance with the **residual distribution $R$**, and returns as soon as one is accepted; the acceptance test uses the ratio $R[s_i]/D[s_i]$ (the same form as standard rejection sampling, except that $R,D$ are updated dynamically after rejections).

## Dynamic shrinking of the residual R and the draft D

The key lies in the two updates after a rejection:
- $R\leftarrow\mathrm{norm}(\max(R-D,0))$: subtract the target probability mass already covered by the draft, leaving only the residual that has not yet been explained. Hence a later fallback sample $x\sim R$ still corrects the overall distribution back to the target $P$ (this is precisely why the procedure is lossless).
- $D[s_i]\leftarrow0$ followed by normalization: a rejected token is no longer proposed again as a candidate; when $D$ is whittled down to all zeros (all draft mass rejected), it is reset to uniform over the **non-rejected set**, avoiding division by zero.

## Boundaries and determinism

- $\mathrm{cdf}$ always takes the **smallest** $j$ with cumulative sum strictly $>u$; acceptance uses the strict $r_i < R[s_i]/D[s_i]$; $\mathrm{norm}$ degenerates to uniform for the zero vector. Any inconsistency in these boundaries drifts away from the oracle step by step, which is why the statement pins down every inequality direction.
- The $D[s_i]>0$ short-circuit: if at some step $D[s_i]=0$ (in theory cdf cannot sample zero mass except at wrap-around boundaries), the test is treated as not accepted and the rejection branch is taken, matching the paper.

## Relation to the other speculative problems

- 175 Medusa: only one deepest candidate path is taken per position (typical acceptance), with no residual distribution.
- **176 (this problem)**: multiple branches per position + dynamic residual $R$ updating, i.e. lossless tree verification.
- 177 Sequoia optimal tree: abstracts "how much can be accepted per layer" into a positional acceptance-rate vector $P[b]$ and picks the tree shape by DP (offline), complementing the online verification here.

## Edge-case self-checks

- `check_accept_first_branch`: the first branch is accepted, returning `[s0, 0]`.
- `check_example`: branch 0 is rejected → residual update → branch 1 is accepted.
- `check_residual_fallback`: $k=1$ and rejected, so the loop ends and the fallback token is sampled from $R$: `[x, 1]`.
- `check_matches_oracle`: random $P/Q/k$ and random uniform numbers, checked step by step against an independent pure-Python implementation.
