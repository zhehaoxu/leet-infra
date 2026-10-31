# Reference Approach: Alternating Scaling, Where eps Goes, and When Double Stochasticity Breaks

## Implementation

1. `q = torch.softmax(logits / tau, dim=-1)`, which already produces a fresh float64 tensor and never touches the input.
2. Two statements per round: `q = q / (q.sum(dim=0, keepdim=True) + eps)` (column normalization), immediately followed by `q = q / (q.sum(dim=1, keepdim=True) + eps)` (row normalization). Both must allocate; do not reuse the input tensor as scratch space.
3. Loop exactly `iters` times; with `iters=0` nothing runs.
4. If `square=True`, do `q = q * q` and then one more row normalization with the same `sum + eps` divisor. The squaring must come after the loop, otherwise later rounds smooth the sharpening away.
5. `col_load = q.sum(dim=0)`, taken over the **returned** matrix, that is after the squaring branch.

Three common mistakes:

- **Reversed alternation**. Rows-then-columns also drives column sums to 1, but the limit differs: with `T=1` and `iters=1`, columns-then-rows returns a row of all $0.25$ (the row constraint is exact, the column constraint is only approximated), while rows-then-columns returns a row of all $1$ (columns happen to be satisfied, row sums become $E$). The single-token case exists to separate these two implementations.
- **eps in the wrong place**. `(q + eps) / (sum + 2*eps)` lifts an all-zero column to $0.5$ per row ($\varepsilon/(2\varepsilon)$), handing mass to experts that had none; `q / (sum + eps)` keeps it at 0.
- **Masking 0/0 with `clamp_min`**. When a column sum is 0, `clamp_min(1e-30)` yields a matrix that "looks normalized", while the contract demands a faithful NaN.

Not stopping early on `allclose(col_sum, 1)` is deliberate: an early exit makes the number of rounds depend on the order of float comparisons, so the same input with a different reduction order could stop one round sooner and the judge could not state an exact expected value. A fixed round count turns the whole thing into plain arithmetic.

## Correctness checks

The 7 `check_*` functions cover: the hand-computed example (`iters=0`, `iters=1`, `square`), 288 randomized differential cases over the full seed × iters × tau × square × eps grid, properties of the doubly stochastic approach (column error decreasing monotonically, exact row sums, row/column permutation equivariance, invariance to per-row constant shifts, the systematic bias eps introduces in the divisor), NaN and eps semantics on a dead column, degenerate shapes (`T=1`, `E=1`, all-zero logits, 16 rounds, noncontiguous input), invalid inputs, and the maximum shape.

The oracle is a pure Python list implementation: `math.fsum` per column, `math.fsum` per row, and scalar division that reproduces IEEE semantics itself (`0/0 -> NaN`, `positive/0 -> inf`). It borrows no torch `softmax`, `sum`, or broadcasting, so it cannot share the reference solution's reduction order.

Mutation testing: reversing the alternation, moving eps to the numerator, dropping the renormalization after squaring, hoisting the squaring before the loop, running one round short, hiding NaN behind `clamp_min`, and returning row sums instead of column sums — all seven mutants are caught by this suite.

## Scope and sources

The Sinkhorn–Knopp theorem (1952) states that a matrix with strictly positive entries and equal row and column counts converges to a doubly stochastic matrix under alternating row and column scaling. This problem uses that iteration as a MoE load-balancing operator and deliberately removes the theorem's assumptions to see how an implementation handles failure — `eps=0` with an all-zero column is exactly "no finite scaling can make the column sums 1", where the matrix has no solution at all.

Two same-shaped implementations exist in the transformers 5.x source used by this repository's judge environment:

- `models/superglue/modeling_superglue.py`, `log_sinkhorn_iterations`: a `for _ in range(num_iterations)` loop alternating row and column scaling updates, carried out in log space to avoid underflow, with a fixed round count and no tolerance-based early exit.
- `models/deepseek_v4/modeling_deepseek_v4.py`, the multi-stream residual (mHC) `comb` weight: `comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)` followed by `hc_sinkhorn_iters - 1` rounds of row and column normalization, constraining `comb` to be doubly stochastic so residual mixing cannot inflate norms. That is hyper-connections rather than MoE routing, but it confirms that "eps in the divisor, fixed round count, columns before rows" is standard practice in real models.

Expert-Choice routing (Shazeer 2022) satisfies the column constraint exactly with a hard "each expert keeps its top-$k_c$ tokens" selection; this collection already covers the hard version in `moe-expert-choice-routing`. This problem is the soft version, where the column constraint is only approached, so one has to answer "how close, and when does it collapse".

The `square` sharpening step is a rule invented for this problem: I could not obtain a reliable primary source tying it to a specific paper's implementation, so no attribution is claimed. The `iters` bound of 16, the `eps` default of 0, and computing `col_load` from the returned matrix are all judging conventions, not defaults of any framework.
