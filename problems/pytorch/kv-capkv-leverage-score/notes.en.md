# Solution Notes · CapKV's Leverage Score

## In one sentence

A two-stage score: first turn the query into an anchor with the **averaged future RoPE** and read off a per-token query-relevance weight; then compute a leverage score against a **capacity matrix** in value space and multiply the two.

## What the leverage score measures

`A = I + Σ (w_j + eps) v_j v_j^T` sums the weighted outer products of all values into a "capacity" matrix. `ℓ_j = v_j^T A^{-1} v_j` measures **how much energy `v_j` still has after the span of the other values is removed**: collinear values get a small `ℓ_j`, a value that owns a direction of its own gets close to `|v_j|^2`. So `w_j * ℓ_j` is "query relevance × non-redundancy".

Note that `A` is built from `√(w+eps)·v` while the quadratic form uses the **raw** `v`: upstream treats `values_rep` as the output-direction proxy (`capacity_vectors = values_rep`), so the scaling enters `A` only. The asymmetry is by design, not a typo.

## The pipeline (matching upstream `cap_press.py`)

1. `n_sink/n_future_positions/epsilon/tau` are `4 / 512 / 1e-6 / 5.0` (L62-67); the query side drops the sinks too (L84).
2. Average rotation: build `cos·I + sin·rot` for 512 future positions and `mean(dim=0)` (L136-137).
3. `matmul(query_states, rotation.T)` (L162), then `mean` over the query axis to get the anchor (L175) — equivalent to rotating each token and averaging.
4. `F.normalize` both the anchor and the keys (default eps=1e-12), `einsum` for the cosine, `clamp(-1, 1)` (L200-204).
5. `logits = tau * statistic`, subtract `amax`, then `exp` (L219-221).
6. Scale the values by `sqrt(weights + epsilon)` (L270) and accumulate `A` with `einsum` (L279-283).
7. Cast `A` to float32, `linalg.solve`, `leverage = (v^T · A^{-1}v^T)` summed over the column axis, `scores = weights * leverage` (L285-291).
8. `view(..., num_kv_groups, ...).mean(dim=2)` back to kv heads (L294-295); `F.pad(..., n_sink, value=scores.max().item() + 1)` (L300).

## Three deviations from upstream (pinned by the statement)

1. **Where epsilon goes**: upstream adds it inside `sqrt(weights + epsilon)` (L270), not to the diagonal of `A`. `A = I + a PSD term` is already definite, so no jitter is needed; the tests require a `v=0` column to give `ℓ = 0`, and a diagonal jitter would turn it into `1/eps`.
2. **Internal precision**: upstream casts `A` and the values to `.to(torch.float32)` before solving (L285/287). Following the repo rules, this problem computes in float64 and casts back to the input dtype; float32 cases get a 3e-5 tolerance.
3. **Input shape**: upstream derives q/k/v from `hidden_states` and builds the 512-row RoPE table internally; here q and the `(cos, sin)` tables are explicit inputs (any table length, only the mean is used), `keep` is explicit instead of being derived from `compression_ratio`, and ties go to the smaller index (`topk` leaves them undefined upstream).

## Division of labour with `kv-expected-attention`

Both problems share a skeleton: drop `n_sink=4`, average the RoPE of 512 future positions, fill the sink with the **global max + 1**. What differs is what the score means:

- **EA** stands on the query side: it evaluates `E[e^{q^Tk/√d}]` under a Gaussian query model, softmaxes, and multiplies by `||v||` — "how much future attention will this key get, and how large is its value";
- **CapKV** stands on the value side: the query only produces the weight `w_j`, and the main term is the leverage of `v_j` under the capacity matrix — "how much of this value's information cannot be explained by the others".

So EA scores element-wise, while CapKV needs one `D×D` solve per head and then averages over GQA groups. The EA problem asks you to derive `mu/cov` yourself; this problem's anchor only needs mean-then-rotate and has no covariance term.

## What is judged, and the edges

- `statistic` stays within `[-1, 1]`; the max of `weights` in every head is exactly 1 (a consequence of the max shift).
- `k_j = 0` or a zero anchor must give a cosine of 0 (normalize eps=1e-12), never a NaN.
- `v_j = 0` gives `ℓ_j = 0` and a zero score; with all-`v` zero every real score is 0 and `kept` is exactly `0..keep-1` (sinks plus the smallest indices).
- The sink score is the **cross-head** global max + 1; one test deliberately makes head 1 score all zeros.
- Every `kept` row is ascending, duplicate-free, and starts with `0..3`.
