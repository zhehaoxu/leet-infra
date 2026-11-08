# Captum NoiseTunnel smoothing ledger: the repeat_interleave noise layout, the min-and-floor partition schedule of nt_samples_batch_size, the smoothgrad / smoothgrad_sq / vargrad aggregation semantics, the baseline draw sampling and repeat_interleave condition, and the torch.cat concatenation of delta

## 1. Background

Captum's NoiseTunnel is a smoothing wrapper placed around any attribution method: it perturbs the input with Gaussian noise, repeats the attribution several times, and aggregates the resulting attributions. This problem decomposes the whole pipeline into five independently checkable numerical ledgers, all implemented with plain Python floats, with the random noise and sampling indices always passed in explicitly so that no randomness source is involved.

### Official source locations

- `NoiseTunnel.attribute` and its four nested functions in `captum/attr/_core/noise_tunnel.py`
- `_validate_noise_tunnel_type` in `captum/attr/_utils/common.py`
- `_expand_and_update_baselines` in `captum/_utils/common.py`

## 2. Conventions

### Inputs and outputs

All tensors are represented as nested lists: `inputs` is `[bsz][d]` and a perturbed matrix is `[bsz*k][d]`. Each mode returns a dictionary whose keys mirror the official variable names. Every invalid input raises `ValueError` with a message that names the violated constraint.

### Explicit randomness

The official code samples with `torch.normal(0, stdev_expanded)` and `np.random.choice`. Here `noise` and `baseline_indices` are explicit arguments: the test cases supply them and the reference solution performs deterministic arithmetic only. All values are Python floats (float64).

## 3. Mode 1: noise_layout

### Noise layout

The official `add_noise_to_input` first repeats the input contiguously with `repeat_interleave`, then adds elementwise noise:

```text
repeat_interleave(x, 3) = [x0, x0, x0, x1, x1, x1]
repeat(x, 3)            = [x0, x1, x0, x1, x0, x1]
```

`source_index` is `i // k`; the wrong candidate `repeat_alt_source` is `i % bsz`. `expanded_size` is `[bsz*k, d]` and every element of `stdev_expanded` equals the scalar `stdev`.

```text
input.repeat_interleave(nt_samples_partition, dim=0) + noise
```

### stdevs type branches

The official `add_noise_to_inputs` accepts only `float`: an `int` trips `assert isinstance(stdevs, float)`, while the `tuple` branch first asserts `len(stdevs) == len(inputs)` but `stdevs_` is assigned only in the float branch, so a length-matching tuple raises `UnboundLocalError` (the pyre-fixme[61] comment in the source already admits this). This problem raises an explicit `ValueError` explaining the official bug for the length-matching tuple case instead of reproducing `UnboundLocalError`.

```text
stdevs_ = (stdevs,) * len(inputs)
```

## 4. Mode 2: batch_schedule

### Partition arithmetic

`eff = nt_samples if bs is None else min(nt_samples, bs)`, `partition` uses floor division rather than ceiling, and `remaining` runs as one extra smaller batch:

```text
eff = nt_samples if bs is None else min(nt_samples, bs)
partition = nt_samples // eff
remaining = nt_samples - partition * eff
batch_sizes = [eff] * partition + ([remaining] if remaining > 0 else [])
```

When `bs > nt_samples`, `min` clamps it to `nt_samples`, so `partition` is always 1 and never 0.

## 5. Mode 3: aggregate

### Three aggregation semantics

The three branches of the official `compute_smoothing`: `smoothgrad_sq` is `E[X^2]` and not `E[X]^2`, and `vargrad` is the population variance (divide by n, no Bessel correction):

```text
smoothgrad     = E[X]
smoothgrad_sq  = E[X^2]
vargrad        = E[X^2] - E[X]^2
```

### Cross-partition accumulation

`update_sum_attribution_and_sq` reshapes each partition's attributions with `view(bsz, k, d)`, accumulates `sum(dim=1)` and `sum(X**2, dim=1)` separately, and finally multiplies both by `1 / nt_samples` once:

```text
view(bsz, k, d).sum(dim=1)
expected_attributions    = sum_attributions    * 1 / nt_samples
expected_attributions_sq = sum_attributions_sq * 1 / nt_samples
```

The square is taken elementwise before summing, not after summing.

## 6. Mode 4: baseline_expand

### The draw branch

With `draw_baseline_from_distrib=True`, indices are drawn by `np.random.choice(num_ref_samples, n_samples * bsz)` with `replace=True` by default, so the count is `n_samples * bsz` and not `n_samples`:

```text
np.random.choice(num_ref_samples, n_samples * bsz)
```

### The default branch

The default branch applies `repeat_interleave(n_samples, dim=0)` only when `baseline.shape[0] == input.shape[0]` and `> 1`; a scalar or single-sample baseline is kept unchanged. With draw enabled, `_validate_input` only requires `input.shape[1:] == baseline.shape[1:]`, not equal batch sizes.

## 7. Mode 5: noise_tunnel

### Full orchestration

The official code calls `add_noise_to_inputs(k)` independently for each partition, so the overall `perturbed` is `concat_p(repeat_interleave(inputs, k_p))` and **not** a single `repeat_interleave(inputs, nt_samples)`. The noise offset must be accumulated separately per partition.

```text
perturbed = concat_p(repeat_interleave(inputs, k_p) + noise_p)
```

### Delta concatenation

With `return_convergence_delta=True`, the implementation is `torch.cat(delta_partial_list, dim=0)`, not the "arithmetic mean across all nt_samples" claimed by the docstring:

```text
delta = torch.cat(delta_partial_list, dim=0)
```

## 8. Common mistakes

1. Writing `repeat_interleave` as the alternating `repeat` order.
2. Passing `stdevs` as `int 1` instead of `1.0`.
3. Assuming a `stdevs` tuple works (an official bug).
4. Computing `smoothgrad_sq` as `E[X]^2`.
5. Using the `n-1` sample-variance correction for `vargrad`.
6. Computing `sum_sq` as `(sum)^2`.
7. Using `ceil` instead of floor division for `partition`.
8. Forgetting that `remaining` forms its own final batch.
9. Not clamping `bs > nt_samples` with `min`.
10. Dividing by the partition count instead of `nt_samples`.
11. Drawing `n_samples` indices instead of `n_samples * bsz`.
12. Expanding for every `num_ref == bsz` (dropping the `> 1` condition).
13. Making `delta` an arithmetic mean instead of a concatenation.
14. Building `perturbed` with a single `repeat_interleave(inputs, nt_samples)`.
15. Using the wrong noise offset from the second partition onward.
16. Not rejecting an unknown `nt_type`.
17. Not rejecting out-of-range `baseline_indices`.
18. Not rejecting a `delta_parts` count that differs from the partition count.
19. `repeat_alt` and `repeat_interleave` coincide naturally when `bsz=1` or `k=1`, so two-way assertions must pick data with `bsz>1` and `k>1`.
20. Asserting only that "an exception is raised" for domain constraints without checking the message, which lets a deleted check pass by having downstream validation raise instead.

## 9. Self-check

- With `stdevs=0`: `smoothgrad = x`, `smoothgrad_sq = x^2`, and `vargrad = 0`.
- The three semantics are pairwise different on data with non-zero variance.
- `batch_sizes` always sums to `nt_samples`.
- `delta` length is always `bsz * nt_samples`.
- Each partition's attributions are indexed in C order under `view(bsz, k, d)`: the `bi`-th sample occupies `part[bi*k:(bi+1)*k]`.
