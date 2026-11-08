# 851 Solution Notes: Captum NoiseTunnel Smoothing Ledger

## 1. Official source retrieval

### Three source files

Three files were pulled from the `main` branch of `pytorch/captum`: `noise_tunnel.py` (504 lines, containing `NoiseTunnel.attribute` and all nested functions), `common.py` (927 lines, containing `_expand_and_update_baselines` and `_validate_input`), and `attr_common.py` (386 lines, containing `_validate_noise_tunnel_type`). A TLS / upstream failure once overwrote a file with a single-line error message; a diagnostic re-fetch fixed it.

### Literal execution environment

The judging environment has no `captum` package, so `official_nt.py` was built: it injects minimal fake captum modules, supplies the annotation names and dependencies, and `exec`s the three official sources verbatim in the order `common.py -> attr_common.py -> noise_tunnel.py`. A `_FakeMethod` acts as the underlying attribution method, and the `__wrapped__` behavior of the official `log_usage` decorator is simulated. The probes therefore run the real official implementation, not a rewrite.

## 2. Ten official mechanism probes

### Probe conclusions

All ten sections of `probe851.py` ran green and pinned down: the three supported `nt_type` values and the `assert` validation; the contiguous `repeat_interleave` layout with elementwise independent Gaussian sampling; the `float` / `int` / `tuple` branches of `stdevs`; the separate `sum(dim=1)` and elementwise sum-of-squares accumulation after `view(bsz, k, d)`; the three aggregation semantics; the `min`-and-floor partition schedule; the `cat` concatenation of `delta`; the determinism of `stdevs=0`; the `replace=True` baseline sampling and default expansion condition; and the inheritance of `is_delta_supported` / `is_gradient_method`.

```text
SUPPORTED_NOISE_TUNNEL_TYPES = ['smoothgrad', 'smoothgrad_sq', 'vargrad']
```

## 3. Three-layer cross-check

### Official layer

`official_nt.py` executes the official `NoiseTunnel.attribute` verbatim; a spy replaces `_FakeMethod.attribute` to capture each partition's perturbed tensor (the real input of the underlying attribution method); the baseline branch calls `_expand_and_update_baselines` from the official namespace directly.

### Reference layer

The problem's reference `solution.py`; all five modes go through `solve(mode, **kw)`.

### Oracle layer

`verify851.py` rewrites the formulas independently in pure Python without importing any function from the reference: partition arithmetic, per-partition segmented accumulation, three aggregations, noise layout, both baseline branches, and delta concatenation are all implemented separately.

### Cross-check conclusion

The five modes produce 4843 check items. official-vs-reference uses a `1e-4` tolerance (float32 official versus float64 reference), oracle-vs-reference uses `1e-9`, and domain constraints plus official-quirk comparisons are asserted separately. Final result: ALL GREEN.

## 4. Official quirk list

### The stdevs tuple bug

`stdevs_` is assigned only in the `float` branch, so the `tuple` branch's `zip(inputs, stdevs_)` raises `UnboundLocalError` when the length matches; the pyre-fixme[61] comment in the source already admits this. The reference raises an explicit `ValueError` explaining the official bug.

### Per-partition noise layout

The main loop `for _ in range(partition)` calls `add_noise_to_inputs(k)` independently each time, and the `remaining` branch calls it once more. The overall perturbed tensor is therefore `concat_p(repeat_interleave(inputs, k_p))`, not a single `repeat_interleave(inputs, nt_samples)`. The first cross-check round exposed this difference on the `stdevs=0` case.

### Delta cat versus docstring

The docstring claims delta is the "arithmetic mean across all nt_samples", but the implementation is `torch.cat(delta_partial_list, dim=0)` with length `bsz * nt_samples`.

### Default baseline expansion condition

`repeat_interleave` applies only when `baseline.shape[0] == input.shape[0]` and `> 1`; with draw enabled, validation only requires the feature dimensions to match.

```text
delta = torch.cat(delta_partial_list, dim=0)
```

## 5. Implementation points of the five modes

### noise_layout

`_validate_stdevs` rejects `bool` first, accepts `float`, and raises keyword-bearing `ValueError` for `int` and `tuple`; `stdev_expanded` is a broadcast of the scalar; `repeat_alt` and `repeat_alt_source` are returned as wrong candidates for two-way assertions.

### batch_schedule

`eff` uses `min`; `partition` uses `//`; `remaining` is appended separately; `total` is `sum(batch_sizes)` and always equals `nt_samples`.

### aggregate

`_accumulate` segments by `batch_sizes`, each `[bsz*k][d]`, with the `bi`-th sample occupying `part[bi*k:(bi+1)*k]`; `sum` and `sum_sq` accumulate separately across partitions; both are divided by `nt_samples` once at the end; `vargrad` is `E[X^2] - E[X]^2`.

```text
expected_attributions    = sum_attributions    * 1 / nt_samples
expected_attributions_sq = sum_attributions_sq * 1 / nt_samples
```

### baseline_expand

The draw branch requires `baseline_indices` of length `n_samples * bsz`, validates each index range, and gathers baselines by index; the default branch chooses `repeat_interleave` or `unchanged` by `num_ref == bsz and bsz > 1`.

### noise_tunnel

It builds perturbed per partition from `batch_sizes` with an accumulated noise offset; `_accumulate` reuses the same schedule; with `return_convergence_delta` it validates the `delta_parts` count and each part's length, then `extend`s them into `cat` semantics.

```text
vargrad = E[X^2] - E[X]^2
```

## 6. Domain constraints and error messages

Every invalid input raises `ValueError` with a message naming the violated constraint: `float` / `official bug` / `must equal` for `stdevs`, `must be either` for unknown `nt_type`, `positive integer` for `nt_samples` and `bs`, `divisible` for row counts, `must be provided` / `6 elements` / `out of range` for indices, `must be provided` / `one entry per batch partition` for `delta_parts`, and `unknown mode` for an unrecognized mode. The test cases assert not only that an exception is raised but also its message content, so a deleted check cannot survive by letting downstream shape validation raise instead.

## 7. Test case design

### Independent oracle

Each check carries a pure-Python oracle rewritten from the official formulas and compares every returned field; data comes from a deterministic LCG and depends on no randomness source.

### Two-way assertions

Besides matching the oracle, the result must differ from a plausible wrong candidate: `repeat_interleave` versus alternating `repeat`, `sum_sq` versus `(sum)^2`, `smoothgrad_sq` versus `E[X]^2`, `vargrad` versus the Bessel-corrected sample variance, `delta` versus a broadcast mean, and per-partition perturbed versus a single overall layout. Note that the two layouts coincide naturally when `bsz=1` or `k=1`, so assertions must pick data with `bsz>1` and `k>1`.

### Domain constraint cases

Invalid inputs must still have legal shapes so that deleting the target check is not masked by downstream shape validation; the error message keyword is asserted as well.

## 8. Mutation coverage

`mut851.py` contains 76 mutants covering both repeat layouts, six `stdevs` branches, `nt_type` validation, nine arithmetic sites in `_batch_schedule`, seventeen accumulation and alias sites in `_accumulate`, the returned fields and orchestration of all five modes, and the `solve` dispatch. Three mutants reported ANCHOR-ERROR in the first round because their anchor text was not unique; after switching to context-bearing unique anchors, the rerun gave **76 CAUGHT / 0 SURVIVED / 0 ANCHOR-ERROR**.

## 9. Common pitfalls

- Test cases must use `check_*(solve)` and must not `from solution import solve`; the mutation runner must likewise use `getattr(T, name)(S.solve)`.
- Official is float32 while the reference is float64, so official-vs-reference tolerance must be `1e-4`.
- Asserting only the exception type for domain constraints is insufficient; the message keyword must be asserted or a deleted check survives.
- Two-way assertions themselves need suitable data: `bsz=1`, `k=1`, and zero-variance data all make the wrong candidate equal the correct result.
- The end-to-end `stdevs=0` case must build attributions with the per-partition layout, otherwise `smoothgrad` becomes a mixed-sample mean.

## 10. Self-check list

- Three-layer cross-check: 4843 items ALL GREEN.
- Test cases: 5/5 pass against the reference.
- Mutation: 76/76 CAUGHT.
- Bilingual validation: zero new errors / warnings.
- selftest: all ACCEPTED in both run and submit modes.
