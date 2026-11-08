# 853 Solution Notes: Captum GradientShap Expectation Ledger

## 1. Official source retrieval

### Three source files and dependency reuse

Three files were pulled from the `main` branch of `pytorch/captum`: `captum/attr/_core/gradient_shap.py` (470 lines / 22110 bytes), `captum/_utils/gradient.py`, and `captum/attr/_utils/attribution.py`; `noise_tunnel.py` / `common.py` / `attr_common.py` were reused directly from the staged sources of order 851. The original `attribution.py` URL timed out repeatedly, so a raw source with a query string yielded 24275 bytes / 536 lines, cross-checked against the web page.

### Literal execution environment

`official853.py` was built: it injects 14 fake Captum modules, sets `ModuleOrModuleList` to a `TypeVar` to satisfy `Generic[...]`, reproduces `log_usage.__wrapped__` with `functools.wraps`, and `exec`s verbatim in the order `common -> attr_common -> gradient -> attribution -> noise_tunnel -> gradient_shap` sharing one namespace. `SeqRNG` replaces `np.random.choice` / `np.random.uniform` / `torch.normal` and records `choice_calls` / `uniform_calls` / `normal_calls` and `order`. A smoke test confirmed 112 namespace symbols and all four official symbol families.

## 2. Twelve official mechanism probes

### Probe conclusions

All twelve sections of `probe853.py` ran green and pinned down: `GradientShap.attribute` hardcoding `nt_type="smoothgrad"` and `draw_baseline_from_distrib=True` without forwarding `nt_samples_batch_size`; the three forms of `_format_callable_baseline` and the assertion that `baselines[0]` must be a Tensor; the flat `k*bsz` indices of `_expand_and_update_baselines` and non-distrib `repeat_interleave(k)`; `_validate_input` checking only `shape[1:]`; the `(n,1,1,..)` broadcast and `requires_grad_()` of `_scale_input`; the RNG order choice -> normal -> uniform with the numpy / torch dual stream; the `repeat_interleave(k)+normal` layout; the float / tuple constraint on `stdevs`; the two `multiply_by_inputs` branches; the single-partition smoothgrad mean; and the delta convention.

```text
NS keys: 112
choice_calls: [(4, 3, True)]
uniform_calls: [(0.0, 1.0, 3)]
normal_calls: [(3, 3)]
order: [('choice', 3), ('normal', 9), ('uniform', 3)]
```

P7 / P11 were initially False; diagnosis showed the layout was exactly right and only about `1e-8` of numeric drift remained, caused by the official `torch.tensor(stdev)` defaulting to float32.

## 3. Three-layer cross-check

### Official layer

`official853.py::run_gradient_shap` executes the real official `GradientShap.attribute` with float64 tensors, driven by the explicit `SeqRNG` sequences.

### Reference layer

The problem's reference `solution.py`; all five modes go through `solve(mode, **kw)` in pure Python float64.

### Oracle layer

`verify853.py` rewrites exact arithmetic with `Fraction` without importing any function from the reference: baseline draws, noise injection, interpolation, analytic gradients, the multiplier, the smoothgrad mean, and delta are all implemented separately.

### Cross-check conclusion

Twelve cases cover single / tuple, linear / quadratic, the mul switch, the delta switch, and varying bsz / d / k / num_ref, for 340 check items, final result ALL GREEN.

```text
checks: 340   failures: 0
RESULT: ALL GREEN
```

## 4. Official quirk list

### Single partition and the dual RNG stream

`GradientShap.attribute` calls `NoiseTunnel.attribute.__wrapped__` without passing `nt_samples_batch_size`, so it is always `None` and a single partition. Choice and uniform share the numpy global stream while normal uses the independent torch stream; with tuple inputs, choice / normal happen once per tensor and uniform only once.

### float32 quantization error

The official `add_noise_to_input` calls `torch.tensor(stdev)` without a dtype, defaulting to float32. Probes P7 / P11 therefore showed about `1e-8` of drift. The fix is a temporary raise of the global default dtype in the driver, restored in a finally block, without modifying the official source itself.

```text
_prev_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.float64)
try:
    ...
finally:
    torch.set_default_dtype(_prev_dtype)
```

### Baseline index layout

`np.random.choice(num_ref, k*bsz, replace=True)` gives `k*bsz` flat indices; `_validate_input` in distrib mode checks only `shape[1:]`. The non-distrib `applicable` predicate requires both `num_ref == bsz` and `num_ref > 1`.

## 5. Implementation points of the five modes

### baseline_draw

Distrib takes `rows[i] for i in indices` with `drawn_shape = [k*bsz, d]`; non-distrib branches into `repeat_interleave` / `unchanged` according to `applicable` and returns the `non_distrib` detail.

### scale_input

`coeff_shape = [n] + [1]*(ndim-1)`, interpolation `alpha*x + (1-alpha)*b`, and a `requires_grad=True` marker.

### noise_layout

Noise is taken in `(i*k+j)*d + t` C order, `expected = bsz*k*d*n_in`, and `_stdevs_check` uniformly validates a float or an equal-length float tuple.

### schedule

`effective = k`, `partitions = 1`, `remaining = 0`, three groups of RNG call ledgers plus `rng_order`, and `compare_batch_size` giving the direct-NoiseTunnel partition comparison.

### gradient_shap

Slices `idx_t` / `noi_t` per tensor and chains draws -> noise -> interpolation -> gradient -> multiplier -> mean; delta accumulates attr row sums across tensors but subtracts the F difference only once.

## 6. Domain constraints and error messages

Every invalid input raises `ValueError` with a message naming the violated constraint: `n_samples must be positive` / `indices length` / `indices must be ints` / `baseline index out of range` / `baselines rows must match inputs rows` / `baselines feature dim must match inputs feature dim` / `alpha must come from np.random.uniform` / `ndim must be` / `k must be positive` / `n_in must be positive` / `noises length` / `noises must be finite numbers` / `stdevs must be type float` / `number of stdevs` / `each stdev` / `alphas must be in` / `multiply_by_inputs must be bool` / `all input tensors must share bsz and feature dim` / `tuple lengths must match` / `unknown mode`.

```text
noises must be finite numbers
baseline index out of range [0, 4)
```

## 7. Test case design

### Independent oracle

Among the 8 `check_*(solve)` cases, `check_gradient_shap` / `check_convergence_delta` use an independently rewritten `_exp_gs` (explicit per-sample loops plus polynomial expansion) and compare every returned field; `_case` generates deterministic sequences with no randomness source. The test cases must not `import solution`, and every `check_*` must accept a `solve` argument.

### Two-way assertions

Besides matching the oracle, the result must differ from plausible wrong candidates: the `repeat()` layout, an alpha upper bound of `1.0`, the grads-versus-attr relation under the mul switch, single versus multiple partitions, and reuse of the second noise / indices slice for tuples.

### Domain constraint cases

Invalid inputs must still have legal shapes so that deleting the target check is not masked by downstream shape validation; the message keyword is asserted as well. `num_ref == bsz == 1` pins the non-distrib `applicable`, and `bsz > 1 and k > 1` pins the C-order layout.

## 8. Mutation coverage

`mut853.py` contains 125 mutants covering the common helpers, baseline_draw, scale_input, noise_layout, schedule, gradient_shap, and the solve dispatch. The first round gave 110 CAUGHT / 14 SURVIVED / 1 ANCHOR-ERROR: the root cause was that the `_case` noise sequence `(i*5+seed)%5` is constant, making multi-tensor slices and the `(i*k+j)` layout mutants unobservable; changing it to `(i*3+seed)%7` restored per-element variation, 12 more domain / metadata assertions were added, and the `gs-delta-flag` anchor mismatch caused by an intervening comment line was fixed. The rerun was fully green.

```text
125 effective mutants: 125 CAUGHT / 0 SURVIVED / 0 ERROR
```

## 9. Common pitfalls

- The judging environment has no `captum`, so a literal official execution must inject 14 fake modules and handle `Generic[TypeVar]` and `log_usage.__wrapped__`.
- The official `torch.tensor(stdev)` defaults to float32; raise the global default dtype temporarily rather than editing the official source.
- In `SeqRNG`, choice and uniform share the numpy stream while normal is independent; the order must not be reversed.
- A constant noise sequence makes multi-tensor slice and layout mutants look truly equivalent; the test sequence must vary per element.
- Test cases use `check_*(solve)` and the mutation runner likewise uses `getattr(T, name)(S.solve)`.
- Asserting only the exception type for domain constraints is insufficient; the message keyword must be asserted.
- Anchors must be unique; a multi-line anchor with an intervening comment line will not match.
- Equivalent mutants should keep their record but be excluded from the statistics.

## 10. Self-check list

- Three-layer cross-check: 340 items ALL GREEN.
- Test cases: 8/8 pass against the reference.
- Mutation: 125 effective mutants all CAUGHT.
- Bilingual validation: zero new errors / warnings.
- selftest: all ACCEPTED in both run and submit modes.
