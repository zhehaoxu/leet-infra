# 854 Solution Notes: Captum Lime Interpretable-Sampling Ledger

## 1. Official Sourcing

### Source list and dependencies

Main source: `meta-pytorch/captum main::captum/attr/_core/lime.py` (1295 lines / 62667 bytes), with dependencies `common.py` (927 lines), `attr_common.py` (386 lines), `batching.py` (245 lines), `attribution.py` (536 lines), `_utils/models/linear_model/model.py`, `_utils/models/model.py`, and `progress.py`. `raw.githubusercontent.com/pytorch/...` now returns `HTTP=000` everywhere, so sources were fetched via `api.github.com/repos/meta-pytorch/captum/contents/<path>?ref=main` and base64-decoded. Note that `_utils/models/linear_model` is a **directory**, not a single file.

### Verbatim execution environment

```text
fake: captum._utils.typing / progress / log / attr._utils.attribution
      / _utils.models(.linear_model) / approximation_methods
exec: common.py -> attr_common.py -> batching.py -> lime.py   (shared NS)
inject: deterministic torch.bernoulli sequence
        explicit RidgeModel (fit / representation)
```

`log_usage` is reproduced with `functools.wraps` and an explicit `__wrapped__`, otherwise `super().attribute.__wrapped__` inside `_attribute_kwargs` breaks. `PerturbationAttribution` / `Attribution` are minimal stand-ins (storing `forward_func` and `get_name`); `SkLearnLasso` is a spy recording constructor arguments, used to verify the default `alpha=0.01`. Not a single line of official source is modified.

## 2. Twelve Official Mechanism Probes

### Probe conclusions

```text
P1  Lime.__init__: SkLearnLasso(alpha=0.01), cosine kw=1.0, Bernoulli 0.5
P2  Lime.attribute n_samples=25 (docstring says 50), ppe=1, return_input_shape=True
P3  construct_feature_mask: default per-scalar / global min shift / F = max+1
P4  default_perturb_func: bernoulli(ones(1,F)*0.5).long(), shape (1,F)
P5  default_from_interp_rep_transform: sample[0][feature_mask].bool()
P6  exp kernel: exp(-d^2/(2*kw^2)), cosine 1-cos / euclidean norm
P7  assembly: interp x / outputs / weights in sample order
P8  ppe scheduling: ceil(n/ppe) batches, loader batch_size = n_samples
P9  dataset tensors all float32, cat vs stack branches
P10 return_input_shape: coefs[f]*(mask==f) accumulation
P11 tuple inputs: cross-tensor same index shares coefficient
P12 bsz>1 per-example branch, no delta, multiplies_by_inputs False
```

Final result: `PROBE 854 DONE: ok=76 fail=0`.

### The float32 conventions

P6 initially disagreed on the cosine weight: the official code computes `distance**2 / (2*kw**2)` on a **float32 tensor** before handing it to `math.exp`, differing from a pure-double recomputation by about `3.3e-9`; reproducing the expectation with the same tensor arithmetic made it exact. P9 confirmed all three `TensorDataset` columns are float32; P7 compares kernel weights with a `1e-6` tolerance. With `alpha>0`, `attribution` also differs by about `2e-7` because `_convert_output_shape` accumulates in float32, so the cross-check uses `1e-6` for that field.

## 3. Three-Way Cross-Check

### Official layer

`official854.run_lime(cfg)` injects a deterministic Bernoulli sequence and optional explicit `weights` (replacing `similarity_func`), runs `Lime.attribute` verbatim, and captures the `(x, y, w)` seen by `RidgeModel.fit`, the DataLoader `batch_size`, the raw dtypes, and the final `out` / `coefs`.

### Reference layer

`solution_staged.solve("lime", **cfg)`: pure Python, Gram and rhs aggregated with `math.fsum`, the linear system solved exactly with `Fraction` Gaussian elimination, and every intermediate quantity returned as a ledger field.

### Oracle layer

`verify854.o_lime(cfg)` and `test_cases_staged._exp_lime(cfg)` are independent implementations: neither reuses any reference-solution helper. Each normalizes masks, builds perturbations, computes forward values and kernel weights, solves the system with `Fraction`, backfills coefficients, and derives batching and shape metadata directly from cfg.

### Cross-check conclusion

15 cases cover single/tuple inputs, default/shifted/gapped masks, linear/quadratic forwards, `alpha=0` and `alpha>0`, freshly computed cosine/euclidean kernels, explicit weights, the `return_input_shape` switch, and several `ppe` and `n_samples` values. Component comparisons also target `construct_feature_mask`, `default_from_interp_rep_transform`, and `get_exp_kernel_similarity_function`. Final result: 374 checks, 0 failures, `RESULT: ALL GREEN`.

## 4. Official Quirk List

### Defaults and the docstring quirk

`Lime.attribute` defaults to `n_samples=25` while its docstring says 50; only `LimeBase.attribute` defaults to 50. The default `SkLearnLasso(alpha=0.01)` requires sklearn, `perturb_interpretable_space=True`, `to_interp_rep_transform=None`, and `from_interp_rep_transform` needs both `feature_mask` and `baselines` kwargs, asserting if either is missing.

### DataLoader batch_count

`batch_count += 1` sits inside every sampling iteration and is **never cleared**, so `DataLoader(dataset, batch_size=batch_count)` always uses the total sample count `n_samples`, not `perturbations_per_eval`. `_evaluate_batch` is still called once per `ppe` samples plus one shorter tail call. `combined_outputs` / `combined_sim` choose `torch.cat` or `torch.stack` based on `len(outputs[0].shape) > 0`.

### float32 backfill

`_convert_output_shape` starts from `torch.zeros_like(single_inp, dtype=torch.float)` and accumulates `+= coefs[f].item() * (feature_mask == f).float()` per feature; all accumulation happens in float32, leaving about `1e-7` relative error against the double coefficient solution. For tuple inputs the loop runs once per tensor, and `_format_output` decides whether a tuple is returned.

## 5. Implementation Notes for the Five Modes

### feature_mask

Normalize `inputs` into a tensor list; the default mask numbers scalars with an accumulating `idx`, while an explicit mask is validated as non-negative ints with matching shape, shifted by the global `min`, and yields `F = max + 1`. Returns `mask / num_interp_features / shifted / default / min_index / shapes`.

### interp_transform

Take the normalized mask, validate `sample` length and int 0/1 entries, derive `b = [sample[m] for m in mask_row]` and `p[j] = row[j] if b[j] else baseline[j]`; the tuple branch records `binary` and `perturbed` per tensor.

### kernel

Flatten inputs, compute the distance per `distance_mode`; cosine guards zero vectors with `max(na*nb, 1e-8)`, euclidean uses the square root of summed squares; both use `exp(-d^2/(2*kw^2))`. Invalid mode / width / length raise `ValueError` with a named keyword.

### surrogate

```text
X = [[float(v) for v in row] for row in samples]
gram = [[sum(W[i]*X[i][a]*X[i][b] for i in range(n))
         for b in range(F)] for a in range(F)]
for a in range(F): gram[a][a] += alpha
rhs  = [sum(W[i]*X[i][a]*Y[i] for i in range(n)) for a in range(F)]
coefs = solve_with_Fraction(gram, rhs)
```

A singular matrix (for example two identical columns with `alpha=0`) raises `surrogate Gram matrix is singular`; judge cases and cross-check cases avoid degenerate designs.

### lime

Chains `feature_mask -> interp_transform -> _forward_value -> kernel/weights -> surrogate` while recording 21 fields including `perturbed`, `binary_masks`, `batches`, `dataset_shapes`, `gram`, and `rhs`; `return_input_shape` selects backfill versus raw `coefs`.

## 6. Domain Constraints and Error Messages

`n_samples` / `perturbations_per_eval` must be positive ints (bool rejected); `return_input_shape` must be bool; `samples` must have `n_samples` rows of length `F` with int 0/1 entries; explicit `weights` must have matching length and be non-negative; `distance_mode` and `kernel_width` are checked; `forward_weights` must be non-empty and cover every feature; `alpha` must be non-negative; baselines shape and feature_mask indices/shape are validated; an unknown mode raises `unknown mode`. Judge cases assert a **specific message keyword** for every rule, so a downstream check with a similar message cannot mask the mutation.

## 7. Judge Case Design

Eight `check_*(solve)` functions: `feature_mask` / `interp_transform` / `kernel` / `surrogate` / `lime` / `domain` / `metadata` / `unknown_mode`. `lime` uses 13 cases for full-field comparison, while `metadata` pins down mutation-prone fields such as `batches`, `batch_count`, `loader_batch_size`, `dataset_shapes`, `default_mask`, and `shifted`. Every explicit sequence varies element by element, so constant sequences cannot disguise layout mutations as true equivalents.

## 8. Mutation Coverage

158 mutants cover the common helpers, `feature_mask`, `interp_transform`, `kernel`, `surrogate`, `_forward_value`, the `lime` return fields, and `solve` dispatch.

```text
round 1: 130 CAUGHT / 24 SURVIVED / 4 ANCHOR-ERROR
fix    : 12 weaker domain assertions + 4 unique anchors + 2 true equivalents
final  : 156 CAUGHT / 0 SURVIVED / 0 ERROR  (ALL MUTANTS CAUGHT)
```

The two true equivalents: `it-mt-count` (the mask tensor count is already guaranteed equal to the input tensor count by `feature_mask`) and `lm-samples-01` (`interp_transform` performs the identical int 0/1 check with the identical message downstream).

## 9. Common Pitfalls

- Treating `feature_mask` as a boolean mask, or computing `F` as the number of distinct indices.
- Dropping the factor 2 in the exponential kernel denominator, or using `cos` itself as the distance.
- Reordering samples with a `repeat` layout, or clearing `batch_count` after every batch.
- Adding `alpha` to the whole Gram matrix, dropping kernel weights, or using an inexact floating-point solve.
- Backfilling coefficients in positional order instead of by mask group; returning input shape when `return_input_shape=False`.
- Using constant weight/noise sequences in judge cases, which makes layout and slicing mutations unobservable.
- Asserting only a broad keyword (such as `n_samples`) for a domain rule, letting a downstream message like `samples length must equal n_samples` slip through.

## 10. Self-Check List

- Official verbatim environment and 12 probes all green (76/76).
- Three-way cross-check 374 checks ALL GREEN; driver 8/8; mutation 156/156 CAUGHT.
- Chinese and English description / notes have matching heading counts, levels, fenced code block counts, and contents; English files contain zero CJK.
- `meta.json` has all ten fields with a CJK-free `title_en`; starter, solution, and test_cases pass local `compile()`.
- 11 landed files, selftest ACCEPTED in both run and submit modes, zero new bilingual warnings library-wide, registration and unique order, short guard and full guard all green.
