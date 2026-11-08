# Captum Lime interpretable-sampling ledger: Bernoulli binary masks and feature_mask index mapping, the 2*kw^2 denominator of the exponential kernel, the DataLoader convention where batch_count always equals n_samples, the weighted-ridge surrogate and coefs[mask] backfill, plus the default n_samples=25 and float32 backfill quirks

## 1. Background

Captum `Lime` explains a single example through "binary sampling in interpretable space -> mapping back to input space -> forward scoring -> similarity weighting -> fitting a linear surrogate -> backfilling coefficients into the input shape". This problem splits that chain into five independently judgeable numerical ledgers, all implemented in pure Python with no dependency on `captum`, `torch`, `sklearn`, or `numpy`, and no global RNG: the Bernoulli draws `samples` and the optional kernel `weights` are always explicit inputs. Unlike order 849 (Kernel SHAP sampling), order 851 (NoiseTunnel smoothing), and order 853 (GradientShap expected gradients), this problem pins down Lime's **perturbation-based surrogate** conventions: binary interpretable features, feature_mask grouping, the exponential kernel, `perturbations_per_eval` batching, and coefficient backfill.

### Official source locations

```text
captum/attr/_core/lime.py                     LimeBase / Lime / default_* / construct_feature_mask
captum/_utils/common.py                       _run_forward / _flatten_tensor_or_tuple / _reduce_list
captum/attr/_utils/common.py                  _construct_default_feature_mask / _format_input_baseline
captum/attr/_utils/batching.py                _batch_example_iterator
captum/attr/_utils/attribution.py             PerturbationAttribution
captum/_utils/models/linear_model/model.py    SkLearnLasso
```

The repository owner has moved from `pytorch/captum` to `meta-pytorch/captum`; the official main source is 1295 lines.

### Verbatim execution environment

The judge environment has neither `captum` nor `sklearn`. The official layer of the three-way cross-check execs `common.py`, `attr_common.py`, `batching.py`, and `lime.py` byte-for-byte into a shared namespace, fakes only genuinely unavailable dependencies (typing / progress / log_usage / attribution base classes / models / approximation_methods), and injects a deterministic `torch.bernoulli` sequence plus an explicit weighted-ridge stand-in `RidgeModel`. The probes therefore exercise the real official implementation, not a rewrite.

## 2. Conventions

### Inputs and outputs

Every tensor is a 2D list `[1][d]` (only `bsz=1` is supported); tuple inputs are a list of tensors flagged with `tuple_inputs=True`. `samples` is an `n_samples`-row binary matrix with `F` int 0/1 entries per row; `weights`, when supplied, holds `n_samples` non-negative kernel weights, and when omitted is computed by the `kernel` formula. Each mode returns a dictionary whose keys mirror official variables or ledger fields. Every invalid input raises `ValueError` with a message naming the violated constraint; an unknown mode raises `unknown mode`.

### Numerical precision and the float32 quirk

The reference solution uses Python float (float64) throughout and solves the surrogate system exactly with `Fraction`. The official side has two float32 conventions: the kernel casts inputs with `.float()` and computes `distance**2` on float32 tensors before `math.exp`, differing from a pure-double recomputation by about `1e-9`; and `_convert_output_shape` accumulates `coefs[f].item() * (mask == f).float()` on a `torch.zeros_like(..., dtype=torch.float)` buffer, so the backfilled values differ from the double solution by about `1e-7`. The three-way cross-check applies matching tolerances at these two points; the reference itself is unaffected.

## 3. Mode 1: feature_mask

### Default mask and explicit mask

`construct_feature_mask(None, formatted_inputs)` calls `_construct_default_feature_mask`: it assigns each scalar an index that **accumulates across tensors** in order, with `F = sum(numel(t[0]))`. An explicit mask is first tuple-ized, then the global minimum `min_index` over all tensors is taken; if `min_index != 0` the whole mask is **shifted down by that minimum** (`shifted=True`), and finally `F = max(mask) + 1`. Note that `F` is determined by the maximum, so indices need not be contiguous.

```text
default : mask[t] = [idx .. idx+d-1], idx += d ; F = idx
explicit: if min != 0: mask -= min
          F = max(mask) + 1
          shifted = (min != 0), default = False
```

### Domain constraints

`inputs` must be non-empty, each tensor must have exactly one row, and all values must be finite. `feature_mask` indices must be non-negative ints, each row length must match the corresponding input, and the tensor count must match. `tuple_inputs` must agree with the actual tensor count.

## 4. Mode 2: interp_transform

### From a binary vector to input space

The official `default_perturb_func` returns a `long` binary vector of shape `(1, F)` via `torch.bernoulli(torch.ones(1, F) * 0.5)`. `default_from_interp_rep_transform` uses `curr_sample[0][feature_mask].bool()` to **index-map** that vector back to the input shape, then mixes element-wise:

```text
binary_mask = sample[feature_mask].bool()
perturbed   = binary_mask * original + (~binary_mask) * baselines
F           = num_interp_features
sample[i] in {0, 1} : 1 keeps original, 0 uses baseline
```

The tuple branch takes `feature_mask[j]` once per tensor; the same index used in different tensors groups those positions into one interpretable feature, toggled simultaneously.

### Domain constraints

`sample` length must equal `F` and its entries must be int 0/1 (bools and non-ints are rejected); the baselines tensor count and feature dimension must match `inputs`.

## 5. Mode 3: kernel

### Exponential kernel formula

The closure returned by `get_exp_kernel_similarity_function(distance_mode, kernel_width)` flattens original and perturbed inputs and casts to float32. `cosine` uses `1 - CosineSimilarity(dim=0)(a, b)` (denominator `max(||a||*||b||, 1e-8)`), `euclidean` uses `torch.norm(a - b)`; both then pass through the exponential kernel:

```text
distance(cosine)    = 1 - dot(a,b) / max(|a|*|b|, 1e-8)
distance(euclidean) = sqrt(sum((a[i]-b[i])^2))
weight              = exp( -(distance^2) / (2 * kernel_width^2) )
```

The denominator is `2 * kernel_width**2`, **not** `kernel_width**2`; defaults are `distance_mode="cosine"` and `kernel_width=1.0`.

### Domain constraints

`distance_mode` must be `cosine` or `euclidean`, `kernel_width` must be a positive finite number (bool is not a number), and `original` / `perturbed` must have equal flattened lengths.

## 6. Mode 4: surrogate

### Weighted ridge regression

The official default surrogate is `SkLearnLasso(alpha=0.01)`, which requires sklearn. Following the general discipline of removing upstream dependencies, this problem uses an exactly judgeable **weighted ridge regression**: for the `n x F` binary design matrix `X`, outputs `y`, and weights `w`,

```text
gram[a][b] = sum_i w[i] * X[i][a] * X[i][b]   (a,b = 0..F-1)
gram[a][a] += alpha
rhs[a]     = sum_i w[i] * X[i][a] * y[i]
coefs      = solve(gram, rhs)
```

With `alpha=0` this is weighted least squares; with `alpha>0` the coefficients shrink. A singular Gram raises `ValueError`.

### Domain constraints

`interp_samples` must be non-empty with equal-length rows of int 0/1; `outputs` and `weights` must have length equal to the sample count, be finite numbers, and weights must be non-negative; `alpha` must be non-negative.

## 7. Mode 5: lime

### End-to-end chain

Each sample goes through `interp_transform` to produce a perturbed input, then is scored by `forward_kind` (`linear`: `sum(w[j]*x[j])`; `quadratic`: `sum(w[j]*x[j]^2)`, summed over tensors for tuple inputs), then takes either explicit `weights[i]` or a freshly computed `kernel` weight; finally `samples / outputs / weights` are handed to `surrogate` for the coefficients. The result also exposes every intermediate quantity: `perturbed`, `binary_masks`, `gram`, `rhs`, and more.

### Batching and the DataLoader

Official `LimeBase.attribute` calls `_evaluate_batch` once every `perturbations_per_eval` accumulated samples, with a separate call for the shorter tail; `batch_count` is incremented by `+= 1` at every sample and **never reset**, so `DataLoader(dataset, batch_size=batch_count)` always has batch_size equal to `n_samples` and the surrogate sees all samples in one batch.

```text
batches       = [ppe] * (n // ppe) + ([n % ppe] if n % ppe else [])
batch_count   = n_samples            (never reset)
loader_batch_size = n_samples
dataset_shapes    = [[n, F], [n], [n]]
```

### Coefficient backfill

With `return_input_shape=True`, coefficients are spread back over the input shape by mask: `attr[t][j] = sum_f coefs[f] * (mask[t][j] == f)`, so positions sharing a mask value share a coefficient. With `False`, the flat length-`F` `coefs` vector is returned directly.

```text
attr[t][j] = sum(coefs[f] for f in range(F) if mask[t][j] == f)
return_input_shape=False -> attribution = coefs
```

### Official quirk list

`Lime.attribute` defaults to `n_samples=25`, yet the same docstring says `Default: 50`; only `LimeBase.attribute` defaults to 50. `perturb_interpretable_space=True` and `to_interp_rep_transform=None`. When `bsz>1` and the forward returns one value per example, `_attribute_kwargs` trains a separate surrogate per example and calls `_reduce_list`; when the forward returns a single scalar per batch it asserts `perturbations_per_eval == 1`. Both `has_convergence_delta()` and `multiplies_by_inputs` are `False`.

## 8. Common mistakes

- Treating `feature_mask` as a boolean mask instead of an **index array**: official code reads `sample[feature_mask]`, not an element-wise `sample & mask`.
- Forgetting the global shift when `min_index != 0`, or computing `F` as `len(unique(mask))` instead of `max + 1`.
- Writing the exponential kernel denominator as `kernel_width**2`, or using the cosine value itself as the distance.
- Reordering samples with a `repeat`-style layout, resetting `batch_count` after each batch, or setting the DataLoader batch_size to `perturbations_per_eval`.
- Dropping kernel weights when fitting the surrogate, adding `alpha` to every matrix entry instead of the diagonal, or replacing ridge regression with unregularized ordinary least squares.
- With `return_input_shape=True`, tiling coefficients in positional order instead of grouping them by mask index.
- Allowing bool or float entries in `samples`, negative `weights`, or sampling Bernoulli from a global random source.

## 9. Self-check

- All five modes are callable independently through `solve(mode, **kw)`; `unknown mode` raises `ValueError`.
- Eight judge cases `check_*(solve)` pass 8/8 on the reference; the three-way cross-check reports 374 checks ALL GREEN; mutation testing catches 156/156.
- The Chinese and English statements have matching heading counts and levels, matching fenced code block counts and contents; `meta.json` contains `title_en`; starter, solution, and test_cases all compile.
