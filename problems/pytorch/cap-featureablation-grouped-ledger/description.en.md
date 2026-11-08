# Captum FeatureAblation grouped-ablation ledger: the dual max+1 and max-min+1 conventions, perturbations_per_eval chunking and tail re-expansion, initial/ablated eval difference accumulation, use_weights division, and the full attribute pipeline

## 1. Background

This problem reconstructs the **numerical ledger of grouped ablation** in `captum.attr.FeatureAblation`, not a call into the library. The judge environment has no `captum` and forbids depending on `torch`; every mode must be a deterministic pure-Python ledger over nested lists, returning the composite dictionaries specified here. The core of FeatureAblation is not "slide a contiguous window": each scalar position carries an integer `feature_mask` id, all positions with the same id form one group, and one forward pass replaces that whole group with the baseline at once. The contribution of the group is `initial_eval - modified_eval`. The batching parameter `perturbations_per_eval` (abbreviated `ppe`) packs several groups into one batch.

### Official source location

The upstream target is `meta-pytorch/captum main::captum/attr/_core/feature_ablation.py` (1075 lines / 47472 bytes; the owner has moved to `meta-pytorch`). We also reuse `captum/_utils/common.py` and `captum/attr/_utils/common.py` from the same repository (fetched for 854), which supply official helpers such as `_format_feature_mask`, `_format_input_baseline`, `_expand_target` and `_expand_additional_forward_args`. This problem targets only the methods relevant to the numerical ledger: `_get_feature_range_and_mask`, `_get_feature_counts`, `_attribute_progress_setup`, `_construct_ablated_input`, `_ith_input_ablation_generator`, `_process_initial_eval`, `_process_ablated_out`, `_accumulate_for_single_input`, `_generate_result`, and the orchestration of `attribute`.

### Verbatim execution environment

The three official `.py` files are `exec`'d as whole files into one shared namespace without changing a single line of upstream source; only genuinely unavailable dependencies are faked: typing aliases, the `log_usage` decorator, the `progress` bar, the three ABCs `Attribution` / `PerturbationAttribution` / `GradientAttribution`, `SkLearnLasso` and `FeatureAblationFutureError`. `torch.futures.Future` / `collect_all` are real. The driver converts nested lists into `float64` tensors and `long` masks and uses a deterministic `SpyForward` (linear / quadratic / multi-output / aggregate / Python scalar) that records the input shapes and values of every forward call.

## 2. Conventions

### Inputs and outputs

The single entry point is `solve(mode, **kw)`. `inputs` is always a list whose elements are individual input tensors, with dimension 0 being `num_examples`. Required parameters and defaults for the five modes:

```text
solve("mask_range",  input, feature_mask=None, offset=0,
                     perturbations_per_eval=1)
solve("ablate",      expanded_input, input_mask, baseline=0,
                     start_feature, end_feature)
solve("schedule",    inputs, feature_mask=None, baselines=None,
                     index=0, perturbations_per_eval=1,
                     target=None, additional_forward_args=None,
                     extra_args={})
solve("eval_out",    initial_eval, input_shapes, modified_evals,
                     masks, batch_sizes, num_examples,
                     perturbations_per_eval=1, index=0,
                     use_weights=False, initial_dtype="float64")
solve("attribution", inputs, forward_weights, feature_mask=None,
                     baselines=None, tuple_inputs=False,
                     perturbations_per_eval=1, forward_kind="linear",
                     target=None, n_classes=2, use_weights=False,
                     show_progress=False, additional_forward_args=None,
                     extra_args={})
```

### Numerical precision

Everything uses Python `int` / `float` (IEEE double). Nested lists are flattened and refolded in C order, and broadcasting follows NumPy rules: align from the trailing dimension, where each pair of dimensions must be equal or one of them must be 1. No global RNG; all perturbations, baselines, masks and forward weights are explicit arguments. An unknown mode raises `ValueError` whose message contains `unknown mode`; every invalid input raises `ValueError` naming the violated constraint.

## 3. mode 1: mask_range

### The dual conventions

The official code uses **two distinct counts** that must never be conflated:

```text
mn            = min(flatten(feature_mask))
mx            = max(flatten(feature_mask))
num_features  = mx + 1                  # loop bound, denominator of ppe clamp
feature_count = mx - mn + 1            # _get_feature_counts, used by progress
loop_start    = mn                     # start at the smallest id, not 0
num_steps     = num_features - mn
ppe_clamped   = min(ppe, num_features) # uses max+1, not max-min+1
chunks        = [[mn, mn+ppe_clamped], ...] until num_features
progress_total= ceil(feature_count / ppe) + 1
```

When `feature_mask=None`, the official code generates a default mask per input with shape `[1] + input.shape[1:]` and consecutive integers starting at `offset`; with multiple inputs the `offset` accumulates globally (the second input's ids continue after the first's). An explicit mask must have the same rank as the input, dimension 0 equal to 1 or `num_examples`, every remaining dimension equal to 1 or the corresponding input dimension, and non-negative integer values. A mask whose trailing dimension is 1 is legal (broadcasting).

### Domain constraints

`input` needs at least 2 dimensions (a leading example dimension); `offset >= 0`; `ppe` must be an integer `>= 1`, with `bool` accepted as 1 per the official `isinstance(True, int)` quirk. Wrong rank raises `feature mask must have the same rank as the input`, a wrong first dimension raises `feature mask first dim must be 1 or num_examples`, a wrong remaining dimension raises `feature mask shape is not broadcastable to input`, and invalid values (float / bool / negative) raise `feature mask values must be non-negative integers`.

## 4. mode 2: ablate

### Stacked masks and blending

This mirrors the official `_construct_ablated_input(expanded_input, input_mask, baseline, start_feature, end_feature)`. `expanded_input` has shape `(K, num_examples, ...)`, where `K = end_feature - start_feature` is the number of groups ablated simultaneously in this step. For each `j in range(start, end)` a 0/1 mask is produced:

```text
mask[j - start] = (input_mask == j) ? 1 : 0
keep            = 1 - mask
ablated        = expanded_input * keep + baseline * mask
```

With `start_feature > 0`, if `input_mask` contains no matching id, that mask is all zeros and the input is preserved verbatim. `baseline` may be a scalar or a tensor broadcast-compatible with the mask (for example a row-dependent baseline). The return contains `mask` / `mask_shape` / `ablated` / `ablated_shape` / `num_ablated_features` / `ones_per_feature` (number of 1s in each mask) / `baseline_shape`.

### Broadcasting and domain constraints

The stacked shape `[K] + input_mask.shape` must be broadcastable to `expanded_input.shape`, and the broadcast result **must not change** the shape of `expanded_input`; when it does, raise `input_mask broadcast changes the expanded_input shape`. `input_mask` must contain integers; `start_feature` / `end_feature` must be integers with `0 <= start < end`.

## 5. mode 3: schedule

### Chunking, repetition and the tail

This mirrors `_ith_input_ablation_generator`. Let `N = num_examples` and `ppe_eff = min(ppe, num_features)` (again using `max+1`). When `ppe_eff > 1`, **all** inputs are repeated row-wise `ppe_eff` times (`[row for _ in range(k) for row in x]`, i.e. the whole input is repeated, not element-wise repeat), and `target` plus tensor-valued `additional_forward_args` are expanded in step; scalar extra arguments stay unchanged. The loop starts at `mn`, each chunk taking `K = min(ppe_eff, num_features - pos)`:

```text
repeated_batch = [repeat_rows(x, ppe_eff) for x in inputs]   # when ppe_eff>1
cur[i]         = reshape(cur[i], [K, N] + feat_shape)
cur_mask       = [ (mask_i == j) for j in range(pos, pos+K) ]
cur[i]         = cur[i] * (1-cur_mask) + baseline * cur_mask
cur[i]         = reshape(cur[i], [K*N] + feat_shape)
batch_size     = K * N
```

★ **Tail-chunk quirk**: when `K != ppe_eff`, the official code does not simply truncate the repeated tensors. It re-expands from the **original** `inputs` / `target` / `additional_forward_args` by `K`, then slices the first `K*N` rows of **every** input (not only the ablated one). Hence the tail chunk's tensor extra argument has length `K`, not `ppe_eff`, and the tails of the other inputs are also removed. Full chunks reuse the already-expanded versions.

### Argument expansion and kwargs

A list `target` is expanded as `list(target) * k`; a scalar stays as-is. List (tensor) elements of `additional_forward_args` are repeated row-wise `k` times while scalar elements are untouched. A **tuple** value in `extra_args` is indexed by `index`, while a non-tuple value passes through; the returned `extra_args_selected` records that choice. A tensor baseline is reshaped to `[1] + baseline.shape`, while a scalar baseline has `baseline_reshape == []`. Progress is `progress_total = sum(ceil(_mask_count(m) / ppe) for m in masks) + 1`, where `_mask_count = max-min+1`.

### Domain constraints

`inputs` must be non-empty; the counts of `feature_mask` / `baselines` must equal the number of `inputs`; a baseline's first dimension must be 1 or `num_examples`; `additional_forward_args` must be a list; `index` must be in range; `ppe` validation is the same as mode 1.

## 6. mode 4: eval_out

### Initial state

This mirrors `_process_initial_eval`: a Python `int` gives `attrib_type == "int64"`, while a Python `float` or a tensor gives `"float64"`; `n_outputs = numel(initial_shape)`, which is 1 for a scalar; `flattened_initial_eval` always has shape `[1, n_outputs]`. Accumulator `total[i]` has shape `[n_outputs] + input_shapes[i][1:]` and starts at 0; with `use_weights=True` an equally shaped `weights` list is allocated, otherwise it is the empty list.

### Difference accumulation and weight division

This mirrors `_process_ablated_out` / `_generate_result`. For each record:

```text
diff      = initial_eval_flat[None, :] - reshape(modified_eval, (-1, n_outputs))
diff      = reshape(diff, shape(diff) + [1]*(input_rank-1))   # broadcast to feature dims
total[i] += sum_axis0(diff * mask)
weights[i] += sum_axis0(float(mask))                 # only with use_weights
attrib    = total / weights                          # only with use_weights, 0 -> nan
```

When `ppe > 1`, the first record must satisfy `modified_eval.shape[0] == (batch_size / num_examples) * initial_shape[0]`, otherwise raise `output should be a tensor whose 1st dim grow` (the official assertion that aggregated outputs grow with the batch). The flattened length of `modified_eval` must be divisible by `n_outputs`, otherwise raise `modified eval cannot reshape to (-1, n_outputs)`. The return contains `attributions` / `weights` / `flattened_initial_eval` / `attrib_type` / `n_outputs` / `total_attrib_shapes` / `eval_diffs` / `initial_shape`.

### Domain constraints

`input_shapes` must be non-empty, `index` in range; the three lists `modified_evals` / `masks` / `batch_sizes` must have equal lengths; `initial_eval` cannot be a `bool`. With no records, accumulator shapes and the initial shape are returned unchanged and attributions are all 0.

## 7. mode 5: attribution

### The six forward_kind values

A deterministic forward with explicit weights replaces a real model. `linear` computes `sum(w_j * x_j)`, `quadratic` computes `sum(w_j * x_j^2)`, `multi` emits `n_classes` values `sum(w[c][j] * x[j])` per batch row and then selects by `target`, `aggregate` sums the whole batch into a Python float, and `python_int` / `python_float` do the same with a forced cast to exercise dtype propagation. In attribution mode `additional_forward_args` may contain only scalar numbers, accumulated into a bias.

### End-to-end orchestration

One forward on the original inputs produces `initial_eval`; then, input by input and chunk by chunk, perturbed forwards are run and their differences accumulated; finally `use_weights` decides whether to divide by the weights. The return contains `attributions` (only element 0 when `tuple_inputs=False`) / `weights` / `n_outputs` / `attrib_type` / `flattened_initial_eval` / `initial_shape` / `num_forwards` / `forward_shapes` / `chunks` (with `input_index` / `start` / `end` / `batch_size`) / `progress_total` (only when `show_progress=True`, otherwise `None`). ★ The official code **skips entire inputs** whose `numel == 0` (empty tensors): their attributions are empty and they do not count as forwards. With multiple inputs the default masks use globally consecutive ids, so the second input's chunks start at the accumulated id.

### Domain constraints

Single-input mode requires exactly one element in `inputs`; multiple inputs need `tuple_inputs=True`. An unknown `forward_kind` raises `unknown forward_kind`. A `multi` forward needs an int `target` or a list as long as the batch, otherwise `Target list length does not match output!` / `multi forward needs an integer or list target`.

## 8. Official quirk checklist

1. `num_features = max+1` (loop bound / ppe clamp) and `feature_count = max-min+1` (progress) are two separate conventions; they differ when `min>0`.
2. `isinstance(True, int)` is true, so `ppe=True` is accepted by the official code as 1.
3. A tail chunk (shorter than `ppe_eff`) re-expands from the original inputs and slices `K*N` rows of **all** inputs, not only the ablated one.
4. A tuple in `extra_args` is indexed by `index`; a non-tuple passes through unchanged.
5. With `ppe>1`, an aggregated scalar forward triggers an assertion requiring output dim 0 to grow with the batch.
6. With `use_weights`, weights are the floating accumulation of masks and division by zero yields `nan`.
7. An input with `numel == 0` is skipped in its entirety.
8. Python `int` / `float` forward dtypes propagate into `attrib_type` (`int64` / `float64`).

## 9. Common mistakes

- Writing `ppe_clamped` as `min(ppe, max-min+1)` fails for masks with `min>0`.
- Restarting default-mask ids at 0 for the second input instead of accumulating globally.
- Implementing `repeat_rows` as element-wise repeat, which corrupts the within-batch row order.
- Reusing the `ppe_eff`-expanded inputs for the tail chunk, or expanding tensor extra arguments by `ppe_eff` instead of `K`.
- Multiplying `diff` by the mask without appending the `[1]*(input_rank-1)` broadcast dimensions.
- Accumulating perturbation rows into `sum_axis0` (the sum belongs inside a chunk; the result is then added to total).
- Returning a non-`None` `progress_total` when `show_progress=False`.
- Forgetting to skip empty inputs, or flattening the `tuple_inputs` result structure.
- Using local rather than global ids for `chunks` `start/end` with multiple inputs.

## 10. Self-check

The judge cases cover the formulas of all five chains, the dual conventions, tail chunks, broadcasting, weight division, dtype, progress and quirks; the reference solution must pass 12 official probe segments (87 assertions), a three-layer comparison (official / reference / independent oracle) and full mutation testing. Any unknown mode must raise `ValueError` containing `unknown mode`.
