# 856 solution notes: Captum FeatureAblation grouped-ablation ledger

## 1. Official source retrieval

### Source list and dependencies

The main source is `meta-pytorch/captum main::captum/attr/_core/feature_ablation.py` (1075 lines / 47472 bytes). Captum's owner has moved to `meta-pytorch`; raw URLs under `pytorch/captum` return 404, so we fetch base64 content via `api.github.com/repos/meta-pytorch/captum/contents/...?ref=main`. We also reuse `captum/_utils/common.py` and `captum/attr/_utils/common.py` fetched for 854. The original candidates Integrated Gradients and DeepLift were dropped because the bank already has `int-captum-integrated-gradients-ledger` and `cap-deeplift-nonlinear-rule-ledger`.

### Verbatim execution environment

The three official files are `exec`'d as whole files into a shared `NS`; only typing aliases, `log_usage`, `progress`, the three ABCs, `SkLearnLasso` and `FeatureAblationFutureError` are faked. `torch.futures.Future` / `collect_all` are genuinely available. After each exec, `NS` is synchronized into module objects such as `captum._utils.common` to resolve the circular `from ... import ...` references between official modules.

```text
for name, folder in (("common.py", D854), ("attr_common.py", D854),
                     ("feature_ablation.py", D856)):
    exec(compile((folder / name).read_text(), name, "exec"), NS)
    for mod in (common, attr_common, batching):
        for k, v in NS.items():
            if not k.startswith("__"):
                setattr(mod, k, v)
```

The driver uses `SpyForward` to record `shapes` / `values` / `extras` for every forward call and exposes eight comparison entry points: `comp_mask_range` / `comp_construct` / `comp_generator` / `comp_range` / `comp_progress` / `comp_process_initial` / `comp_process_ablated` / `run_attribute`, all converted to nested lists. Not a single line of upstream source is changed; all comparisons run in `float64`.

## 2. Twelve official mechanism probes

### Probe conclusions

`probe856.py` has 12 segments with 87 assertions, all green: P1 default masks (single input / 3D / tuple global offsets); P2 the `max+1` vs `max-min+1` split when `min>0` and gap masks; P3 the 0/1 stacking of `_construct_ablated_input`, scalar/tensor baselines, and the all-zero mask for an out-of-range start; P4 `ppe=1` modifying only the indexed input; P5 `ppe=3` repeating all inputs, row-wise cat of tensor extra arguments, and tail target re-expansion by K; P6 empty masks for holes with `min>0` and a large ppe, plus the `K*N` batch; P7 tuple kwargs indexed by `index` and the two expand helpers; P8 `_process_initial_eval` dtype / `n_outputs` / flattened eval / accumulator shapes / `use_weights` allocation; P9 `_process_ablated_out` differences, mask broadcasting, sum over dim 0 and weight division; P10 ppe validation, the `bool` quirk, the aggregated-output assertion, and Python int output; P11 forward counts, expanded batch shapes, progress total/updates/closed; P12 the five forward kinds, target selection, baselines, use_weights, and the tail chunk of 5 features with ppe=3.

★ One assertion failed on the first run: P12 expected `[[-2., -2.], [-2., -2.]]` for a scalar baseline, while the official result is `[[-1., 0.], [1., 2.]]` (baseline=2 replaces only ablated positions; non-ablated originals are preserved). Fixing the expectation made all 12 segments green.

## 3. Three-layer comparison

### official layer

The verbatim official class covers 5 mask_range cases, 6 ablate cases, 10 schedule cases, 7 eval_out cases and 20+ attribution cases, including 3D inputs, tuple inputs, `min>0` masks, tensor baselines, ppe=2/3/5, every forward_kind, use_weights, show_progress and empty inputs.

### reference layer

The five modes of `solution_staged.py`, pure Python with no torch. `shape_of` / `flatten` / `unflatten` / `broadcast_to` / `bop` / `sum_axis0` handle flattening, broadcasting and element-wise operations over arbitrary ranks; `_mask_eq` / `_expand_add` / `_expand_target` / `_mask_count` reproduce the semantics of official helpers.

### oracle layer

An independent implementation: mask ranges use naive `min/max` and an explicit chunk loop; ablate uses per-position `if mask[p] == j: baseline else expanded`; schedule is a naive generator that constructs one complete input per feature id; eval_out accumulates explicit `(init[j] - mod[j]) * mask[p]` terms; attribution runs one full forward per feature group by definition, deliberately different in structure from the official batched implementation.

### Comparison conclusion

```text
VERIFY 856 DONE checks=5291 fails=0 ALL GREEN
```

The three layers are structurally independent, avoiding same-source errors. First-round failures came from mistaking the generator's yield order for the internal batch order of `attribute`, and from an inverted row order in the independent oracle; after fixing those, all 5291 checks are green.

## 4. Official quirk checklist

### Dual conventions and bool ppe

`_get_feature_range_and_mask` returns `num_features = max+1` and `min_feature = min`; `_get_feature_counts` returns `max-min+1`. `ppe_eff = min(ppe, max+1)`, but progress total uses `ceil((max-min+1)/ppe)+1`. They differ when `min>0`, which is the easiest place to get wrong. `assert isinstance(ppe, int)` lets `True` pass validation and behave as 1.

### Tail re-expansion

When `K != ppe_eff`, `_ith_input_ablation_generator` re-expands by K from the **non-repeated** original `inputs` / `target` / `additional_forward_args`, then slices `[0:K*N]` of every input. Therefore the tail chunk's tensor extra argument has length K and other inputs are also truncated to `K*N` rows; full chunks reuse the already-expanded versions.

### Empty inputs and scalar forwards

`attribute` uses `continue` for any input with `numel(inputs[i]) == 0`; its attributions are empty and it contributes no forward calls. Python `int` / `float` forwards yield `attrib_type` `int64` / `float64` through `_process_initial_eval`; aggregate-style scalar outputs trigger the official assertion with `ppe>1` because dim 0 does not grow with the batch.

## 5. Implementation points of the five modes

### mask_range

`_ppe` centralizes validation (bools become `int(p)`); `_validate_mask` checks rank / first dimension / remaining dimensions / value types. The default mask uses consecutive integers from `offset` with shape `[1] + ishape[1:]`. Thirteen fields are returned, `chunks` is a list of `[start, end)` pairs, and `progress_total = ceil(feature_count/ppe)+1`.

### ablate

Validate `expanded_input` rank and `start/end` first, then broadcast `[K] + mshape` against `eshape`; `merged != eshape` raises the dedicated message. `current_mask = [_mask_eq(mask, j) ...]`, `keep = 1 - mask`, and `bop` performs the two broadcast multiply-adds; `ones_per_feature` counts each mask.

### schedule

`_generate` computes `ppe_eff` first; with `ppe_eff > 1` it applies `repeat_rows` to all inputs and reshapes tensor baselines to `[1]+shape`. Each loop takes `K = min(ppe_eff, num_features - pos)`; when `K != ppe_eff` it re-expands and slices all inputs. `cur[index]` is reshaped to `[K, N] + feat_shape`, blended with the mask and folded back to `[K*N] + feat_shape`. A tuple in `extra_args` is indexed by `index`.

### eval_out

`_initial_state` handles int / float / array initial evals, building a `[1, n_outputs]` flattened form and `[n_outputs]+s[1:]` accumulators. `_accumulate` performs the `ppe>1` shape assertion, the `(-1, n_outputs)` reshape, diff broadcasting, `sum_axis0(diff*mask)` accumulation, and floating mask accumulation when `use_weights` is set. `_generate_result` divides by weights, yielding `nan` for zero denominators.

### attribution

The six forward kinds share `_forward`; the initial forward plus per-input per-chunk perturbed forwards accumulate `forward_shapes` and `chunks`. Empty inputs with `numel == 0` are skipped. `progress_total` is computed only when `show_progress` is set. `tuple_inputs` decides whether all attributions are returned or only element 0.

## 6. Domain constraints and error messages

Every invalid input raises `ValueError` naming the constraint: `perturbations_per_eval must be an integer and at least 1`, `input must have a leading example dimension`, `offset must be non-negative`, `feature mask must have the same rank as the input`, `feature mask first dim must be 1 or num_examples`, `feature mask shape is not broadcastable to input`, `feature mask values must be non-negative integers`, `expanded_input must have shape (n_features, n_examples, ...)`, `input_mask values must be integers`, `start_feature and end_feature must be integers`, `start_feature must be less than end_feature`, `start_feature must be non-negative`, `input_mask is not broadcastable to expanded_input`, `input_mask broadcast changes the expanded_input shape`, `inputs must be a non-empty list of input arrays`, `feature mask count must match inputs`, `baselines count must match inputs`, `baseline must be a number or a broadcastable array`, `baseline first dim must be 1 or num_examples`, `additional_forward_args must be a list`, `index out of range`, `input_shapes must be a non-empty list of shapes`, `modified_evals, masks and batch_sizes lengths must agree`, `initial_eval must be a number or an array`, `output should be a tensor whose 1st dim grow`, `modified eval cannot reshape to (-1, n_outputs)`, `single-input attribution needs exactly one input array`, `unknown forward_kind`, `additional_forward_args must be scalar numbers in attribution mode`, `Target list length does not match output!`, `multi forward needs an integer or list target`, `unknown mode`.

## 7. Judge case design

Ten `check_*(solve)` functions: `check_mask_range` (8 groups: default/offset/3D/min>0/gap/trailing-1/oversized ppe clamp/bool quirk), `check_mask_range_domain` (5 bad ppe + 6 bad mask + unknown mode), `check_ablate` (6 groups including tensor baselines, out-of-range start, row-dependent baseline), `check_ablate_domain` (7 assertions), `check_schedule` (12 groups including tail re-expansion, all-input slicing, tuple extra_args, ppe clamp, min>0 progress), `check_schedule_domain` (8 assertions), `check_eval_out` (6 groups including int64 dtype, no records, one ppe>1 record, use_weights), `check_eval_out_domain` (6 assertions), `check_attribution` (20 groups including six forward kinds, tuple inputs, empty inputs, bias, min>0 progress), `check_attribution_domain` (7 assertions). The driver emulates the judge injecting `solve`; the reference solution is **10/10 ALL ACCEPTED**.

## 8. Mutation coverage

```text
257 raw mutants, 9 true equivalents removed, 248 active: 248 CAUGHT, 0 SURVIVED, 0 ERROR
ALL MUTANTS CAUGHT
```

First-round SURVIVED entries were mostly deleted domain guards without matching invalid inputs, plus several "unobservable" points: `_mask_count` with `min>0`, the clamp when ppe exceeds the feature count, tail tensor extra arguments expanded by K, row-dependent tensor baselines, skipped empty inputs, and scalar forwards with bias. Precise message assertions and dedicated cases made them green; duplicate anchors (the `n_classes` indentation, the ppe validation line) were replaced with unique context-bearing anchors.

Nine true `EQUIVALENT` mutants: `h-unflatten-scalar` (the scalar branch is unreachable in this input domain), `h-reshape-guard`, `h-bto-equal`, `ab-fullshape`, `ev-weights-float`, `at-shapevalid-once`, `at-usew-float`, `at-generate-div`, `at-generate-nodiv` — each is covered by a stronger upstream constraint or an identity transformation.

## 9. Common pitfalls

- Clamping `ppe_clamped` with `max-min+1` loses chunks for masks with `min>0`.
- Implementing `repeat_rows` as element-wise repeat corrupts row order in both tail and full chunks.
- Reusing `ppe_eff`-expanded inputs / tensor arguments for the tail chunk, or slicing only the ablated input.
- Forgetting the `[1]*(input_rank-1)` broadcast dimensions on `diff`, which breaks 3D inputs outright.
- `broadcast_to` must special-case scalars (`shape == []`) or indexing fails.
- Writing `sum_axis0` as a sum over all perturbation rows mixes differences from different chunks.
- Not accumulating a global offset for default masks with multiple inputs makes `chunks` start/end restart at 0.
- Returning a non-`None` `progress_total` when `show_progress=False`.

## 10. Self-check list

The 12 official probe segments at 87/87, the three-layer comparison at 5291 checks ALL GREEN, driver 10/10, and mutation at 257 raw / 9 true equivalents / 248 all CAUGHT come first; only then the four bilingual docs, meta, starter precheck, 11-file landing, dual-mode selftest, bank-wide bilingual validation, collections registration, todos update, short guard and full guard.
