# Reference: COO accumulating scatter and additive merging

## Implementation

1. Validation order: weight 2-D and finite → indices non-empty, per-item pairs with integer coordinates (reject bool) and range checks → **budget** `K <= in*out` → values (list or 1-D tensor, length = K, each finite) → scaling non-negative finite (reject bool and inf) → x at least 1-D, last dim = in, finite.
2. **Scatter rebuild**: `delta[r, c] += v * scaling`. Accumulate in float64, then cast back to the weight dtype, matching the upstream sparse path's float behavior. The equivalent form is `torch.sparse_coo_tensor(idx, values*scaling, shape).to_dense()` — do **not** rebuild with an "overwrite dict": that is this problem's first mine.
3. **Additive merge**: `merged = weight + delta`. Dropping the delta (`merged = weight`) or computing the forward with the base weight (`x @ weight.T`) are both mines — upstream forward adds the delta into the base and runs one `F.linear`.

## Verification

4 `check_*` functions cover: the hand example (duplicate accumulation 6, merged, y), a fixed-seed 20-coordinate parity (the oracle rebuilds via `torch.sparse_coo_tensor(...).to_dense()` — upstream's own primitive, with plenty of natural duplicates), scaling=0.5 ≡ halved values, values as a 1-D tensor, 1-D x, the W=0 additive anchor (`y=[[4,0]]`), a budget exactly at `K=in*out` (legal), and 14 invalid inputs.

Key oracle design: instead of rewriting the scatter loop, call `torch.sparse_coo_tensor`'s `to_dense` directly — upstream's `get_delta_weight` primitive, which carries the accumulate-duplicates semantics natively and cross-checks this implementation's Python loop. torch emits a sparse-invariant UserWarning (routine without explicit opt-in); harmless.

Mutation testing (6/6 caught): duplicates changed to overwrite, scaling dropped, merged losing the delta, y via the base weight, budget check removed, negative scaling allowed. The budget mutation is caught by the `K=7 > 6` case in `check_validation`.

## Scope and sources

Upstream huggingface/peft `src/peft/tuners/shira/layer.py`:

- `ShiraLayer.update_layer` (L78-126): `num_shira_weight = r*(in+out)`, error when `> in*out` (budget check); mask → `torch.where(mask == 1.0)` → `torch.cat` into a 2×K int index; `shira_weight` is a **dense 1-D parameter** of length K (stores a vector, scatters into a sparse tensor on demand — avoiding PyTorch's sparse-parameter autograd gaps).
- `Linear.get_delta_weight` (L210-226): `torch.sparse_coo_tensor(indices, weight*scaling, (out, in))` — scaling multiplies the values, the delta is a COO sparse tensor.
- `Linear.forward` (L228-259): `new_weight = get_base_weight().clone()`; `new_weight += delta`; `F.linear(x, new_weight, bias)` — forward on the merged weight.
- `Linear.merge` (L158-196): `orig_weight += delta_weight` — additive merging.

Deliberate deviations:

- **Coordinates as direct inputs**: the `r` and mask indirection is dropped (K is the coordinate count; the budget check stays); upstream's mask-path row-major ordering (`torch.where`) out of scope.
- Explicit return contract: delta is delivered as a dense `[out, in]` (upstream's intermediate is a COO tensor; densifying is the `to_dense` semantics); `y` returned explicitly.
- `safe_merge` NaN checks, `unmerge` subtraction, the quantization `supports_merge` branch, and `_warn_once_about_module_hooks` out of scope.
- `scaling` pinned to non-negative finite: upstream allows arbitrary inference-time scaling; negative values are meaningless here and would blur with "values may be negative", hence tightened.
