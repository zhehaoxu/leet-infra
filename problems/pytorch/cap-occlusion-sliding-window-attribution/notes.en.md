# Captum Occlusion source walkthrough

Official sources covered:

- `captum/attr/_core/occlusion.py::Occlusion.attribute`
- `captum/attr/_core/occlusion.py::Occlusion._occlusion_mask`
- `captum/attr/_core/feature_ablation.py::FeatureAblation.attribute` / `_process_ablated_out_full` / `format_result`
- `captum/attr/_utils/common.py::_format_and_verify_sliding_window_shapes` / `_format_and_verify_strides` / `_format_baseline`
- `captum/_utils/common.py::_select_targets` / `_verify_select_column`

## 1. The complete official flow

`Occlusion.attribute` itself does only three things:

1. formats and validates `strides` and `sliding_window_shapes`;
2. computes the per-dimension `shift_counts`;
3. hands `sliding_window_tensors` (one all-ones tensor per window) and `shift_counts` to the parent class `FeatureAblation`.

The actual window enumeration and attribution live in `FeatureAblation`'s generic implementation. The **one key difference** between `Occlusion` and `FeatureAblation` is:

```python
# FeatureAblation.__init__
self.use_weights = use_weights        # defaults to False

# Occlusion.__init__
super().__init__(model, multiply_by_inputs=multiply_by_inputs)
self.use_weights = True                # <- Occlusion forces it to True
```

`use_weights=True` is where "overlapping regions are averaged" comes from.

## 2. ★ Grid enumeration: base-conversion, dim0 is the least significant digit

The official `_occlusion_mask` (inside `FeatureAblation`):

```python
for current_index in itertools.product(*[
    range(shift_count) for shift_count in shift_counts
]):
```

The semantics of this `itertools.product` equal base-conversion counting: **the last dimension varies fastest**. But the official code then **transposes** it:

```python
# how the official _occlusion_mask computes the offsets
current_index = []
for i, shift_count in enumerate(shift_counts):
    current_index.append((remaining_total % shift_count) * stride)
    remaining_total = remaining_total // shift_count
```

This is **hand-written base-conversion with dim0 as the least significant digit** (fastest changing). Confirmed empirically:

```
shape=[3,4], window=[2,2], stride=[1,2] -> enumeration order
(0,0) -> (2,0) -> (0,2) -> (2,2)
```

`itertools.product` yields the raw order `(0,0),(0,2),(2,0),(2,2)` (last dimension fastest); the official code adds the transposition, so dim0 ends up fastest. **This is the opposite of the intuition and is trap #1 of this problem.**

## 3. ★ The `shift_counts` formula and the two assertions

```python
current_shape = np.subtract(inp.shape[1:], sliding_window_shapes[i])
assert (np.array(current_shape) >= 0).all(), "Sliding window dimensions ... cannot exceed input dimensions"
assert np.logical_or(
    np.array(current_shape) == 0,
    np.array(strides[i]) <= sliding_window_shapes[i],
).all(), "Stride dimension ... cannot be larger than sliding window shape dimension ..."
shift_counts.append(
    tuple(np.add(np.ceil(np.divide(current_shape, strides[i])).astype(int), 1))
)
```

- `shift_counts[d] = ceil((shape[d] - window_shape[d]) / stride[d]) + 1`.
- Assertion 1: the window must not exceed the input dimension.
- Assertion 2: **when a dimension's window does not cover it exactly (`current_shape != 0`), the stride must not exceed the window length.**
  ★ Exception: when `shape[d] == window_shape[d]` (the window covers the whole dimension), `current_shape == 0` and the stride may be arbitrarily large — then `shift_counts[d] = ceil(0/s)+1 = 1` and only one window is enumerated. Empirically `shape=[5], window=[5], strides=[7]` is legal.

## 4. ★★ Window truncation: negative padding in `F.pad`

The official mask construction locates an all-ones tensor inside the input size using `F.pad`:

```python
padding = []
for i in range(num_dims - 2, -1, -1):
    padding.append(current_index[i])
    padding.append(input_shape[i] - current_index[i] - sliding_window_shape[i])
mask = torch.nn.functional.pad(sliding_window_tensor, padding)
```

For dimension `i`: left padding = offset `current_index[i]`, right padding = `shape[i] - current_index[i] - window_shape[i]`.

★ **The right padding can be negative** — `F.pad` with a negative right padding **clips the tail off the window** (equivalently, "drop `-right_pad` elements from the end"). This is exactly the mechanism behind the last-window truncation.

Empirically with `shape=[6], window=[4], stride=[3]`:

- Window 0: left 0, right `6-0-4 = 2` → `F.pad(ones(4), (0,2))` → mask `[1,1,1,1,0,0]`
- Window 1: left 3, right `6-3-4 = -1` → `F.pad(ones(4), (3,-1))` → mask `[0,0,0,1,1,1]`

The negative padding semantics of `F.pad`: a negative right padding trims elements off the end of the tensor. `ones(4)` has length 4; padding 3 on the left gives length 7; padding -1 on the right trims one element off the end → length 6. The result `[0,0,0,1,1,1]` matches "covered interval `[3, 6)`" exactly.

⇒ The covered interval is therefore `[start, min(start + w, shape[d]))`, which is equivalent to computing the interval directly.

## 5. ★ `use_weights=True`: overlapping regions are averaged

The core accumulation in `FeatureAblation` (`_process_ablated_out_full`):

```python
for feature in range(num_features):
    if ablation_masks[feature]:          # this feature is covered by this window
        attrib[feature] += attr_value
        weights[feature] += 1
```

Then in `format_result`:

```python
if use_weights:
    # single_attrib is the float64 accumulated value
    single_attrib = single_attrib.float() / weight
```

Two details:

1. **`weights[feature]` is the number of windows covering that feature** → when a region is covered by several windows, their differences are **averaged**.
   ★ Controlled experiment (1-D, `shape=[5]`, `window=[3]`, `stride=[1]`, model is `sum`):
   - `Occlusion` (`use_weights=True`): `[6.0, 7.5, 9.0, 10.5, 12.0]`
   - `FeatureAblation` (`use_weights=False`): `[1.0, 2.0, 3.0, 4.0, 5.0]`
   The two differ by the coverage counts.
2. ★ **`single_attrib.float()`**: the official code first **demotes the float64 accumulated value to float32**, and only then divides in float32. So the attribution is not the exact float64 quotient.

A third detail: positions with **`weights` all zero** (covered by no window at all) go through `0.0 / 0.0` → **NaN**. Under this problem's conventions (sliding windows with stride ≤ window guarantee full coverage), `weights` is always ≥ 1 and that branch is unreachable.

## 6. ★★ The float32 demotion in the official `format_result`

```python
single_attrib = single_attrib.float() / weight
```

`single_attrib` is the float64 accumulated value; `.float()` demotes it to float32, and the division is then performed in float32.

This produces an observable difference whenever the attribution is **not a binary finite decimal**. For an accumulated value of `1.0` and a weight of `3`:

- float64: `1.0 / 3 = 0.333333333333333314829616256247...`
- float32: `0.3333333432674407958984375` (the closest float32 to `1/3`)

This problem requires reproducing the float32 result.

## 7. ★★ Negative-index semantics for `target`

The official `_select_targets` routes an `int` target through:

```python
def _verify_select_column(output, target):
    target = (target,) if isinstance(target, int) else target
    assert len(target) <= len(output.shape) - 1, ...
    return output[(slice(None), *target)]
```

★ `output[(slice(None), -1)]` is **PyTorch's negative indexing** — `target=-1` selects the last output and `target=-2` the second to last. Empirically:

```
target=0  -> [6.0, 7.5, 9.0, 10.5, 12.0]     (first of 3 outputs)
target=1  -> [12.0, 15.0, 18.0, 21.0, 24.0]
target=2  -> [18.0, 22.5, 27.0, 31.5, 36.0]
target=-1 -> [18.0, 22.5, 27.0, 31.5, 36.0]   <- same as target=2
target=-3 -> [6.0, 7.5, 9.0, 10.5, 12.0]      <- same as target=0
target=3  -> IndexError: index 3 is out of bounds for dimension 1 with size 3
target=-4 -> IndexError: index -4 is out of bounds for dimension 1 with size 3
```

The official implementation raises `IndexError` (PyTorch's) when out of range; this problem uniformly expresses it as `ValueError`.

## 8. ★ Scalar broadcast for `strides`

The official `_format_and_verify_strides`:

```python
if strides is None:
    strides = tuple(1 for input in inputs)
if len(inputs) == 1 and not (isinstance(strides, tuple) and len(strides) == 1):
    strides = (strides,)
```

★ When there is only one input tensor and `strides` is not "a tuple of length 1", `strides` gets wrapped into `(strides,)`. At that point `strides` may be a **scalar int**. Empirically (`shape=[5]`, `window=[3]`):

```
strides=1 -> [6.0, 7.5, 9.0, 10.5, 12.0]
strides=2 -> [6.0, 6.0, 9.0, 12.0, 12.0]     <- shift_counts = ceil(2/2)+1 = 2
```

For the 2-D case (`shape=[3,4]`, `window=[2,2]`, `strides=2` as a scalar), every dimension uses stride 2.

★ **In 1-D, "apply the scalar to the first dimension only" and "broadcast to all dimensions" are equivalent**, so the test suite must use a 2-D case to pin this down.

## 9. The three shapes of `baseline` in the official code

`_format_baseline` supports:

1. a tensor with the **same shape** as the input;
2. a scalar (broadcast);
3. a tuple of scalars (one per input tensor).

This problem's convention: broadcast a scalar, or a list whose flattened length equals `prod(shape)`.

★ The official length check is "broadcastable to match the dimensions of inputs"; this problem simplifies it to "the length must equal `prod(shape)`".

## 10. The three-layer verification structure

1. **Official layer**: call `captum.attr.Occlusion` directly with a `float64` network (ReLU only between layers) → the authoritative output.
2. **Reference-solution layer**: this problem's pure-Python + `Fraction` implementation → **bit-for-bit identical** to the official output (`0.000e+00`).
3. **Independent oracle layer**: re-implements the enumeration and accumulation straight from the statement, without reusing the reference solution → cross-validation.

Empirically all 14 cases (1-D / 2-D / 3-D / multi-output / negative target / per-element baseline / truncation) come out at **`0.000e+00`**.

## 11. ★ New lesson this round: actively drop equivalent mutations

When mutation testing reached 46 items, one survived:

```python
("target-negative-not-wrapped",
 "        return target + n_outputs",
 "        return target")
```

It replaces the negative-index folding `target + n` with returning `target` directly. **This mutation is semantically equivalent** — the downstream access is `probe[tgt]` on a Python list, and Python list negative indexing is naturally equivalent to `n + target`. This is a **true equivalent mutation** (same family as `xs-base-num-rejects-odd` from round 843) and should be deleted rather than treated as a test-coverage gap.

It was replaced with a genuinely observable mutation:

```python
("target-negative-clamped-to-zero",
 "        return target + n_outputs",
 "        return 0")
```

⇒ **When a mutation survives, first ask "is this a semantic gap or an equivalent mutation?"** Delete the equivalent ones; close the gaps with test cases.

## 12. ★★ New lesson this round: validation layers must be independent defenses

`x-length-check-dropped` (removing the "input length must equal the shape size" check) initially **survived**. The reason:

- that check lives in `_occlusion_core` (`len(flat_x) != total`);
- but `_forward` also has a check comparing the layer's input dimension against the actual input length (`len(h) != n_in`);
- my test case for "x length differs from shape" happened to use a layer whose **input dimension equals the shape size**, so both checks pointed at the same error → removing one leaves the other to catch it → the mutation survives.

This is exactly the "independent defenses" principle distilled in round 844: do not validate the same semantic constraint at multiple layers.

The fix (make the test case a shape that **only the intended check** can reject):

```python
# a layer with input dim 2
LAYER_2IN = [{"W": [[1.0, 1.0]], "b": [0.0]}]
...
_expect_error(
    lambda: solve("occlusion_diff", layers=LAYER_2IN, x=[1.0, 2.0], baseline=0.0,
                  shape=[5], window_shape=[3], strides=[1], target=0),
    "diff.x-length",
)
```

Now the `x` length (2) equals the layer input dimension (2) but differs from `prod(shape)` (5) → only the "x length vs shape size" check can raise → removing it is guaranteed to be caught (it surfaces as an `IndexError` instead of a `ValueError`).

⇒ **For a check to be individually caught by mutation, the test case must create a scenario where only that check can fail.**

## 13. ★★ New lesson this round: mutations invisible in 1-D need a 2-D case

`grid-scalar-stride-not-broadcast` (a scalar stride broadcast only to the first dimension) **survived** the 1-D cases — because with a single dimension, `[s] + [1] * 0` and `[s] * 1` are literally the same.

The fix is a 2-D scalar-stride case:

```python
rs2 = solve("occlusion_grid", shape=[3, 4], window_shape=[2, 2], strides=2)
# shift_counts = [2, 2]
```

⇒ **Equivalent mutations are invisible in low dimensions.** Any "broadcast / per-dimension" semantics must have a case with at least 2 dimensions.

## 14. Ten reusable lessons

1. **Enumeration order**: `itertools.product` makes the last dimension fastest, but the official base-conversion makes the **first** dimension fastest — read the code, do not guess.
2. **Negative padding in `F.pad` means truncation**, equivalent to clamping the covered interval's right end to `shape[d]`.
3. **`use_weights=True` means overlapping regions are averaged**, which differs from `FeatureAblation`'s default.
4. **float32 demotion**: `.float()` happens **before** the division, so the attribution is not the exact float64 quotient.
5. **A negative `target` follows PyTorch negative indexing**, so `-1` is the last output.
6. **A scalar `strides` broadcasts to every dimension**.
7. **The `+1` in `shift_counts` must not be dropped.**
8. **Exception to stride ≤ window**: when the window covers the whole dimension exactly, the stride may be larger.
9. **Checks must be independent defenses**: never validate the same semantics at multiple layers, or mutations will survive.
10. **Drop equivalent mutations and fill in cases with insufficient dimensionality.**

## 15. The three-layer verification structure

```
official captum.attr.Occlusion  ->  reference solution (Fraction)  ->  independent oracle
        |                                     |                          |
   authoritative output            bit-for-bit identical        bit-for-bit identical
```

Empirically: **all 14 cases at `0.000e+00` (bit-for-bit identical)**.
