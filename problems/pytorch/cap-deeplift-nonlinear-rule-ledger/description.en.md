# Captum DeepLIFT non-linear propagation rule ledger

## Background

DeepLIFT (Deep LIFT) is an attribution method built on *multiplicative attribution*:
instead of computing gradients directly, it asks every non-linear layer in the network
one question — **how much did this layer's output change relative to the baseline
(reference input)?** — and then distributes that change back onto the inputs.

The distribution rule is the *propagation rule*. In Captum
(`captum/attr/_core/deep_lift.py`), one rule function is attached to each kind of
non-linear layer:

| Layer | Rule function |
|---|---|
| `nn.ReLU` / `nn.ELU` / `nn.LeakyReLU` / `nn.Sigmoid` / `nn.Tanh` / `nn.Softplus` | `nonlinear` |
| `nn.Softmax` | `softmax` |
| `nn.MaxPool1d` / `nn.MaxPool2d` / `nn.MaxPool3d` | `maxpool` |

Your job is to re-implement the **exact numerical semantics** of those three rules
(plus the shared difference function `_compute_diffs`) in pure Python. Several details
look reasonable but are *not* what the official code does; the judge pins each of them
down.

## Conventions

Every mode takes **flattened one-dimensional lists**; no tensor shape inference is
involved:

- `inputs` / `outputs`: length must be **even**. Officially they are split with
  `chunk(2)` along dim 0 into **first half = actual input**, **second half =
  reference input**.
- `grad_input` / `grad_output`: gradients, shaped like the corresponding `inputs` /
  `outputs`.
- Every returned value is a `float` (internally, exact rationals go through
  `fractions.Fraction`).
- Illegal input raises `ValueError`.

---

## mode 1: `chunk_diffs`

Re-implement the official `_compute_diffs`:

```python
def _compute_diffs(inputs, outputs):
    input, input_ref = inputs.chunk(2)
    output, output_ref = outputs.chunk(2)
    delta_in = input - input_ref
    delta_out = output - output_ref
    return torch.cat(2 * [delta_in]), torch.cat(2 * [delta_out])
```

**Signature**

```python
chunk_diffs(inputs, outputs)
    → {"delta_in": [...], "delta_out": [...], "half": int}
```

**Key points**

1. The difference direction is `actual - reference` (not the other way round).
2. ★ The return value is `torch.cat(2 * [delta])` — each difference is **duplicated
   into two copies**, so the returned length equals the **input length**, not half
   of it.
3. The two halves are exactly identical.
4. `half` = `len(inputs) // 2`.

---

## mode 2: `nonlinear_rule`

Re-implement the official `nonlinear`:

```python
def nonlinear(module, inputs, outputs, grad_input, grad_output, eps=1e-10):
    delta_in, delta_out = _compute_diffs(inputs, outputs)
    new_grad_inp = torch.where(
        abs(delta_in) < eps, grad_input, grad_output * delta_out / delta_in
    )
    return new_grad_inp
```

**Signature**

```python
nonlinear_rule(inputs, outputs, grad_input, grad_output, eps=1e-10)
    → {"delta_in", "delta_out", "new_grad_in", "used_fallback", "n_fallback"}
```

**Key points**

1. ★ The predicate is **strictly less than**: `abs(delta_in) < eps`. When
   `abs(delta_in)` is **exactly equal to** `eps`, the **division branch** is taken.
2. The division branch computes `grad_output * delta_out / delta_in`
   (multiply first, divide second).
3. A negative `delta_in` yields a result with the corresponding negative sign.
4. When `delta_in` is 0 (or `abs(delta_in) < eps`), the `grad_input` branch is taken.
5. `grad_input` / `grad_output` must have the same length as `delta_in` — that is,
   the same length as `inputs` (because `_compute_diffs` duplicated the differences
   into two copies).
6. The **default value of `eps` is `1e-10`** (used when `eps` is not passed).
7. `used_fallback` is a per-element boolean list marking which positions took the
   `grad_input` branch; `n_fallback` counts the `True` entries.

---

## mode 3: `softmax_rule`

Re-implement the official `softmax`:

```python
def softmax(module, inputs, outputs, grad_input, grad_output, eps=1e-10):
    delta_in, delta_out = _compute_diffs(inputs, outputs)
    grad_input_unnorm = torch.where(
        abs(delta_in) < eps, grad_input, grad_output * delta_out / delta_in
    )
    n = grad_input.numel()
    new_grad_inp = grad_input_unnorm - grad_input_unnorm.sum() * 1 / n
    return new_grad_inp
```

**Signature**

```python
softmax_rule(inputs, outputs, grad_input, grad_output, eps=1e-10)
    → {"unnorm", "total", "n", "shift", "new_grad_in"}
```

**Key points**

1. First compute the un-normalized version `unnorm` (identical to the
   `new_grad_in` of `nonlinear_rule`).
2. ★ `n = grad_input.numel()` — the **total element count**, not the batch count
   and not the "single-side" length of the difference. Since `_compute_diffs`
   duplicated the differences, `n` effectively equals `len(inputs)`.
3. ★ What is subtracted is `unnorm.sum() * 1 / n` — a **global scalar**, and
   **not** a per-row (per-half) mean.
4. After the correction, all elements sum to 0.
5. `total` = the element sum of `unnorm`; `shift` = the subtracted scalar
   (`total / n`).

---

## mode 4: `maxpool_rule`

Re-implement the official `maxpool` (shown for `MaxPool1d`; `2d` / `3d` merely add
extra spatial axes and have the same semantics under this mode's 1-D convention):

```python
def maxpool(module, pool_func, unpool_func, inputs, outputs,
            grad_input, grad_output, eps=1e-10):
    with torch.no_grad():
        input, input_ref = inputs.chunk(2)
        output, output_ref = outputs.chunk(2)

        delta_in = input - input_ref
        delta_in = torch.cat(2 * [delta_in])

        delta_out_xmax = torch.max(output, output_ref)
        delta_out = torch.cat([delta_out_xmax - output_ref, output - delta_out_xmax])

        _, indices = pool_func(module.input, module.kernel_size, module.stride,
                               module.padding, module.dilation, module.ceil_mode, True)

        grad_output_updated = grad_output
        unpool_grad_out_delta, unpool_grad_out_ref_delta = torch.chunk(
            unpool_func(
                grad_output_updated * delta_out,
                indices,
                module.kernel_size, module.stride, module.padding,
                list(module.input.shape),
            ),
            2,
        )

    unpool_grad_out_delta = unpool_grad_out_delta + unpool_grad_out_ref_delta
    unpool_grad_out_delta = torch.cat(2 * [unpool_grad_out_delta])

    new_grad_inp = torch.where(
        abs(delta_in) < eps, grad_input[0], unpool_grad_out_delta / delta_in
    )
    return new_grad_inp
```

**Signature**

```python
maxpool_rule(inputs, outputs, grad_input, grad_output, indices, eps=1e-10)
    → {"delta_in", "delta_out_xmax", "delta_out", "scaled",
       "unpool_delta", "unpool_ref", "merged", "new_grad_in", "used_fallback"}
```

**Conventions**

Let `n_feat = len(inputs) // 2` (single-side input length) and
`n_out = len(outputs) // 2` (single-side output length):

| Quantity | Length | Meaning |
|---|---|---|
| `inputs` | `2 * n_feat` | first half = actual input, second half = reference input |
| `outputs` | `2 * n_out` | first half = actual output, second half = reference output |
| `grad_input` | `2 * n_feat` | same shape as `inputs` |
| `grad_output` | `2 * n_out` | same shape as `outputs` |
| `indices` | `2 * n_out` | first half = window hit positions on the actual side, second half = on the reference side |

**Key points**

1. ★ `delta_out_xmax = torch.max(output, output_ref)` — the **element-wise**
   maximum of the actual and reference halves.
2. ★ `delta_out = cat([xmax - output_ref, output - xmax])`:
   - the first half (`n_out` elements) = `xmax - output_ref`, i.e. "how much xmax
     exceeds the reference", and is **always non-negative**;
   - the second half (`n_out` elements) = `output - xmax`, i.e. "the gap between the
     actual output and xmax", and is **always non-positive**; it becomes negative
     when the reference side is larger.
3. ★ `grad_output` has the **same shape** as `outputs` (length `2 * n_out`), and
   `grad_output * delta_out` is an **element-wise** product, not a broadcast.
4. `scaled` = that element-wise product (length `2 * n_out`).
5. **The unpool allocation**: `indices` says where each output window's gradient
   belongs.
   - the first half uses `indices[:n_out]` (actual-side window hits) together with
     `scaled[:n_out]`;
   - the second half uses `indices[n_out:]` (reference side) together with
     `scaled[n_out:]`;
   - they are written into `unpool_delta` (length `n_feat`) and `unpool_ref`
     (length `n_feat`) respectively.
   - ★ **When several windows hit the same position, the later write OVERWRITES the
     earlier one — it does not accumulate** (this is the actual behaviour of
     `torch.max_unpool`).
6. `merged` = `unpool_delta + unpool_ref` (element-wise, length `n_feat`), then
   **duplicated into two copies** (length `2 * n_feat`).
7. ★ The eps branch takes `grad_input[0]` — the **first half along dim 0**
   (not the 0-th element of `grad_input`). Hence in the eps branch **both halves
   use the first half's `grad_input` values**.
8. Every other position yields `merged / delta_in`.

---

## Common mistakes

1. After `chunk(2)`, computing only one copy of the difference instead of
   duplicating it — the returned length is then wrong.
2. Writing `delta_in` as `reference - actual`.
3. In `chunk_diffs`, treating `half` as the "single-side length" but returning only
   the single side.
4. Writing the eps predicate as `<=` (officially it is strict `<`).
5. Writing the eps predicate as `abs(delta_in) < eps` but remembering the default
   `eps` as `1e-6` (officially it is `1e-10`).
6. Using the "single-side length" for `n` in `softmax` instead of
   `grad_input.numel()`.
7. Implementing `softmax` as a per-row (per-half) mean subtraction instead of
   subtracting one global scalar.
8. Normalizing in `softmax` before computing `unnorm`.
9. Treating `grad_output` in `maxpool` as having "single-side length" and relying on
   broadcasting.
10. Taking `xmax` from the actual side only (not the element-wise max of both
    halves).
11. Writing the two halves of `maxpool`'s `delta_out` in the wrong order.
12. Using **accumulation** instead of **later-write-overwrites** for the maxpool
    unpool.
13. Taking `grad_input[0]` as the "0-th element" instead of the "first half".
14. Returning only one half of `maxpool`'s `merged` (forgetting to duplicate it).
15. Using the wrong half of `indices` for `unpool_delta` / `unpool_ref`.

## Self-check

```bash
python starter.py            # skeleton self-check
```

Before submitting, confirm:

- The returned key names of the four modes exactly match the spec above.
- Every returned value is a `float` (not a `Fraction`).
- Illegal input raises `ValueError`.
