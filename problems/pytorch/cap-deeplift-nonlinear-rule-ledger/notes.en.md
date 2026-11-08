# Captum DeepLIFT non-linear propagation rule ledger — source dissection

> Official repository: `pytorch/captum` (GitHub repo id `204734444`, default branch `main`)
> Official file: `captum/attr/_core/deep_lift.py`
> The four functions this problem focuses on: `_compute_diffs` / `nonlinear` / `softmax` / `maxpool`

---

## 1. Where these rules sit inside DeepLIFT

DeepLIFT works by attaching forward hooks and backward hooks to every non-linear
layer of the model. The forward hook records that layer's `input` and `output`; when
the backward hook fires, it takes `grad_input` / `grad_output`, calls the **propagation
rule** attached to that layer, and replaces the original gradient with the "multiplier"
the rule computes.

The rule function is bound through a lookup table:

```python
SUPPORTED_NON_LINEAR = {
    nn.ReLU: nonlinear,
    nn.ELU: nonlinear,
    nn.LeakyReLU: nonlinear,
    nn.Sigmoid: nonlinear,
    nn.Tanh: nonlinear,
    nn.Softplus: nonlinear,
    nn.MaxPool1d: maxpool1d,
    nn.MaxPool2d: maxpool2d,
    nn.MaxPool3d: maxpool3d,
    nn.Softmax: softmax,
}
```

The three `maxpool*` entry points merely hand different `pool_func` / `unpool_func`
callables (`F.max_pool1d` and `F.max_unpool1d`, etc.) to the same `maxpool` body:

```python
def maxpool1d(module, inputs, outputs, grad_input, grad_output, eps=1e-10):
    return maxpool(
        module, F.max_pool1d, F.max_unpool1d,
        inputs, outputs, grad_input, grad_output, eps=eps,
    )
```

So `maxpool_rule` only has to implement the semantics of the `maxpool` body.

### 1.1 `module.input` and `inputs` are the same tensor

This is what determines where `indices` comes from. In `_forward_pre_hook`:

```python
def _forward_pre_hook(self, module, inputs):
    inputs = _format_tensor_into_tuples(inputs)
    module.input = inputs[0].clone().detach()
```

and `_backward_hook` passes that very tensor into the rule:

```python
multipliers = SUPPORTED_NON_LINEAR[type(module)](
    module,
    module.input,
    module.output,
    grad_input,
    grad_output,
    eps=self.eps,
)
```

⇒ Inside `maxpool`, `module.input` **is** `inputs` (the same tensor).
Hence `pool_func(module.input, ...)` computes `indices` on the **concatenated**
input, and `unpool`'s `output_size = list(module.input.shape)` also refers to that
concatenated shape.

> ★ This is the easiest place to come up empty: if `module.input` is filled with the
> "single-side" input, the shape of `indices` will not match
> `grad_output * delta_out`, and the official code raises immediately.

---

## 2. `_compute_diffs`: the "duplicate into two copies" semantics

```python
def _compute_diffs(inputs: Tensor, outputs: Tensor) -> Tuple[Tensor, Tensor]:
    input, input_ref = inputs.chunk(2)
    output, output_ref = outputs.chunk(2)
    delta_in = input - input_ref
    delta_out = output - output_ref

    return torch.cat(2 * [delta_in]), torch.cat(2 * [delta_out])
```

### ★ Point 2.1: `cat(2 * [delta])` doubles the length

`torch.cat(2 * [delta_in])` is equivalent to `torch.cat([delta_in, delta_in])`,
i.e. the same difference is **concatenated with itself**.

Therefore:

- the length of `delta_in` equals the length of `inputs` (**not** half of it);
- the first half and the second half are exactly identical;
- this directly implies that in `nonlinear`, `grad_input` / `grad_output`
  must have the **same length as `inputs`** (not the single-side length).

Measured by calling the official function directly:

```python
inputs  = torch.tensor([1., 2., 3., 4., 0.5, 1.5, 2.5, 3.5])
outputs = torch.tensor([10., 20., 30., 40., 5., 15., 25., 35.])
di, do = _compute_diffs(inputs, outputs)
# di.shape == (8,)  do.shape == (8,)
# di.tolist() == [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
```

### ★ Point 2.2: the difference direction is `actual - reference`

`chunk(2)` splits along dim 0: **first half = actual, second half = reference**.
So `input - input_ref` is actual minus reference.

Getting the direction backwards flips the sign of the entire chain.

---

## 3. `nonlinear`: the two branches of `where`

```python
def nonlinear(module, inputs, outputs, grad_input, grad_output, eps=1e-10):
    delta_in, delta_out = _compute_diffs(inputs, outputs)

    new_grad_inp = torch.where(
        abs(delta_in) < eps, grad_input, grad_output * delta_out / delta_in
    )
    return new_grad_inp
```

### ★ Point 3.1: the predicate is **strictly** less than

`torch.where(cond, a, b)` takes `a` when `cond` holds. The condition reads
`abs(delta_in) < eps` — **strictly less than**.

Measured (with `delta_in` exactly `0.5`):

| `eps` | `abs(0.5) < eps` | branch taken |
|---|---|---|
| `0.5` | `0.5 < 0.5` → False | **division branch** |
| `0.5000001` | `0.5 < 0.5000001` → True | `grad_input` branch |
| `0.4999999` | `0.5 < 0.4999999` → False | division branch |

⇒ The judge needs a case where the magnitude is **exactly equal to** `eps`,
otherwise `<` and `<=` cannot be told apart.

### ★ Point 3.2: the division branch is `grad_output * delta_out / delta_in`

Multiply first, divide second. In floating point `a * b / c` and `a / c * b`
generally differ by one ulp, but this problem's reference solution uses exact
rationals (`Fraction`), so the two are **exactly equal** ⇒ this is an
**equivalent mutation** and must not be used as a scoring point.

### ★ Point 3.3: a negative `delta_in` yields a negative result

Measured: `delta_in = -2`, `delta_out = 5` → result `-2.5`.
There is no `abs` and no clamping.

### ★ Point 3.4: `grad_input` / `grad_output` must match `delta_in` in length

Because `_compute_diffs` duplicated the differences, the length of `delta_in`
equals the length of `inputs`. A mismatched `grad_input` makes the official code raise:

```
RuntimeError: The size of tensor a (4) must match the size of tensor b (8)
```

### ★ Point 3.5: the default `eps = 1e-10`

The official signature hard-codes `eps: float = 1e-10`.
The judge needs a case that **omits `eps`** to pin the default down — use an
input whose `|delta_in|` falls between `1e-10` and `1e-6`, so that
"default 1e-10" and "default 1e-6" give different results.

---

## 4. `softmax`: the global normalization correction

```python
def softmax(module, inputs, outputs, grad_input, grad_output, eps=1e-10):
    delta_in, delta_out = _compute_diffs(inputs, outputs)

    grad_input_unnorm = torch.where(
        abs(delta_in) < eps, grad_input, grad_output * delta_out / delta_in
    )
    # normalizing
    n = grad_input.numel()

    # updating only the first half
    new_grad_inp = grad_input_unnorm - grad_input_unnorm.sum() * 1 / n
    return new_grad_inp
```

### ★ Point 4.1: `n = grad_input.numel()` is the **total element count**

- not the batch count;
- not the "single-side" length of the difference;
- since the differences were duplicated, `n` effectively equals `len(inputs)`.

Measured (`grad_input` of length 8): `n == 8`.

### ★ Point 4.2: a **global scalar** is subtracted

`grad_input_unnorm.sum() * 1 / n` is a 0-dim tensor, broadcast onto every
element of `grad_input_unnorm`.

⇒ This is **not** a per-row (per-batch / per-half) mean subtraction.

Measured: `unnorm = [10]*4 + [20]*4` → `sum = 120`, `n = 8` → subtract `15` →
`[-5]*4 + [5]*4`.

> ⚠️ **A pitfall in test design**: if the two halves of `unnorm` happen to be
> symmetric (equal sums), then "subtract a global scalar" and "subtract a per-half
> mean" give the **same** result ⇒ the case loses its discriminating power.
> The judge must use **asymmetric** data and explicitly assert that the two
> candidates differ.

### ★ Point 4.3: after the correction the total sums to 0

This is a direct corollary of the global scalar correction, so it is a cheap
property to assert — but note that subtracting per-half means also yields a zero
total, so **on its own this does not discriminate**.

### ★ Point 4.4: the comment `updating only the first half` contradicts the code

The official source carries a comment reading `# updating only the first half`,
but the code is actually
`grad_input_unnorm - grad_input_unnorm.sum() * 1 / n`, which **corrects every
element**. The comment is a leftover from an earlier implementation.

⇒ Follow the **code**, not the comment.

---

## 5. `maxpool`: cross-maximum and the unpool allocation

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

        _, indices = pool_func(
            module.input, module.kernel_size, module.stride,
            module.padding, module.dilation, module.ceil_mode, True,
        )
        grad_output_updated = grad_output
        unpool_grad_out_delta, unpool_grad_out_ref_delta = torch.chunk(
            unpool_func(
                grad_output_updated * delta_out,
                indices,
                module.kernel_size, module.stride, module.padding,
                list(cast(torch.Size, module.input.shape)),
            ),
            2,
        )

    unpool_grad_out_delta = unpool_grad_out_delta + unpool_grad_out_ref_delta
    unpool_grad_out_delta = torch.cat(2 * [unpool_grad_out_delta])

    if grad_input.shape != inputs.shape:
        raise AssertionError(...)

    new_grad_inp = torch.where(
        abs(delta_in) < eps, grad_input[0], unpool_grad_out_delta / delta_in
    )
    return new_grad_inp
```

### ★ Point 5.1: `delta_out_xmax` is an **element-wise** maximum

`torch.max(output, output_ref)` is the element-wise maximum of two tensors
(not the "reduce over a dimension" flavour of `torch.max`).

Measured (`output = [2, 4]`, `output_ref = [6, 8]`): `xmax = [6, 8]`.

### ★ Point 5.2: the two halves of `delta_out` mean completely different things

```python
delta_out = torch.cat([delta_out_xmax - output_ref, output - delta_out_xmax])
```

- the first half = `xmax - output_ref` — "how much xmax exceeds the reference",
  **always non-negative**;
- the second half = `output - xmax` — "the gap between the actual output and xmax",
  **always non-positive**; it becomes negative when the reference side is larger.

This is exactly the intent of the "cross-maximum" design: **the gradient is
attributed to whichever side produced the larger output**.
When actual is larger, the increment (positive) lands in the first half; when
reference is larger, the gap (negative) lands in the second half.

Measured (`output = [2, 4]`, `output_ref = [6, 8]`):
`delta_out = [0, 0, -4, -4]`.

### ★ Point 5.3: `grad_output` has the **same shape** as `outputs`

The `grad_output` handed in by the official backward hook has the same shape as
that layer's output, and `outputs` is already **concatenated**
(actual + reference). So the length of `grad_output` is `2 * n_out`.

Consequently `grad_output_updated * delta_out` is an **element-wise** product of
length `2 * n_out`.

> ⚠️ **A common mistake**: treating `grad_output` as having the "single-side
> length" and relying on broadcasting. Measured: with `grad_output` of shape
> `(1, 1, n_out)`, `prod` broadcasts to `(2, 1, n_out)` — it *runs* numerically,
> but the semantics are wrong (both halves share one gradient), and the judge
> must exclude it.

### ★ Point 5.4: the unpool scatter is a **later-write overwrite**

`unpool_func` is `torch.nn.functional.max_unpool1d` in the official code. It
writes each output window's gradient to the input position named by `indices`.
**When several windows hit the same position, the later write overwrites the
earlier one — it does not accumulate.**

Measured (`indices = [1, 1, 2]`, `prod = [10, 20, 30]`):
the result is `[0, 20, 30, 0, 0]` — position 1 received **20** (the second
window's value), not `10 + 20 = 30`.

Reversing `prod` (`[30, 20, 10]`) → the result is `[0, 20, 10, 0, 0]`; position 1
is still **20** (the later write) ⇒ confirmed to be a later-write overwrite,
independent of the numeric magnitudes.

### ★ Point 5.5: the two scattered halves are summed first, then duplicated

```python
unpool_grad_out_delta = unpool_grad_out_delta + unpool_grad_out_ref_delta
unpool_grad_out_delta = torch.cat(2 * [unpool_grad_out_delta])
```

`torch.chunk(unpool(...), 2)` splits the unpool result (whose shape matches
`module.input`, i.e. the concatenated shape) along dim 0 into the actual half and
the reference half. The two are **added element-wise**, and the result is then
**duplicated into two copies** by `cat(2 * [...])`.

⇒ the final `merged` has length `2 * n_feat`, with identical halves.

### ★ Point 5.6: the eps branch takes `grad_input[0]` (the first half)

`grad_input[0]` is the slice at index 0 along **dim 0**, of shape
`(1, ..., n_feat)` — **not** the 0-th element of `grad_input` (which would be a
scalar).

Measured (`grad_input = [[7,8,9,10],[11,12,13,14]]`):
`grad_input[0]` = `[7,8,9,10]` (shape `(1,4)`).

Since `torch.where` broadcasts, in the eps branch **both halves use the first
half's values**.

Measured (`delta_in` all zero, `grad_input = [7,8,9,10,11,12,13,14]`):
the output is `[7,8,9,10, 7,8,9,10]` — the second half is `7,8,9,10` too, not
`11,12,13,14`.

### ★ Point 5.7: `grad_input` must have the same shape as `inputs`

The official code has an explicit assertion:

```python
if grad_input.shape != inputs.shape:
    raise AssertionError("A problem occurred during maxpool modul's backward pass. ...")
```

i.e. `grad_input` and `inputs` are **the same shape** (both concatenated).

---

## 6. Three reproduction pitfalls in the official source

### Pitfall 1: `maxpool` depends on torch's pooling implementation

The `maxpool` body itself is pure tensor arithmetic, but it takes `pool_func` /
`unpool_func` as parameters (`F.max_pool1d` / `F.max_unpool1d`, etc.).
Reproducing it means accepting two external constraints from torch pooling:

**(a) `indices` is computed by pooling on `module.input`**

- `pool_func(module.input, kernel_size, stride, padding, dilation, ceil_mode, True)`
  — the third positional argument is `return_indices=True`;
- the shape of `indices` is that of `module.input` with the spatial dims reduced,
  i.e. the **concatenated** `(2, 1, n_out)`;
- `max_pool1d` breaks ties at the **first** maximum (measured: `[5,5,1,1]`
  with k=2 s=2 → `indices=[0,2]`).

**(b) `torch.max_unpool1d` validates `output_size`**

The official code passes `list(module.input.shape)`, and torch checks it against
the "natural" output size implied by `indices` plus kernel/stride:

```
ValueError: invalid output_size "[6]" (dim 0 must be between 4 and 6)
```

⇒ One must satisfy `M == floor((L - k) / s) + 1` (`L` = single-side input
length, `M` = single-side output length). This is not captum's restriction, it
is torch's; but the official-function comparison must respect it, otherwise the
official function simply raises.

### Pitfall 2: `module.input` must be the concatenated tensor

If `module.input` is filled with the single-side input:

- `pool_func` yields `indices` with only half the batch dimension, which does
  not match the `(2, 1, n_out)` of `grad_output * delta_out` ⇒ the official code
  raises `Expected shape of indices to be same as that of the input tensor`;
- `unpool`'s `output_size` would be wrong too.

### Pitfall 3: the shape of `grad_output` can "accidentally" work

With a single-sided `grad_output`, `grad_output * delta_out` broadcasts to
`(2, 1, n_out)`, the official function **does not raise**, and it even produces
plausible numbers. But the semantics are wrong (both halves share one gradient).
⇒ The judge must adopt the "same shape as `outputs`" convention and assert the
length explicitly, otherwise this wrong implementation survives silently.

---

## 7. The three-layer verification structure

This problem uses a three-layer comparison directly against the official source:

| Layer | Content | Role |
|---|---|---|
| Official | `_compute_diffs` / `nonlinear` / `softmax` / `maxpool` extracted **verbatim** from `captum/attr/_core/deep_lift.py` via AST slicing | the single source of truth |
| Reference | pure Python + `Fraction` implementation | what the submitter writes |
| Oracle | an independent re-write of the official formulas (pure float), not derived from the reference | guards against the reference and the oracle failing in the same way |

Comparison result:

```
WORST official-vs-ref = 5.551e-17
WORST oracle-vs-ref   = 5.551e-17
```

(`5.551e-17` is floating-point rounding in the softmax normalization; a magnitude
around 1e-17 is expected.)

The official layer uses `ast.parse` + `ast.get_source_segment` to extract the
source, so **what runs is the upstream original**, not a hand-copied duplicate.

---

## 8. Three principles of mutation design (re-confirmed here)

### Principle 1: actively remove equivalent mutations

- `go * delta_out / delta_in` → `go / delta_in * delta_out`:
  under exact rationals these are **exactly equal** ⇒ equivalent mutation, remove it.
- `delta_out[0] + delta_out[1]` → `delta_out[1] + delta_out[0]`:
  the two halves have identical content ⇒ equivalent mutation, remove it.

Scoring points must sit where values genuinely change.

### Principle 2: unobservable differences need dedicated cases

- "eps strictly less than vs less-or-equal": needs a case where `|delta_in|`
  is **exactly equal to** `eps`.
- "global scalar vs per-half mean": needs **asymmetric** data, plus an explicit
  assertion that the two candidates differ — otherwise the case has no
  discriminating power.
- "default eps 1e-10 vs 1e-6": needs a case that **omits `eps`** and whose
  `|delta_in|` falls between the two.

### Principle 3: do not validate the same semantics at several layers

The "duplicate into two copies" semantics of `delta_in` / `delta_out` shows up in
all four modes (`chunk_diffs`, `nonlinear`, `softmax`, `maxpool`). The judge
tests each layer's **own** distinctive behaviour exactly once, instead of
measuring "the length doubles" four times over — otherwise a single wrong length
convention produces the same error reported four times, wasting scoring
information.

---

## 9. Reusable lessons

1. **`torch.cat(2 * [x])` doubles the length** — when reading official code and
   seeing `cat(2 * [...])`, immediately note "output length = 2 × input length".
2. **Whether the condition in a three-argument `torch.where` is strict depends on
   the source**; a single `=` is an independent scoring point.
3. **`grad_input.numel()` is the element count and `sum()` is a *global* scalar**
   — for any normalization-style operation, first establish whether it is global
   or per-row.
4. **`grad_input[0]` and `grad_input[0][0]` are completely different**: the
   former is a slice (shape preserved), the latter is a scalar.
5. **Official comments can be stale** (the `# updating only the first half` in
   `softmax` contradicts the code) — trust the code.
6. **"It runs" does not mean "the semantics are right"**: the official `maxpool`
   broadcasts the correct shape and yields plausible numbers even when
   `grad_output` has the wrong shape. Such silent errors must be excluded by
   explicit convention assertions in the judge.
7. **Compare in layers**: only when the official / reference / independent oracle
   all converge can a shared failure between the reference and the oracle be
   ruled out.
8. **Test the cases in reverse**: inject known-wrong implementations into the
   reference and confirm the judge catches them — far more reliable than only
   seeing "the reference passes".

---

## 10. Index of the official source locations

| Content | Location |
|---|---|
| `SUPPORTED_NON_LINEAR` mapping table | end of `captum/attr/_core/deep_lift.py` |
| `_compute_diffs` | same file, near the end |
| `nonlinear` | same file |
| `softmax` | same file |
| `maxpool` | same file |
| `maxpool1d` / `maxpool2d` / `maxpool3d` | same file |
| `_forward_pre_hook` (sets `module.input`) | inside the `DeepLift` class |
| `_backward_hook` (calls the rule) | inside the `DeepLift` class |
