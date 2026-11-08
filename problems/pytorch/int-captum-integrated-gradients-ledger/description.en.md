# Captum Integrated Gradients: the Quadrature and Attribution Ledger

> Official source: `pytorch/captum` (main)
> `captum/attr/_utils/approximation_methods.py` · `captum/attr/_core/integrated_gradients.py` · `captum/attr/_utils/common.py` · `captum/attr/_utils/attribution.py`

Integrated Gradients (IG) writes "how much did this input feature contribute to
the final output" as a path integral from a baseline to the input:

```
F(input) - F(baseline) = ∫₀¹ (∇F(baseline + α·(input - baseline))) · (input - baseline) dα
```

The official implementation approximates that integral with a **finite-step
quadrature rule**: it samples a set of path parameters `alphas`, evaluates the
gradient at each of them, weights them by a set of `step_sizes`, sums along the
step axis to obtain `total_grads`, and finally multiplies by
`(input - baseline)` to obtain `attributions`.

This problem splits that pipeline into four **pure numeric ledgers** and
rebuilds the official behaviour ledger by ledger. No forward pass, no automatic
differentiation, no GPU kernel — only "where do the parameters come from, how
is the layout arranged, how is the weighting done, how is the delta
subtracted".

---

## 1. Mode one: `quadrature_params`

> Inputs: `method` (string), `n_steps` (int)
> Returns: `{"step_sizes": [...], "alphas": [...]}`

### 1.1 The official method families

The official `approximation_parameters(method)` recognizes exactly two families:

```python
SUPPORTED_RIEMANN_METHODS = ["riemann_left", "riemann_right",
                             "riemann_middle", "riemann_trapezoid"]
SUPPORTED_METHODS = SUPPORTED_RIEMANN_METHODS + ["gausslegendre"]
```

Any other name must raise `ValueError`.

### 1.2 The four Riemann variants

The two official closures in `riemann_builders`:

```python
def step_sizes(n):
    assert n > 1, "The number of steps has to be larger than one"
    deltas = [1 / n] * n
    if method == Riemann.trapezoid:
        deltas[0] /= 2
        deltas[-1] /= 2
    return deltas

def alphas(n):
    assert n > 1, "The number of steps has to be larger than one"
    if method == Riemann.trapezoid:
        return torch.linspace(0, 1, n).tolist()
    elif method == Riemann.left:
        return torch.linspace(0, 1 - 1 / n, n).tolist()
    elif method == Riemann.middle:
        return torch.linspace(1 / (2 * n), 1 - 1 / (2 * n), n).tolist()
    elif method == Riemann.right:
        return torch.linspace(1 / n, 1, n).tolist()
```

**The four variants use different endpoint conventions for `alphas`** — this is
the group of facts most easily missed:

| method | official linspace | `alphas[0]` | `alphas[n-1]` | has 0 | has 1 |
|---|---|---|---|---|---|
| `riemann_left` | `linspace(0, 1-1/n, n)` | `0` | `(n-1)/n` | yes | no |
| `riemann_right` | `linspace(1/n, 1, n)` | `1/n` | `1` | no | yes |
| `riemann_middle` | `linspace(1/(2n), 1-1/(2n), n)` | `1/(2n)` | `1-1/(2n)` | no | no |
| `riemann_trapezoid` | `linspace(0, 1, n)` | `0` | `1` | yes | yes |

Expanding the mathematical definition of `torch.linspace` (step size
`(end-start)/(n-1)`), all four `alphas` sequences are **exact rational
numbers**:

```
left      alphas[i] = i/n          (i = 0..n-1)
right     alphas[i] = (i+1)/n
middle    alphas[i] = (2i+1)/(2n)
trapezoid alphas[i] = i/(n-1)
```

And likewise for `step_sizes`:

```
left / right / middle :  step_sizes = [1/n] * n
trapezoid             :  step_sizes = [1/(2n)] + [1/n]*(n-2) + [1/(2n)]
```

> ⚠️ **Official quirk**: only `trapezoid` halves its first and last step, so
> its `step_sizes` sum to `1 - 1/n`, **not 1**. The other three variants sum
> to 1.

### 1.3 Gauss-Legendre

The official `gauss_legendre_builders` follows a **completely different
semantics**:

```python
def step_sizes(n):
    assert n > 0, "The number of steps has to be larger than zero"
    return list(0.5 * np.polynomial.legendre.leggauss(n)[1])   # ← quadrature weights

def alphas(n):
    assert n > 0, "The number of steps has to be larger than zero"
    return list(0.5 * (1 + np.polynomial.legendre.leggauss(n)[0]))  # ← nodes
```

Two key differences:

1. **`step_sizes` are not uniform steps but Legendre quadrature weights**
   (scaled to an interval of length 1). The weights sum to exactly 1, but no
   two weights are equal.
2. **`alphas` are not evenly spaced points but the quadrature nodes** scaled
   from `[-1, 1]` to `[0, 1]`: `alphas[i] = 0.5 * (1 + node[i])`, with the
   nodes in ascending order.

The assertion boundary changes too: Riemann requires `n > 1`, while
Gauss-Legendre requires `n > 0` (`n = 1` is legal).

### 1.4 The convention used in this problem (important)

The official `torch.linspace` runs in **float32** by default, which introduces
tail differences of order 1e-8 (example: for `riemann_trapezoid` with `n = 4`
the second alpha is `0.3333333432674408` in float32, while the exact value is
`1/3 = 0.3333333333333333`).

To keep the ledger decidable, this problem pins the convention:

- **The four Riemann variants**: `step_sizes` and `alphas` are computed as
  **exact rational numbers** (via `fractions.Fraction`), then converted to
  `float` on output (Fraction → float is correctly rounded and therefore
  uniquely determined). These results are judged by **bit-exact equality**.
- **Gauss-Legendre**: the nodes are irrational roots of Legendre polynomials
  and cannot be expressed as rationals, so they are judged with a **relative
  tolerance of 1e-12**. The solver is free to use any method (a pure-Python
  Newton iteration suffices; measured against numpy `leggauss` at `n = 50` the
  node difference is ≤ 1.11e-16 and the weight difference ≤ 2.67e-15).

### 1.5 Invalid inputs

| situation | behaviour |
|---|---|
| `method` not in `SUPPORTED_METHODS` | raise `ValueError` |
| Riemann variant with `n_steps <= 1` | raise `ValueError` |
| `gausslegendre` with `n_steps <= 0` | raise `ValueError` |

---

## 2. Mode two: `scaled_features`

> Inputs: `input` (`[B][F]`), `baseline` (`[B][F]`), `method`, `n_steps`
> Returns: `{"scaled_features": [S*B][F]}`

The first step of the official `IntegratedGradients._attribute`:

```python
scaled_features_tpl = tuple(
    torch.cat(
        [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
    ).requires_grad_()
    for input, baseline in zip(inputs, baselines)
)
```

That is: for each alpha, compute a whole batch of
`baseline + alpha * (input - baseline)`, then concatenate along `dim=0`. The
resulting shape is `(S*B, F)`.

> ⚠️ **The layout is step-major, not batch-major**.
> Row `s*B + b` belongs to **sample `b` under the `s`-th alpha**.
>
> Example: with `B = 2, S = 2`:
> ```
> row 0 = alpha0 × sample0        row 1 = alpha0 × sample1
> row 2 = alpha1 × sample0        row 3 = alpha1 × sample1
> ```
> Laying it out batch-major (row 0 = alpha0×sample0, row 1 = alpha1×sample0, …)
> shifts every subsequent weighted aggregation.

Only the four Riemann variants participate in this mode's judgement (the
Gauss-Legendre alpha is irrational, and mixing it with the mode-one tolerance
convention would destroy the uniqueness of the ledger). This mode is computed
entirely with exact rational arithmetic.

---

## 3. Mode three: `ig_aggregate`

> Inputs: `input` (`[B][F]`), `baseline` (`[B][F]`), `grads` (`[S*B][F]`),
> `method`, `n_steps`, `multiply_by_inputs` (default `True`)
> Returns: `{"total_grads": [B][F], "attributions": [B][F]}`

`grads` is given in the **official post-flatten layout**: length `S*B`, where
row `s*B + b` is the gradient of sample `b` under the `s`-th alpha (same order
as the `scaled_features` of mode two).

The official aggregation block:

```python
scaled_grads = [
    grad.contiguous().view(n_steps, -1)
    * torch.tensor(step_sizes).float().view(n_steps, 1).to(grad.device)
    for grad in grads
]

total_grads = tuple(
    _reshape_and_sum(scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:])
    for (scaled_grad, grad) in zip(scaled_grads, grads)
)

if self.multiplies_by_inputs:
    attributions = tuple(
        total_grad * (input - baseline)
        for total_grad, input, baseline in zip(total_grads, inputs, baselines)
    )
else:
    attributions = total_grads
```

where the official `_reshape_and_sum` is:

```python
def _reshape_and_sum(tensor_input, num_steps, num_examples, layer_size):
    tensor_input = tensor_input.reshape((num_steps, num_examples) + layer_size)
    return torch.sum(tensor_input, dim=0)
```

### The three ledger steps

1. **Weighting**: every gradient row in segment `s` is multiplied by
   `step_sizes[s]`.
2. **Sum along the step axis**:
   `total_grads[b][f] = Σ_s scaled_grads[s*B + b][f]`. Note that the
   reduction runs along the **step** axis (`dim=0`), not the batch axis.
3. **Multiply by the input difference**: with `multiply_by_inputs=True`,
   `attributions[b][f] = total_grads[b][f] * (input[b][f] - baseline[b][f])`;
   with `False`, `attributions == total_grads`.

> ⚠️ The official `torch.tensor(step_sizes).float()` demotes the step sizes to
> float32 before multiplying. This problem pins the exact rational convention
> and does not chase that tail difference (same as in mode one).

Only the four Riemann variants participate in this mode's judgement.

---

## 4. Mode four: `ig_completeness`

> Inputs: `attributions` (`[B][F]`), `start_out` (`[B][K]`), `end_out` (`[B][K]`)
> Returns: `{"attr_sum": [B], "start_out_sum": [B], "end_out_sum": [B], "delta": [B]}`

The **completeness axiom** of IG states that the sum of the attributions should
equal the difference of the function's output between the end and the start
point. The official `Attribution.compute_convergence_delta` turns this into a
residual:

```python
row_sums = [_sum_rows(attribution) for attribution in attributions]
attr_sum = torch.stack([cast(Tensor, sum(row_sum)) for row_sum in zip(*row_sums)])
_delta = attr_sum - (end_out_sum - start_out_sum)
```

where the official `_sum_rows` is `input.reshape(input.shape[0], -1).sum(1)`,
i.e. it flattens each sample's row and sums it.

With a single input tensor:

```
attr_sum[b]      = Σ_f attributions[b][f]
end_out_sum[b]   = Σ_k end_out[b][k]
start_out_sum[b] = Σ_k start_out[b][k]
delta[b]         = attr_sum[b] - (end_out_sum[b] - start_out_sum[b])
```

> ⚠️ **Sign order**: `attr_sum` comes first, `end_out_sum - start_out_sum`
> second. Reversing them negates the whole delta, and samples where
> completeness holds (delta = 0) would still pass — so use non-zero samples to
> pin this point down.

> ⚠️ `zip(*row_sums)` zips along the **sample** axis: with multiple input
> tensors, `attr_sum[b]` is the **sum** of each tensor's row sum for sample
> `b`. This problem supplies a single input tensor, so it is simply the sum of
> all attributions of that sample.

This mode is independent of `method` and is computed entirely with exact
rational arithmetic.

---

## 5. Checklist of common mistakes

| # | pitfall | correct behaviour |
|---|---|---|
| 1 | assuming the `trapezoid` step sizes sum to 1 | they sum to `1 - 1/n` (ends halved) |
| 2 | giving all four variants the same `alphas` endpoints | follow the table in §1.2 |
| 3 | writing `middle`'s alpha as `i/n` | it is `(2i+1)/(2n)` |
| 4 | writing `trapezoid`'s alpha as `i/n` | it is `i/(n-1)` |
| 5 | writing Gauss-Legendre `step_sizes` as uniform `1/n` | they are half the Legendre quadrature weights |
| 6 | writing Gauss-Legendre `alphas` as evenly spaced points | they are the scaled quadrature nodes |
| 7 | asserting `n > 1` for Gauss-Legendre too | it is `n > 0` |
| 8 | laying out `scaled_features` batch-major | it is step-major |
| 9 | interpreting `grads` batch-major | it is step-major (row `s*B+b`) |
| 10 | summing along the batch axis | sum along the step axis |
| 11 | ignoring the `multiply_by_inputs` switch | with `False`, `attributions == total_grads` |
| 12 | writing delta as `(end - start) - attr_sum` | it is `attr_sum - (end - start)` |
| 13 | assuming delta = 0 proves completeness | completeness gives delta = 0, but so does the negated form; use non-zero samples |
| 14 | using the float32 `linspace` values | the Riemann branches use exact rationals here |
