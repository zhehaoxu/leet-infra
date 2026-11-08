# notes: Captum Integrated Gradients Quadrature and Attribution Ledger

> Companion to `description.md`. This file holds the line-by-line official
> source dissection, measured data, and reusable lessons.

---

## 1. Official source dissection

### 1.1 `captum/attr/_utils/approximation_methods.py`

```python
class Riemann(Enum):
    left = 1
    right = 2
    middle = 3
    trapezoid = 4

SUPPORTED_RIEMANN_METHODS = [
    "riemann_left", "riemann_right", "riemann_middle", "riemann_trapezoid",
]
SUPPORTED_METHODS: List[str] = SUPPORTED_RIEMANN_METHODS + ["gausslegendre"]

def approximation_parameters(method):
    if method in SUPPORTED_RIEMANN_METHODS:
        return riemann_builders(method=Riemann[method.split("_")[-1]])
    if method == "gausslegendre":
        return gauss_legendre_builders()
    raise ValueError("Invalid integral approximation method name: {}".format(method))
```

`method.split("_")[-1]` takes the last segment and looks it up in the `Riemann`
enum — so `riemann_trapezoid` → `Riemann.trapezoid`.

### 1.2 The two closures of `riemann_builders` (source of §1.2 above)

```python
def step_sizes(n: int) -> List[float]:
    assert n > 1, "The number of steps has to be larger than one"
    deltas = [1 / n] * n
    if method == Riemann.trapezoid:
        deltas[0] /= 2
        deltas[-1] /= 2
    return deltas

def alphas(n: int) -> List[float]:
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

**The key observation (the origin of this problem's convention)**: the step
size of `torch.linspace(start, end, n)` is `(end - start) / (n - 1)`. Plugging
in the four variants:

| variant | start | end | step size |
|---|---|---|---|
| left | `0` | `1-1/n` | `(1-1/n)/(n-1) = 1/n` |
| right | `1/n` | `1` | `(1-1/n)/(n-1) = 1/n` |
| middle | `1/(2n)` | `1-1/(2n)` | `(1-1/n)/(n-1) = 1/n` |
| trapezoid | `0` | `1` | `1/(n-1)` |

Hence the exact rational forms of the four `alphas` sequences:

```
left      i/n
right     (i+1)/n
middle    (2i+1)/(2n)
trapezoid i/(n-1)
```

**All four are exact rational numbers** — which is precisely why this problem
can judge the Riemann branches bit-exactly under an exact-rational convention.

### 1.3 `gauss_legendre_builders`

```python
def step_sizes(n: int) -> List[float]:
    assert n > 0, "The number of steps has to be larger than zero"
    return list(0.5 * np.polynomial.legendre.leggauss(n)[1])

def alphas(n: int) -> List[float]:
    assert n > 0, "The number of steps has to be larger than zero"
    return list(0.5 * (1 + np.polynomial.legendre.leggauss(n)[0]))
```

`leggauss(n)` returns `(nodes, weights)` with the nodes in ascending order and
the weights paired one-to-one.

- `step_sizes = 0.5 * weights`: **quadrature weights**, not uniform steps.
- `alphas = 0.5 * (1 + nodes)`: the **nodes** scaled from `[-1,1]` to `[0,1]`.

Both use `n > 0` (not the Riemann `n > 1`).

### 1.4 `captum/attr/_core/integrated_gradients.py::IntegratedGradients._attribute`

```python
def _attribute(self, inputs, baselines, target, additional_forward_args,
               n_steps=50, method="gausslegendre", step_sizes_and_alphas=None):
    if step_sizes_and_alphas is None:
        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
    else:
        step_sizes, alphas = step_sizes_and_alphas

    # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
    scaled_features_tpl = tuple(
        torch.cat(
            [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
        ).requires_grad_()
        for input, baseline in zip(inputs, baselines)
    )
    ...
    grads = self.gradient_func(forward_fn=self.forward_func,
                               inputs=scaled_features_tpl, target_ind=expanded_target, ...)

    scaled_grads = [
        grad.contiguous().view(n_steps, -1)
        * torch.tensor(step_sizes).float().view(n_steps, 1).to(grad.device)
        for grad in grads
    ]

    total_grads = tuple(
        _reshape_and_sum(scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:])
        for (scaled_grad, grad) in zip(scaled_grads, grads)
    )

    if not self.multiplies_by_inputs:
        attributions = total_grads
    else:
        attributions = tuple(
            total_grad * (input - baseline)
            for total_grad, input, baseline in zip(total_grads, inputs, baselines)
        )
    return attributions
```

Points to note:

1. **`torch.cat(..., dim=0)` is step-major**. Each alpha contributes one
   `(B, F)` block, so the result is `(S*B, F)` and row `s*B + b` belongs to
   sample `b` under the `s`-th alpha.
2. **`grad.contiguous().view(n_steps, -1)`** squeezes `(S*B, F)` into
   `(S, B*F)`; multiplying by `step_sizes.view(S, 1)` broadcasts row-wise, so
   every element of row `s` is multiplied by `step_sizes[s]`.
3. **`torch.tensor(step_sizes).float()`** demotes the step sizes to float32
   before the multiplication. This problem does not chase that tail difference.
4. **The official defaults are `method="gausslegendre"` and `n_steps=50`.**

### 1.5 `captum/attr/_utils/common.py::_reshape_and_sum`

```python
def _reshape_and_sum(tensor_input: Tensor, num_steps: int,
                      num_examples: int, layer_size: Tuple[int, ...]) -> Tensor:
    tensor_input = tensor_input.reshape((num_steps, num_examples) + layer_size)
    return torch.sum(tensor_input, dim=0)
```

`reshape((S, B) + (F,))` followed by `sum(dim=0)` — the reduction runs along
the **step** axis.

### 1.6 `captum/attr/_utils/attribution.py::Attribution.compute_convergence_delta`

```python
start_out_sum = _sum_rows(start_out_eval)
end_out_sum   = _sum_rows(end_out_eval)
row_sums = [_sum_rows(attribution) for attribution in attributions]
attr_sum = torch.stack([cast(Tensor, sum(row_sum)) for row_sum in zip(*row_sums)])
_delta = attr_sum - (end_out_sum - start_out_sum)
```

with `captum/attr/_utils/common.py::_sum_rows`:

```python
def _sum_rows(input: Tensor) -> Tensor:
    return input.reshape(input.shape[0], -1).sum(1)
```

In `zip(*row_sums)`, `row_sums` is a list holding one `(B,)` row-sum vector per
input tensor, and the zip iterates the **sample** axis — so `attr_sum[b]` is
the **sum** of each tensor's row sum at sample `b`.

---

## 2. Measured data

### 2.1 The float32 tail difference of `torch.linspace`

```
torch.linspace(0, 1, 4) = [0.0, 0.3333333432674408, 0.6666666269302368, 1.0]
dtype: torch.float32
```

The exact values should be `[0, 1/3, 2/3, 1]`. The discrepancy of the second
and third entries is of order 1e-8 — this is why the Riemann branches are
pinned to an exact-rational convention here.

### 2.2 The five methods at `n = 4`

```
left        deltas=[0.25, 0.25, 0.25, 0.25]           sum(d)=1.0
left        alphas=[0.0, 0.25, 0.5, 0.75]
right       deltas=[0.25, 0.25, 0.25, 0.25]           sum(d)=1.0
right       alphas=[0.25, 0.5, 0.75, 1.0]
middle      deltas=[0.25, 0.25, 0.25, 0.25]           sum(d)=1.0
middle      alphas=[0.125, 0.375, 0.625, 0.875]
trapezoid   deltas=[0.125, 0.25, 0.25, 0.125]         sum(d)=0.75   ← quirk
trapezoid   alphas=[0.0, 0.3333333432674408, 0.6666666269302368, 1.0]
gausslegendre deltas=[0.17392742256872673, 0.32607257743127327,
                      0.32607257743127327, 0.17392742256872673]  sum(d)=1.0
gausslegendre alphas=[0.06943184420297371, 0.33000947820757187,
                      0.6699905217924281,  0.9305681557970262]
```

Note that the Gauss-Legendre `deltas` are **clearly not uniform**
(0.1739 / 0.3261 alternating) — they are distinguishable from the Riemann
families at a glance.

### 2.3 A pure-Python Newton iteration reproduces numpy `leggauss`

Using the Legendre recurrence `P_k = ((2k-1)x·P_{k-1} - (k-1)P_{k-2})/k`,
the derivative `P'_n = n(x·P_n - P_{n-1})/(x²-1)`, the weight
`w = 2/((1-x²)·P'_n²)`, and the Chebyshev-type initial guess
`x = cos(π(i+0.75)/(n+0.5))`:

```
n= 1 max|dx|=0.000e+00 max|dw|=0.000e+00
n= 2 max|dx|=0.000e+00 max|dw|=2.220e-16
n= 3 max|dx|=0.000e+00 max|dw|=2.220e-16
n= 4 max|dx|=5.551e-17 max|dw|=4.441e-16
n= 6 max|dx|=1.110e-16 max|dw|=1.110e-16
n= 8 max|dx|=1.110e-16 max|dw|=3.469e-16
n=12 max|dx|=2.776e-17 max|dw|=5.551e-16
n=20 max|dx|=1.110e-16 max|dw|=9.298e-16
n=50 max|dx|=1.110e-16 max|dw|=2.670e-15
```

At `n = 50` the node difference is still only 1.11e-16 — pure Python is
adequate as a replacement for numpy.

> Caution: after the Newton iteration you **must re-sort by ascending node**
> before aligning with the official output (the official `leggauss` returns
> ascending nodes, whereas the initial guesses above are descending). Using
> them as-is reverses the whole table.

### 2.4 Structural properties of Gauss-Legendre

- The weights sum to 2 (by the `leggauss` definition), so after the halving
  `step_sizes` sums to 1.
- The nodes sum to 0, hence `alphas[i] + alphas[n-1-i] = 1` (symmetric about
  0.5).
- `n = 1`: node = 0, weight = 2 → `step_sizes = [1.0]`, `alphas = [0.5]`.
- `n = 2`: node = ±1/√3, weight = [1, 1] → `step_sizes = [0.5, 0.5]`,
  `alphas = [0.5 ∓ 1/(2√3)]`.

---

## 3. Verification strategy (three layers)

### L1 Semantic layer: bit-exact agreement with the official mathematical definition

For the Riemann branches, recompute the official `linspace` mathematical
definition in float64 (`start + i*(end-start)/(n-1)`) via
`torch.linspace(dtype=torch.float64)` and compare with the solver's
`Fraction → float` values; the difference must be ≤ 1e-15 (measured: bit-exact).

For the Gauss-Legendre branch, compare against the official
`approximation_parameters("gausslegendre")` which calls numpy `leggauss`
directly; the difference must be ≤ 1e-12.

### L2 Tail-difference layer: locating the official float32 deviation

For the Riemann branches, compare element by element against the official
`torch.linspace` (float32) output. Every component deviation must be ≤ 1e-7
(float32 magnitude); otherwise the check reports the **offending index**. This
guarantees that "our exact value" differs from "the official float32 value" by
tail noise only, never by semantics.

### L3 Aggregation layer: cross-check against the official tensor logic

Transcribe the official `scaled_features` / `view(S,-1)` / `_reshape_and_sum` /
`multiply_by_inputs` tensor code (in float64) and compare with the solver
(difference ≤ 1e-12); on the completeness side, transcribe the official
`compute_convergence_delta` `attr_sum` / `delta` sign order and compare.

**Verification scale**: 4 Riemann variants × 9 values of `n` plus 12
Gauss-Legendre values of `n`, plus 160 aggregation cases (each with both
values of `multiply_by_inputs`), plus 120 completeness cases = **328 groups**.

---

## 4. Test design notes

### 4.1 Why the Riemann branches use an exact-rational convention instead of tracking the official float32

The official `torch.linspace` runs in float32, with tail differences of order
1e-8. Judging bit-exactly against the official values would force the solver
to **replicate float32 rounding**, which is neither natural nor stable.
Loosening to a 1e-6 tolerance, on the other hand, lets real errors slip
through: writing `middle`'s alpha as `i/n` instead of `(2i+1)/(2n)` is a
deviation of `1/(2n)` (still caught at `n = 50`, but only barely), and
writing `trapezoid`'s alpha as `i/n` instead of `i/(n-1)` deviates by only
~1/n² at large `n` and would slip through.

The exact-rational convention solves three things at once:
1. `trapezoid`'s `i/(n-1)` vs `i/n` is detectable at any `n`;
2. `middle`'s `(2i+1)/(2n)` vs `i/n` is detectable;
3. the step-size sum `1-1/n` vs `1` is detectable.

### 4.2 Why Gauss-Legendre uses a tolerance instead of exact equality

Legendre nodes are irrational and have **no** exact rational representation.
That branch can only use a tolerance (1e-12). Fortunately the
Gauss-Legendre branch appears only in mode one (the quadrature parameters),
and its semantic differences (weights vs uniform steps, nodes vs evenly spaced
points) are **magnitude-level**, so a 1e-12 tolerance is easily enough to
separate them from any equivalence error.

### 4.3 Why modes two, three and four exclude Gauss-Legendre

If `scaled_features` (mode two) and `attributions` (mode three) admitted an
irrational alpha, the ledger would no longer have a unique solution (the same
floating-point input could correspond to several "correct" results). Isolating
Gauss-Legendre in mode one keeps modes two/three/four on "exact rational →
bit-exact", which maximizes the strength of the judgement.

### 4.4 Division of labour between anchors and grids

- **Hand-computed anchors** pin the official quirks and sign conventions (the
  step-size sum `1-1/n`, the four endpoint conventions, the step-major
  layout, the `delta` sign, the `multiply_by_inputs` switch).
- **Grids** pin the combinatorial explosion (the interaction of `n × B × F`).
- **Non-zero delta samples** pin the sign of `attr_sum - (end - start)` (the
  negated form also yields delta = 0 on samples where completeness holds, so
  dedicated non-zero samples are required).

---

## 5. Reusable lessons (10 items)

1. **When the upstream is floating-point and contains ties, a pointwise oracle
   does not hold** (the lesson of problem 841, reproduced here). This problem
   goes one step further: the Riemann float error comes **entirely from the
   float32 of `torch.linspace`**, not from the algorithm itself. Falling back
   to exact rational forms (here all of them are `i/n`-shaped) sidesteps the
   tail difference without losing semantics.
2. **First ask whether the upstream mathematical object is rational.** Rational
   → exact convention, judged bit-exactly. Irrational → tolerance, and keep
   the irrational branch **isolated in its own mode** so it does not pollute
   the uniqueness of the other modes.
3. **`torch.linspace` defaults to float32.** Any problem sourced from it must
   state its precision convention explicitly, otherwise the cross-check runs
   into the "the official value differs from the mathematical value by 1e-8"
   wall.
4. **Official quirks need their own hand-computed anchor.** The halved end
   steps of `trapezoid` (sum `1-1/n`) are the easiest thing to get wrong and
   the easiest thing for a grid to miss.
5. **A multi-block `torch.cat(..., dim=0)` is step-major** (one block per
   alpha), not batch-major. This "multi-block concatenation" layout is a
   high-frequency pitfall.
6. **`reshape((S,B)+shape).sum(dim=0)` reduces along the step axis.** Whenever
   a `view(n_steps, -1)` is paired with a reduction, the axis is almost always
   the step axis.
7. **`zip(*row_sums)` zips along the sample axis**, so the row sums of
   multiple input tensors are summed **per sample**, not per tensor.
8. **Sign order needs its own non-zero sample.** Writing the completeness
   residual as `a - (b - c)` versus `(b - c) - a` cannot be distinguished on
   samples where completeness happens to hold.
9. **Pure Python can replace numpy's `leggauss`**: the Legendre recurrence
   plus a Newton iteration, with a node difference of 1.11e-16 at `n = 50`.
   **Remember to re-sort by ascending node** (the Newton initial guesses are
   descending). This keeps the solution free of third-party dependencies (the
   in-repo discipline: `grep -rl "^import numpy" problems/pytorch/*/solution.py`
   hits 0 files).
10. **Assertion boundaries can be a pitfall too**: Riemann is `n > 1` while
    Gauss-Legendre is `n > 0`. Two different assertions in the same library
    are easy to half-remember.
