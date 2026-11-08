# Captum Source Walkthrough: Kernel SHAP Sampling and Evaluation-Count Ledger

> Upstream repository: `pytorch/captum` (branch `main`)
> Files involved: `captum/attr/_core/kernel_shap.py`, `captum/attr/_core/shapley_value.py`

## 1. Where These Rules Live in Captum

Captum's attribution methods split into two families: **white-box propagation**
(Integrated Gradients, DeepLIFT, Layer Conductance and the like, which push
gradients backwards from intermediate activations) and **black-box sampling**
(LIME, Kernel SHAP, Shapley Values and the like, which rely on repeated forward
passes plus a surrogate model fit).

Round 847 covered DeepLIFT (white-box multiplicative propagation); round 849
covers Kernel SHAP (black-box sampling). The two mechanisms do not overlap at
all:

| | DeepLIFT (847) | Kernel SHAP (849) |
|---|---|---|
| Core | per-layer propagation rules (`nonlinear` / `softmax` / `maxpool`) | sampling distribution + similarity weights + surrogate fit |
| Cost | linear in input size | tied to `n_samples` |
| Key quirks | eps strictly smaller, global normalization, unpool overwrite | `probs[0]=0` manual patch, float32 precision, strict `>` drops ties |

Kernel SHAP is a **specialized configuration of the LIME framework**. LIME owns
the whole pipeline:

```
sampling (perturb_func) -> evaluation (forward_func) -> similarity (similarity_func)
      -> assemble TensorDataset -> interpretable_model.fit() -> representation()
```

Kernel SHAP swaps out three of those parts (official `KernelShap.__init__`):

```python
Lime.__init__(
    self,
    forward_func,
    interpretable_model=SkLearnLinearRegression(),
    similarity_func=self.kernel_shap_similarity_kernel,
    perturb_func=self.kernel_shap_perturb_generator,
)
self.inf_weight = 1000000.0
```

`captum/attr/_core/shapley_value.py` is where the other two routes live (the
sampling variant `ShapleyValueSampling` and the exact variant `ShapleyValues`).

---

## 2. `probs`: The Sampling Distribution of the Shapley Kernel

Official code (tail of `kernel_shap.py::KernelShap.attribute`):

```python
num_features_list = torch.arange(num_interp_features, dtype=torch.float)
denom = num_features_list * (num_interp_features - num_features_list)
probs = (num_interp_features - 1) / denom
probs[0] = 0.0
return self._attribute_kwargs(
    ...,
    num_select_distribution=Categorical(probs),
    ...
)
```

### ★ Point 2.1: The range of k is `0 .. M-1` (M excluded)

`torch.arange(M)` produces `0, 1, ..., M-1`. Therefore:

- **k=0** has a slot (but its probability is patched to zero);
- **k=M is out of range** => the "select everything" endpoint is **never sampled
  at random**.

Measured (M=5): the set of sampled k values = `{1, 2, 3, 4}`.

### ★ Point 2.2: `probs[0] = 0.0` Is a **Manual Erasure of inf**

At k=0, `denom = 0 * M = 0` => `(M-1)/0 = inf`. The official code covers it
with a single line, `probs[0] = 0.0`.

This is not the "mathematically correct" approach -- theoretically the k=0
Shapley kernel weight is infinite as well (symmetric to k=M). The official code
chooses to **drop the k=0 sample entirely** rather than give it a finite weight.

### ★ Point 2.3: When k = M-1, `probs` Is Exactly **1.0**

`denom = (M-1) * (M - (M-1)) = (M-1) * 1 = M-1`
=> `probs = (M-1)/(M-1) = 1.0`.

Measured (M=5): `probs = [0.0, 1.0, 0.6667, 0.6667, 1.0]`.
Note that k=1 and k=M-1 both carry weight 1.0 (symmetric), while the middle k
values are smaller -- that is the shape of the Shapley kernel.

### ★ Point 2.4: `probs` Is **Not Normalized**

For M=5, `sum(probs) = 10/3 ~= 3.3333`, not 1.
For M=8 it is `4.5375`.

What actually takes effect is the `probs / probs.sum()` computed inside
`Categorical(probs)`. Measured normalized result for M=8:

```
[0.0, 0.220386, 0.128558, 0.102847, 0.096419,
 0.102847, 0.128558, 0.220386]
```

### ★ Point 2.5: The Official Path Is **float32** All the Way

`torch.arange(num_interp_features, dtype=torch.float)` -- note it is
`torch.float` (**float32**), not float64.

So the official `probs` carries float32 rounding error. Measured at M=16: the
maximum difference between the official float32 value and the exact rational
value is `4.257e-08`.

### ★ Point 2.6: M=1 Crashes Immediately

At M=1, `probs = [0.0]` (denominator 0, patched to 0), `sum = 0`
=> `Categorical` normalizes to **NaN** => raises:

```
ValueError: Expected parameter probs (Tensor of shape (1,)) of distribution
Categorical(probs: tensor([nan])) to satisfy the constraint Simplex(),
but found invalid values: tensor([nan])
```

**Official Kernel SHAP simply crashes when there is only one interpretable
feature.**

---

## 3. `kernel_shap_similarity_kernel`: Endpoint Weights

```python
def kernel_shap_similarity_kernel(self, _, __, interpretable_sample, **kwargs):
    assert "num_interp_features" in kwargs, ...
    num_selected_features = int(interpretable_sample.sum(dim=1).item())
    num_features = kwargs["num_interp_features"]
    if num_selected_features == 0 or num_selected_features == num_features:
        similarities = self.inf_weight
    else:
        similarities = 1.0
    return torch.tensor([similarities])
```

### ★ Point 3.1: The Endpoint Weight Is `1000000.0`, **Not Infinity**

The official comment is explicit:

```
# weight should be theoretically infinite when
# num_selected_features = 0 or num_features
# enforcing that trained linear model must satisfy
# end-point criteria. In practice, it is sufficient to
# make this weight substantially larger so setting this
# weight to 1000000 (all other weights are 1).
```

This weight enters the surrogate model's **weighted least squares**:
the outputs at the endpoints (original input / baseline) must be fitted exactly
by the linear model, so they get an overwhelming weight.

### ★ Point 3.2: The Criterion Is the "Number Selected", Position-agnostic

Measured: at M=3, the weight is `1000000.0` for both k=0 and k=3,
and `1.0` for both k=1 and k=2.

---

## 4. `kernel_shap_perturb_generator`: Perturbation Generation

```python
def kernel_shap_perturb_generator(self, original_inp, **kwargs):
    ...
    num_features = kwargs["num_interp_features"]
    yield torch.ones(1, num_features, device=device, dtype=torch.long)
    yield torch.zeros(1, num_features, device=device, dtype=torch.long)
    while True:
        num_selected_features = kwargs["num_select_distribution"].sample()
        rand_vals = torch.randn(1, num_features)
        threshold = torch.kthvalue(
            rand_vals, num_features - num_selected_features
        ).values.item()
        yield (rand_vals > threshold).to(device=device).long()
```

### ★ Point 4.1: The First Two Samples Are **Fixed Endpoints**

`yield ones` (original input) and `yield zeros` (baseline) are produced
**unconditionally**, consuming no random budget. Measured: the first sample is
`[[1,1,1,1]]`, the second is `[[0,0,0,0]]`, and the dtype is `torch.int64`.

=> **With `n_samples = 2` there are only the endpoints and not a single random
sample.**

### ★ Point 4.2: The Threshold Is the **(M-k)-th Smallest**

`torch.kthvalue(rand_vals, num_features - num_selected_features)`.
The k of `kthvalue` is the **1-based** "which smallest value".
So passing `M-k` picks the **(M-k)-th smallest** value -- exactly the dividing
point needed to select the largest k.

### ★ Point 4.3: The Comparison Is **Strictly Greater** => Ties Select Fewer

`rand_vals > threshold`. Without ties this yields exactly k elements.

**But when there are ties**: `rand_vals = [5, 5, 5, 1]`, k=2, M=4
=> threshold = 2nd smallest = 5.0 => strictly greater than 5.0 matches
**nothing**
=> **0** elements are selected instead of 2.

Measured side by side (`rand_vals` with 6 elements, k=2):

| Comparison | Number selected |
|---|---|
| `rand_vals > threshold` | 2 |
| `rand_vals >= threshold` | 3 |

### ★ Point 4.4: Official Raises When `k = M`

`kthvalue(rand_vals, 0)` has an out-of-range k:

```
RuntimeError: kthvalue(): selected number k out of range for dimension 1
```

Measured: both k=0 and k=M+1 raise this error.

(In practice k=M never happens, because `p(M) = 0`. But it is part of the
**domain constraint**.)

---

## 5. `_get_n_evaluations`: Two Evaluation-Count Conventions

```python
# shapley_value.py::ShapleyValueSampling
def _get_n_evaluations(self, total_features, n_samples, perturbations_per_eval):
    return math.ceil(total_features / perturbations_per_eval) * n_samples

# shapley_value.py::ShapleyValues
def _get_n_evaluations(self, total_features, n_samples, perturbations_per_eval):
    return math.ceil(total_features / perturbations_per_eval) * math.factorial(
        total_features
    )
```

### ★ Point 5.1: The Sampling Variant = `ceil(total / ppe) * n_samples`

Measured (total=5, n_samples=4, ppe=2) -> `ceil(5/2) * 4 = 3 * 4 = 12`.

The `ceil` comes **first**, the `* n_samples` comes **after**. Writing
`ceil(total * n_samples / ppe)` gives `ceil(10) = 10` at total=5/n=4/ppe=2,
which differs from the correct 12.

### ★ Point 5.2: The Exact Variant **Ignores `n_samples` Completely**

`ShapleyValues` enumerates all `M!` permutations, so the multiplicative factor
is `factorial(total)`. Measured:

| total | n_samples | result |
|---|---|---|
| 3 | 4 | 18 |
| 3 | 25 | 18 |
| 5 | 4 | 600 |
| 5 | 25 | 600 |

**`n_samples` makes no difference whatsoever.**

### ★ Point 5.3: `ppe` Appears Only Inside the Denominator of `ceil`

`perturbations_per_eval` is "how many perturbations one forward pass evaluates
in parallel"; it only affects how many batches are needed, not the size of each
batch. So it does **not** take part in the multiplication.

---

## 6. `_all_perm_generator`: `num_samples` Is Completely Ignored

```python
def _all_perm_generator(num_features: int, num_samples: int) -> Iterable[Sequence[int]]:
    for perm in itertools.permutations(range(num_features)):
        yield perm
```

### ★ Point 6.1: `num_samples` Is a Dead Parameter

Measured: `_all_perm_generator(3, 10)` yields **6** items (= 3!),
`_all_perm_generator(5, 3)` yields **120** items (= 5!) -- the `num_samples`
passed in has absolutely no effect.

This is a typical "the signature was kept but the implementation never uses
it" trace: upstream passes `num_samples` in to keep the signature consistent
with `_perm_generator` (the random variant, which **does** really use
`num_samples`).

### ★ Point 6.2: The Order Is the Lexicographic Order of `itertools.permutations`

The first item is the ascending `[0, 1, ..., M-1]`.

---

## 7. Three Reproduction Pitfalls in the Official Source

### Pitfall 1: `probs` Is Unnormalized and Easy to "Normalize by Hand"

The official `probs` sums to `10/3` (M=5), not 1.
If you "helpfully" normalize it inside your implementation, the later
`Categorical(probs)` will **normalize a second time** -- the final result
happens to be the same (normalization is idempotent), but the `probs` output
then disagrees with the official one.

Verification trick: assert `total == 10/3` directly (M=5).

### Pitfall 2: float32 vs Exact Rationals

The official path is float32 throughout while the reference solution uses
Fraction. At M=16 the difference is `4.3e-08`.
The judge must offer **both** conventions (`probs` exact / `probs_f32` float32),
otherwise you either fail against the official values or fail against
submissions that compute exact rationals as the statement demands.

### Pitfall 3: The k of `kthvalue` Is 1-based, and 0 Is Out of Range

Writing `sorted(vals)[M-k-1]` (0-based) instead of `sorted(vals)[M-k]`.
And when `k = M`, `M-k = 0`, so `sorted(vals)[-1]` **silently picks the maximum**
without raising -- whereas the official code raises `RuntimeError`. This is a
classic case of "a Python negative index masks an official error".

---

## 8. The Three-Layer Verification Structure

1. **Official layer**: from the upstream `kernel_shap.py` / `shapley_value.py`,
   use `ast.get_source_segment` to extract **verbatim** the
   `kernel_shap_similarity_kernel` / `kernel_shap_perturb_generator` /
   `_all_perm_generator` / `_perm_generator` / the two `_get_n_evaluations`,
   stripping only the whole-`captum` package dependency (the evaluation
   environment has no `sklearn` / `tqdm`).
2. **Reference layer**: `/tmp/captum849_solution_staged.py` (pure Python +
   `Fraction`).
3. **Oracle layer**: a pure-float implementation rewritten independently of the
   reference solution, straight from the official formulas.

Measured over **132 cases**:
- `official-vs-ref <= 5.960e-08` (float32 precision, carried by the `probs_f32`
  convention);
- `oracle-vs-ref = 0.000e+00` (the exact rational convention agrees exactly).

**Why two precision conventions**: the official code is float32, so if the
statement only demanded exact rationals your answer would not match the
official run; if it only demanded float32, the stricter "exact value"
convention would not be pinned down. Emitting both covers both audiences.

---

## 9. Three Principles for Mutation Testing

The mutation list for this round has 43 entries and ends at **43/43 CAUGHT**.
The pitfalls hit along the way:

1. **Equivalent mutants must be removed on purpose**. Two of the 42 entries in
   the first round survived, and after checking each one they turned out to be
   **truly equivalent**:
   - changing `probs[0] = Fraction(0)` into `pass` -- because the earlier
     `d == 0` branch already appended `Fraction(0)`, so the **values are
     identical**;
   - changing `nf = [f32(k) for k in range(M)]` into `nf = [float(k) ...]` --
     because every later operation is wrapped in `f32()`, so the **precision
     difference of `nf` is fully absorbed**.

   Both are "identical output over the entire legal input domain"; the
   criterion is global equivalence, not "identical on this set of cases".

2. **The validation layer must be split into independent defenses**. The
   `perturb_sample` check, for instance, asserts all of:
   - `sample == oracle`;
   - the "threshold is the k-th largest" candidate **must differ**;
   - the "use `>=` instead of `>`" candidate **must differ**;
   - under ties, `num_selected == 0`.

   With only the first assertion, an implementation that writes `>=` where `>`
   belongs might happen to pass on the tie-free cases.

3. **Unobservable differences need dedicated cases, and the anchor must land
   on an input that really changes.** The only case gap this round appeared
   **at the position of the bidirectional assertion**: at first the "k-th
   largest vs (M-k)-th smallest" bidirectional assertion was written inside
   the loop over every case, but for some data the k-th largest happens to
   equal the (M-k)-th smallest (e.g. `[3,2,1]` with k=2 and M=3 gives 2.0 for
   both), so the assertion **falsely fired**. The fix was to move it out of
   the loop and use a **deliberately asymmetric** dataset `[9,1,7,3,5]` with
   k=2 and M=5 (2nd largest = 7.0, 3rd smallest = 5.0 -- different).

   Lesson: **bidirectional assertions need carefully chosen data too** -- they
   require the two candidates to differ **on that data**, not some universal
   property that holds on all data.

---

## 10. Index of Official Source Locations

| Symbol | Location |
|---|---|
| `KernelShap.__init__` and `inf_weight` | `kernel_shap.py:31-46` |
| probs construction in `KernelShap.attribute` | `kernel_shap.py:276-286` |
| `num_features_list = torch.arange(...)` | `kernel_shap.py:280` |
| `probs = (num_interp_features - 1) / denom` | `kernel_shap.py:284` |
| `probs[0] = 0.0` | `kernel_shap.py:286` |
| `Categorical(probs)` | `kernel_shap.py:296` |
| `kernel_shap_similarity_kernel` | `kernel_shap.py:307-330` |
| `kernel_shap_perturb_generator` | `kernel_shap.py:332-373` |
| `yield torch.ones(...)` / `yield torch.zeros(...)` | `kernel_shap.py:365-366` |
| `torch.kthvalue(rand_vals, num_features - num_selected_features)` | `kernel_shap.py:370-372` |
| `(rand_vals > threshold)` | `kernel_shap.py:373` |
| `_all_perm_generator` | `shapley_value.py:35-37` |
| `_perm_generator` | `shapley_value.py:40-42` |
| `ShapleyValueSampling._get_n_evaluations` | `shapley_value.py:582-584` |
| `ShapleyValues._get_n_evaluations` | `shapley_value.py:875-881` |
