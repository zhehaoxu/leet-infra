# Captum Kernel SHAP Sampling and Evaluation-Count Ledger

## Background

Attribution methods answer one question: **which inputs contributed to this
output?**

Shapley values give a beautiful but expensive answer: turn each feature "on" and
"off", enumerate every ordering, and measure how much each feature contributes
on average. The math is unimpeachable; the cost is `M!` forward evaluations —
3.6 million at M=10.

Kernel SHAP (arXiv **1705.07874**) offers an engineering escape hatch:
**sample instead of enumerate**. Its insight is that in Shapley's weighted
regression, each sample's weight happens to be a closed-form function (the
Shapley kernel), so you can draw a modest number of samples according to that
weight and solve a small weighted least-squares problem.

Captum implements this in `captum/attr/_core/kernel_shap.py`. It is a
specialized configuration of the **LIME framework** — LIME owns the
"sampling → evaluation → fit a surrogate model" pipeline, and Kernel SHAP swaps
out three of its parts:

| Part | LIME default | Kernel SHAP |
|---|---|---|
| sampling distribution | independent `bernoulli(0.5)` per feature | Shapley kernel `p(k) = (M-1)/(k(M-k))` |
| similarity kernel | Gaussian kernel over cosine distance | endpoint `inf_weight` / interior `1.0` |
| surrogate model | `SkLearnLasso(alpha=0.01)` | `SkLearnLinearRegression()` |

Meanwhile `captum/attr/_core/shapley_value.py` provides two more Shapley routes
(the sampling variant `ShapleyValueSampling` and the exact variant
`ShapleyValues`). They share the same "permutation → swap in one feature at a
time → accumulate contributions" mechanism, but their **forward-evaluation
count conventions are completely different**.

Your job is to reproduce the exact numerical semantics of this sampling and
counting machinery in pure Python. There are a dozen or so places where the
official code does something other than what "looks reasonable", and the judge
pins every one of them down.

## Conventions

- `num_interp_features` (call it **M**): the number of interpretable features,
  `>= 1`.
- `interpretable_sample`: a length-M **0/1 vector**; bit i says whether the i-th
  interpretable feature is "selected".
- `rand_vals`: a length-M float vector (the official code uses `torch.randn(1, M)`;
  here it is an explicit input ⇒ fully deterministic).
- All outputs are plain `float` (anything that can be an exact rational goes
  through `fractions.Fraction` internally).
- Illegal inputs raise `ValueError`.

---

## mode 1: `select_distribution`

Reproduce the official sampling-distribution construction:

```python
num_features_list = torch.arange(num_interp_features, dtype=torch.float)
denom = num_features_list * (num_interp_features - num_features_list)
probs = (num_interp_features - 1) / denom
probs[0] = 0.0
# ... num_select_distribution=Categorical(probs)
```

**Signature**

```python
select_distribution(num_interp_features)
    → {"probs": [...], "denom": [...], "total": float,
       "normalized": [...] | None, "sampling_range": [lo, hi],
       "probs_f32": [...], "normalized_f32": [...] | None}
```

**Key points**

1. ★ `num_features_list = arange(M)` ⇒ **k ranges over `0 .. M-1`**,
   **excluding M**. So the "select everything" endpoint (k = M) is **never**
   drawn at random.
2. ★ `denom = k * (M - k)`. **At k=0 the denominator is 0** ⇒
   `(M-1)/0 = inf`, and the official code **manually wipes it out** with
   `probs[0] = 0.0`.
3. ★ **At k = M-1, `denom = (M-1) * 1 = M-1`** ⇒
   `probs = (M-1)/(M-1) = **1.0**`, **not** infinity. In other words the
   "nearly everything" endpoint gets weight 1, whereas theoretically it should
   weigh the same as k=1 (by symmetry).
4. ★ probs are **not normalized**: at M=5 they sum to `10/3`, not 1. The
   distribution that actually takes effect is `probs / probs.sum()`, computed
   inside `Categorical`.
5. ★ **The M=1 degenerate case**: `probs = [0.0]`, the sum is 0 ⇒ normalization
   yields **NaN** ⇒ `Categorical` raises `ValueError` (the Simplex constraint
   fails). **Official Kernel SHAP simply crashes when there is only one
   interpretable feature.** In that case `normalized` and `normalized_f32` must
   both be `None`.
6. ★ The official code runs in **float32** throughout
   (`torch.arange(M, dtype=torch.float)`), so `probs_f32` differs from the exact
   rationals at float32 precision (about `4.3e-08` at M=16). `probs` / `denom`
   / `total` / `normalized` follow the **exact rational** convention, while
   `probs_f32` / `normalized_f32` follow the **official float32** one.
7. `sampling_range` is the range of k values that can actually be **drawn** —
   the index interval where `probs > 0`. For M>=2 that is `[1, M-1]`; for M=1
   it is the empty interval `[0, -1]`.

**Example**

```python
solve("select_distribution", num_interp_features=5)["probs"]
# → [0.0, 1.0, 2/3, 2/3, 1.0]     sum = 10/3
solve("select_distribution", num_interp_features=5)["normalized"]
# → [0.0, 0.3, 0.2, 0.2, 0.3]      sum = 1
solve("select_distribution", num_interp_features=1)["normalized"]
# → None
```

---

## mode 2: `similarity_weight`

Reproduce the official similarity weight:

```python
num_selected_features = int(interpretable_sample.sum(dim=1).item())
num_features = kwargs["num_interp_features"]
if num_selected_features == 0 or num_selected_features == num_features:
    similarities = self.inf_weight
else:
    similarities = 1.0
return torch.tensor([similarities])
```

**Signature**

```python
similarity_weight(interpretable_sample, num_interp_features, inf_weight=1e6)
    → {"weight": float, "num_selected": int, "is_endpoint": bool}
```

**Key points**

1. ★ The endpoint weight (**all 0** or **all 1**) is
   `inf_weight = **1000000.0**`; every other sample gets weight `1.0`.
2. ★ That `1000000.0` is **not** a true infinity. The upstream comment is
   explicit: the endpoint weight should theoretically be infinite, "but in
   practice it is sufficient to make this weight substantially larger so
   setting this weight to 1000000 (all other weights are 1)" — it forces the
   trained linear model to satisfy the **end-point criteria**. Since all other
   weights are 1, this value is also a relative magnitude.
3. The criterion is the **count of selected features**, entirely independent of
   *where* the 1s sit.
4. `inf_weight` may be supplied by the caller (default `1000000.0`).
5. `interpretable_sample` must be a 0/1 vector of length exactly M, otherwise
   `ValueError`.

**Example**

```python
solve("similarity_weight", interpretable_sample=[1, 0, 0],
      num_interp_features=3)["weight"]     # → 1.0
solve("similarity_weight", interpretable_sample=[0, 0, 0],
      num_interp_features=3)["weight"]     # → 1000000.0
```

---

## mode 3: `perturb_sample`

Reproduce the official perturbation-vector generation:

```python
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

**Signature**

```python
perturb_sample(rand_vals, num_selected_features, num_interp_features)
    → {"sample": [...], "threshold": float, "num_selected": int,
       "endpoint": bool}

perturb_sample(kind="endpoints", num_interp_features)
    → {"samples": [[...], [...]], "num_endpoints": 2}

perturb_sample(kind="sequence", rand_vals_list, selected_counts,
               num_interp_features)
    → {"samples": [[...], ...], "num_endpoints": 2, "num_random": int}
```

**Key points**

1. ★ The first two samples are **fixed endpoints**: **all 1** (the original
   input) and **all 0** (the baseline), produced **unconditionally** and not
   drawing on the random budget. So at `n_samples = 2` there are only endpoints
   and no random sample at all.
2. ★ The threshold is the **`(M - k)`-th smallest** value
   (`kthvalue`'s k is the 1-based "which one counting from the smallest"),
   **not** the k-th largest.
3. ★ The comparison is **strictly greater than**, `rand_vals > threshold`.
   Without ties exactly k features are selected; **with ties the code selects
   fewer** — e.g. `rand_vals = [5, 5, 5, 1]`, `k = 2`, `M = 4`: the threshold is
   the 2nd smallest (also 5.0), and nothing is strictly greater than 5.0, so
   **0** features are selected instead of 2.
4. `num_selected_features` must lie in `1 .. M-1`. At `k = M` the official
   `kthvalue(rand_vals, 0)` index is out of range (`kthvalue`'s k is 1-based, so
   0 is illegal) ⇒ it raises ⇒ your implementation should raise `ValueError`.
5. `rand_vals` must have length M.
6. With `kind="sequence"`, `rand_vals_list` and `selected_counts` correspond
   item by item and must have the same length; the output sequence is
   `[all 1, all 0] + ` the per-item random samples.

**Example**

```python
solve("perturb_sample", rand_vals=[3.0, 2.0, 1.0],
      num_selected_features=1, num_interp_features=3)["sample"]
# → threshold = the (3-1)=2nd smallest = 2.0 ⇒ strictly greater ⇒ [1, 0, 0]
solve("perturb_sample", rand_vals=[5.0, 5.0, 5.0, 1.0],
      num_selected_features=2, num_interp_features=4)["num_selected"]
# → 0 (ties cause under-selection)
```

---

## mode 4: `n_evaluations`

Reproduce the forward-evaluation counts of the two Shapley routes:

```python
# ShapleyValueSampling
def _get_n_evaluations(self, total_features, n_samples, perturbations_per_eval):
    return math.ceil(total_features / perturbations_per_eval) * n_samples

# ShapleyValues
def _get_n_evaluations(self, total_features, n_samples, perturbations_per_eval):
    return math.ceil(total_features / perturbations_per_eval) * math.factorial(
        total_features
    )
```

**Signature**

```python
n_evaluations(total_features, n_samples, perturbations_per_eval, exact=False)
    → {"n_evaluations": int, "mode": "exact" | "sampling",
       "permutation_factor": int}
```

**Key points**

1. The sampling variant is
   `ceil(total_features / perturbations_per_eval) * n_samples` —
   **ceiling first, then multiply**.
2. ★ The exact variant (`ShapleyValues`) **ignores `n_samples` entirely** and
   uses `factorial(total_features)` instead — because it enumerates all M!
   permutations.
3. `perturbations_per_eval` appears only in the denominator of the `ceil`; it
   does **not** affect the multiplication factor.
4. The `mode` output says which convention was used (`"sampling"` / `"exact"`),
   and `permutation_factor` is that multiplication factor itself.

---

## mode 5: `all_permutations`

Reproduce the official exhaustive permutation enumeration:

```python
def _all_perm_generator(num_features: int, num_samples: int) -> Iterable[Sequence[int]]:
    for perm in itertools.permutations(range(num_features)):
        yield perm
```

**Signature**

```python
all_permutations(num_features, num_samples=None)
    → {"permutations": [[...], ...], "count": int, "num_samples_ignored": bool}
```

**Key points**

1. ★ The `num_samples` parameter is **completely ignored** — the generator
   yields all `M!` permutations, regardless of the `num_samples` passed in.
2. The permutation order is `itertools.permutations`' lexicographic order (the
   first entry is the ascending `[0, 1, ..., M-1]`).
3. `count` = `M!`.

---

## Common Mistakes

1. Leaving `probs[0]` as `inf` (skipping the manual `probs[0] = 0.0` wipe-out).
2. Normalizing `probs` up front (the official code returns the **unnormalized**
   raw probs).
3. Using the range `0 .. M` for k (the official code uses `arange(M)`, which
   **excludes M**).
4. Writing `denom` as something like `k * (M - k + 1)`.
5. Treating `probs[M-1]` as "theoretically infinite" (it is exactly **1.0**).
6. Still returning a normalized distribution at M=1 (the official code crashes;
   `normalized` must be `None`).
7. Computing `normalized_f32` in double (the official code is **float32**
   throughout).
8. Writing the endpoint weight as a true `inf` (the official value is the finite
   **1000000.0**).
9. Testing only "all 0" or only "all 1" for the endpoint (the official code
   treats **both** as endpoints).
10. Ignoring the `inf_weight` argument.
11. Taking the **k-th largest** threshold instead of the **`(M - k)`-th
    smallest**.
12. Using `>=` for the threshold comparison instead of **strictly greater**.
13. Forgetting that ties cause **under-selection** (`[5,5,5,1]` with k=2 ⇒ 0
    selected).
14. Not rejecting `k = M` (the official `kthvalue` index is out of range).
15. Swapping the two fixed endpoints (all 1 first, all 0 second).
16. Putting the endpoint samples at the **end** of the sequence instead of the
    front.
17. Using `n_samples` instead of `factorial(total)` for the exact variant's
    evaluation count.
18. Writing `ceil` as `floor`, or getting the **order** of the ceiling and the
    multiplication wrong.
19. Letting `perturbations_per_eval` participate in the multiplication.
20. Truncating the exhaustive permutation enumeration by `num_samples`.

## Self-check

```bash
python starter.py            # skeleton self-check
```

Before submitting, confirm:

- The return key names of the five modes exactly match the statement.
- All outputs are `float` (not `Fraction`).
- Illegal inputs raise `ValueError`.
- At M=1, both `normalized` and `normalized_f32` are `None`.
