# Captum GradientShap expectation ledger: flat k*bsz baseline draws under draw_baseline_from_distrib, the (n,1,1,..) alpha broadcast of _scale_input, the repeat_interleave(k)+normal noise layout, the numpy/torch dual RNG stream order, single-partition smoothgrad aggregation, and the non-forwarded nt_samples_batch_size and float32 noise quantization quirks

## 1. Background

Captum's `GradientShap` approximates Expected Gradients as an expectation over baseline-distribution draws, random interpolation points, injected noise, and gradients. This problem decomposes the whole chain into five independently checkable numerical ledgers, all implemented with plain Python floats, with every random sequence (baseline indices / alphas / noises) passed in explicitly so that it depends on neither `captum` / `torch` / `numpy` nor any global RNG. Unlike order 849 (Kernel SHAP sampling) and order 851 (NoiseTunnel smoothing), this problem pins down the hardcoded semantics of GradientShap relative to its underlying attribution methods, the RNG call order, and the delta convention.

### Official source locations

- `captum/attr/_core/gradient_shap.py`: `GradientShap.attribute` / `InputBaselineXGradient.attribute` / `_scale_input`
- `captum/attr/_core/noise_tunnel.py`: `NoiseTunnel.attribute` / `add_noise_to_inputs`
- `captum/_utils/common.py`: `_expand_and_update_baselines` / `_format_callable_baseline`
- `captum/attr/_utils/attribution.py`: `GradientAttribution.compute_convergence_delta`

The judging environment has no `captum`, so a literal official execution environment was built by injecting 14 fake Captum modules and `exec`ing in the order `common -> attr_common -> gradient -> attribution -> noise_tunnel -> gradient_shap`; the probes run the real official implementation rather than a rewrite.

## 2. Conventions

### Inputs and outputs

Each tensor is a two-dimensional list `[bsz][d]`, and tuple inputs are a list of tensors. `indices` are `k*bsz*n_in` flat baseline indices, `alphas` are `k*bsz` coefficients in `[0,1)`, and `noises` are `k*bsz*d*n_in` noise values. Each mode returns a dictionary whose keys mirror the official variable names. Every invalid input raises `ValueError` with a message that names the violated constraint.

### Numerical precision and the float32 quirk

All values are Python floats (float64). The official `add_noise_to_input` builds its standard-deviation tensor with `torch.tensor(stdev)` without a dtype, which defaults to float32 and quantizes the injected noise by about `1e-8`. During the three-layer cross-check the driver temporarily calls `torch.set_default_dtype(torch.float64)` and restores it in a finally block, **without modifying the official source itself**, so that official intermediates match the pure-Python reference exactly; the reference solution itself is unaffected by this quirk.

## 3. Mode 1: baseline_draw

### Flat k*bsz indices and sampling with replacement

`GradientShap.attribute` hardcodes `draw_baseline_from_distrib=True`, so the official `_expand_and_update_baselines` takes `np.random.choice(num_ref, k*bsz, replace=True)`, producing `k*bsz` **flat** indices rather than indices grouped by example:

```text
indices = np.random.choice(num_ref, k*bsz, replace=True)
drawn   = [baselines[i] for i in indices]
layout  = flat_k_times_bsz
```

In distrib mode `_validate_input` checks only `shape[1:]`, not the batch dimension, so `num_ref` and `bsz` may differ. `GradientShap` asserts that `baselines[0]` must be a Tensor; `_format_callable_baseline` supports 0-argument / 1-argument callables, tensors, and scalars.

### repeat_interleave and applicable in non-distrib mode

With `draw_baseline_from_distrib=False` the official path is `baseline.repeat_interleave(k, dim=0)`, but it applies only when `num_ref == bsz and num_ref > 1`; otherwise the baseline is kept unchanged.

```text
applicable = (num_ref == bsz and num_ref > 1)
applicable : layout = repeat_interleave, drawn_shape = [num_ref*k, d]
otherwise  : layout = unchanged,        drawn_shape = [num_ref, d]
```

### Domain constraints

Non-positive `n_samples` / `bsz`, an index length other than `k*bsz`, non-int indices, out-of-range indices, and a mismatched feature dimension all raise `ValueError`, each with a message naming the corresponding constraint.

## 4. Mode 2: scale_input

### The (n,1,1,..) alpha broadcast

The alpha of `InputBaselineXGradient.attribute` comes from `np.random.uniform(0.0, 1.0, bsz*k)` and is viewed as `(n,1,1,..)` inside `_scale_input` before broadcasting:

```text
inp_shape = (bsz,) + (1,) * (ndim - 1)
scaled    = alpha * input + (1.0 - alpha) * baseline
scaled    = scaled.requires_grad_()
```

Note that the alpha length is `bsz*k` (the noised batch), and the two interpolation endpoints are the **noised input** and the **drawn baseline**, not the original input. `requires_grad_()` is an explicit official call.

### Domain constraints

The baseline row count / feature dimension must match the inputs, the alphas length must equal the batch and lie in `[0,1)`, and `ndim` must be an int `>= 2`.

## 5. Mode 3: noise_layout

### repeat_interleave(k) + normal

The official `add_noise_to_inputs` applies `input.repeat_interleave(k, dim=0) + torch.normal(0, stdev_expanded)` to each input tensor, with noise laid out in C order as `(i*k+j, t)`:

```text
stdev_expanded = torch.tensor(stdev).repeat((bsz*k,) + feature_shape)
noise          = torch.normal(0, stdev_expanded)
perturbed      = input.repeat_interleave(k, dim=0) + noise
```

The `bsz*k` rows of `perturbed` follow `(i*k+j)`, i.e. the k noise copies of example i are contiguous, not the `repeat()` layout grouped by noise sample.

### stdevs validation

`stdevs` must be a Python `float` scalar, or a tuple whose length equals the number of input tensors with every element a `float`; an `int` is rejected by the official assertion. A mismatched tuple length also raises.

### Domain constraints

`k` / `n_in` must be positive, the `noises` length must equal `bsz*k*d*n_in`, and every value must be a finite number (bool does not count).

## 6. Mode 4: schedule

### Single partition and the dual RNG stream

`GradientShap.attribute` **does not forward `nt_samples_batch_size`**, so `NoiseTunnel` receives `None` and is always a single partition. The RNG call order is fixed as `choice -> normal -> uniform`: choice and uniform share the numpy global stream, while normal uses the independent torch stream.

```text
order         = [choice]*n_in + [normal]*n_in + [uniform]
choice_calls  = [[num_ref, k*bsz, True] for each tensor]
normal_shapes = [[k*bsz, d] for each tensor]
uniform_calls = [[0.0, 1.0, k*bsz]]              # once, shared by all tensors
```

With tuple inputs, choice / normal happen once per tensor and uniform only once; with a single tensor, all three happen once.

### Direct NoiseTunnel partition comparison

Calling `NoiseTunnel` directly with `nt_samples_batch_size=b` splits into `p = k // b` full partitions plus one remainder partition; choice is called only once at the start (`kwargs_copy`), while normal / uniform happen once per partition. This problem exposes that comparison ledger through `compare_batch_size`, but GradientShap itself always has `partitions = 1`, `remaining = 0`, and `batch_size_forwarded = False`.

## 7. Mode 5: gradient_shap

### End-to-end chain

```text
drawn     = baseline[indices[ti]]                 # choice, per tensor
perturbed = repeat_interleave(input, k) + noise   # normal, per tensor
scaled    = alpha*perturbed + (1-alpha)*drawn     # uniform, shared once
grads     = dF(scaled)                            # linear: w / quadratic: 2*w*scaled
attr      = (perturbed - drawn)*grads or grads    # multiply_by_inputs
out[i]    = sum_j attr[i*k+j] / k                 # single-partition smoothgrad mean
```

### smoothgrad mean and multiply_by_inputs

`nt_type` is hardcoded to `smoothgrad`, and aggregation is `view(bsz,k,..).sum(dim=1) * 1/nt_samples`, i.e. the mean of the k copies grouped per example, not a mean over all `bsz*k` rows and not a bare sum without dividing by `k`. With `multiply_by_inputs=True`, `InputBaselineXGradient` uses `(input_noisy - baseline_drawn) * grad`; with `False` it returns `grad` directly.

### convergence delta

With `return_convergence_delta=True`, delta has shape `(bsz*k,)`, one row per noised sample:

```text
delta[r] = sum(attrs[r]) - (F(end) - F(start))
F(end)   = sum_t F(perturbed_t[r])
F(start) = sum_t F(drawn_t[r])
```

With tuple inputs, `sum(attrs)` accumulates across all tensors while `F(end) - F(start)` is subtracted only once (the forward_func receives all tensors).

## 8. Official quirk list

1. `GradientShap.attribute` hardcodes `nt_type="smoothgrad"` and `draw_baseline_from_distrib=True`, does not forward `nt_samples_batch_size`, and is always a single partition.
2. Baseline indices are `k*bsz` flat `np.random.choice(..., replace=True)` results, not grouped by example.
3. Distrib mode validates only `shape[1:]`, not the batch dimension.
4. The two endpoints of `_scale_input` are the noised input and the drawn baseline, and the alpha length is `bsz*k`.
5. The RNG order is choice(numpy) -> normal(torch) -> uniform(numpy), with choice and uniform sharing the numpy stream.
6. With tuple inputs, choice / normal happen once per tensor and uniform only once.
7. `torch.tensor(stdev)` has no dtype, defaults to float32, and gives about `1e-8` of quantization error on the injected noise.
8. `stdevs` must be a float or an equal-length float tuple; int is rejected.
9. `delta` has shape `(bsz*k,)`, not `(bsz,)`; attrs accumulate across tensors but the F difference is subtracted only once.
10. With `multiply_by_inputs=False`, the attribution is the gradient itself, without the `(input - baseline)` factor.

## 9. Common mistakes

1. Grouping baseline indices by example instead of flat `k*bsz`.
2. Writing `replace` as `False`, or dropping sampling with replacement.
3. Validating batch-dimension equality in distrib mode.
4. Dropping `num_ref > 1` from the non-distrib `applicable` predicate.
5. Writing `alpha.view` as `(n, d)` instead of `(n,1,1,..)`.
6. Interpolating with the original input instead of the noised perturbed input.
7. Forgetting `requires_grad_()`.
8. Writing the alpha upper bound as `<= 1.0`.
9. Using `repeat(k)` instead of `repeat_interleave(k)` for noise.
10. Indexing noise as `(j*bsz+i)` instead of the `(i*k+j)` C order.
11. Accepting an int `stdevs` or int elements inside the tuple.
12. Computing the `noises` length as `bsz*k*d`, forgetting the `n_in` factor.
13. Treating `nt_samples_batch_size` as forwardable and producing multiple partitions.
14. Writing uniform before choice in the RNG order, or making uniform per-tensor for tuple inputs.
15. Averaging smoothgrad over all `bsz*k` rows, or summing without dividing by `k`.
16. Reversing the `multiply_by_inputs` branch: still multiplying by `(input - baseline)` when False.
17. Writing the quadratic gradient as `w*scaled` instead of `2*w*scaled`.
18. Giving delta shape `(bsz,)`, or subtracting the F difference once per tensor for tuples.
19. Writing the delta sign as `(F(end)-F(start)) - sum(attrs)`.
20. Corrupting the reported `partition_count` / `n_in` / `feature_dim` fields.
21. Still returning delta when `return_convergence_delta=False`.
22. Not raising `unknown mode` for an unrecognized mode.
23. Asserting only that "an exception is raised" for domain constraints without checking the message keyword.
24. Accepting bool as a finite number in numeric validation.
25. A constant noise sequence making multi-tensor slices and layout differences unobservable.

## 10. Self-check

- When `num_ref == bsz == 1`, non-distrib stays `unchanged` and `drawn_shape` remains `[num_ref, d]`.
- With tuple inputs, the two per_tensor noise / indices slices must differ.
- With `bsz > 1 and k > 1`, the `(i*k+j)` and `(j*bsz+i)` layouts must differ.
- The `return_convergence_delta` switch must precisely control the presence of the `delta` field.
- The three-layer cross-check, test cases, and mutation must all pass before landing.
