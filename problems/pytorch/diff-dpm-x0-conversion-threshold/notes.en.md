# Deep Dive: DPM-Solver Output Conversion and Dynamic Thresholding

## Source

HuggingFace diffusers, `src/diffusers/schedulers/scheduling_dpmsolver_multistep.py`: `convert_model_output` (split first by `algorithm_type` into the data and noise formulations, then by `prediction_type` into four / three branches), `_threshold_sample` (annotated as copied from `scheduling_ddpm.DDPMScheduler._threshold_sample`), and `_sigma_to_alpha_sigma_t` (`alpha = 1/√(σ²+1)`, `sigma_t = σ·alpha`, or `alpha = 1-σ`, `sigma_t = σ` under `use_flow_sigmas`). The thresholding formula is from Imagen (2205.11487).

## Key Points and Pitfalls

1. **The two formulations return different things**: `dpmsolver++` yields \(x_0\), `dpmsolver` yields \(\epsilon\). The same `prediction_type="epsilon"` takes genuinely different branches — the data formulation converts \(\epsilon\) into \(x_0\), the noise formulation returns it untouched. Same output shape, opposite meaning.
2. **`sample` prediction is the identity branch** in the data formulation (`x0_pred = model_output`), but in the noise formulation it must first compute `eps = (x - α·model_output)/σ_t`. Easy to collapse into "return model_output" by mistake.
3. **`flow_prediction` exists only in the data formulation**: it uses the raw \(\sigma\) (`x0 = sample - σ·model_output`) rather than \(\alpha/\sigma_t\), and here `model_output` is the velocity \(v=\epsilon-x_0\). The noise formulation's `else` raises `ValueError`. Writing the flow branch with \(\alpha/\sigma_t\) is tempting because the numbers coincide when `use_flow_sigmas=True` — follow the source: the flow branch does no \(\alpha\) conversion at all.
4. **Thresholding in the noise formulation round-trips twice**: `x0 = (x-σ_t·ε)/α`, then `x0 = threshold(x0)`, then `ε = (x-α·x0)/σ_t`. Thresholding \(\epsilon\) directly is wrong — the threshold acts on \(x_0\); \(\epsilon\) is only the transport.
5. **`learned` variance only affects the epsilon branch**: `model_output[:, :3]` takes the first three channels (DDPM's mean), the rest are variance, and DPM-Solver only needs the mean. Hence `model_output` is \((B,6,\dots)\) while `sample` is \((B,3,\dots)\) — shape validation has to slice before comparing.
6. **The quantile is per sample**: `torch.quantile(abs_sample, ratio, dim=1)` runs along dim=1 of the flattened \((B, C\cdot H\cdot W)\), so every sample gets its own \(s\) and batches are independent. A global quantile would make the whole batch share one scale.
7. **`s` is clamped on both sides**: `clamp(s, min=1, max=sample_max_value)`. The `min=1` lets samples with \(|x_0|\le 1\) pass through unchanged (plain clipping), while `max` keeps an extreme percentile from blowing up. Consequently the output **always** lands in \([-1,1]\), even with a large `sample_max_value`.
8. **`sample_max_value` below 1 is meaningless**: `min=1` conflicts with `max=sample_max_value<1`, and `torch.clamp` is unspecified there, so this problem requires it to be \(\ge 1\).
9. **`use_flow_sigmas` with \(\sigma\ge1\) divides by zero**: \(\alpha=1-\sigma\), so at \(\sigma=1\) the data-formulation `epsilon` branch divides by 0. This problem requires \(0<\sigma<1\).
10. **Numerics**: upstream upcasts half to float32 for the quantile and casts back; this problem computes entirely in float64 and returns float32, so the example numbers match a float64 hand calculation exactly.

## Reference Comparison

Judging uses a numpy oracle (`_oracle` + `_threshold_oracle`, with `np.quantile(method="linear")` for the percentile) that redoes the whole conversion branch by branch, then compares against the reference solution:

- **Example values**: three hard-coded six-decimal groups (data-formulation \(\epsilon\) / \(v\), noise-formulation \(v\), thresholded) plus bitwise equality for "noise formulation with \(\epsilon\) returns the input unchanged";
- **Prediction identities**: starting from \(x=\alpha x_0+\sigma_t\epsilon\), invert to get \(\epsilon\) and \(v=\alpha\epsilon-\sigma_t x_0\) and check the data formulation recovers \(x_0\); the `sample` branch is bitwise identical; the flow branch is verified through \(v=\epsilon-x_0\);
- **Noise-formulation inverses**: the `dpmsolver` `sample` / `v_prediction` branches recover \(\epsilon\), and the two formulations' \(\epsilon\) branches are mutually inverse;
- **Thresholding maths**: inputs with \(|x_0|\le1\) pass through bitwise, the output stays in \([-1,1]\), `ratio=1.0` divides by the per-sample max absolute value, changing one sample leaves the other untouched, and a tight versus loose `sample_max_value` gives different results;
- **Flow formulation**: recovers \(x_0\) under \(\alpha=1-\sigma\) and visibly differs from the VP formulation (so the two branches cannot collapse into one);
- **Learned variance**: a 6-channel input matches the direct 3-channel call bitwise (once per formulation), also with thresholding stacked on top;
- **Combined comparison**: 2 shapes × 3 values of \(\sigma\) × 14 configurations (all four `prediction_type`s, both formulations, thresholding on/off, varied `sample_max_value` / `ratio`, flow combinations) plus 2 learned configurations;
- **Invalid inputs**: 21 cases (non-tensor / integer / mismatched or empty shapes / 1-D, \(\sigma\) zero / negative / NaN / `bool`, unknown `prediction_type` / `algorithm_type` / `variance_type`, `thresholding=1`, `use_flow_sigmas="yes"`, `ratio` 0 or >1, `sample_max_value<1`, flow with \(\sigma\ge1\), noise formulation with flow, learned with non-epsilon, learned with wrong shapes) must all raise `ValueError`.

Eight assertion groups in total.

## Complexity

A constant number of multiply-adds per element plus one per-sample sort for the quantile: \(O(B\cdot N\log N)\) time (\(N=C\cdot H\cdot W\)) and \(O(B\cdot N)\) space.
