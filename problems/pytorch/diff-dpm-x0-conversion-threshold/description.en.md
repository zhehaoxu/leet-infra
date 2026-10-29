# DPM-Solver Model-Output Conversion and Dynamic Thresholding

## Problem Statement

DPM-Solver and DPM-Solver++ want **opposite model-output formulations**: the former discretizes the integral of the noise prediction \(\epsilon_\theta\), the latter the integral of the data prediction \(x_0\). The same weights can be paired with either, so the scheduler has to normalize the raw network output before numerical integration — that logic lives in `convert_model_output`.

The same function also does something independent: **dynamic thresholding** (Imagen, 2205.11487). Under large guidance scales the predicted \(x_0\) saturates; dynamic thresholding pushes each sample back into \([-1,1]\) using a per-sample percentile, then converts back.

This problem implements both: convert the raw output to \(x_0\) or \(\epsilon\) according to `algorithm_type` / `prediction_type`, and insert thresholding when the flag is on.

## Rules

Let \(\sigma\) be the current noise level, \(x\) the sample, and `model_output` the raw network output.

**1. \(\alpha/\sigma_t\) conversion:**

\[ \text{use\_flow\_sigmas} \Rightarrow \alpha = 1-\sigma,\ \sigma_t = \sigma; \qquad \text{otherwise} \Rightarrow \alpha = \frac{1}{\sqrt{\sigma^2+1}},\ \sigma_t = \sigma\alpha \]

**2. Data formulation** (`algorithm_type` in `{"dpmsolver++", "sde-dpmsolver++"}`), output is \(x_0\):

- `epsilon`: \(x_0 = (x - \sigma_t\,\epsilon)/\alpha\)
- `sample`: \(x_0 = \text{model\_output}\)
- `v_prediction`: \(x_0 = \alpha x - \sigma_t\,\text{model\_output}\)
- `flow_prediction`: \(x_0 = x - \sigma\,\text{model\_output}\) (uses the **raw \(\sigma\)**, bypassing \(\alpha/\sigma_t\); here `model_output` is the velocity \(v=\epsilon-x_0\))

**3. Noise formulation** (`algorithm_type` in `{"dpmsolver", "sde-dpmsolver"}`), output is \(\epsilon\):

- `epsilon`: \(\epsilon = \text{model\_output}\)
- `sample`: \(\epsilon = (x - \alpha\,\text{model\_output})/\sigma_t\)
- `v_prediction`: \(\epsilon = \alpha\,\text{model\_output} + \sigma_t x\)
- `flow_prediction`: unsupported, raise `ValueError`

**4. Dynamic thresholding** (`thresholding=True`), per sample:

1. flatten \(x_0\) to \((B, -1)\) and take the `dynamic_thresholding_ratio` quantile \(s\) of \(|x_0|\) (linear interpolation);
2. \(s \leftarrow \min(\max(s, 1), \text{sample\_max\_value})\);
3. \(x_0 \leftarrow \operatorname{clip}(x_0, -s, s)/s\).

The data formulation thresholds \(x_0\) directly; the noise formulation first recovers \(x_0\) from \(\epsilon\), thresholds it, then maps back via \(\epsilon = (x-\alpha x_0^{\text{thr}})/\sigma_t\).

**5. Variance channels**: for `variance_type` in `"learned"` / `"learned_range"`, epsilon prediction only, `model_output` is \((B, 6, \dots)\) while `sample` is \((B, 3, \dots)\); only the first 3 channels are used as \(\epsilon\).

Return a float32 tensor shaped like `sample`.

## Example

With `sample=[[1.0, -2.0, 0.5], [0.25, 3.0, -1.5]]`, `model_output=[[0.5, 0.25, -0.75], [0.1, -0.2, 0.3]]`, `sigma=0.5` (\(\alpha=0.894427\), \(\sigma_t=0.447214\)):

```
dpmsolver++ / epsilon        -> [[0.868034, -2.361068, 0.934017], [0.229508, 3.454102, -1.827051]]
dpmsolver++ / v_prediction   -> [[0.67082, -1.900658, 0.782624], [0.178885, 2.772724, -1.475805]]
dpmsolver   / v_prediction   -> [[0.894427, -0.67082, -0.447214], [0.201246, 1.162755, -0.402492]]
dpmsolver   / epsilon        -> returns model_output unchanged
thresholding ratio=0.9 max=1.2
                             -> [[0.723362, -1.0, 0.778347], [0.191257, 1.0, -1.0]]
variance_type="learned" (first 3 channels) -> identical to the first line
```

In the first line \(x_0 = (1 - 0.447214\times0.5)/0.894427 = 0.868034\). In the thresholding line, sample 0 has \(|x_0| = [0.868, 2.361, 0.934]\); its 0.9 quantile is \(0.934+0.8\times(2.361-0.934)=2.0756\), clipped by `sample_max_value=1.2` to \(s=1.2\), hence \(0.868/1.2=0.723362\) and the excess is clamped to \(\pm1\).

## Constraints

- `model_output` / `sample` must be non-empty floating tensors with at least 2 dims; without `learned` their shapes must match exactly.
- `sigma` must be a finite positive real (`bool` does not count); with `use_flow_sigmas=True` it must additionally satisfy \(0<\sigma<1\).
- `prediction_type` accepts only `epsilon` / `sample` / `v_prediction` / `flow_prediction`; `algorithm_type` only one of the four values; `variance_type` only `fixed_small` / `fixed_large` / `learned` / `learned_range`.
- `thresholding` / `use_flow_sigmas` must be `bool`; `dynamic_thresholding_ratio` lies in \((0,1]\); `sample_max_value` must be \(\ge 1\).
- `learned` / `learned_range` support epsilon prediction only, requiring `model_output` of shape \((B,6,\dots)\) and `sample` of shape \((B,3,\dots)\).
- Compute in float64 internally, return float32.
