# EulerAncestral: σ_up/σ_down Split and Ancestral Noise Injection

## Problem Statement

The Euler Ancestral sampler turns the Euler ODE step into an SDE step that re-injects noise every iteration: when going from \(\sigma_{\text{from}}\) to \(\sigma_{\text{to}}\) it does not drop the noise level in one shot, but splits the drop — part of it is a deterministic Euler step down to \(\sigma_{\text{down}}\), and the rest is injected as fresh noise with magnitude \(\sigma_{\text{up}}\). That is what "ancestral" means: randomness is replenished at every step, so different seeds give different images while the \(x_0\) prediction still follows what the model learned.

This problem implements one full step of `EulerAncestralDiscreteScheduler.step`: model-input scaling, \(x_0\) prediction, the \(\sigma_{\text{up}}/\sigma_{\text{down}}\) split, the Euler update, and noise injection.

## Rules

Let \(\sigma = \text{sigmas}[i]\), \(\sigma' = \text{sigmas}[i+1]\), \(x\) the sample, and `model_output` the network output.

**1. Model-input scaling** (upstream `scale_model_input`):

\[ \text{scaled\_input} = \frac{x}{\sqrt{\sigma^2+1}} \]

**2. \(x_0\) prediction**:

- `epsilon`: \(x_0 = x - \sigma\,\text{model\_output}\)
- `v_prediction`: \(x_0 = \text{model\_output}\cdot\dfrac{-\sigma}{\sqrt{\sigma^2+1}} + \dfrac{x}{\sigma^2+1}\)

(The second branch **copies the upstream implementation**; note the sample term is \(x/(\sigma^2+1)\), not \(x/\sqrt{\sigma^2+1}\). See the notes.)

**3. Noise-level split**:

\[ \sigma_{\text{up}} = \sqrt{\frac{\sigma'^2\,(\sigma^2-\sigma'^2)}{\sigma^2}},\qquad \sigma_{\text{down}} = \sqrt{\sigma'^2-\sigma_{\text{up}}^2} \]

**4. Euler step + noise injection**:

\[ \text{derivative} = \frac{x-x_0}{\sigma},\qquad \text{prev\_sample} = x + \text{derivative}\cdot(\sigma_{\text{down}}-\sigma) + \text{noise}\cdot\sigma_{\text{up}} \]

`noise` is an **explicit argument** (a float tensor shaped like the sample); it plays no part when \(\sigma_{\text{up}}=0\), in which case it may be omitted.

Return `(scaled_input, prev_sample, pred_original_sample, sigma_up, sigma_down)`: the first three are float32 tensors shaped like `sample`, the last two are python floats.

## Example

With `sample=[[1.0, -2.0, 0.5], [0.25, 3.0, -1.5]]`, `model_output=[[0.5, 0.25, -0.75], [0.1, -0.2, 0.3]]`, `sigmas=[2.0, 1.0, 0.5, 0.0]`:

```
i=0  sigma_up=0.866025 sigma_down=0.5   scaled=[[0.447214, -0.894427, 0.223607], ...]
     x0=[[0.0, -2.5, 2.0], [0.05, 3.4, -2.1]]
     prev=[[0.683013, -2.808013, 1.841506], [-0.116506, 3.949519, -1.841747]]
i=1  sigma_up=0.433013 sigma_down=0.25
     prev=[[0.841506, -2.404006, 1.170753], [0.066747, 3.47476, -1.670873]]
i=2  sigma_up=0.0      sigma_down=0.0     prev is exactly x0 (no noise needed)
```

In step 0, \(\sigma_{\text{up}} = \sqrt{1^2(4-1)/4} = \sqrt{0.75} = 0.866025\) and \(\sigma_{\text{down}} = \sqrt{1-0.75} = 0.5\), whose squares sum to exactly \(\sigma'^2=1\). On the final step \(\sigma'=0\) zeroes both terms and \(dt=-\sigma\), so `prev_sample` equals \(x_0\) exactly.

## Constraints

- `sample` / `model_output` must be non-empty floating tensors of the same shape; `sigmas` must be a 1-D float tensor with at least 2 entries, finite, non-negative, and non-increasing.
- `step_index` is an integer (`bool` does not count) in \([0, K-1]\); the current \(\sigma\) must be positive.
- `prediction_type` accepts only `"epsilon"` / `"v_prediction"`; anything else raises `ValueError` (upstream raises `NotImplementedError` for `"sample"`; this problem uses `ValueError` uniformly).
- When \(\sigma_{\text{up}}>0\) a float `noise` shaped like `sample` must be supplied, otherwise `ValueError`; when \(\sigma_{\text{up}}=0` the `noise` argument is ignored.
- Compute in float64 internally; the three tensor outputs are float32.
