# DDIM Inversion Step: Pushing a Sample Back to a Noisier Level

## Problem Statement

DDIM sampling walks from pure noise to a clean sample; running it backwards walks from a clean sample back into noise. That is DDIM inversion, the workhorse of "encode to noise, edit the noise, decode again" image editing. Every inversion step asks for a sample one level **noisier** than the current one, so besides the same "predict x0 plus a direction term" arithmetic as sampling, there is an asymmetry: the output noise level is the current `timestep`, while the noise level used to predict x0 sits one step ratio further back — and if that index falls below 0, a configurable initial alpha takes over.

This problem covers a **single** inversion step: given the cumulative-alpha schedule, the current timestep, a sample and the model output, compute this step's `prev_sample` and the predicted `pred_original_sample`.

## Rules

Let the schedule have length \(T\) and the number of inference steps be \(N\), with step ratio \(r = \lfloor T/N \rfloor\) (integer division).

1. **Two noise levels**: the output level is `timestep`; the prediction level is \(t=\min(\texttt{timestep}-r,\;T-1)\). Hence

   \[
   \bar\alpha_t=\begin{cases}\bar\alpha[\texttt{timestep}],&t\ge 0\\ \text{initial},&t<0\end{cases}
   \qquad
   \bar\alpha_{\text{prev}}=\bar\alpha[\texttt{timestep}]
   \]

   where `initial` is \(1.0\) when `set_alpha_to_one=True` (the default) and \(\bar\alpha[0]\) otherwise.
2. **Split the model output into x0 and noise** (\(\beta_t=1-\bar\alpha_t\)):

   ```
   epsilon:      x0 = (sample - sqrt(beta_t) * model_output) / sqrt(alpha_t)
                 eps = model_output
   sample:       x0 = model_output
                 eps = (sample - sqrt(alpha_t) * x0) / sqrt(beta_t)
   v_prediction: x0 = sqrt(alpha_t) * sample - sqrt(beta_t) * model_output
                 eps = sqrt(alpha_t) * model_output + sqrt(beta_t) * sample
   ```
3. **Optional clamp**: when `clip_sample=True`, clamp x0 to \([-c, c]\) with \(c\) equal to `clip_sample_range`.
4. **Compose the noisier sample**:

   \[
   \texttt{prev\_sample}=\sqrt{\bar\alpha_{\text{prev}}}\,x_0+\sqrt{1-\bar\alpha_{\text{prev}}}\,\epsilon
   \]

Return `(prev_sample, pred_original_sample)`: two float32 tensors shaped like `sample`. Note that the returned x0 is the **clamped** one.

## Example

`alphas_cumprod = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]` (\(T=10\)), `num_inference_steps = 10` (so \(r=1\)), `timestep = 5`, `sample = 2.0`, `model_output = 0.5`, `prediction_type = "epsilon"`:

```
prediction level t = 4    alpha_prod_t = 0.5       beta_prod_t = 0.5
output level timestep = 5 alpha_prod_t_prev = 0.4

x0  = (2.0 - sqrt(0.5) * 0.5) / sqrt(0.5) = 2.328427
eps = 0.5
```

- Without clamping: `prev_sample = sqrt(0.4) * 2.328427 + sqrt(0.6) * 0.5 = 1.859925`;
- With the default `clip_sample=True`, `clip_sample_range=1.0`: x0 is clamped to 1.0 and `prev_sample = sqrt(0.4) * 1.0 + sqrt(0.6) * 0.5 = 1.019754`.

With `timestep = 0`, `sample = 1.0`, `model_output = 0.2`, the prediction level is \(t=-1\) and \(\bar\alpha_t\) takes the initial value: `set_alpha_to_one=True` gives `(1.011929, 1.0)`, while `False` (using \(\bar\alpha[0]=0.9\)) gives `(1.0, 0.987426)`.

## Constraints

- `alphas_cumprod` must be a 1-D float tensor (float32 / float64), non-empty, free of nan / inf, with every element in \((0,1]\);
- `timestep` is a Python integer (booleans are rejected) in \([0,T)\) — it indexes `alphas_cumprod` directly, i.e. it is the noise level of the output;
- `num_inference_steps` is an integer with \(1\le N\le T\);
- `sample` and `model_output` must be same-shaped, non-empty float tensors free of nan / inf;
- `prediction_type` is one of `"epsilon"`, `"sample"`, `"v_prediction"`;
- `clip_sample` and `set_alpha_to_one` must be booleans; `clip_sample_range` must be a positive finite real;
- Raise `ValueError` when any of the above fails; the `"sample"` prediction type also raises when \(\beta_t\le 0\) (i.e. \(\bar\alpha_t=1\), possible only when `timestep` falls below 0 with `set_alpha_to_one=True`);
- Properties: without clamping, feeding \((\bar\alpha_t, x_0, \epsilon)\) produced by the forward process reproduces exactly the forward sample at level `timestep`; all three prediction types give the same `prev_sample` for the same \((x_0,\epsilon)\).
