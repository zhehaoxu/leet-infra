# Diffusion: Converting between ε, x0 and v Prediction Targets

## Problem

A diffusion model can be asked to predict the noise \(\epsilon\), the clean sample \(x_0\), or the "rotated combination" \(v\) of the two at the same \(\bar\alpha_t\) (v-prediction, used by SD 2.x / SDXL / Imagen). The three targets are linearly invertible: with \(\alpha=\sqrt{\bar\alpha_t}\) and \(\sigma=\sqrt{1-\bar\alpha_t}\), the forward process and the definition of v are

\[ x_t = \alpha x_0 + \sigma \epsilon, \qquad v = \alpha \epsilon - \sigma x_0 \]

so knowing any one of them determines the other two (one branch per case, spelled out in `notes.md`). Note that \((x_0,\epsilon)\mapsto(x_t,v)\) is an orthogonal rotation: switching targets keeps the information and only changes the numerical scale the network sees during training.

## Rules

Given `x_t`, the scalar `alpha_bar`, the model output `pred` and the target parameterization `mode` (`"eps"` / `"x0"` / `"v"`, with `"epsilon"` and `"x_zero"` treated as aliases), return `(x0, eps, v)` as three same-shaped float32 tensors (compute internally in float64). `pred` is exactly the quantity named by `mode` and is passed through unchanged.

## Examples

\[ x_t=[0.5,-1.0,2.0],\ \bar\alpha_t=0.64\ (\alpha=0.8,\ \sigma=0.6),\ \epsilon=[0.3,0.0,-0.9] \]

```
x0  = [0.4, -1.25, 3.175]
eps = [0.3, 0.0, -0.9]
v   = [0.0, 0.75, -2.625]
```

## Constraints

- \(\bar\alpha_t\in[0,1]\); `mode="eps"` requires \(\bar\alpha_t>0\) and `mode="x0"` requires \(\bar\alpha_t<1\) (otherwise the formula divides by zero and must raise `ValueError`).
- **Degeneracies**: at \(\bar\alpha_t=1\), \(x_0=x_t\) and \(v=\epsilon\); at \(\bar\alpha_t=0\), \(v=-x_0\) and \(\epsilon=x_t\).
- The three modes are mutual inverses: feeding the computed \(x_0\) back must return the same \(\epsilon\), and likewise for \(v\).
