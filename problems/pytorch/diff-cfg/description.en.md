# Diffusion: Classifier-Free Guidance (CFG)

## Problem

**Classifier-Free Guidance** (Ho & Salimans, arXiv:2207.12598) lets a single network produce both the "conditional" and "unconditional" noise predictions \(\epsilon_\text{cond}\), \(\epsilon_\text{uncond}\) at the same time, and then extrapolates linearly between them to amplify the influence of the conditioning signal (e.g. the text prompt), trading "diversity ↔ adherence to the condition" for higher fidelity.

The combination most commonly used in practice (diffusers / Stable Diffusion), with guidance weight \(w\ge 1\):

\[ \tilde\epsilon = \epsilon_\text{uncond} + w\,\bigl(\epsilon_\text{cond} - \epsilon_\text{uncond}\bigr) \]

This is equivalent to \((1-w)\,\epsilon_\text{uncond} + w\,\epsilon_\text{cond}\); it differs from the paper's \((1+w')\epsilon_\text{cond}-w'\epsilon_\text{uncond}\) by the reparameterization \(w'=w-1\).

## Rules

Given `eps_cond` and `eps_uncond` of the same shape and a scalar `w` (guidance scale), return \(\tilde\epsilon\) (same shape). float32. Ready-made diffusion libraries are forbidden.

## Examples

\[ \epsilon_\text{cond}=\begin{bmatrix}1.0&-0.5&0.2\\ 0.3&0.7&-1.1\end{bmatrix},\ \epsilon_\text{uncond}=\begin{bmatrix}0.4&0.1&-0.3\\ 0.6&-0.2&0.5\end{bmatrix},\ w=7.5 \]

```
eps_guided = [[ 4.9,  -4.4,   3.45],
              [-1.65,  6.55, -11.5 ]]
```

## Constraints

- `eps_cond` and `eps_uncond` have the same shape; `w` is a float scalar, usually \(w\ge1\).
- **Degenerate properties**: when \(w=0\), \(\tilde\epsilon=\epsilon_\text{uncond}\); when \(w=1\), \(\tilde\epsilon=\epsilon_\text{cond}\) (no guidance); \(w>1\) extrapolates along the cond direction to amplify the condition.
- Element-wise linear combination; the result may fall outside the input's numeric range (caused by guidance amplification, which is expected).
