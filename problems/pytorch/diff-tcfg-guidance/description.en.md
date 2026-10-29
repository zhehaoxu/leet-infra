# Diffusion: Tangential CFG with SVD Truncation

## Problem

Classic CFG extrapolates along the line through \(\epsilon_c-\epsilon_u\), but the two predictions span a 2-D subspace: only one direction carries most of the energy, while the other is mostly noise or random texture jitter. Pushing hard along the latter produces oversaturation and structural artifacts. **Tangential CFG** (TCFG) therefore truncates the unconditional prediction onto the dominant direction before extrapolating:

Stack the two predictions into a \(2\times D\) matrix, take the SVD and keep the first right singular vector \(v_1\); project \(\epsilon_u\) onto it:

\[ p=(\epsilon_u\cdot v_1)\,v_1 \]

\[ \text{out}=p+s\,(\epsilon_c-p) \]

With only two rows, \(v_1\) necessarily lies in the plane spanned by \(\{\epsilon_c,\epsilon_u\}\), so the projection needs just one scalar multiple.

## Rules

Given same-shaped `eps_cond` / `eps_uncond` and `guidance_scale=s`, return a float32 tensor:

1. per sample, flatten \(\epsilon_c,\epsilon_u\) into two rows and form the \(2\times D\) matrix;
2. take its first right singular vector \(v_1\) and compute \(p=(\epsilon_u\cdot v_1)v_1\);
3. return \(p+s(\epsilon_c-p)\).

**Ties**: when the two singular values are equal (the dominant direction is not unique), the convention is to take \(\epsilon_c\)'s direction, in which case the projection collapses to 0 (for orthogonal branches of equal norm the output is simply \(s\,\epsilon_c\)).

## Examples

\[ \epsilon_c=\begin{bmatrix}0.4&-1.2&0.6\\0.0&0.5&-0.5\end{bmatrix},\ \epsilon_u=\begin{bmatrix}0.1&-0.4&0.2\\0.2&-0.1&0.0\end{bmatrix} \]

```
s=0 -> [[ 0.127716, -0.392791,  0.196395], [ 0.002381, -0.055423,  0.054233]]
s=3 -> [[ 0.944568, -2.814419,  1.407210], [-0.004763,  1.610847, -1.608465]]
```

The `s=0` row is the projection itself.

## Constraints

- Inputs are at least 2-D (dim 0 is the batch) and must share a shape; `s` is non-negative and finite.
- **Degeneracies**: at \(s=1\) the output equals `eps_cond` bit-for-bit; for parallel branches \(v_1\) is the shared direction, the projection leaves \(\epsilon_u\) untouched and the output reduces to bare CFG; for a zero \(\epsilon_u\) the projection is 0 and the output is \(s\,\epsilon_c\).
- The projection residual is orthogonal to the projection: \(\epsilon_u-p\perp p\).
- Both branches zero yields an all-zero output, never NaN.
