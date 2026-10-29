# Diffusion: MAMBO-G Magnitude-Aware Guidance

## Problem Statement

At large guidance scales the most visible failure mode of CFG is **over-saturation**: once the magnitude of the update \(\epsilon_c-\epsilon_u\) dwarfs the unconditional prediction itself, samples get pushed into blotchy, over-textured territory. **Magnitude-Aware Mitigation for Boosted Guidance** (MAMBO-G) is an almost free patch — it never touches the direction, it only pulls the effective guidance strength from \(w\) back towards 1 in proportion to the ratio of the update magnitude to the unconditional magnitude:

\[ r=\frac{\lVert \epsilon_c-\epsilon_u\rVert_2}{\lVert \epsilon_u\rVert_2},\qquad w'=1+(w-1)\,e^{-\lambda r},\qquad \text{out}=\epsilon_u+w'\,(\epsilon_c-\epsilon_u) \]

When \(r\) is small (a gentle update) \(w'\approx w\) and the behaviour matches CFG; when \(r\) is large \(w'\to 1\) and the output falls back to the conditional prediction — "this sample is already being pushed too hard, stop pushing". The coefficient \(\lambda\) controls how aggressively this happens.

Note that the reduction is **per sample**: for each sample, one L2 norm over every dimension except dim 0 (the batch) gives one scalar \(r\). Do not normalize per channel or per spatial position.

## Rules

Given same-shaped `eps_cond` and `eps_uncond` (dim 0 is the batch), a base guidance strength `guidance_scale = w ≥ 0`, and a suppression strength `alpha = λ ≥ 0`:

1. \(\delta=\epsilon_c-\epsilon_u\);
2. per sample, \(r=\lVert\delta\rVert_2/\lVert\epsilon_u\rVert_2\);
3. \(w'=1+(w-1)e^{-\lambda r}\);
4. \(\text{out}=\epsilon_u+w'\delta\).

Special case: if a sample has \(\lVert\epsilon_u\rVert_2=0\), treat \(r\) as \(+\infty\), so \(w'=1\) and the output is exactly \(\epsilon_c\). This case must not produce `NaN` or `inf`.

Return a float32 tensor of the same shape as `eps_cond`.

## Example

\[ \epsilon_c=\begin{bmatrix}1.5&-2.5&1.25\\-0.25&1.0&-1.5\end{bmatrix},\quad \epsilon_u=\begin{bmatrix}1.0&-2.0&1.0\\0.5&1.0&-1.0\end{bmatrix},\quad w=10 \]

```
alpha=0.0 -> [[ 6.000000, -7.000000,  3.500000], [-7.000000,  1.000000, -6.000000]]
alpha=0.5 -> [[ 5.361224, -6.361224,  3.180612], [-5.248210,  1.000000, -4.832140]]
alpha=2.0 -> [[ 3.939285, -4.939285,  2.469642], [-2.279302,  1.000000, -2.852868]]
```

The two rows have magnitude ratios \(r\approx 0.3062\) and \(r\approx 0.6009\), so the second row gets suppressed harder. Both rows keep \(w'\) strictly between \(1\) and \(10\), moving closer to 1 as \(\lambda\) grows. The second element of the second row is \(1\) in both predictions, so its output is always 1.

## Constraints

- Inputs have at least two dimensions: dim 0 is the batch and every other dimension takes part in the norm reduction; both inputs must share the same shape.
- `guidance_scale` and `alpha` are finite non-negative reals; `bool` does not count as a real.
- **Degenerate cases**: \(\lambda=0\) gives \(e^{0}=1\), so the output is exactly \(\epsilon_u+w(\epsilon_c-\epsilon_u)\) (plain CFG); \(w=1\) gives \(w'=1\), so the output equals `eps_cond` bit for bit regardless of \(\lambda\) or the magnitude ratio; for \(w>1\), \(w'\) decreases monotonically in \(\lambda\) and converges to 1 as \(\lambda\to\infty\).
- Samples with \(\lVert\epsilon_u\rVert_2=0\) output `eps_cond` with no `NaN`/`inf` (a naive \(\lVert\delta\rVert/\lVert\epsilon_u\rVert\) yields `inf`, and `inf * 0` is `NaN`).
- Compute in float64 internally if you like, but return float32.
