# In Depth: Converting between ε, x0 and v Prediction Targets

## Origin

Salimans & Ho, *Progressive Distillation for Fast Sampling of Diffusion Models* (arXiv:2206.00364) train with \(v=\alpha\epsilon-\sigma x_0\): in distillation a student takes half a step while the teacher takes a full one, and only in v space is that half step an interpolation. SD 2.x / SDXL / Imagen all ship a v-prediction switch.

## Core points and pitfalls

1. **Do not flip the sign of v**: it is \(v=\alpha\epsilon-\sigma x_0\), not \(\sigma x_0-\alpha\epsilon\). In rotation form, \([x_t;v]=\begin{bmatrix}\alpha&\sigma\\-\sigma&\alpha\end{bmatrix}[x_0;\epsilon]\), which has determinant 1 and preserves the norm.
2. **Inverting \(v\to(x_0,\epsilon)\) is just undoing the rotation**: \(\epsilon=\alpha v+\sigma x_t\) and \(x_0=\alpha x_t-\sigma v\) carry opposite signs on the \(\sigma\) term; copying one into the other shows up clearly as \(\bar\alpha_t\to1\).
3. **Singularities at both ends**: at \(\alpha=0\) you cannot recover \(x_0\) from \(\epsilon\) (division by zero), and at \(\sigma=0\) you cannot recover \(\epsilon\) from \(x_0\). The problem requires `ValueError` there instead of returning inf.
4. **The known quantity must be passed through bit-exactly**: for `mode="v"` the returned `eps`/`x0` are computed, but `v` has to equal the input element by element — the easiest thing for a "round the output" implementation to break.

## Verification approach

The reference solution vectorizes the three branches in float64 and casts to float32 at the exit. The judge's `_oracle` recomputes element by element with `math.sqrt`, expanding the formulas so no intermediate is shared (e.g. \(x_0=x_t/\alpha-(\sigma/\alpha)\epsilon\)), and adds three structural groups: the forward relations \(x_t=\alpha x_0+\sigma\epsilon\) and \(v=\alpha\epsilon-\sigma x_0\) must hold simultaneously; \(\epsilon\to x_0\to\epsilon\) and \(v\to(x_0,\epsilon)\) must round-trip; and the degeneracies at \(\bar\alpha_t=0,1\) plus invalid inputs. Tolerance `allclose(atol=1e-5, rtol=1e-5)`.

## Complexity

Element-wise \(O(N)\), with no reductions and no randomness.
