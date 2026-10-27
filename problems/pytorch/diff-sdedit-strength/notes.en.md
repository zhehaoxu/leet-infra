# In Depth: SDEdit Strength and the Truncated Start Step

## Origin

Meng et al., *SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations* (arXiv:2108.01073): a strength \(s\) noises the input up to an intermediate time (with \(t_0\) increasing in \(s\)), and the reverse SDE/ODE starts from there. In diffusers' `StableDiffusionImg2ImgPipeline`, `init_timestep = min(int(num_steps * strength), num_steps)` and `t_start = max(num_steps - init_timestep, 0)` are the engineering form of the same accounting.

## Core points and pitfalls

1. **The index floors**: use `floor`, not `round`, otherwise \(s=1\) may land past \(T-1\), or a slightly larger \(s\) jumps a level early. The judge probes \(s=0.99\) and \(s=1\) precisely for this.
2. **Order of the table**: `alpha_bars` is sorted by ascending noise level (index 0 cleanest), and this convention must match the direction the sampler walks; reversing it makes larger \(s\) mean cleaner.
3. **ε is an input**: no randomness is drawn here, so inverting ε and plugging it back must round-trip exactly — this is how the judge checks the coefficient uses \(\sqrt{\bar\alpha_i}\) rather than \(\sqrt{\bar\alpha_{T-1}}\).
4. **The \(s=0\) boundary**: no noise, zero steps, so the output equals \(x_0\) (given \(\bar\alpha_0=1\)). Many implementations carry an off-by-one "run at least one step" bug here.
5. **Remaining steps equal the index**: \(i\) is both the start index and the number of steps down to 0; do not derive them from two separate formulas.

## Verification approach

The reference solution looks up \(\bar\alpha_i\) and vectorizes with `math.sqrt` (float64). The judge's `_oracle` expands element-wise into \(a\cdot x_0+s\cdot\epsilon\), and adds four structural groups: the `floor` semantics and monotonicity of the index, the \(s=0/1\) endpoints, the \(x_i ↔ ε\) round trip, and an element-wise comparison of the ε inverted from the output against the input to pin down the coefficients. Tolerance `allclose(atol=1e-6)`.

## Complexity

Element-wise \(O(N)\), with an \(O(1)\) index computation.
