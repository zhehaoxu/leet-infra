# Solution notes · Bias Compensation: streaming per-channel output correction

## Provenance

- Bias Compensation (Gong, Zheng, Hu, Lin, Fan, Zhang & Li, CAAI AIR 2024, arXiv:2404.01892, "Minimize Quantization Output Error with Bias Compensation"): instead of optimizing the non-convex quantization procedure, it appends a per-channel bias **after** the quantized layer and reads it straight off the "float output minus quantized output" difference on calibration data; the paper shows that minimizing the output error this way is convex, so no training or fine-tuning is needed.
- Every line of the rules here is transcribed from `bias_compensation/quantizers/BiasCompensation.py` in the official repository `GongCheng1919/bias-compensation` (obtained on 2026-10-30 through a `codeload` tarball, HTTP 200, checked clause by clause against this note): `compute_shape`'s "normalize negative axes by `output.dim()`, write the size at channel axes and 1 elsewhere, append non-channel axes in ascending order to `mean_axis`"; `update`'s "`self.nsamples += 1` happens before `(bias*(nsamples-1)+bias_update)/nsamples` is read"; the `post_bias=True` branch `float_output - quantized_output + self.bias.view(bias_shape)`; and the single `if len(self.mean_axis) > 0:` guard around the mean. `forward`'s `output += bias.view(self.bias_shape)` is exactly what `corrected_output` means here.
- **Listing confirmed**: the entry sits inside the `### Quantization` section of HuangOwen/Awesome-LLM-Compression with `[[Paper]](arxiv 2404.01892) [[Code]](GongCheng1919/bias-compensation)` (the README was fetched directly on 2026-10-30 and that line is present).

## Source fidelity and deliberate simplifications

- **The multi-batch presentation changed.** Upstream is a `torch.nn.Module` plus forward hooks: each calibration forward triggers one `update()` call and the state (`bias`, `nsamples`) is carried implicitly between calls. This exercise hands all $T$ forwards at once along axis 0 of `float_batches`/`quant_batches` and additionally returns the trajectories `bias_updates` and `bias_after`, so that judging only looks at the returned value. The upstream behaviour "the counter grows by exactly 1 per call" is kept verbatim, which is why a larger calibration batch does **not** make an update count more.
- **The `act_func` / `clone_output` branches are dropped.** They are hook plumbing (add the bias then apply the activation, or clone before mutating) and irrelevant to the bias estimate, so this exercise has no "error measured after the activation" branch.
- **`channel_size` became a validation instead of an assertion.** Upstream's `compute_shape` raises `AssertionError` from `assert np.prod(bias_shape) == channel_size`; following the house convention this exercise raises `ValueError`. `channel_size=None` is an addition of this exercise (upstream requires it at construction) and exists only so that the "derive the channel count from the shape" case is testable too.
- **`init_bias` and the `T=0` boundary are additions of this exercise.** Upstream's `bias` is an `nn.Parameter` initialized to zeros with no way to seed it, and it never sees an empty calibration set. Here `init_bias` is allowed (it becomes $b_0$) and $T=0$ is rejected, so both edges can be probed. Its reach splits by mode: with `post_bias=False` the first update multiplies it by weight 0, so it changes nothing at all, while with `post_bias=True` it enters the observation through the bias added back into $\Delta_0$ and keeps coefficient exactly 1 at every later step, so the whole trajectory shifts by `init_bias`.
- **`residual_mse` is a metric invented by this exercise**; the source has no such quantity (the paper reports downstream task numbers). `corrected_output` applies the **final** bias to every forward, whereas the real `forward` applies the bias as it stands **at that moment** (it moves every step) — an intentional de-sequencing to keep the returned value unique.
- **float64 inside, input dtype on the way out.** Upstream keeps float32 module state; here the mean and the streaming average run in float64 and are cast back to the input dtype, so the pure-Python oracle and the torch implementation only have to agree at the final cast.
- Out of scope: the paper's `scale` correction side path (commented out in the source), the `Ternary`/`fixedpointQ`/`ul2ptq` quantizers, layer-level mounting and evaluation on ViT/LLM, and anything that would require a real model forward.

## The core and the traps

- The denominator when normalizing a negative axis is the **single-forward rank** $d=\text{dim}-1$, not the tensor's `dim()`; on $(T,B,C,S)$, `channel_axis=-2` means $C$, not $B$.
- `bias_shape` describes a single forward only (length $d$); axis 0 ($T$) is not part of it. Broadcasting as if it were turns $T$ into the channel count.
- `nsamples` is **incremented before it is used**: the first step is $(b_0\cdot 0+u_0)/1$, i.e. the initial value is overwritten; only from the second step on is it a real running mean. Writing `(bias*(t-1)+u)/t` with a 0-based `t` either divides by zero or weights wrongly.
- With `post_bias=True` the quantity added back is the bias after the **previous** update, not this one; using `bias_after[t]` instead gets everything wrong.
- When `mean_axis` is empty the upstream code **does not average** and just flattens, so `bias_updates[t]` has one entry per element of a single forward rather than one per channel.
- `bias_updates` and `bias_after` are different objects: the first is this step's observation, the second is that observation blended with the old bias. Judging checks them separately, so conflating them fails.
- What `init_bias` does is decided entirely by `post_bias`: with `post_bias=False` no $\Delta_t$ contains the bias, so the very first update multiplies the seed by weight 0 and it never reaches the output; with `post_bias=True` it enters $\Delta_0$ and its coefficient stays exactly 1 thereafter, so the whole trajectory is the zero-seed trajectory shifted by `init_bias`. Treating it as "just a starting guess" with regularization-style intuition is wrong.
- Enlarging the calibration batch leaves the result unchanged (that axis is averaged away), but growing $T$ from 2 to 3 changes every subsequent sample's weight $1/(t+1)$.

## Cross-checking approach

- The tests use an independent pure-Python `_oracle`: three nested `for` loops compute the difference, sum and divide along the `mean_axis` list one axis at a time, then blend with weight $1/(t+1)$ — plain `math`-level arithmetic compared against the torch implementation in float64; tolerance `1e-5` for float32 and `1e-9` for float64.
- The three tuple keys (`channel_axis_norm` / `bias_shape` / `mean_axis`) are asserted exactly, covering negative axes, several channel axes, and the empty `mean_axis` case.
- Additional probes: dtype sweep (float32/float64 agree within tolerance), non-contiguous views (transposed or strided inputs give the same result), inputs not mutated, invalid inputs (shape/dtype mismatch, empty axis tuple, out-of-range or duplicate axes, wrong `channel_size`, wrong `init_bias` length, $T=0$), and degenerates ($F=Q$ with a zero seed keeps the bias identically zero; $T=1$ gives `bias == bias_updates[0]`; `post_bias=False` and `True` coincide when $T=1$).

## Complexity

$O(T\cdot \lvert R\rvert)$ where $\lvert R\rvert$ is the element count of one forward: one difference, one reduction over the given axes, one weighted update of length $C$. No iterative solver, no matmul, no memory peak.

## Related problems

- `quant-rtn-gptq-error`, `quant-adaround-recon-search`: both also attack the output error introduced by quantization, but the first redistributes error column by column inside the layer and the second searches the rounding direction; this problem **never touches the quantizer** and only appends a bias, which is the third route.
- `quant-hqq-scale-solve`: likewise a closed form, but for the scale; here the unknown is the output bias, plus the streaming counter trap.
- `quant-omniquant-transform-clipping`: OmniQuant's equivalent transform also reduces output error, but through grid-searched initializations of learnable parameters; this update is purely statistical and exact in one step.
