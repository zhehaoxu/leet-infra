# In Depth: CFG Rescaling and the Guidance Interval

## Origin

Lin et al., *Common Diffusion Noise Schedules and Sample Steps are Flawed* (arXiv:2305.08891) §3.2 introduces guidance rescaling by \(\operatorname{std}(\epsilon_c)/\operatorname{std}(\hat\epsilon)\) to fix oversaturation at large \(w\); Kynkäänniemi et al., *Applying Guidance in a Limited Interval Improves Sample and Distribution Quality* (arXiv:2211.11559) contribute the guidance interval. A related but different trick is Imagen's dynamic thresholding (clipping \(x_0\)) — do not conflate them.

## Core points and pitfalls

1. **The std is one scalar per sample**: reduce over (channel × spatial) within each batch element, so `dims` must exclude dim 0. A whole-batch std couples samples and immediately fails the "scaling is one constant per sample" assertion.
2. **Population, not unbiased, std**: the convention is `unbiased=False`. The default \(N-1\) normalization introduces a systematic \(\sqrt{N/(N-1)}\) bias in every sample.
3. **Magnitude only, not mean**: \(\hat\epsilon\cdot r\) scales the mean too; "subtract the mean, scale, add back" is a different algorithm and breaks the \(\varphi=0.5\) blend assertion.
4. **The interval is closed**: both \(t=t_{\text{low}}\) and \(t=t_{\text{high}}\) guide; the judge samples both endpoints. Outside it the output must equal `eps_cond` exactly (not approximately).
5. **Zero std must raise**: with both branches constant the rescaling becomes 0/0 or inf, so `ValueError` is required — a real "degenerate batch" failure seen in samplers.
6. **The φ blend is applied after scaling**: compute \(\hat\epsilon\cdot r\) and then interpolate with \(\hat\epsilon\), which equals \(\hat\epsilon\cdot((1-\varphi)+\varphi r)\); numerically the same, but only the first form expresses "partial correction".

## Verification approach

The reference solution vectorizes in float64 and reduces with `torch.std(..., unbiased=False)`. The judge's `_oracle` flattens each sample to a python list and computes the mean and mean square deviation itself (no `torch.std`), adding four structural groups: exact identity at \(w=1\) and outside the interval, guidance active at the interval endpoints, the std match at \(\varphi=1\) plus the linear blend at \(\varphi=0.5\), and explicit `dims` reducing only the last axis. Tolerance `allclose(atol=1e-5)`.

## Complexity

Element-wise \(O(N)\), plus one per-sample variance reduction (also \(O(N)\)).
