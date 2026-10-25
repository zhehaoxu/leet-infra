# Asymmetric Quantization Explained

## Quantization mapping

Symmetric quantization assumes the data is symmetric about 0 and wastes half the range covering a side that does not exist. Asymmetric quantization uses an **affine** map to shift the whole $[\min,\max]$ segment onto $[q_{\min},q_{\max}]$:

$$ s=\frac{\max-\min}{q_{\max}-q_{\min}},\qquad z=\operatorname{round}\!\Big(\frac{-\min}{s}\Big) $$
$$ q=\operatorname{clamp}(\operatorname{round}(x/s)+z,\;q_{\min},\;q_{\max}),\qquad \hat x=(q-z)\,s $$

uint8 uses $[0,255]$ ($q_{\max}-q_{\min}=255$). $z$ is "which integer grid point float 0 lands on".

## When to use it

- **Activations after ReLU are all $\ge0$**: the symmetric version wastes nearly half of the 255 grid points (the negative half-idle), while the asymmetric version spreads $[0,\max]$ across $[0,255]$, a significant accuracy gain.
- The same benefit applies to clearly skewed distributions (e.g. feature maps with a positive bias).
- Weights are usually approximately zero-mean, so symmetric is more common there (saves a zero_point, and matmuls need not handle the zero-point offset).

## zero_point must be clamped

$z=\operatorname{round}(-\min/s)$ goes outside $[0,255]$ when the tensor is **all positive** ($\min>0\Rightarrow z<0$) or **all negative** ($\max<0\Rightarrow z>255$). `torch.fake_quantize_per_tensor_affine` requires $z\in[q_{\min},q_{\max}]$ and errors otherwise. The standard fix is to clamp $z$ into range — the cost is that for extremely one-sided distributions the $\min$ side is no longer restored exactly, but the grid points remain in the legal domain.

## Quantization noise and the all-zero degeneration

- The round-trip error is $\le s/2$, and $s$ is determined by the range: **outliers stretch $[\min,\max]$ → $s$ grows → overall precision drops**. Mitigations are percentile/clipping (use the 99.99% percentile instead of the true max) or per-channel.
- For a constant tensor $\max=\min$ → $s=0$ → division by zero, so it must be special-cased and returned unchanged.

## Trade-offs versus the symmetric version

| | Symmetric | Asymmetric |
|--|------|--------|
| scale | absmax/127 | (max-min)/255 |
| zero_point | always 0 | must be stored/computed |
| suited to | near-zero symmetric (weights) | skewed (activations) |
| matmul | clean | must handle the zero-point cross term |

## Related

symmetric-quantize is the $z=0$ special case; the fake-quant concept (STE forward) is shared by both problems; the quantization family is central to on-device deployment and inference speedups.
