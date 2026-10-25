# Symmetric Quantization in Depth

## The quantization triple

Any uniform quantization is described by three quantities:

- **scale** $s$: the floating-point step each integer grid point represents.
- **zero_point** $z$: the integer grid point corresponding to float 0. Symmetric quantization has $z=0$.
- **range** $[q_{\min},q_{\max}]$: int8 symmetric usually uses $[-127,127]$ (leaving -128 for asymmetric/saturation headroom).

$$ \hat x = (q - z)\,s,\quad q=\operatorname{clamp}(\operatorname{round}(x/s),\,q_{\min},\,q_{\max}) $$

The symmetric version has $z=0$ and $s=\text{absmax}/127$.

## Why absmax/127 and not /128

int8 symmetric takes $[-127,127]$ (the positive end stops at 127), so that $\pm\text{absmax}$ both land exactly on the range boundaries, symmetric about 0. Using 128 would map $-\text{absmax}$ to -128 while $+\text{absmax}$ could only reach 127, breaking the symmetry. Many frameworks (including the symmetric convention of PyTorch `quantize_per_tensor`) use 127.

## fake-quant and STE

Inference really stores int8; during training one uses **fake-quantize**: a quantize→dequantize round trip whose values remain floats but now carry quantization noise, letting the model "get used to" low precision ahead of time. The gradient of round is 0 everywhere, so one uses the **straight-through estimator (STE)**: round in the forward pass, treat round as the identity in the backward pass. This problem only checks the forward numbers (the judge does not test gradients), but conceptually fake-quant is exactly the forward part of STE.

## Error

The round-trip error satisfies $|\hat x - x|\le s/2$ (rounding to nearest half step). The larger the absmax, the larger $s$, the larger the noise — outliers raise absmax and thereby "dilute" the precision, which is exactly what per-channel / asymmetric / clipping try to mitigate.

## Pitfalls

- **Division by zero**: for an all-zero tensor absmax=0, $s=0$ → $x/s$ is NaN, so it needs a special case returning the input unchanged.
- **Rounding convention**: use `torch.round` (banker's rounding, 0.5→nearest even), consistent with hardware/`fake_quantize`; do not use floor(x+0.5).
- clamp is necessary: numerical error can push some $|x/s|$ slightly past 127.

## Related

asymmetric-quantize is its $z\ne0$ counterpart (using min/max instead of absmax); the "clamping" idea in hardsigmoid/relu6 is akin to clamp; the quantization family is the key to saving memory and speeding up on the deployment side.
