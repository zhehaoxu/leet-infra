# QuaRot: Hadamard-Rotation Quantization

## Problem

**QuaRot** (Ashkboos et al., 2024, [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)) cures outliers with an **orthogonal rotation**: since outlier channels blow up the quantization range, rotate the vector into a new coordinate system where "no coordinate stands out" and quantize there. A **Hadamard matrix** is chosen as the rotation because every entry is $\pm 1/\sqrt{n}$ — a single multiply spreads one outlier coordinate evenly across all coordinates and markedly reduces kurtosis. The rotation is an orthogonal transform ($H H^\top = I$), so multiplying by $H$ again after quantization returns to the original coordinate system.

## Rules

Given a weight $W$ of shape $(O, C)$ where $C$ must be a **power of two**, and a bit width `bits`.

1. Build the $C\times C$ **normalized Hadamard** matrix $H$ recursively:
   $$H_1=[1],\quad H_{2m}=\begin{bmatrix}H_m & H_m\\ H_m & -H_m\end{bmatrix},\quad H=\frac{1}{\sqrt{C}}H_C$$
   $H$ is symmetric and orthogonal, so $H^{-1}=H^\top=H$.
2. **Rotate**: $\widetilde{W} = W\,H$ (right-multiply each row by $H$).
3. **Quantize**: apply per-tensor symmetric RTN to $\widetilde{W}$ (scale $=\max|\widetilde{W}|$, $q_{max}=2^{bits-1}-1$, round to nearest and clamp, then dequantize) to get $\widetilde{W}_q$.
4. **Rotate back**: $\hat{W} = \widetilde{W}_q\,H$.

Return $\hat{W}$ of shape $(O, C)$.

## Examples

```
W = [[1.0, 2.0, 3.0, 4.0],
     [0.5, -1.0, 2.0, 0.0]],  C=4, bits=8

H = (1/2)·[[1, 1, 1, 1],[1,-1, 1,-1],[1, 1,-1,-1],[1,-1,-1, 1]]
Ŵ = W·H → rows are more balanced after rotation (outliers are spread out)
apply 8-bit RTN to Ŵ (nearly lossless) → Ŵ_q
Ŵ = Ŵ_q·H restored:
≈ [[1.0039, 1.9882, 3.0118, 3.9961],
   [0.4921, -1.0039, 1.9882, 0.0197]]
```

(At 8 bits, rotation + inverse rotation is nearly reversible; with fewer bits you can observe that once outliers are "spread out" the overall error becomes more uniform.)

## Constraints

- CPU float32; `allclose(atol=1e-4)`.
- $C$ must be a power of two ($1,2,4,8,\dots$); normalize $H$ with $\frac{1}{\sqrt C}$.
- RTN rounds to nearest with ties to even (same as `torch.round`).
