# GPTQ: Column-wise Quantization + Hessian Error Compensation

## Problem

RTN (round-to-nearest) rounds each weight independently and ignores the **coupling** between them: the error introduced by quantizing one weight can be "absorbed" by the other weights that have not yet been quantized. **GPTQ** (Frantar et al., 2022, [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)) builds on the column-by-column coordinate-descent idea of OBQ/GPT: quantize each column in order, and immediately feed the quantization error of that column back into the remaining columns through second-order information (the inverse of the Hessian), so that the layer's overall output error stays as unchanged as possible. With a single matrix inversion plus column-wise updates, the quantization loss at 3–4 bit is greatly reduced.

## Rules

Given weights $W$ of shape $(O, C)$, a per-layer Hessian approximation $H = 2XX^\top$ of shape $(C,C)$, a bit width `bits`, and damping `damp`.

1. Per-tensor symmetric quantization parameters: $q_{max}=2^{bits-1}-1$, $s=\max|W|$ (take $1$ when $s=0$).
2. Damped inverse: $H^{-1}=(H+\texttt{damp}\cdot I)^{-1}$ ($H$ is positive semi-definite; the damping guarantees invertibility).
3. Let the working copy be $W_q \leftarrow W$, and the output $O\leftarrow W$. For column $i=0,1,\dots,C-1$:
   - Quantize the current column $w=W_{q,:,i}$ (at this point $w$ already contains the compensation propagated from earlier columns):
     $$q=\mathrm{clamp}\big(\mathrm{round}(w/s\cdot q_{max}),\,-q_{max}-1,\,q_{max}\big),\qquad \hat w=\frac{q}{q_{max}}\,s$$
   - The column error is $e = w - \hat w$; write back $O_{:,i}=\hat w$.
   - **Error compensation**: spread $e$ over the not-yet-quantized columns along row $i$ of $H^{-1}$ (with $d_i=H^{-1}_{i,i}$):
     $$W_{q,\,[:,\,i+1:]} \;-\!=\; \big(e/d_i\big)\otimes H^{-1}_{i,\,i+1:}$$

Return the reconstructed weights $\hat W = O$ of shape $(O, C)$.

## Examples

```
W = [[1.0, 2.0, 0.5],       H = [[2.0, 0.5, 0.1],
     [0.3, 1.0, 1.5]],           [0.5, 1.5, 0.2],
                                 [0.1, 0.2, 1.0]]
bits = 2, damp = 0.1
s = max|W| = 2.0, qmax = 1  →  quantization grid ∈ {-2, 0, 2} (after ×s/qmax)

Wq = [[0.0, 2.0, 0.0],
      [0.0, 2.0, 2.0]]
```

(At 2-bit the grid is extremely coarse; column-wise compensation pushes the error of the earlier-quantized columns onto the later-quantized ones, so the final $\hat W$ reproduces the original layer output more closely than plain RTN.)

## Constraints

- CPU float32; results are compared with `torch.allclose(atol=1e-3)`.
- $H$ is symmetric positive semi-definite and $C\times C$; the number of columns of $W$ must match the dimension of $H$.
- Round to nearest, ties-to-even for `.5` (same as `torch.round`); columns must be processed **left to right** (compensation only propagates forward).
