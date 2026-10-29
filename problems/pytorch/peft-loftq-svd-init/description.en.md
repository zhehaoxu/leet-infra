# PEFT: LoftQ Quantized Base with Alternating SVD Initialization

## Description

**LoftQ** (LoRA-Fine-Tuning-aware Quantization, arXiv:2310.08602) quantizes the base weight while initializing LoRA \(A/B\) via alternating SVD so that:

\[W_0 \approx W_q + B A,\]

where \(W_q\) is the quantized base. The alternating procedure is:

1. Symmetric absmax quantize \(W_0\) to \(b\) bits, obtaining \(W_q^{(0)}\).
2. For iteration \(t=0,\dots,\text{iterations}-1\):
   - Residual \(R = W_0 - W_q^{(t)}\).
   - SVD of \(R\): \(R = U S V^\top\).
   - Take the top \(r\) singular values: \(s_r = \sqrt{S_{:r}}\).
   - Initialize \(B = U_{:, :r} \cdot s_r\), \(A = s_r \cdot (V^\top)_{:r, :}\).
   - Update quantized base \(W_q^{(t+1)} = \operatorname{quantize}_b(W_0 - B A)\).
3. Return merged weight \(W' = W_q^{(T)} + (\alpha/r)\,B A\).

When `iterations=0`, \(B,A\) are zero matrices and \(W'\) equals the \(b\)-bit quantization of \(W_0\).

## Task

Given \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\), rank \(r\), scaling \(\alpha\), quantization bits `bits` (positive integer), and alternating rounds `iterations` (non-negative integer), return the merged weight \(W'\) following the LoftQ procedure above.

**Quantization function** is symmetric absmax quantization:

\[\operatorname{quantize}_b(W) = \mathrm{round}\!\left(\frac{W}{s}\cdot q_\max\right)\cdot \frac{s}{q_\max},\qquad s=\max|W|,\ q_\max=2^{b-1}-1.\]

When \(s=0\) return \(W\) directly (an all-zero matrix needs no quantization).

- `iterations=0` degenerates to pure quantized base.
- \(\alpha=0\) also returns only the final quantized base \(W_q\).
- Output \(W'\in\mathbb R^{d_\text{out}\times d_\text{in}}\), float32.

## Hint

- Use `torch.linalg.svd(R, full_matrices=False)` to obtain \(U,S,Vh\) (where \(Vh=V^\top\)).
- \(B A\) equals \(U_{:, :r}\,\operatorname{diag}(S_{:r})\,(V^\top)_{:r, :}\), i.e. the best rank-\(r\) approximation of residual \(R\); therefore only the final merged weight \(W'\) is judged, not the signs of \(U/V\).
- Take the square root of \(S_{:r}\) and multiply it onto \(U\) and \(Vh\) respectively to obtain balanced \(B/A\) initialization.
