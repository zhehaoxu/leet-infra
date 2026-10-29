# PEFT: LoftQ Quantized Base with Alternating SVD Initialization — Algorithm Notes

## 1. Formula and Source

LoftQ (arXiv:2310.08602) aims to approximate the full-precision base weight \(W_0\) as the sum of a low-bit quantized weight and a low-rank adapter:

\[W_0 \approx W_q + BA.\]

The alternating algorithm uses the residual of the current \(W_q\) to compute the best rank-\(r\) approximation (via SVD), then updates the quantization target with this approximation and repeats.

## 2. Symmetric Absmax Quantization

This problem uses a simplified per-tensor symmetric absmax quantization:

\[\operatorname{quantize}_b(W)=\mathrm{round}\!\left(\frac{W}{s}\cdot q_\max\right)\cdot \frac{s}{q_\max},\quad s=\max|W|,\ q_\max=2^{b-1}-1.\]

- When \(s=0\) return \(W\) directly.
- Only the quantization + SVD initialization flow is tested; complex codebooks such as NF4 are not used. The final merged weight does not depend on the specific quantization grid details.

## 3. SVD Initialization Details

- Residual \(R = W_0 - W_q\).
- `torch.linalg.svd(R, full_matrices=False)` returns \(U,S,Vh=V^\top\).
- \(s_r = \sqrt{S_{:r}}\), \(B = U_{:, :r} \cdot s_r\), \(A = s_r \cdot Vh_{:r, :}\).
- Therefore \(BA = U_{:, :r}\,\operatorname{diag}(S_{:r})\,Vh_{:r, :}\), i.e. the best rank-\(r\) approximation of \(R\).

## 4. Degenerate Cases and Grading Points

- `iterations=0`: \(B,A\) are zero, and \(W'\) equals the quantized result of \(W_0\).
- \(\alpha=0\): LoRA term is zero, and \(W'\) equals the final quantized base \(W_q\).
- All-zero \(W_0\): scale is zero, quantization is identity, result remains zero.
- Grading asserts only the final merged weight \(W'\), because SVD singular-vector signs / degenerate ordering are indeterminate and would fail direct assertions.

## 5. Split with Existing Problems

- `peft-qlora-nf4-forward`: forward pass after NF4 quantization (with dequantized matmul).
- `quant-double-quant-scale`: re-quantization of the scale tensor.
- This problem: LoftQ-specific "quantized base + alternating SVD-initialized LoRA" merged result.
