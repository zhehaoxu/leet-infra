# PEFT: QLoRA NF4 Dequantized Forward and Memory Account — Algorithm Notes

## 1. The NF4 codebook and its quantile construction

NF4 in QLoRA (Dettmers et al., NeurIPS 2023, arXiv:2305.14314) is "fixed 16-value codebook + block scale" lookup quantization: the 16 reconstruction values are the quantiles of the standard normal distribution within \([-1,1]\) with equal probability mass (1/16 per segment), hence information-theoretically optimal for normally distributed weights (the paper's ablation against uniform int4/int8 grids exists precisely for this design). Division of labor with #85① in this collection (MX FP8/E8M0 microscaling): MX is a floating-point format with "per-group shared exponents", NF4 is a fixed-codebook lookup — the two share the block structure (64) but differ in mechanism.

Three anchors of the codebook: code **7** is exactly \(0.0\) (the symmetric midpoint of the normal), codes **0/15** are \(\pm1\) (the saturation boundary) — `check_nf4_code_values` asserts them as independent anchors.

## 2. Block scaling and double quantization (judged points)

- Basic block quantization: one FP32 absmax per 64 weights (consecutive columns of a row), overhead \(32/64=0.5\) bit/param.
- **Double quantization**: each block's FP32 absmax is quantized to FP8 (1 byte), and every 256 blocks share one FP32 second-level scale:

\[8/64+32/(64\cdot 256)=0.125+0.001953\approx 0.127\ \text{bit/param}\]

  saving 0.373 bit/param over 0.5 — the paper uses this to compress a 65B model's scale overhead from the ~4GB order down to the ~1GB order.
- This problem's byte count (\(n\) = number of weights, \(d_\text{in}\) a multiple of 64):
  - Weights: \(n/2\) (4-bit packing, \(n\) even)
  - First-level scales: \(n/64\) FP8 values → \(n/64\) bytes
  - Second-level scales: \(\lceil (n/64)/256\rceil\) FP32 values → \(\times 4\) bytes

`check_bytes_account` asserts \(37/136/1324\) for \((1,64)\), \((2,128)\), \((4,640)\).

## 3. Forward composition (judged points)

QLoRA forward = dequantized base (frozen) + BF16 LoRA bypass (trainable):

\[y=xW^{\!\top}+\frac{\alpha}{r}(xA^{\!\top})B^{\!\top}\]

In the engineering implementation (bitsandbytes `Linear4bit`) dequantization happens on the fly inside the kernel and the full \(W\) is never materialized — this problem materializes \(W\) for judging determinism. `check_lora_branch_scales` asserts the bypass output is strictly linear in \(\alpha\):

\[y(2\alpha)-y(0)=2\,\bigl(y(\alpha)-y(0)\bigr)\]

pinning down the semantics "quantized base frozen, bypass additive"; getting either path wrong (e.g. multiplying the scale into the base, or misremembering \(\alpha/r\) as \(\alpha\)) breaks this identity.

## 4. Division of labor within the collection

- `peft-lora-forward`/`peft-lora-merge`: LoRA forward and merge on a non-quantized base.
- `#85①` (MX/E8M0): the scaling primitive of another quantization format — do not copy it into the NF4 codebook or vice versa.
- This problem: QLoRA's "dequantized forward pass + double-quant memory account"; the quantization (encoding) direction, pivot-vector backprop, and the paged optimizer are out of scope.

## 5. Verification approach

- `check_example`: \(1\times64\) single block, absmax \(=2\), all-7 base (code value 0.0) plus three special code positions (0/8/15) → \(y=0.159\), nbytes \(=37\).
- `check_matches_oracle`: random \(4\times128\), \(N=3\), compared against an independent scalar-by-scalar recomputation (float64 internally, float32 out, tolerance \(10^{-5}\)).
- The code anchors, the byte account, and the bypass linearity each get fixed assertions.
