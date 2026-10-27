# PEFT: QLoRA NF4 Dequantized Forward and Memory Account

## Problem

**QLoRA** (Dettmers et al., NeurIPS 2023, arXiv:2305.14314) quantizes the frozen base model to 4-bit and trains BF16 LoRA adapters attached on top of it, so that a 65B model fits into a single 48GB card for fine-tuning. The base uses the **NF4** (4-bit NormalFloat) codebook: assuming the weights follow a normal distribution, take its quantiles to split \([-1,1]\) into 16 equal-probability segments, and use each segment's quantile as the reconstruction value — information-theoretically optimal for normally distributed weights. Every **64** weights (consecutive columns of the same row) share one FP32 scale (absmax), and dequantization is

\[W_{ij}=\text{NF4}[c_{ij}]\cdot s_{i,\lfloor j/64\rfloor}\]

The NF4 codebook (paper Appendix E, 4-decimal precision):

| code | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| value | -1.0 | -0.6962 | -0.5251 | -0.3949 | -0.2844 | -0.1848 | -0.0911 | 0.0 | 0.0795 | 0.1609 | 0.2461 | 0.3379 | 0.4407 | 0.5626 | 0.7226 | 1.0 |

At forward time \(W\in\mathbb R^{d_\text{out}\times d_\text{in}}\) is dequantized and combined additively with the LoRA bypass (\(A\in\mathbb R^{r\times d_\text{in}}\) down-projection, \(B\in\mathbb R^{d_\text{out}\times r}\) up-projection):

\[y=xW^{\!\top}+\frac{\alpha}{r}\,(xA^{\!\top})B^{\!\top}\]

**Double quantization** then compresses the scales themselves: each block's FP32 absmax (0.5 bit/param overhead) is quantized to FP8 (1 byte per block), and every **256** blocks share one FP32 second-level scale. The base's storage in bytes is

\[\texttt{nbytes}=\frac{n}{2}+\frac{n}{64}+\left\lceil\frac{n/64}{256}\right\rceil\cdot 4,\qquad n=d_\text{out}\cdot d_\text{in}\]

## Task

Implement the NF4 dequantized forward: dequantize the base weights, add the LoRA bypass output, and report the base's storage in bytes after double quantization.

- Inputs: \(\texttt{codes}\in\{0,\dots,15\}^{d_\text{out}\times d_\text{in}}\) (\(d_\text{in}\) a multiple of 64), \(\texttt{absmax}\in\mathbb R^{d_\text{out}\times(d_\text{in}/64)}\), \(x\in\mathbb R^{N\times d_\text{in}}\), \(A\in\mathbb R^{r\times d_\text{in}}\), \(B\in\mathbb R^{d_\text{out}\times r}\), real \(\alpha\), positive integer \(r\).
- Output \((y,\ \texttt{nbytes})\): \(y\in\mathbb R^{N\times d_\text{out}}\) (float32), \(\texttt{nbytes}\) an integer.

**Convention**: compute in float64 internally, output float32; \(\lceil\cdot\rceil\) is the integer ceiling; the quantization (encoding) direction is out of scope. Tolerance \(10^{-5}\).

## Hints

- Dequantization is elementwise table lookup times scale: the block's column boundary is \(j\,\text{//}\,64\).
- The LoRA bypass and the quantized base are **additive**: compute each separately, then add.
- The byte count has three parts: weights \(n/2\), one FP8 byte per block scale (\(n/64\) of them), and second-level scales at one FP32 (4 bytes) per 256 blocks.
