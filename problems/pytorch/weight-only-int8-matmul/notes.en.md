# Weight-Only INT8 Dequantized Matrix Multiplication In Depth

## Why "weight-only"

In inference the memory/bandwidth bottleneck is usually the weights: across one forward pass all activations
are read only once, while each layer's weights are shuttled repeatedly for every token. Weight-only
quantization compresses the weights to int8/int4 (size ÷2~4) while keeping activations in fp16 — **the accuracy
loss is almost nil** (activations have a large dynamic range; quantizing them is what hurts accuracy), and the
implementation is simple too (no need to calibrate the activation distribution). This is the default
W8A16/W4A16 tier of llama.cpp and AutoGPTQ/AWQ.

## Pulling the scale out of the matrix multiply

$$\text{out}[m,n]=\text{w\_scale}[n]\sum_k x[m,k]\,Wq[n,k]$$

The per-channel `w_scale[n]` is independent of the summation variable $k$, so we can first compute
`x @ Wqᵀ` (accumulated in integer / low precision) and finally **multiply by a vector along the output
channels**. Two benefits:

1. There is no need to dequantize the whole int8 weight to fp16 and store another copy (saving memory); the
   accumulator in the kernel multiplies by the scale directly.
2. This is exactly the mathematical form of the GPU int8 tensor-core pattern "int32 accumulate → per-channel
   rescale"; W4A16 is the same, only with nibble unpacking.

This pull-out formulation is the solution; `F.linear(x, Wq.float()*w_scale[:,None])` is the equivalent
"materialize the dequantized weight first" reference, and the two floating-point results agree within a `1e-5`
tolerance (`check_matches_oracle`).

## Dimensions and conventions

- Weights $[N,K]$ ($N$ outputs, $K$ inputs), `w_scale` has length $N$ — **per output channel**, not the input
  dim; don't multiply along the wrong axis (`[:, None]` vs `[None, :]`).
- int8 covers -128..127; `Wq.float()` is lossless (float32 represents every integer in that range exactly).
- Transpose: `x[M,K] @ Wq[N,K]ᵀ -> [M,N]`, consistent with `F.linear` (which multiplies by `Wᵀ` internally).

## Relation to the other quantization problems

- `per-channel-quantize`: produces the `Wq` and `w_scale` that are inputs here (the true integer version of
  the fake-quant step).
- `group-wise-quantize`: a finer `w_scale` (multiple groups per channel), so the matmul must switch scale per
  group.
- This collection connects "how to quantize" (forward fake-quant) to "how to compute once quantized"
  (inference matmul).

## Grading

- oracle: `F.linear(x, Wq.float()*w_scale[:,None])`.
- `_ref` recomputes with a triple `einsum`/explicit loop, a different path from the solution's "matrix
  multiply + column rescale".
- There are also float64, single-output-channel and random-shape cases; none modifies the inputs.
