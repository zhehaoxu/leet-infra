# RMSNorm in Detail

## Why It Is the Standard in LLMs

LayerNorm does two steps: subtracting the mean (re-centering) and dividing by the standard deviation (re-scaling), plus the two parameters $\beta,\gamma$.
Zhang & Sennrich (2019) found that **subtracting the mean contributes little to the outcome**, so RMSNorm keeps only the re-scaling:

$$ \text{RMS}(x)=\sqrt{\tfrac1d\sum_j x_j^2},\qquad \text{out}_i = \frac{w_i\,x_i}{\text{RMS}(x)+\epsilon\ (\text{inside the square root})} $$

- One fewer mean computation and one fewer bias parameter → faster and more memory-efficient;
- It is sensitive to the **scale** of token vectors but indifferent to mean shifts, which suits exactly the per-position normalization that Transformers need.

LLaMA, Qwen, Mistral, and DeepSeek all use it, always as **pre-norm** (normalize before entering attn/mlp).

## Three Pitfalls When Replicating transformers Semantics

1. **Compute in float32**: HF first upcasts `hidden_states` to fp32 to compute the mean square and the normalization, then applies `.to(input_dtype)`, and only at the end multiplies by `weight`. If you stay in low precision throughout, or change the order of the multiplication with weight, boundary values drift slightly.
2. **eps goes inside the square root**: `x * rsqrt(variance + eps)`, not `(x / rsqrt(variance)) + eps`.
3. **No mean subtraction**: `variance` is really the **mean square** `mean(x^2)`, not `mean((x-mean)^2)`. Writing LayerNorm's variance instead is wrong.

## Adjacent Concepts

- **LayerNorm**: subtract the mean + divide by the standard deviation + $\beta$; see `layernorm-forward` / `batchnorm-forward`.
- **Later in this collection**: the two `Qwen2RMSNorm` pre-norms in `llm-decoder-layer` and the final `model.norm` in `llm-qwen2-model` all use this implementation.
- Qwen3's **QK-Norm** also wraps q/k in an RMSNorm layer (`llm-qk-norm`, planned).

## Cross-check Criteria

The judge uses `transformers`' `Qwen2RMSNorm` (same weights, same eps) as the oracle, with `allclose` relative tolerance $10^{-5}$.
The reference solution is hand-written and does not call HF; the tests additionally recompute with an independent formula in double precision as a cross-check, giving dual assurance of correctness.
