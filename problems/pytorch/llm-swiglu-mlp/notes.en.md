# SwiGLU MLP in Detail

## From FFN to SwiGLU

The classic FFN is $\text{ReLU}(xW_1+b_1)W_2$: expand the dimension, apply a nonlinearity, then reduce it again. Shazeer (2020) observed that, at the **same parameter count / compute budget**, adding a **multiplicative gate** to the activation (the GLU family) is clearly better. The **SwiGLU** used by LLaMA/Qwen takes SiLU (swish) as the gate:

$$ \text{FFN}(x) = \text{SiLU}(xW_{gate}) \odot (xW_{up})\,W_{down} $$

The matrices $W_{gate},W_{up}$ raise hidden to inter and $W_{down}$ lowers it back. **None of them has a bias**.

## Why Three Matrices Instead of Two

A GLU needs one branch to compute the gate and one to compute the value, plus the output projection, which gives three matrices. To keep a parameter count comparable to the old FFN,
the inter dimension is usually $\approx \tfrac{2}{3}\cdot 4\,hidden = \tfrac{8}{3}hidden$ (LLaMA uses $2\cdot\text{hidden}/3$ rounded up to a multiple of 64).

## Implementation Pitfalls

- **Elementwise multiply, not addition**: `silu(gate)*up`. Writing the gate as an addition changes both the capacity and the semantics. The test case `up=0 -> output 0` exists precisely to catch this.
- **Weight layout [out, in]**: `F.linear(x, W)=x@Wᵀ`. This matches HF `nn.Linear.weight`, so do not transpose the wrong way.
- **SiLU = z·sigmoid(z)**: the same as `swish`/`F.silu`; note it is not the parameterized `x·sigmoid(βx)` variant.

## Related

- Upstream: `llm-rmsnorm`, `llm-rope-apply`.
- Downstream: the second half of `llm-decoder-layer`, `post_attention_layernorm → mlp → residual`, uses this.
- DeepSeek-V3's MoE replaces "one MLP" with "N expert MLPs + routing", and each expert is still SwiGLU internally (`llm-moe-experts`, to be done).
- Cross-check: `transformers` `Qwen2MLP` (same weights, `hidden_act="silu"`), `allclose` $10^{-5}$.
