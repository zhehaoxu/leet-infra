# Swish / SiLU Explained

## Definition

$$ \text{swish}(x) = x\cdot\sigma(\beta x) $$

With $\beta=1$ this is **SiLU** (what this collection implements), which is also the activation of LLaMA-style models. It is a self-gating activation: it uses its own sigmoid as the gate.

## Why use it

- **Smooth and non-monotonic**: for $x<0$ there is a small negative valley (about $-0.278$ at $x\approx-1.28$), so it keeps a little negative signal unlike ReLU and does not saturate away like sigmoid; it is differentiable everywhere, giving more stable gradients.
- Experimentally it beats ReLU in deep networks, at the cost of one sigmoid.

## Implementation notes

- `x * torch.sigmoid(x)` is enough; do not write an approximation such as `x*0.5*(1+erf(...))` (the exact SiLU is $x\sigma(x)$).
- Large negative $x$: sigmoid underflows to 0, the product is 0, safe. Large positive $x$: sigmoid→1, it degenerates to the identity $x$ and cannot overflow.

## Gradient

$$ \text{swish}'(x) = \sigma(x) + x\,\sigma(x)(1-\sigma(x)) = \sigma(x)\,[1 + x(1-\sigma(x))] $$

## Related

GELU $=x\Phi(x)$ gates with the Gaussian CDF, Mish $=x\tanh(\text{softplus}(x))$ gates another way — all part of the same "$x$ × some gate" family. The three differ only in the shape of the gate.
