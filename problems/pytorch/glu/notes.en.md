# GLU (Gated Linear Unit) Explained

## Definition

Split the input **into two halves** along the last dimension: one half is the value, the other is the gate:

$$ \text{glu}(x)=A\otimes\sigma(B),\qquad x=[A,B]\ \text{split along dim} $$

$\sigma$ is the sigmoid (this level); variants with ReLU/tanh gates also exist.

## Why It Matters

Gating is at the core of modern Transformer variants: the GLU family (GeGLU, SwiGLU) replaces the FFN activation with a gated form and beats a plain ReLU/GELU FFN. **LLaMA uses SwiGLU**—the gate is SiLU and the hidden dimension is cut to $2/3$ to keep the parameter count. Understanding GLU is a prerequisite for understanding these architectures.

## Implementation and Splitting Pitfalls

1. **Split dimension and shape**: `x.shape[dim]` must be even, with $A=x[\dots,0{:}h]$, $B=x[\dots,h{:}2h]$ and $h=\text{shape}[dim]/2$. `chunk(2, dim)` does it in one step.
2. **The gate only applies to its own half**: the first half is gated and the gate function comes from the second half—do not swap them, otherwise every value changes.
3. **The output dimension is halved**: this is intrinsic to GLU, not a bug.
4. sigmoid numerical stability: $\sigma$ has built-in protection; large negatives → 0, large positives → 1.

## Validation Advice

The reference implementation pairs `torch.sigmoid` with `chunk`, matching a hand-written version. Note that `dim` may be negative (e.g. `-1`), so slicing must use the resolved axis.
