# FlashAttention Online Softmax Explained

## Why Tiling + Online Reduction

Standard attention must explicitly store the full $L_q\times L_k$ matrix of $P=\text{softmax}(QK^\top)V$, which costs $O(L^2)$ memory,
and repeatedly reading and writing HBM is the bottleneck. FlashAttention puts query tiles into SRAM, streams and accumulates over K/V blocks for each tile,
**never materializing the full attention matrix**, reducing memory to $O(L)$ and making IO rather than compute the bottleneck.

## The Online (Streaming) Softmax Recurrence

The softmax denominator needs the global maximum, but when processing in blocks you cannot see all the data. The solution is to maintain a triple and update it with each new block:
for the current block's logits $s$,

$$ m_{new}=\max(m_{old}, \max_j s_j),\quad \alpha=e^{m_{old}-m_{new}} $$
$$ l_{new}=\alpha\, l_{old} + \sum_j e^{s_j-m_{new}},\quad \text{acc}_{new}=\alpha\, \text{acc}_{old} + e^{s-m_{new}}V $$

Finally $\text{out}=\text{acc}/l$. $\alpha$ is the key: when a larger $m$ appears, the **previously** accumulated $l,\text{acc}$ are
rescaled to the new reference, which is equivalent to bringing all blocks under the same subtrahend $m$.

## Pitfalls in This Problem

- **First block $m_{old}=-\infty$**: $\alpha=\exp(-\infty-m_{new})=0$, which exactly keeps the initial zero accumulators at zero; do not produce NaN.
- **Rescale order**: you must first use the old $m$ to compute $\alpha$ and $p=e^{s-m_{new}}$, then update $m\leftarrow m_{new}$; the wrong order skips the rescaling.
- **Non-divisible lengths**: the last block is not full `block`; cover it with the slice `min(i0+block, L)`, and do not assume divisibility.
- **$L_k \neq L_q$**: the number of query blocks follows $L_q$ and the key loop follows $L_k$; the two are independent.
- The large-logits scenario is a probe for "forgetting to subtract $m$"—without subtracting the max, `exp(±880)` goes straight to inf.

## Differences from a Real Implementation

Real FlashAttention also fuses the backward recomputation (not storing P) with the forward pass into a kernel, and performs warp-level tiling;
this problem only requires numerical equivalence of the forward pass—understanding "rescale running max" is the core. Causal masking makes each query
block iterate only over key blocks near the diagonal.
