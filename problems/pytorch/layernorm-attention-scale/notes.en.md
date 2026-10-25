# QK-Norm in Depth (attention with normalized Q/K)

## The problem: exploding attention logits

When training large models, the magnitude of $q\cdot k$ keeps growing with the hidden dimension and with training progress, saturating softmax, killing gradients, and even causing loss spikes and divergence. **QK-Norm** normalizes $Q$ and $K$ separately before the dot product, locking the logit magnitude into a bounded range and markedly improving training stability (adopted by ViT-22B, Chameleon, and most modern LLMs).

## Approach in this problem: RMSNorm then dot product

$$ \tilde q=\text{RMSNorm}(q)\cdot\gamma_q,\quad \tilde k=\text{RMSNorm}(k)\cdot\gamma_k,\qquad \text{scores}=\frac{\tilde q\,\tilde k^\top}{\sqrt d} $$

$$ \text{RMSNorm}(t)=\frac{t}{\sqrt{\frac1d\sum t_i^2+\epsilon}}\quad(\text{no mean subtraction}) $$

- Normalization runs along the **last dimension $d$**, row by row per token; $\gamma$ has shape $(d,)$ and broadcasts.
- Afterwards the usual scaled dot product is applied, and $\sqrt d$ is still kept (identical to the SDPA problem).

## Why RMSNorm instead of LayerNorm

RMSNorm does not subtract the mean and saves one reduction; moreover, its **scale invariance** is exactly what is wanted here: scaling $q$ up does not change the normalized result, which suppresses norm growth. LayerNorm would also work, but RMSNorm is cheaper and stable enough. See the "RMSNorm" problem for details.

## Pitfalls

- eps goes **inside** the square root; use `torch.rsqrt(ms+eps)`.
- Normalization is done separately for $Q$ and $K$ (with different $\gamma_q,\gamma_v$); do not apply it to $V$ (V is not normalized).
- Normalize only before the dot product; do not confuse this with the post-softmax weighting.
