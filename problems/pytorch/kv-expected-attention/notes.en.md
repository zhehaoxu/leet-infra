# Solution Notes · ExpectedAttention

## Deriving the Expected Attention

If $q \sim \mathcal{N}(\mu, \Sigma)$, then for a fixed $k$: $q^\top k \sim \mathcal{N}(\mu^\top k,\, k^\top \Sigma k)$. Hence:

$$ \mathbb{E}[e^{q^\top k}] = e^{\mu^\top k + \frac{1}{2} k^\top \Sigma k} \quad\text{(Gaussian moment-generating function)} $$

Dividing by $\sqrt{d}$ gives $u_{h,n}$ in the problem.

## RoPE Averaging

Real kvpress first averages the RoPE rotation matrices over the future $n_\text{future}$ positions to get $\bar{R}$, then transforms $\mu$ and $\Sigma$ into the "future coordinate frame" (`mu ← mu @ R.T`, `cov ← R @ cov @ R.T`). **For easy cross-checking, this problem feeds the already-transformed mu / cov as input**—the RoPE part is not implemented.

## vnorm Scaling

Raw attention scores only reflect "how much attention", not "how large the value is". Scaling by $\|v\|_2$ pushes pairs with "high attention × large value" to the front. Adding $\varepsilon$ prevents the score from collapsing to zero when $\|v\|=0$ or $\text{softmax}=0$.

## Sink Pad

The first `n_sink` positions (default 4) get the top score regardless of content—protecting the attention sinks exactly as StreamingLLM does. The fill value is the **global max + 1**, so topk(n_kept) always retains them first.

## kv-group

A real implementation has `num_key_value_groups = num_heads / num_kv_heads` and averages along the group after softmax. This problem fixes `num_heads = num_kv_heads` (no group folding).
