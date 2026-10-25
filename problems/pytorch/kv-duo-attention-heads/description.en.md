# DuoAttention: Splitting Retrieval Heads vs Streaming Heads

## Problem

**DuoAttention** (Xiao et al., 2024, [arXiv:2410.10819](https://arxiv.org/abs/2410.10819)) found that attention heads come in two kinds: a **retrieval head** really looks back and retrieves tokens from far away (its full KV must be kept), while a **streaming head** only cares about "the first few + the most recent few" (a StreamingLLM-style sink+window suffices). So the **least "retrieval-like" heads** are downgraded to streaming heads that keep only sink+recent, while the rest keep everything—a mixed compression **along the head dimension**.

## Rules

You are given `head_scores` of shape $(H,)$ (the more a head looks like a retrieval head, the higher its score); `seq_len` $=L$, `sink_size` $=S$, `recent_size` $=R$, and `head_compression_ratio` $\in[0,1]$: return the boolean retention mask `mask` of shape $(H, L)$.

1. Number of heads converted to streaming $n\_pruned = \lfloor H \cdot \text{head\_compression\_ratio} + 0.5\rfloor$ (half rounds up).
2. Take the $n\_pruned$ heads with the **smallest** `head_scores` as the **streaming heads** (ties: smaller head index wins). The rest are **retrieval heads**.
3. Build the mask per head:
   - **Retrieval head**: the whole row is `True` (everything kept).
   - **Streaming head**: keep only the first $S$ and the last $R$ positions, i.e. position $t$ is `True` when $t<S$ or $t\ge L-R$, and `False` in the middle stretch.

## Examples

```
H=5, L=8, S=2, R=2, head_compression_ratio=0.4
head_scores = [0.9, 0.2, 0.7, 0.1, 0.5]
n_pruned = round(5*0.4) = 2 → the two smallest heads = {3(0.1), 1(0.2)} → streaming heads 1, 3

mask:
h=0 (retrieval): [T, T, T, T, T, T, T, T]
h=1 (streaming): [T, T, F, F, F, F, T, T]
h=2 (retrieval): [T, T, T, T, T, T, T, T]
h=3 (streaming): [T, T, F, F, F, F, T, T]
h=4 (retrieval): [T, T, T, T, T, T, T, T]
```

## Constraints

- CPU; `mask` is `torch.bool`.
- If $S+R\ge L$, a streaming head is equivalent to a fully `True` row (the head and tail windows overlap and cover the whole sequence).
- When $H\cdot\text{ratio}$ lands exactly on $.5$, **half rounds up** (equivalent to $\lfloor x+0.5\rfloor$, not banker's rounding).
- The length of `head_scores` is the head count $H$; there is no batch dimension.
