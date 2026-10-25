# Sequence/Context Parallelism: Activation Sharding and the Communication Account

## Problem

Tensor parallelism shards along the hidden dimension and pipeline parallelism along the layer dimension, whereas **sequence / context parallelism** shards along the **sequence-length** dimension: a long sequence is cut into several segments distributed to different ranks, and each rank holds only the activations over its own segment, thereby amortizing the most memory-hungry activations (and the handling of very long contexts in attention) across multiple GPUs. Sequence parallelism in Korthikanti et al. (2022) and Ring Attention (Liu et al., 2023) both take this route. The cost is that operators needing to "see the whole sequence", such as attention and reductions, must shuttle data back and forth between the sharded and unsharded regions using **All-Gather + Reduce-Scatter**.

This problem computes a simplified account: given the total activation volume for one full step on a single GPU, the fraction that can be sharded along the sequence, and the parallelism degree, find the **per-rank activation memory** and the **per-step communication bytes**.

## Model (follow this definition exactly)

Let the **total activation memory** of one forward step on a single GPU be $A=\texttt{total\_activation}$ (bytes, already including all layers).

- **Shardable fraction** $f=\texttt{shardable\_fraction}\in[0,1]$: of which $f\cdot A$ is activations **that can be sharded along the sequence dimension**, and $(1-f)\cdot A$ is activation that cannot be sharded and must be **fully replicated** on every rank.
- **Parallelism degree** $C=\texttt{cp\_degree}$ (number of ranks). The shardable part is split evenly across the $C$ ranks, each keeping only $\frac{f\cdot A}{C}$; the unshardable part is fully replicated on each rank as $(1-f)\cdot A$. Hence the per-rank activation memory is:
  $$ \text{per\_rank} = (1-f)\cdot A + \frac{f\cdot A}{C} $$
- **Communication**: entering and leaving the "full-sequence view" from the sharded region requires collectives; in each training step all layers' sharded activations are aggregated and subjected to one **All-Gather** and one **Reduce-Scatter**. Going around the $C$ ranks in a ring, one such primitive moves, per rank and with ring coefficient
  $$ c=\frac{C-1}{C} $$
  times the data volume. Hence the per-rank per-step communication bytes are:
  $$ \text{comm} = 2\cdot(f\cdot A)\cdot c = 2\,f\,A\cdot\frac{C-1}{C} $$
  (2 = one All-Gather plus one Reduce-Scatter; the same ring-coefficient convention as `distributed-zero-communication`.)

Degeneracy check: when $C=1$, $c=0$, the shardable part needs no distribution, $\text{per\_rank}=A$ and $\text{comm}=0$.

## Rules

Implement `solve(total_activation, shardable_fraction, cp_degree)` and return two float byte counts `(per_rank, comm)`.

- `total_activation` $A$: the total activation bytes for one full step on a single GPU;
- `shardable_fraction` $f\in[0,1]$: the fraction shardable along the sequence;
- `cp_degree` $C\ge 1$: the sequence/context parallelism degree (an integer rank count).
- When $C=1$, `comm` must be `0.0` and `per_rank == A`.

## Examples

```
A=1000, f=0.8, C=4
  per_rank = 0.2*1000 + 0.8*1000/4 = 200 + 200 = 400
  comm     = 2 * 0.8*1000 * (3/4) = 2*800*0.75 = 1200
→ (400.0, 1200.0)

A=8000, f=0.75, C=8
  per_rank = 2000 + 6000/8 = 2750.0
  comm     = 2*6000*7/8 = 10500.0
→ (2750.0, 10500.0)
```

## Constraints

- `total_activation ≥ 0`; `0 ≤ shardable_fraction ≤ 1`; `1 ≤ cp_degree ≤ 4096` (integer)
- Return `(float, float)`, compared within relative error $10^{-6}$
- Compute only with the simplified model above; the attention ring-exchange schedule of specific layers, overlap, and real bandwidth are not involved
