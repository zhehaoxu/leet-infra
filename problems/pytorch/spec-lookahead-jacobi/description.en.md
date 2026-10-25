# Lookahead Decoding: Jacobi n-gram Candidate Pool

## Problem

**Lookahead Decoding** (Fu et al., 2024, [arXiv:2402.02057](https://arxiv.org/abs/2402.02057)) needs no separate draft model. Instead it treats decoding as solving a set of Jacobi iterations: it maintains a pool of n-gram candidates (**pool**) in parallel, speculates by matching the longest suffix of the current context against continuations already seen in the pool, falls back when the match fails, and simultaneously writes the newly generated n-gram → successor relations back into the pool, learning as it goes. In this problem you implement the core loop of **pool matching + deterministic fallback + pool update** (the details of the parallel n-gram branches are omitted so that the focus is on single-stream candidate lookup and learning).

## Rules

Inputs:
- `prompt`: the initially generated sequence `list[int]`.
- `steps`: the number of tokens to generate.
- `min_n`, `max_n`: lower / upper bound on the n-gram window length.
- `pool_seed`: several token sequences `list[list[int]]` used to pre-populate the pool.
- `V`: vocabulary size (used for the fallback modulo).

**Seeding the pool**: for each seed sequence $s$, for each start index $i$ and each $n\in[\text{min\_n},\text{max\_n}]$, if $i+n<|s|$, set
$$\text{pool}[(s_i,\dots,s_{i+n-1})] = s_{i+n}.$$
Writes follow the seed order and the $(i,n)$ order, with **later writes overwriting** an n-gram with the same key.

**Each generation step** (`steps` steps in total, $seq$ is the current sequence):

1. **Draft lookup**: for $n$ from $\text{max\_n}$ down to $\text{min\_n}$, if $|seq|\ge n$ and $(seq_{-n},\dots,seq_{-1})\in\text{pool}$, take `drafted = pool[key]` and stop searching (**longest suffix first**).
2. **Fallback**: if no suffix matches, $drafted=(seq_{-1}+1)\bmod V$ (deterministic).
3. Append $drafted$ to $seq$ and record it in the output.
4. **Update the pool**: for each $n\in[\text{min\_n},\text{max\_n}]$, if $|seq|\ge n+1$, set $\text{pool}[(seq_{-(n+1)},\dots,seq_{-2})]=seq_{-1}$ (map the $n$-gram immediately preceding the new token to the new token).

Return a 1-D `torch.long` tensor of length exactly `steps`.

- "Longest suffix first" is what realizes look-ahead: prefer the longer seen context, which makes the draft more reliable.
- The pool is **stateful** and learns while generating; within a step the lookup happens before the write, keeping it step-by-step consistent with the standalone oracle.

## Examples

```
prompt=[0,1], steps=4, min_n=2, max_n=2, pool_seed=[[0,1,2,3]], V=10
seeded pool = {(0,1):2, (1,2):3}

step1 seq=[0,1]        suffix (0,1) hits -> 2   ; update pool[(0,1)]=2
step2 seq=[0,1,2]      suffix (1,2) hits -> 3   ; update pool[(1,2)]=3
step3 seq=[0,1,2,3]    suffix (2,3) misses -> fallback (3+1)%10 = 4 ; update pool[(2,3)]=4
step4 seq=[0,1,2,3,4]  suffix (3,4) misses -> fallback (4+1)%10 = 5
output = [2, 3, 4, 5]
```

## Constraints

- CPU; return a `torch.long` of length `steps`, compared with `torch.equal`.
- All token ids lie in $[0,V)$; the fallback is strictly $(seq_{-1}+1)\bmod V$.
- Suffix lookup runs from `max_n` down to `min_n`; pool updates run from `min_n` up to `max_n` (the order decides which overwrite wins and must match the statement).
