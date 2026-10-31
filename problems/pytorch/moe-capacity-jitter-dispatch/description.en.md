# MoE: Jitter-Priority Capacity Dispatch with Spill-Over

## Task

Implement `solve`: every expert has only `capacity` slots, so line up all tokens' top-k candidates by their **jittered priority key** from high to low and let them claim slots. When an expert is full, a slot **spills forward along its rank** to the token's next candidate; if nothing is available the slot is dropped. Renormalize the gating weights over the **surviving slots only** at the end.

This models one specific decision in capacity dispatch: *who gets dropped*. Dropping in token order (first come, first served) systematically starves the tail of a batch, which is why real routers perturb the priority before queueing. Here the perturbation is an **input tensor** rather than an internal RNG draw, so the judge stays fully deterministic. There is no backward pass and no simulation of all-to-all communication.

## Inputs

- `probs`: CPU float64 `[T,E]`, non-empty, finite, element-wise non-negative; rows are **not** required to sum to 1.
- `jitter`: CPU float64 `[T,E]`, same shape as `probs`, finite, any sign.
- `k`: a Python `int` with $1\le k\le E$ (`bool` does not count as `int`).
- `capacity`: a Python `int`, $\ge 0$ (`bool` does not count).

Noncontiguous tensors are accepted and no input is mutated. Raise `ValueError` for wrong dtype, shape, device, rank, non-finite or negative values, and out-of-range `k`/`capacity`.

## Exact rules

A slot is numbered $s=t\,k+j$, where the rank $j$ is the candidate's position in that token's list.

**Candidates**: $cand_t$ is the top-$k$ of `probs[t]`, taking the first $k$ entries ordered by the key `(-probs[t][e], e)` — **ties go to the smaller expert index**. `jitter` never changes the candidate set; it only affects the processing order below.

**Priority key and processing order**:

$$
key_{t,j}=probs[t][cand_t[j]]+jitter[t][cand_t[j]],\qquad
order=\text{argsort}\bigl((-key,\;s)\bigr).
$$

That is, all $S=Tk$ slots queue up **globally** in descending `key`, breaking equal keys by the smaller slot id. This is not token order: a low-rank slot of some token can easily be processed before a high-rank slot of the same token.

**Greedy claiming**: walk `order`, with `remaining[e]` starting at `capacity`. For slot $(t,j)$ examine

$$
cand_t[j],\;cand_t[j+1],\;\dots,\;cand_t[k-1]\qquad(\text{forward only, no wrap-around}),
$$

and take the first expert with $remaining[e]>0$ that token $t$ has **not** already claimed: set `assign[t][j]=e` and `remaining[e]-=1`. If none qualifies, `assign[t][j]=-1`. Dropped slots consume no capacity.

One invariant follows immediately: **the surviving slots of a token always land on distinct experts** (a top-$k$ list has no repeats, and claiming skips experts the token already used).

**Weights**: with $w^{raw}_{t,j}=probs[t][assign_{t,j}]$ (0 for dropped slots),

$$
weights_{t,j}=\begin{cases}
w^{raw}_{t,j}\big/\sum_{j'} w^{raw}_{t,j'}, & \sum_{j'} w^{raw}_{t,j'} > 0,\\[3pt]
0, & \text{otherwise}.
\end{cases}
$$

The renormalization covers surviving slots, not the original top-$k$. When a whole row is dropped, or when the row's probabilities are all zero, the denominator is 0 and the result must be 0 rather than NaN.

## Return value

`(assign, weights, counts, dropped, order)`:

| Item | Shape / type | Meaning |
| --- | --- | --- |
| `assign` | int64 `[T,k]` | The expert each slot finally landed on; `-1` means dropped |
| `weights` | float64 `[T,k]` | Gating weights renormalized over survivors; 0 on dropped slots |
| `counts` | int64 `[E]` | Slots actually claimed per expert |
| `dropped` | Python `int` | How many entries of `assign` are `-1` |
| `order` | int64 `[T*k]` | Processing order, given as slot ids |

Compare floats with `rtol=atol=2e-11` and integer tensors bit for bit.

## Example

`T=4, E=4, k=2, capacity=1`, `jitter` all zero except `jitter[1][0]=0.25`:

```python
probs = [[0.5,    0.25,   0.125,  0.125 ],
         [0.4375, 0.3125, 0.125,  0.125 ],
         [0.375,  0.25,   0.125,  0.25  ],
         [0.0625, 0.125,  0.25,   0.5   ]]

# cand      t0=[0,1] t1=[0,1] t2=[0,1] t3=[3,2]
# key       (1,0)=0.6875 (0,0)=0.5 (3,0)=0.5 (2,0)=0.375
#           (1,1)=0.3125 (0,1)=0.25 (2,1)=0.25 (3,1)=0.25
# order     [2, 0, 6, 4, 3, 1, 5, 7]
# assign    [[1, -1], [0, -1], [-1, -1], [3, 2]]
# counts    [1, 1, 1, 1]      dropped = 4
# weights   [[1, 0], [1, 0], [0, 0], [2/3, 1/3]]
```

Three things worth spelling out:

- In $cand_2=[0,1]$ the expert e1 wins by a tie: `probs[2][1] == probs[2][3] == 0.25`, ties prefer the smaller index, so e3 never even becomes a candidate.
- `order` handles $(1,0)$ first (key 0.6875) and it claims e0. Then $(0,0)$ finds its own expert full, **moves one step forward along the rank** and spills onto e1, so `assign[0][0]=1` and its weight uses `probs[0][1]=0.25`. At key 0.5, $(0,0)$ and $(3,0)$ tie, and slot id 0 comes before 6.
- Both of token3's slots survive, so its weights are $0.5/0.75=2/3$ and $0.25/0.75=1/3$; token0 keeps a single slot, so its weight is 1 rather than 0.5 — **renormalization is over survivors**.
