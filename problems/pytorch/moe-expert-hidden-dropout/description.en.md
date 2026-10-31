# MoE: Expert-Hidden Dropout with Gate Renormalization

## Task

Implement one top-k sparse MoE forward pass: pick experts by softmax scores and renormalize the gates, then run each selected expert through its two-layer feed-forward — with **dropout applied only to the expert's hidden activation** (after the activation function, before the second projection), using inverted scaling in training and fully disabled at inference.

This is a deterministic forward model of "expert-hidden dropout", the dropout that ST-MoE applies independently of global dropout. The problem fixes the mask semantics, the scaling factor, where the gate renormalization happens, and the tie order; it is not a step-by-step reproduction of any training framework and involves no random numbers, no backward pass, and no parameter updates.

## Inputs

- `logits`: CPU float64 `[T,E]`, the router logits.
- `x`: CPU float64 `[T,D]`, the token inputs.
- `w1`: `[E,D,H]`, `b1`: `[E,H]`; `w2`: `[E,H,D]`, `b2`: `[E,D]`. Each expert is a `D→H→D` two-layer feed-forward.
- `drop_mask`: CPU float64 `[T,E,H]` whose entries are exactly `0.0` (drop) or `1.0` (keep). It is an **explicit input**; the problem generates no random numbers.
- `k`: a genuine Python `int`, `1 <= k <= E`.
- `expert_dropout`: Python `int`/`float`, `0 <= p < 1`.
- `training`: Python `bool`.

`T,E,D,H >= 1`. Every tensor must be CPU float64 and finite; a `drop_mask` entry other than 0/1, a `bool` posing as `int`, an out-of-range `k`, `p` out of range or NaN/±inf, a non-`bool` `training`, or mismatched shapes raise `ValueError`. Do not mutate inputs; accept non-contiguous tensors.

## Exact rules

**Gating.** Row-wise softmax:

$$
s_{t,e}=\frac{e^{z_{t,e}-\max_j z_{t,j}}}{\sum_{j}e^{z_{t,j}-\max_j z_{t,j}}}.
$$

Take the top $k$ experts by descending $s_{t,e}$, denoted $\pi_{t,1},\dots,\pi_{t,k}$; **exact ties go to the smaller expert index**. Renormalize the weights over only the $k$ picks:

$$
g_{t,j}=\frac{s_{t,\pi_{t,j}}}{\sum_{\ell=1}^{k}s_{t,\pi_{t,\ell}}}.
$$

**Expert forward.** For every token and every selected expert $e=\pi_{t,j}$:

$$
a_{t,j}=x_tW_1[e]+b_1[e],\qquad h_{t,j}=\operatorname{relu}(a_{t,j}),
$$

```text
if training and p > 0:
    h = h * drop_mask[t, e] / (1 - p)     # inverted dropout
else:
    h = h                                  # drop_mask is ignored entirely
```

$$
y_{t,j}=h_{t,j}W_2[e]+b_2[e],\qquad \text{output}_t=\sum_{j=1}^{k}g_{t,j}\,y_{t,j}.
$$

Three points:

1. Dropout acts on the **post-relu hidden activation** — not on the token input $x$, the gate scores, or the expert output.
2. Training uses inverted scaling: kept entries are multiplied by $1/(1-p)$, and inference applies no compensation. With `training=False` or `p=0` dropout is off entirely, and an all-zero `drop_mask` must not change the result.
3. The gate renormalization is independent of dropout; it is the ordinary treatment for sparse top-k gating (the mass of the unselected experts is discarded).

## Returns

Return a 4-tuple:

```text
(expert_idx, gates, hidden, output)
```

| Item | Shape / dtype | Meaning |
| --- | --- | --- |
| `expert_idx` | int64 `[T,k]` | Selected expert ids, by descending score |
| `gates` | float64 `[T,k]` | Renormalized gate weights, each row sums to 1 |
| `hidden` | float64 `[T,k,H]` | The hidden activation **after dropout** |
| `output` | float64 `[T,D]` | Gate-weighted sum of the expert outputs |

Float results are compared with `rtol=1e-9, atol=1e-12`.

## Example

```python
logits = [[0.0, 0.0, -1.0]]
x      = [[2.0, -1.0]]
w1     = [[[1, 0], [0, 1]],      # expert 0
          [[0, 1], [1, 0]],      # expert 1
          [[1, 1], [1, 1]]]      # expert 2
b1     = zeros(3, 2)
w2     = three 2x2 identity matrices
b2     = [[0, 0], [1, 1], [2, 2]]
mask   = expert 0 -> [1, 0], expert 1 -> [0, 1], expert 2 -> [0, 0]

k = 2, p = 0.5, training = True
# expert_idx = [[0, 1]]
# gates      = [[0.5, 0.5]]
# hidden     = [[[4, 0], [0, 4]]]
# output     = [[2.5, 2.5]]

training = False
# expert_idx = [[0, 1]]
# gates      = [[0.5, 0.5]]
# hidden     = [[[2, 0], [0, 2]]]
# output     = [[1.5, 1.5]]
```

`softmax([0,0,-1]) = [0.42231880, 0.42231880, 0.15536240]`; the two zeros are an exact tie, so experts 0 and 1 are picked by index and renormalize to `0.5` each.

Expert 0's pre-activation is `[2*1+(-1)*0, 2*0+(-1)*1] = [2, -1]`, which relu turns into `[2, 0]`; mask `[1,0]` and `1/(1-0.5)=2` give `[4, 0]`, and `W2=I`, `b2=[0,0]` leave it at `[4, 0]`. Expert 1 symmetrically yields `[0, 4]`, plus `b2=[1,1]` for `[1, 5]`. The weighted sum is `0.5*[4,0] + 0.5*[1,5] = [2.5, 2.5]`.

At inference there is no scaling: the hidden activations return to `[2,0]` and `[0,2]`, the expert outputs are `[2,0]` and `[1,3]`, and the weighted sum is `[1.5, 1.5]`. **Same inputs, same mask, different train and inference outputs** — that is exactly dropout's train/inference asymmetry.

## Boundary against neighbouring problems

- `moe-router-soft-clamp` handles tanh soft-clamping of logits and a selection bias, stopping at the gate; here the gate is plain softmax + top-k.
- `moe-capacity-jitter-dispatch` covers token dropping and dispatch under jittered capacity; this problem has no capacity limit and drops no tokens.
- `moe-dead-expert-ledger` collects expert load statistics and dead experts; this problem collects none.
- `moe-shared-expert-forward` composes shared and routed experts; every expert here is routed.
- This problem is forward-only: no gate-bias updates, no load-balancing auxiliary loss, no gradients.
