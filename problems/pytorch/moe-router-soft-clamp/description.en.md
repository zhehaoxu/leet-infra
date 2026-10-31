# MoE: Tanh Soft-Clamping of Router Logits and Saturation Ties

## Task

Implement `solve`: apply a tanh soft clamp to the router logits first, then take the sigmoid to get scores; a per-expert bias that **participates only in picking winners, never in weighting** may change which experts are chosen, while the weights always normalize the bias-free sigmoid scores.

This is a restricted numerical model of "soft-capped gating plus an auxiliary-loss-free selection bias". Where the clamp happens, what the bias may touch, tie order, and the fallback under underflow are all fixed explicitly by the statement. It is not a step-by-step reproduction of any training framework and involves no gradients and no bias updates.

## Inputs

- `logits`: CPU float64 `[T,E]`; noncontiguous tensors are accepted and the input is never mutated.
- `bias`: `None`, or CPU float64 `[E]`.
- `k`: a true Python `int` with `1 <= k <= E`.
- `tau`: a non-negative finite `int`/`float`; `tau = 0` disables the soft clamp.
- `T,E = 1..8`.

Raise `ValueError` when a `bool` impersonates an `int`, `k` is a float or out of range, `tau` is NaN, ±inf or negative, `bias` has a length other than `E`, the dtype is not float64, a tensor is not on CPU, the number of dimensions is wrong, or an argument is not a tensor at all.

## Exact rules

The clamp happens **before** the sigmoid. Clamp when `tau > 0`, pass through when `tau = 0`:

$$
c_{ij}=\tau\tanh\!\left(\frac{z_{ij}}{\tau}\right)\ (\tau>0),\qquad c_{ij}=z_{ij}\ (\tau=0).
$$

Scores and choice scores:

$$
s_{ij}=\sigma(c_{ij})=\frac{1}{1+e^{-c_{ij}}},\qquad u_{ij}=s_{ij}+b_i.
$$

Take $b_i=0$ when `bias=None`.

Per row, take the $k$ experts with largest $u_{ij}$ and call their indices $\pi_1,\dots,\pi_k$; **exact ties go to the lower expert index**. Weights normalize the bias-free $s$ only:

$$
w_{i,j}=\frac{s_{i,\pi_j}}{\sum_{\ell=1}^{k}s_{i,\pi_\ell}}.
$$

If that denominator is exactly 0 (every selected sigmoid underflowed to 0), the row gets uniform weights $1/k$ while $\mathit{scores}$ still honestly reports 0.

`tanh` is strictly monotone, so as long as two values remain distinguishable in floating point, the clamp cannot reorder them; what it changes is the **ties created when large logits collapse onto the same value**: with `tau=1`, `tanh(20)` and `tanh(21)` are both exactly 1.0 in float64, and those two experts become indistinguishable.

## Return value

Return a three-item `tuple`:

```text
(weights, indices, scores)
```

| Item | Shape / type | Meaning |
| --- | --- | --- |
| `weights` | float64 `[T,k]` | Normalized weights, each row summing to 1 (uniform `1/k` when the denominator is 0) |
| `indices` | int64 `[T,k]` | Selected expert indices in descending choice order |
| `scores` | float64 `[T,E]` | Clamped and sigmoided scores, including experts that were not selected |

Compare floats with `rtol=2e-12, atol=2e-12`.

## Example

```python
logits = [[2.0, 0.0, -1.0],
          [1.0, 1.0, 0.5]]

solve(logits, None, 2, 1.0)
# scores  = [[0.72392747, 0.5,        0.31830026],
#            [0.68169974, 0.68169974, 0.61351630]]
# indices = [[0, 1], [0, 1]]
# weights = [[0.59147906, 0.40852094], [0.5, 0.5]]

solve(logits, [0.1, -0.2, 0.0], 2, 1.0)
# indices = [[0, 2], [0, 2]]
# weights = [[0.69459625, 0.30540375], [0.52632126, 0.47367874]]
```

Row one: `tanh(2)=0.9640275800758169`, `1/(1+exp(-0.96402758))=0.7239274686640463`; `tanh(0)=0` gives `0.5`; `tanh(-1)=-0.7615941559557649` gives `0.3183002578054738`. The top two are experts 0 and 1, with weight `0.72392747/(0.72392747+0.5)=0.59147906`.

Row two: the two `1.0` entries collapse to the same clamped value, so this is an exact tie, experts 0 and 1 are picked by index order, and each weight is `0.5`.

With `bias=[0.1,-0.2,0.0]` the first row's choice scores become `[0.82392747, 0.3, 0.31830026]`, expert 1 is pushed out and expert 2 takes its place; normalization still uses `s` alone, `0.72392747/(0.72392747+0.31830026)=0.69459625`, so the bias never appears in the weights. Replacing `bias` with all `0.37` changes neither the selection nor the weights — adding the same constant to every expert changes nothing.
