# Top-k Gating with Explicit Noise

## Task

Implement `solve`: add **caller-supplied** Gaussian noise to routing logits, take the top-k by the noisy scores, and apply softmax only over the selected scores. Return the sparse gate matrix, selected indices, and full noisy logits.

This is a restricted numerical exercise for noisy top-k gating in Shazeer-style sparse MoE. Noise is not drawn from a global generator; the training/inference difference is only whether the noise term is used.

## Inputs

- `x`: CPU float64 `[T,D]`, one token per row.
- `gate_w`: CPU float64 `[E,D]`; gate logits are `x @ gate_w.T`.
- `noise_w`: CPU float64 `[E,D]`; noise scale is `softplus(x @ noise_w.T)`.
- `noise`: CPU float64 `[T,E]` of supplied standard-normal samples, each with absolute value at most 4.
- `k`: Python `int`, `1 <= k <= E`. It must be a true `int`; `bool` and `2.0` are invalid and raise `ValueError`.
- `training`: Python `bool`, default `True`. When `False`, neither `noise` nor `noise_w` is used.
- `T,D,E=1..8`; absolute `x` values are at most 3 and absolute weights at most 2. Other inputs are valid; accept noncontiguous tensors and mutate none.

Interpret `softplus(z)=log(1+e^z)` exactly; do not substitute `|z|` or `relu(z)`.

## Exact rules

Gate logits and noisy scores:

$$
z=xW_g^{\mathsf T},\qquad
\tilde z=z+\epsilon\odot\operatorname{softplus}(xW_n^{\mathsf T})\quad(\text{training}).
$$

During inference, $\tilde z=z$. Select the top k experts by **descending** $\tilde z$; exact ties go to the lower expert index.

$$
(\tilde z_{(1)},\ldots,\tilde z_{(k)})=\operatorname{topk}(\tilde z,k),\qquad
a_j=\frac{e^{\tilde z_{(j)}}}{\sum_{\ell=1}^{k}e^{\tilde z_{(\ell)}}}.
$$

The `gates` tensor has shape `[T,E]`: selected positions hold $a_j$ and all others are **exactly 0**. Normalize only across the k selected experts, never including unselected ones. Returned `indices` follow descending-score order.

## Return value

Return the three-item `tuple` `(gates, indices, effective_logits)`.

- `gates`: float64 `[T,E]`, each row sums to 1;
- `indices`: int64 `[T,k]` in descending selection order;
- `effective_logits`: float64 `[T,E]`, equal to $\tilde z$ in training and $z$ in inference.
- Compare floats with `rtol=2e-11, atol=2e-11`.

## Example

```python
x = [[1.0]]
gate_w = [[1.0], [0.9]]
noise_w = [[0.0], [0.0]]
noise = [[0.0, 1.0]]
k = 1
training = True
```

Both expert noise scales equal `softplus(0)=ln 2`, so:

```python
indices = [[1]]
effective_logits = [[1.0, 0.9 + ln 2]]
gates = [[0.0, 1.0]]
```

Noise moves the previously weaker expert 1 to the front. With `noise=[[0.0,0.0]]`, expert 0 is selected and `gates=[[1.0,0.0]]`. With `training=False`, changing `noise` arbitrarily leaves the result equal to the denoised result.
