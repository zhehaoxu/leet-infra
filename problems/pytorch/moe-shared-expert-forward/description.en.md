# Shared and Routed Expert Composition

## Task

Implement `solve`: evaluate the weighted selected routed experts for each token, add one **always-on, ungated** shared SwiGLU expert, and count expert parameters. Routing indices and coefficients are supplied; do not implement a router.

This is a restricted numerical exercise for a DeepSeek-style shared path, not Qwen2-MoE's sigmoid-gated shared expert or a full model forward pass.

## Inputs

All floating tensors are CPU `torch.float64`; `expert_ids` is CPU `torch.int64`. Accept noncontiguous tensors and do not mutate inputs.

| Argument | Shape | Meaning |
| --- | --- | --- |
| `x` | `[T,D]` | Token hidden states |
| `expert_ids` | `[T,K]` | Distinct expert IDs within each row, in `[0,E)` |
| `routing_weights` | `[T,K]` | Nonnegative coefficients aligned with the IDs; **not necessarily summing to 1; do not renormalize** |
| `gate_w`, `up_w` | `[E,H,D]` | Routed gate and up projections |
| `down_w` | `[E,D,H]` | Routed down projections |
| `shared_gate_w`, `shared_up_w` | `[Hs,D]` | Shared gate and up projections |
| `shared_down_w` | `[D,Hs]` | Shared down projection |

`T,E,D,H,Hs` are each 1..8, and `1 <= K <= E`. `|x| <= 5`, projection weights have absolute value at most 3, and routing coefficients lie in `[0,4]`. All inputs are valid. There are no biases, dropout, residual connections, or normalization layers.

## Exact rules

Projections use `x @ W.T`. For a row vector $x$, define:

$$
F_e(x)=\bigl(\operatorname{SiLU}(xW_{g,e}^{\mathsf T})\odot xW_{u,e}^{\mathsf T}\bigr)W_{d,e}^{\mathsf T},\qquad
\operatorname{SiLU}(z)=\frac{z}{1+e^{-z}}.
$$

The shared function $F_s$ uses the three shared weights and hidden width $H_s$. For each token:

$$
r_t=\sum_{j=0}^{K-1}a_{t,j}F_{i_{t,j}}(x_t),\qquad
s_t=F_s(x_t),\qquad y_t=r_t+s_t.
$$

The shared output is neither multiplied by routing coefficients nor disabled by selection. It remains active even when every routing coefficient is zero.

Count only the supplied expert projection matrices, **not an unspecified router or full-model parameters**:

$$
P_{\mathrm{total}}=3D(EH+H_s),\qquad
P_{\mathrm{active}}=3D(KH+H_s).
$$

`active_params` is the **structural parameter count of selected paths per token**, not the union of experts visited by the batch and not multiplied by token count. A selected path still counts when its coefficient is zero.

## Return value

Return the five-item `tuple` `(output, routed, shared, total_params, active_params)`.

- The first three items are CPU float64 tensors of shape `[T,D]`, representing $y,r,s$.
- The last two items are Python `int` values.
- Floating results use `rtol=2e-10, atol=2e-10`.

## Example

Construct all floating values as float64:

```python
x = [[1.0]]
expert_ids = [[1, 0]]
routing_weights = [[2.0, 0.5]]
gate_w = [[[1.0]], [[1.0]]]
up_w = [[[2.0]], [[4.0]]]
down_w = [[[1.0]], [[1.0]]]
shared_gate_w = [[1.0]]
shared_up_w = [[3.0]]
shared_down_w = [[1.0]]
```

Let $q=\operatorname{SiLU}(1)=1/(1+e^{-1})$. The complete output is:

```python
(output, routed, shared, total_params, active_params)
= ([[12*q]], [[9*q]], [[3*q]], 9, 9)
```

The first three items denote tensors. The routed branch is $2\cdot4q+0.5\cdot2q=9q$ and the shared branch is $3q$. Replacing only `routing_weights` with `[[0.0,0.0]]` gives `([[3*q]], [[0.0]], [[3*q]], 9, 9)`; the structural counts are unchanged.
