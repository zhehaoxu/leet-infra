# order 843 · Captum LRP's z-rule Relevance Propagation Ledger

> Upstream source: `pytorch/captum` (main branch) — `captum/attr/_core/lrp.py`, `captum/attr/_utils/lrp_rules.py`

## Background

Layer-wise Relevance Propagation (LRP) distributes "how much the model output contributed to a
given target" back to the input neurons, one layer at a time. It differs fundamentally from
Integrated Gradients (order 842): IG **integrates** gradients along a straight-line path, while LRP
**redistributes** relevance top-down, layer by layer.

Captum's LRP implementation has two easily-missed key points, and this problem makes both of them
mandatory:

1. **Two forward passes.** The official code first runs a forward pass to record each layer's
   activations and to rewrite the weights, then recomputes everything using
   **the rewritten weights times the activations from the first pass**.
2. **`sign(0) = +1`.** The z-rule denominator is `outputs + sign(outputs) * eps`, and the official
   code forces `sign(0)` to `+1`, so when an output happens to be exactly zero the denominator is
   `+eps` rather than zero.

This problem asks you to rebuild that propagation ledger **with exact rational arithmetic**.

---

## Specification (the single source of truth)

- **Every input quantity is an integer** (weights, biases, activations, input vectors).
- **`eps` is the exact rational `1/10^9`** (officially it is the float64 `1e-9`; this problem
  makes explicit the semantics that floating point had been hiding).
- Compute the whole chain with `fractions.Fraction` **exactly**, and convert to `float` only at the
  very end (one conversion per output quantity).
- Float outputs are checked with a relative tolerance of `1e-12`. That tolerance exists only to
  absorb the single exact-rational-to-float rounding and the official float64's own ~1e-16
  tail error; any semantic error is amplified far beyond `1e-12`.
- Do **not** mix float and Fraction across the chain: a single float step lets tail error grow
  along the propagation.

---

## mode 1 · `rewrite_rule`

The weight-rewriting ledger of the official `PropagationRule._manipulate_weights`.

**Arguments**

| Parameter | Type | Meaning |
|---|---|---|
| `rule` | str | `"epsilon"` / `"gamma"` / `"alpha1_beta0"` |
| `weights` | `[[int]]` | weight matrix (list of rows) |
| `bias` | `[int]` | bias vector (length = number of weight rows) |
| `gamma` | str | rational string such as `"1/4"`, `"1/2"`, `"0"`, `"1"` (used by the `gamma` rule only) |
| `set_bias_to_zero` | bool | whether to zero the bias (default `False`) |

**Returns**: `{"weights": [[float]], "bias": [float]}`

**Semantics of the three rules**

- `epsilon`: **weights and bias are returned unchanged**.
  The body of the official `EpsilonRule._manipulate_weights` is `pass` — it only changes
  `STABILITY_FACTOR`, never the weights.
- `gamma`: `W'[i][j] = W[i][j] + gamma * max(W[i][j], 0)`
  That is "**add** gamma times the positive part", equivalent to "scale positive weights by
  `(1 + gamma)` and leave negative weights alone".
  **Negative weights are neither scaled up nor scaled down.**
- `alpha1_beta0`: `W'[i][j] = max(W[i][j], 0)` (negative weights are zeroed — the Deep-Taylor rule)

When `set_bias_to_zero=True` the bias is set entirely to zero (the weights are still rewritten per
the table above).

**An unknown `rule` must raise `ValueError`.**

---

## mode 2 · `z_rule_layer`

The z-rule relevance decomposition for a single linear layer.

**Arguments**: `weights` (`[[int]]`, **already-rewritten** weights), `bias` (`[int]`), `inputs`
(`[int]`, the layer's input activation), `target` (int, index of the target output)

**Returns**: `{"outputs": [...], "relevance_out": [...], "relevance_in": [...], "attribution": [...]}`

**Steps**

1. Forward: `outputs[i] = sum_j W[i][j] * inputs[j] + bias[i]`
2. The upstream gradient is one-hot: `grad = e_target`
3. **output hook** (element-wise):
   ```
   relevance_out[i] = grad[i] / (outputs[i] + sign(outputs[i]) * eps)
   ```
   where `sign(v)`: `v > 0` → `+1`, `v < 0` → `-1`, `v == 0` → **`+1`** (official quirk).
4. **input hook**:
   ```
   relevance_in[j] = (relevance_out @ W)[j] * inputs[j]
   ```
   i.e. treat `relevance_out` as a row vector, multiply it by the weight matrix on the right, then
   multiply element-wise by `inputs`.
5. Normalized output: `attribution[j] = relevance_in[j] * outputs[target]`

**An out-of-range `target` must raise `ValueError`.**

---

## mode 3 · `stack_relevance`

The full propagation ledger for a stack of linear layers. This is the core of the problem.

**Arguments**

- `layers`: `[{weights, bias, rule, gamma?, set_bias_to_zero?}, ...]`, at least one layer; layers
  chain head to tail (layer `i`'s input width equals layer `i-1`'s output width).
- `inputs`: `[int]`, the stack's input vector
- `target`: int, the target index at the **top** (output) layer

**Returns**

```
{
  "activations":    [[...]],   # per-layer output of the FIRST forward pass
  "outputs":        [[...]],   # per-layer output of the SECOND forward pass
  "relevance_out":  [[...]],   # per-layer relevance_out
  "relevance_in":   [[...]],   # per-layer relevance_in
  "attribution":    [...],     # relevance_in of the input-most layer × original-weight target output
}
```

**All lists are in forward order** (index 0 = the first layer, closest to the input).

**Exact semantics of the two forward passes**

1. **First forward pass**: chain the forward using the **original weights** (`W`, `b`, unrewritten),
   giving `activations[k]` = the output of layer `k`.
   Layer 0's input is `inputs`; layer `k`'s input is `activations[k-1]`.

2. **Rewrite the weights**: obtain each layer's `W'`, `b'` per the mode 1 semantics.

3. **Second forward pass**: layer `k`'s output is
   ```
   outputs[k][i] = sum_j W'[k][i][j] * activations[k-1][j] + b'[k][i]
   ```
   **It multiplies the corresponding layer's ORIGINAL activation, not the previous layer's
   `outputs`.**
   This is the direct consequence of the official `forward_pre_hook_activations`: during the second
   forward pass every layer's input is restored to the activation recorded during the first pass.

   ⚠️ Writing it as "chain the forward pass with the rewritten weights" is the single most common
   mistake.

4. **Backward pass** (from the top layer down to the bottom):
   ```
   upstream = e_target                       # only at the top layer
   rel_out[k] = upstream / (outputs[k] + sign(outputs[k]) * eps)
   rel_in[k]  = (rel_out[k] @ W'[k]) * activations[k-1]
   upstream   = rel_in[k]                   # handed to the next layer down
   ```
   Note that `rel_in[k]` multiplies the **original activation** `activations[k-1]`.

5. **Normalized output**:
   ```
   output_target = activations[-1][target]   # the target value under the ORIGINAL weights
   attribution[j] = rel_in[0][j] * output_target
   ```

**An unknown `rule`, empty `layers`, and an out-of-range `target` must all raise `ValueError`.**

---

## mode 4 · `completeness`

The completeness delta.

**Arguments**: `attributions` (`[int]`, the attribution vector for a single sample), `output`
(int, the target output value under the **original** weights)

**Returns**: `{"attr_sum": float, "delta": float}`

```
attr_sum = sum(attributions)
delta    = output - attr_sum
```

⚠️ **Sign order**: the official LRP `compute_convergence_delta` computes `output - attr_sum`
(**output first**). This is the **opposite** of Captum's Integrated Gradients (order 842), where the
formula is `attr_sum - (end_out - start_out)`.

---

## Common mistake checklist

| Mistake | Consequence |
|---|---|
| Implementing the `gamma` rule as "scale all weights by `(1+gamma)`" | negative weights wrongly scaled |
| Forgetting to zero negative weights in `alpha1_beta0` | negative weights survive |
| Also rewriting weights in the `epsilon` rule | weights wrongly truncated |
| Returning 0 from `sign(0)` | division by zero / wrong denominator |
| Using `1e-8` / `1e-10` for `eps` | denominator too large / too small |
| Writing the second forward pass as "chain the rewritten weights" | every layer's `outputs` is wrong |
| Multiplying `rel_in` by `outputs` instead of the original activation | the effect of weight rewriting is erased |
| Using `W` (original) instead of `W'` inside `relevance_in` | the effect of the rewrite rule is lost |
| Resetting `upstream` to one-hot at every layer | cross-layer propagation breaks |
| Multiplying `attribution` by `outputs[-1][target]` (rewritten) | wrong normalization basis |
| Writing `delta` as `attr_sum - output` | sign flipped |
| Mixing float and Fraction mid-chain | tail error grows past `1e-12` |

---

## Self-check

```python
from fractions import Fraction

EPS = Fraction(1, 10 ** 9)

# mode 1: gamma scales only the positive weights
# W = [[-3, 5]]  ->  W' = [[-3, 25/4]]

# mode 2: sign(0) = +1
# W=[[1,1]], b=[-2], inputs=[1,1] -> outputs=[0]
# relevance_out = [1/eps], attribution = [0, 0]

# mode 3: two forward passes
# layer0 = gamma(1/4), layer1 = epsilon
# activations[1] = [44, -66, -38]  (original-weight chain)
# outputs[1]     = [44, -66, -38]  (rewritten weights × original activation a1,
#                                   NOT × outputs[0])
```
