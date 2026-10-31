# Tuned Lens Affine-Probe Ledger: Layer-wise Logit Mapping and KL Convergence

## Problem

Tuned Lens maps intermediate-layer hidden states into vocab space with one affine probe per layer, answering "what would the LM head predict from this layer". This problem splits the probe ledger into four items: per-layer logits, cross-entropy against the final prediction, KL divergence from the final distribution, and argmax agreement — plus a rank ledger for the probe matrices.

## Rules

Implement `solve(hidden_states, probe_w, probe_b, final_logits, temperature=1.0)`, returning:

| Field | Shape | Meaning |
|---|---|---|
| `layer_logits` | `[L, B, V]` | `logits_l = W_l @ h_l + b_l` |
| `layer_ce` | `[L, B]` | cross-entropy of each layer's log-probs against `target = argmax(final_logits)` |
| `layer_kl` | `[L, B]` | `KL(softmax(final/T) ‖ softmax(layer_logits/T))`, summed over V per sample |
| `argmax_match` | `[L, B]` | whether the layer logits' argmax hits the final argmax |
| `probe_rank` | `[L]` | rank of each probe matrix |

Everything computed in float64; `temperature ≤ 0` raises `ValueError`.

## Example

```
V=2，probe rows identical + biases making layer logits = [1,1] (uniform)，final=[-1,3] (hits 1)
layer_ce = -ln(0.5) = ln2
layer_kl = KL(softmax([-1,3]) ‖ uniform) = ln2 + Σ p·ln p (hand-checkable)
```

## Constraints

- `L ≤ 4`, `V ≤ 12`, `D ≤ 8` small scale, pure-Python softmax/KL oracle
- With probe = LM head, the last layer's KL is strictly 0 and earlier layers' KL > 0
