# PEFT: X-LoRA Scaling-Classifier Mixture of LoRA Experts

## Problem Description

**X-LoRA** (Mixture of LoRA Experts, arXiv:2402.07148, HF PEFT `tuners/xlora`) treats a set of trained LoRA adapters as a **frozen expert pool** and trains only a scaling classifier: from the base model's last hidden state it predicts **per-token × per-layer × per-expert** coefficients \(\Lambda \in \mathbb{R}^{b\times s\times l\times n}\), with softmax keeping each expert row summing to 1. Inference runs a dual forward pass (a scaling pass with dummy scalings to obtain hidden states from the base model, then the classifier predicts scalings, and the second pass mixes experts by them).

The deterministic account of the official PEFT implementation (anchors fully re-computable offline):

```
logits = classifier(hidden_last)                    # Linear+ReLU chain -> last Linear
scalings = logits.reshape(b, s, l, n)               # layerwise=True: last outputs n·l, C-order reshape
scalings = softmax(scalings / temperature, dim=-1)  # TemperatureScaledSoftmax
# per-layer application:
sl = scalings[:, :, layer_number, :]                # (b, s, n)
topk path: a bool mask keeps ORIGINAL values (no renormalization)
softmax_topk path: renormalize nonzero slots only (-inf fill -> softmax -> zeros back to 0)
x_mod = x * sl[:, :, adapter].unsqueeze(-1)         # the scaling multiplies the INPUT to the LoRA expert!
delta = Σ_a (x·λ_a) @ (B_a A_a)ᵀ · scaling · global_scaling_weight
```

Three key conventions:

1. **topk is a mask, not a renormalization**: the `top_k_lora` path first zeroes the non-top-k slots (original top-k values are kept); when `enable_softmax_topk=True` it **then** softmax-renormalizes over the nonzero slots; k=1 + softmax_topk equals one-hot. An all-zero row (all experts of that token/layer originally 0) must produce all zeros — not NaN — under softmax_topk; `masked_fill(~nonzero, 0.0)` exists precisely for this.
2. **The scaling multiplies the input side**: `apply_scalings_to_x` broadcasts λ onto the input of each LoRA expert (\((b,s,1)\times(b,s,in)\)), mathematically equivalent to a per-token column scaling of \(B_aA_a\): \((x\cdot\lambda)\,M^{\top} = \mathrm{einsum}(\text{bsi,bs,oi}\to\text{bso})\).
3. **Dummy scalings (scaling pass)**: a \((b,s,l,n)\) tensor filled with `scaling_pass_value` (config default 0; None means \(1/n\)), used by the first pass that runs with adapters disabled.

## Rules

- `mode="temperature_softmax"`: softmax(dim=-1) of logits/temperature.
- `mode="classifier_forward"`: hidden \((b,s,h)\) + weight chain (each a \((W,bias)\); the last is the final Linear) + `layerwise` switch → scalings \((b,s,l,n)\). With layerwise=False the head outputs \(n\) classes, expanded over l layers (identical across layers).
- `mode="dummy_scalings"`: filled with override (None → \(1/n\)).
- `mode="maybe_topk_scalings"`: slice one layer + optional topk mask + optional softmax_topk renormalization → \((b,s,n)\).
- `mode="apply_scalings_to_x"`: input-side scaling.
- `mode="layer_forward"`: base + expert loop (adapter index = expert column), with `scaling` and `global_scaling_weight` multiplying only the LoRA paths; scalings=None degrades to a plain LoRA sum (no global weight).
- All torch float64.

## Example

\(h=8, l=4, n=3\) (layerwise, depth 3):

```
classifier_forward(hidden, [(W1,b1),(W2,b2),(W3,b3)], 4, 3, T=1.7) -> (2,5,4,3)
maybe_topk_scalings(scalings, layer=1, top_k=2, enable_softmax_topk=True) -> (2,5,3), nonzero rows sum to 1
layer_forward(x, scalings, [(A_a,B_a)...], scaling=2.0, gw=0.5, layer=0, W0) -> base + Σ_a scaled paths
```

## Constraints

- After softmax, each per-token per-layer expert row sums to 1; the temperature is observable (changing it changes the distribution).
- topk=1 + softmax_topk outputs one-hot (the max slot gets 1.0).
- `global_scaling_weight` multiplies only the LoRA delta: doubling gw ⇒ output = base + 2·(old output − base).
- An all-zero row through softmax_topk is all zeros and finite (NaN guard).
