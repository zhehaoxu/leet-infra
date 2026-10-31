# Logit Lens: Final-LayerNorm + Unembed Layerwise Readout Ledger

## Problem

Logit lens (nostalgebraist) reads intermediate hidden states as "what would be output right now": apply the **final LayerNorm**, then multiply the **unembedding matrix** to get that layer's logits. This problem turns the readout into a ledger: given per-layer (final-position) hidden states `hiddens [L, D]`, the final LN's `w/b`, unembed `[D, V]`, return per-layer logits, argmax (ties → smallest index), agreement with the final layer's prediction, the rank of the final layer's top token within each layer's logits (`rank_of_final`, 1-based, strict-greater count), and each layer's top logit.

## Solution

```
per layer l:
  mu  = mean(h)                       # over D
  var = mean((h - mu)²)               # biased variance (divide by D)
  n   = (h - mu)/sqrt(var + eps)·w + b     # eps pinned at 1e-5
  logits[l] = n @ unembed
argmax: ties broken by smallest index (hand-written min-index, not torch.argmax's tie behavior)
final_argmax = argmax[L-1]; agree[l] = (argmax[l] == final_argmax)
rank_of_final[l] = 1 + #{v : logits[l][v] > logits[l][final_argmax]}   # always 1 on the final layer
```

Key points: the LayerNorm is part of the logit-lens definition (`apply_norm=False` is a degenerate mode kept for contrast); eps is not negligible (normed = ±2/√(4+1e-5) ≠ ±2); `rank_of_final` counts **strictly greater** logits, so ties do not affect it.

## Data scale

- 1 ≤ L ≤ 8, 1 ≤ D ≤ 64, 1 ≤ V ≤ 128.
- Shape mismatches, empty input, eps ≤ 0, non-bool apply_norm → `ValueError`.
