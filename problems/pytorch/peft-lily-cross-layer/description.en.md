# PEFT: Lily Cross-Layer Expert Mix and Router

## Problem

**Lily** (Low-Rank Interconnected Adaptation across Layers, arXiv:2407.09946, ACL 2025 Findings, official repo `yibozhong/lily`) dismantles LoRA's per-layer A-B pairing: the low-dim projector **LP** stays local to a layer group (a new one every `stride` layers), while the high-dim projectors **HP** are detached into a **model-level shared expert pool** (\(n_e\) matrices of shape \((hidden, out)\)); every LP connects to all HPs. The cross-layer interconnection stops the gradient projection from being pinned to a low-rank subspace by one fixed projection matrix — a **higher-rank** \(\Delta W\) at the same or smaller parameter budget.

The deterministic chain of `lily_adapter.forward` (official `natural-language-understanding/lily.py`):

```
hidden = non_linear(lp(x))                 # ReLU for MLP slots, Identity for attention q/v
logits  = router(hidden)                   # nn.Linear(hidden, ne, bias=False)
probs   = softmax(logits, dim=-1)          # [B, N, ne], data-dependent
p_mean  = probs.mean(dim=(0, 1))           # global mean over batch+tokens -> (ne,)
combined_hp = einsum("e,eio->io", p_mean, hps)
delta   = hidden @ combined_hp
```

The router is a **pure dense softmax weighting** (no top-k, no noise), so all anchors are deterministic. The `monoscale` variant drops the router: `combined_hp = sum(hps, 0) * (1/ne)`. The `set_lily` integration account: `stride = n_layers // ne`, a new LP wherever `idx % stride == 0` (shared within a group); HPs are `torch.zeros(ne, dim, out)` (so \(\Delta W=0\) at the start of training).

## Rules

- `mode="router_probs"`: inputs hidden and the router weight; returns the softmaxed \((\dots, n_e)\) distribution (each token's probabilities sum to 1).
- `mode="expert_mix"`: inputs `probs_mean (n_e,)` and `hps (n_e, hidden, out)`; returns the combined HP \((hidden, out)\).
- `mode="forward_delta"`: the full chain; with `mlp=True` hidden goes through ReLU; `probs_mean` is the mean over **both** batch and token axes.
- `mode="monoscale_delta"`: the uniform variant scaled by \(1/n_e\).
- `mode="lp_groups"`: returns `{"stride", "lp_group", "num_lp"}` — the tail group absorbs the remainder (e.g. \(n=5, n_e=2\): stride=2, groups [0,0,1,1,2], 3 LPs).
- LP/router weights and HPs are all explicit inputs; everything in float64.

## Example

\(in=8, hidden=4, out=6, n_e=3\):

```
forward_delta(x, lp_w, router_w, hps) -> (2,3,6)
  = hidden @ Σ_e p_mean[e]·hps[e], hidden = x @ lp_wᵀ
monoscale_delta(x, lp_w, hps) -> hidden @ (Σ_e hps[e] / 3)
lp_groups(12, 4) -> {stride: 3, lp_group: [0,0,0,1,1,1,2,2,2,3,3,3], num_lp: 4}
```

## Constraints

- Softmax outputs sum to 1 per token; different tokens may have different expert distributions (data dependence).
- The forward equals the explicit expansion \(\sum_e \bar p_e\,(hidden\cdot hps_e)\) (cross-checked in the tests).
- `lp_groups` satisfies: num_lp = number of distinct group ids, realized by tail-group absorption.
