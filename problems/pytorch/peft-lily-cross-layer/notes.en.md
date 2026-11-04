# Notes: Lily Cross-Layer Expert Mix and Router

## Source

Paper: Low-Rank Interconnected Adaptation across Layers (arXiv:2407.09946, ACL 2025 Findings; Yibo Zhong et al., Sichuan University). Theoretical starting point: LoRA is essentially a **gradient compressor** — a fixed projection matrix \(A_0\) projects gradients into a low-rank subspace, pinning \(\Delta W\) to low rank. Lily decouples this with a hierarchical frame: LPs (down-projections) stay local to layer groups, HPs (up-projections) become a model-level shared expert pool (MoE-style), and every LP connects to all HPs; the interconnection plus a selective router breaks the per-layer low-rank bound. The official repo splits by task (commonsense-reasoning / natural-language-understanding / subject-driven-generation / visual-adaptation-benchmark); the NLU `lily.py` is the cleanest implementation (built on the fourierft experiment scaffold).

## Line-by-line translation of the official implementation

```python
class lily_adapter(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, ne, lp, hps, mlp=False):
        self.router = nn.Linear(hidden_dim, ne, bias=False)
        self.non_linear = nn.ReLU() if mlp else nn.Identity()
    def forward(self, x):
        hidden = self.non_linear(self.lp(x))
        router_logits = self.router(hidden)                    # [B, N, ne]
        router_probability = F.softmax(router_logits, dim=-1)
        expert_probabilities = router_probability.mean(dim=(0, 1))   # (ne,)
        combined_hp = torch.einsum("e,eio->io", expert_probabilities, self.hps)
        return torch.matmul(hidden, combined_hp)
```

Three key conventions:

1. **The router is a bias-free Linear + global softmax mean**: no top-k, no load-balancing noise, no sparsification — `expert_probabilities` is the global mean over **both** batch and token axes, so the whole batch shares one expert mixture (data-dependent but deterministic per batch).
2. **The monoscale variant** (`lily_adapter_monoscale`): `combined_hp = sum(hps, 0) * (1/ne)` — a uniform average with no router at all; it is the paper's ablation baseline.
3. **HPs are zero-initialized**: `set_lily` uses `nn.Parameter(torch.zeros(ne, dim, 768))` — at the start of training every expert is zero ⇒ \(\Delta W=0\), the same philosophy as LoRA's "B zero, A random" but more thorough.

**The LP placement account in `set_lily`**:

```python
stride = 12 // ne
if idx % stride == 0: new_lp = nn.Linear(768, dim, bias=False)
```

Layer indices start at 0; a new LP every `stride` layers, shared within the group; n=12/ne=4 → stride=3 → 4 LPs. When n does not divide (5/2 → stride=2), layer 4 falls into group 2 — tail-group absorption.

**The attention-side scale**: `value_layer = ... + self.lily_v(hidden_states) * self.s` — the delta is scaled by a global \(s\) (default 1.0), applied outside the adapter and not taken into this problem.

## Pitfalls

1. **The two-axis mean**: `mean(dim=(0, 1))` is the global batch+token mean; averaging only over batch or only over tokens are both off-by-one-axis mutants (pinned by m-mean-axis/m-mean-axis2). This is also where Lily differs from a standard MoE — the mixture is "this batch's layer profile", not a per-token route.
2. **Softmax over the expert axis**: `dim=-1` (the ne dim); normalizing over the batch axis destroys the semantics entirely (pinned by m-softmax-axis).
3. **Softmax shift invariance**: the numerical-stability term (subtracting the max) can be replaced by any constant shift without changing the output — the mutant `logits+1.0` is genuinely equivalent (observed SURVIVED and removed; recorded here so nobody re-pins it).
4. **The einsum subscripts of expert_mix**: "e,eio->io" sums over e and keeps io free; writing "->oi" keeps the shape (for square hps) but transposes the values (pinned by m-einsum-trans).
5. **The mlp switch acts on hidden** (after the LP, before the router), not on the output; leaking ReLU into the Identity branch or dropping it is numerically observable (pinned by m-relu-leak/m-relu-drop).
6. **monoscale's 1/ne**: it is a uniform average — forgetting the factor inflates by ne (m-mono-scale); taking only the first expert is also wrong (m-mono-sum).
7. **LP placement is `idx % stride == 0`**, equivalently group id `i // stride`; writing `i % ne` destroys group contiguity (pinned by m-lp-mod).

## Verification anchors

Anchors were dumped in a sandbox (numpy float64): F1 uses in=8/hidden=4/out=6/ne=3, with both router and MLP branches and both monoscale branches anchored. Cross-check: `forward = Σ_e p̄_e·(hidden@hps_e)` (explicit expert expansion). Four LP configurations (including tail-group absorption at n5/ne2 and stride=1 at n6/ne4). The tests also pin the softmax properties (probabilities sum to 1, token distributions non-degenerate) and the observable mlp branch difference.

## Complexity

The forward costs \(O(BN(in\cdot h + h\cdot n_e + h\cdot out))\), the same order as a single LoRA (the expert mix only adds an \(O(n_e\cdot h\cdot out)\) combination term, once per batch); LP sharing compresses the LP parameters from n_layers copies to \(\lceil n/st\rceil\) copies.
