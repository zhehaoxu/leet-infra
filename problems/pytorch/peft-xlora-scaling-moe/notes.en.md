# Notes: X-LoRA Scaling-Classifier Mixture of LoRA Experts

## Source

X-LoRA (arXiv:2402.07148, Eric Buehler et al., author of mistral.rs) is in HF PEFT (`src/peft/tuners/xlora/`: classifier.py + model.py + layer.py). Paper account: all pretrained LoRA adapters stay frozen; a scaling classifier (a depth-1 single Linear or a depth-d Linear+ReLU chain) predicts \(\Lambda \in \mathbb{R}^{s\times l\times n}\) from the base model's hidden states; inference is a dual forward pass (a scaling pass pins the scalings to a constant and disables experts to obtain hidden states, then the classifier predicts scalings, and the forward pass mixes experts by them). The PEFT port keeps the topk/clamp/global-weight application account in `layer.py::XLoraLayer` (the original xlora repo had it in the classifier).

## Line-by-line translation of the official implementation

**Classifier (`classifier.py`)**:

```python
class TemperatureScaledSoftmax(nn.Module):
    def forward(self, logits):
        return self.softmax(logits / self.temperature)     # dim=-1

# chain (depth>1):
layers = [Linear(h, xlora_size, bias=True), ReLU(), (Dropout?)]
for _ in range(depth - 2): Linear(xlora_size, xlora_size, bias=True), ReLU(), (Dropout?)
last = Linear(xlora_size, n_classes * n_layers, bias=True)   # layerwise=True
# or Linear(xlora_size, n_classes) + expand(-1,-1,n_layers,-1)  # layerwise=False

scalings = logits.reshape(batch, seq, n_layers, n_classes)   # C-order: (n_layers, n_classes)
if enable_softmax: scalings = temperature_softmax(scalings)

make_dummy_scalings: torch.full((b, s, n_layers, n_classes), override_scaling_pass_value)
# _set_override_scaling_pass_value(None) -> 1/n_classes
```

**Per-layer application (`layer.py::XLoraLayer`)**:

```python
def get_maybe_topk_scalings(self, scalings):
    xl = scalings[:, :, self.layer_number, :]
    if self.config.top_k_lora is not None:
        _, topk_indices = torch.topk(xl, k=self.config.top_k_lora, dim=-1)
        mask = torch.zeros_like(xl, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        xl = xl * mask.to(xl.dtype)                    # original values, NO renorm!
    if self.config.enable_softmax_topk:
        nonzero = xl != 0
        full = xl.masked_fill(~nonzero, float("-inf"))
        new = torch.softmax(full, dim=-1)
        xl = new.masked_fill(~nonzero, 0.0)            # all-zero-row guard: NaN -> 0
    return xl

@staticmethod
def apply_scalings_to_x(x, scalings_layer, adapter):
    return x * scalings_layer[:, :, adapter].unsqueeze(-1)   # input side!

# forward:
for adapter_n, active_adapter in enumerate(active_adapters):    # index = expert column
    x_mod = apply_scalings_to_x(x, xlora_scalings, adapter_n)
    result += lora_B(lora_A(dropout(x_mod))) * scaling * global_scaling_weight
```

## Pitfalls

1. **The topk mask keeps original values**: the selected top-k slots carry the raw scaling values (their sum may be < 1); only `enable_softmax_topk=True` renormalizes. The two stages are **serial** (mask first, then softmax over nonzero slots), not "softmax then topk".
2. **The all-zero-row NaN guard**: softmax over an all--inf row yields NaN; `masked_fill(~nonzero, 0.0)` repairs an all-zero row (no surviving slot after topk) to zeros. The tests pin that an all-zero row through softmax_topk must be finite — this is the only moment masked_fill is observable (in any other row the -inf slots are already 0 after softmax).
3. **The scaling multiplies the input side**: λ multiplies the **input** x of the LoRA expert (not the output). Under linearity the math is equivalent (\((x\lambda)M^{\top} = x(\lambda M)^{\top}\), a per-token column scaling), but the implementation position determines how dropout and other operators order around it; pinned by the einsum cross-check.
4. **The adapter index is the expert column**: the loop index in `for adapter_n, ... in enumerate(active_adapters)` is used directly as the column index of `scalings_layer[:, :, adapter_n]` — the n-dim order of the scaling tensor matches the adapter activation order.
5. **Scaling-pass semantics**: the dummy scalings default to 0 (config `scaling_pass_value=0`); None → \(1/n\) (the paper's \(n^{-1}\) ablation, reported to perform similarly).
6. **global_scaling_weight multiplies only the LoRA paths**: doubling gw ⇒ output = base + 2·(delta); the base is unaffected.
7. **The layerwise reshape order**: the last Linear outputs `n_classes*n_layers` and `reshape(b,s,l,n)` is C-order — W3's output layout is (l0c0, l0c1, l0c2, l1c0, …) (layer-major). A transposed rearrangement is an off-by-layout mutation (pinned by m-cf-reshape).
8. **The scalings=None branch**: degrades to plain LoRA (no scalings, no global weight, sw=1) — the behavior when the classifier is off or adapters are disabled.

## Cross-check conventions

Anchors dumped from the sandbox: F1 full depth-3 layerwise classifier chain (temperature 1.7) + two dummy variants + softmax row property; F2 non-layerwise expand (identical distributions across layers); F3 two-expert layer application (dense per-term sum cross-check + einsum equivalence + topk original-value mask + softmax_topk nonzero renormalization + one-hot + all-zero-row NaN guard + layer_number slicing). Cross-checks: \((x\lambda)M^{\top}=\mathrm{einsum}\), and gw multiplies only the delta.

## Complexity

The classifier costs \(O(h\cdot s_{xlora} + s_{xlora}^2\cdot(d-2) + s_{xlora}\cdot n\cdot l)\) per token; each layer's application is \(O(b\cdot s\cdot (in\cdot r + r\cdot out))\), the same order as plain LoRA (the scaling is elementwise); the dual forward roughly doubles inference cost (the first pass can run no_grad with experts disabled).
