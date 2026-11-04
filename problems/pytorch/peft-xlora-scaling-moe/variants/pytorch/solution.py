"""X-LoRA (Mixture of LoRA Experts, arXiv:2402.07148, PEFT v0.20.0 tuners/xlora).

A scaling classifier predicts per-token, per-layer, per-expert coefficients from the base
model's last hidden state; each LoRA layer multiplies its input by its slice of the scalings
before the frozen LoRA experts, with optional top-k sparsification and softmax renormalization.

Accounts (official PEFT implementation):
- TemperatureScaledSoftmax: softmax(logits / temperature, dim=-1)
- classifier chain: Linear+ReLU(+dropout) x (depth-1) -> last Linear(h, n_classes*n_layers)
  (layerwise) or (h, n_classes) then expand over layers; reshape (b, s, n_layers, n_classes)
- dummy scalings (scaling pass): full tensor of the override value; None -> 1/n_classes
- maybe_topk: slice [:, :, layer_number, :]; top-k keeps ORIGINAL values via a bool mask
  (no renormalization); enable_softmax_topk renormalizes over nonzero slots only
- apply_scalings_to_x: scalings[:, :, adapter].unsqueeze(-1) * x  — the scaling multiplies
  the INPUT to the LoRA expert (equivalent to a per-token column scaling of B@A)
- layer delta: sum_a (x * lambda_a) @ (B_a A_a).T * scaling * global_scaling_weight
"""
import torch


def temperature_softmax(logits, temperature):
    return torch.softmax(logits / temperature, dim=-1)


def classifier_forward(hidden, weights, n_layers, n_classes, temperature, layerwise=True):
    """hidden (b, s, h); weights: chain of (W, bias) with the last being the final Linear.
    layerwise=True: last Linear outputs n_classes*n_layers, reshaped C-order to (n_layers, n_classes).
    layerwise=False: outputs n_classes, expanded over layers."""
    h = hidden
    for W, b in weights[:-1]:
        h = torch.relu(h @ W.T + b)
    W, b = weights[-1]
    logits = h @ W.T + b
    b_, s, _ = hidden.shape
    if not layerwise:
        logits = logits.unsqueeze(2).expand(-1, -1, n_layers, -1)
    scalings = logits.reshape(b_, s, n_layers, n_classes)
    return temperature_softmax(scalings, temperature)


def dummy_scalings(batch, seq, n_layers, n_classes, override=None):
    """Scaling-pass dummy: full tensor of the override value; None -> 1/n_classes."""
    value = 1.0 / n_classes if override is None else override
    return torch.full((batch, seq, n_layers, n_classes), value, dtype=torch.float64)


def maybe_topk_scalings(scalings, layer_number, top_k=None, enable_softmax_topk=False):
    """scalings (b, s, n_layers, n_classes) -> (b, s, n_classes) for one LoRA layer.
    top-k: bool-mask keeps original values (no renorm). softmax_topk: renormalize nonzero slots."""
    xl = scalings[:, :, layer_number, :]
    if top_k is not None:
        _, topk_indices = torch.topk(xl, k=top_k, dim=-1)
        mask = torch.zeros_like(xl, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        xl = xl * mask.to(xl.dtype)
    if enable_softmax_topk:
        nonzero = xl != 0
        full = xl.masked_fill(~nonzero, float("-inf"))
        new = torch.softmax(full, dim=-1)
        xl = new.masked_fill(~nonzero, 0.0)
    return xl


def apply_scalings_to_x(x, scalings_layer, adapter):
    """(b, s, in) * (b, s, 1) — the scaling multiplies the input to the LoRA expert."""
    return x * scalings_layer[:, :, adapter].unsqueeze(-1)


def layer_forward(x, scalings, lora_pairs, scaling, global_scaling_weight, layer_number,
                  base_weight, top_k=None, enable_softmax_topk=False):
    """x (b, s, in); lora_pairs: [(A (r,in), B (out,r))] per expert; base_weight (out, in).
    Returns x @ W0.T + sum_a (x * lambda_a) @ (B_a A_a).T * scaling * global_scaling_weight."""
    if scalings is not None:
        sl = maybe_topk_scalings(scalings, layer_number, top_k, enable_softmax_topk)
    result = x @ base_weight.T
    for adapter_n, (A, B) in enumerate(lora_pairs):
        if scalings is not None:
            x_mod = apply_scalings_to_x(x, sl, adapter_n)
            sw = global_scaling_weight
        else:
            x_mod = x
            sw = 1.0
        result = result + x_mod @ (B @ A).T * scaling * sw
    return result


def solve(mode=None, **kw):
    if mode == "temperature_softmax":
        return temperature_softmax(kw["logits"], kw["temperature"])
    if mode == "classifier_forward":
        return classifier_forward(kw["hidden"], kw["weights"], kw["n_layers"],
                                  kw["n_classes"], kw["temperature"],
                                  kw.get("layerwise", True))
    if mode == "dummy_scalings":
        return dummy_scalings(kw["batch"], kw["seq"], kw["n_layers"], kw["n_classes"],
                              override=kw.get("override"))
    if mode == "maybe_topk_scalings":
        return maybe_topk_scalings(kw["scalings"], kw["layer_number"],
                                   top_k=kw.get("top_k"),
                                   enable_softmax_topk=kw.get("enable_softmax_topk", False))
    if mode == "layer_forward":
        return layer_forward(kw["x"], kw["scalings"], kw["lora_pairs"], kw["scaling"],
                             kw["global_scaling_weight"], kw["layer_number"],
                             kw["base_weight"], top_k=kw.get("top_k"),
                             enable_softmax_topk=kw.get("enable_softmax_topk", False))
    if mode == "apply_scalings_to_x":
        return apply_scalings_to_x(kw["x"], kw["scalings_layer"], kw["adapter"])
    raise ValueError(f"unknown mode {mode!r}")
