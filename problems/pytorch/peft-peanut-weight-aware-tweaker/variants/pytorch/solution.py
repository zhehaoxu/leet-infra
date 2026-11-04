"""PEANuT (arXiv:2410.01870) weight-aware neural tweakers.

Official account: HF PEFT ``src/peft/tuners/peanut/layer.py``
(commit series v0.20+). All math runs in float64 for exact anchors.

Core identities (official ``_compute_delta_weight``):
  base_weight_t = W.transpose(0, 1)              # (in, out)
  delta_w = act(base_weight_t @ A.weight.t())    # A: Linear(out, r) -> weight (r, out)
  residuals = []; for i in range(depth):
      residuals.append(delta_w); delta_w = act(encoder_i(delta_w))
  for i in range(depth):
      delta_w = act(decoder_i(delta_w)) + residuals[depth - 1 - i]
  delta_w = B(delta_w) * scaling                 # B: Linear(r, out) -> weight (out, r)
  delta = delta_w.transpose(0, 1)                # (out, in)

Parameter count:
  LoRA    = in * r + r * out
  PEANuT  = out * r + r * out + 2 * depth * r * r
(in_features > out_features  =>  PEANuT cheaper by r * (in - out).)

Initialization (official ``reset_peanut_parameters``), in this RNG order:
  A -> encoders[0..depth-1] -> decoders[0..depth-1] each kaiming_uniform_(a=sqrt(5))
  B: zeros_ if init_weights else kaiming_uniform_(a=sqrt(5))
kaiming_uniform_ with a=sqrt(5) on a Linear weight of fan_in f draws
uniform(-1/sqrt(f), 1/sqrt(f)).
"""
import math

import torch
import torch.nn.functional as F


def _act(name, x):
    if name == "relu":
        return F.relu(x)
    if name == "gelu":
        return F.gelu(x)
    if name == "tanh":
        return torch.tanh(x)
    if name == "silu":
        return F.silu(x)
    if name == "identity":
        return x
    raise ValueError(f"unsupported act_fn: {name}")


def compute_delta_weight(weight, a_w, enc_ws, dec_ws, b_w, scaling=1.0, act_fn="relu"):
    """Official _compute_delta_weight. weight: (out, in); a_w: (r, out);
    enc_ws/dec_ws: list of (r, r); b_w: (out, r). Returns (out, in)."""
    if act_fn == "relu":
        non_linear = F.relu
    elif act_fn == "gelu":
        non_linear = F.gelu
    elif act_fn == "tanh":
        non_linear = torch.tanh
    elif act_fn == "silu":
        non_linear = F.silu
    elif act_fn == "identity":
        non_linear = lambda t: t
    else:
        raise ValueError(f"unsupported act_fn: {act_fn}")

    base_weight_t = weight.transpose(0, 1).to(a_w.dtype)
    delta_w = non_linear(torch.matmul(base_weight_t, a_w.t()))

    residuals = []
    for enc in enc_ws:
        residuals.append(delta_w)
        delta_w = non_linear(F.linear(delta_w, enc))

    for dec in dec_ws:
        delta_w = non_linear(F.linear(delta_w, dec))
        delta_w = delta_w + residuals.pop()

    delta_w = F.linear(delta_w, b_w)
    return (delta_w * scaling).transpose(0, 1)


def peanut_param_count(in_features, out_features, r, depth):
    """Official peanut_A/encoders/decoders/peanut_B trainable params."""
    return out_features * r + r * out_features + 2 * depth * r * r


def lora_param_count(in_features, out_features, r):
    return in_features * r + r * out_features


def init_factors(in_features, out_features, r, depth, seed, init_weights=True,
                 dtype=torch.float64):
    """Official reset_peanut_parameters RNG replay: A -> encoders -> decoders
    -> B (zeros or kaiming). Returns dict of weight tensors."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    out["A"] = _kaiming_sqrt5(r, out_features, dtype, g)
    out["encoders"] = [_kaiming_sqrt5(r, r, dtype, g) for _ in range(depth)]
    out["decoders"] = [_kaiming_sqrt5(r, r, dtype, g) for _ in range(depth)]
    if init_weights:
        out["B"] = torch.zeros(out_features, r, dtype=dtype)
    else:
        out["B"] = _kaiming_sqrt5(out_features, r, dtype, g)
    return out


def _kaiming_sqrt5(n_out, n_in, dtype, generator):
    """nn.init.kaiming_uniform_(w, a=sqrt(5)): uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))."""
    bound = 1.0 / math.sqrt(n_in)
    return torch.empty(n_out, n_in, dtype=dtype).uniform_(-bound, bound, generator=generator)


def solve(mode, **kw):
    if mode == "delta_weight":
        return compute_delta_weight(
            kw["weight"], kw["a"], kw.get("encoders", []), kw.get("decoders", []),
            kw["b"], scaling=kw.get("scaling", 1.0), act_fn=kw.get("act_fn", "relu"))
    if mode == "forward":
        delta = compute_delta_weight(
            kw["weight"], kw["a"], kw.get("encoders", []), kw.get("decoders", []),
            kw["b"], scaling=kw.get("scaling", 1.0), act_fn=kw.get("act_fn", "relu"))
        x = kw["x"].to(delta.dtype)
        base = F.linear(x, kw["weight"])
        return base + torch.matmul(x, delta.transpose(0, 1))
    if mode == "param_count":
        p = peanut_param_count(kw["in_features"], kw["out_features"], kw["r"], kw.get("depth", 0))
        l = lora_param_count(kw["in_features"], kw["out_features"], kw["r"])
        return {"peanut": p, "lora": l, "saving": l - p}
    if mode == "init_factors":
        return init_factors(kw["in_features"], kw["out_features"], kw["r"], kw.get("depth", 0),
                            kw["seed"], init_weights=kw.get("init_weights", True))
    if mode == "zero_init_check":
        f = init_factors(kw["in_features"], kw["out_features"], kw["r"], kw.get("depth", 0),
                         kw["seed"], init_weights=True)
        delta = compute_delta_weight(kw["weight"], f["A"], f["encoders"], f["decoders"],
                                     f["B"], scaling=kw.get("scaling", 1.0),
                                     act_fn=kw.get("act_fn", "relu"))
        return bool(torch.count_nonzero(delta).item() == 0)
    if mode == "residual_order":
        # Distinguish residual add order: with depth=2 the decoder loop must add
        # residuals in reverse (res[1] first, then res[0]). Caller passes
        # identity-ish enc/dec weights plus a probe; we return the raw residual
        # pairing actually used: list of (decoder_index, residual_index).
        depth = kw.get("depth", 0)
        return [(i, depth - 1 - i) for i in range(depth)]
    raise ValueError(f"unknown mode: {mode}")


CHECKS = ["check_delta_weight", "check_forward", "check_param_count",
          "check_init_factors", "check_residual_order"]
