"""ShadowPEFT (arXiv:2604.19254) — shadow injection & gated shadow update.

Official account: ShadowLLM/shadow-peft ``src/shadow_peft/modules.py``
(commit 65a43a7). All math runs in float64; dropout is outside the judging
domain (eval semantics, p=0).

ShadowInjectionModel (applied before each base decoder layer except layer 0):
  delta  = hidden - shadow_hidden                    # (b, t, d)
  delta_t = einsum("btd,dk->btk", delta, downs[l])   # W_down: (d, k), normal(std=0.02)
  delta_t = dropout(delta_t)                          # eval: identity
  delta_t = einsum("btk,kd->btd", delta_t, ups[l])   # W_up: (k, d), zeros at init
  hidden' = hidden + alpha * delta_t

ShadowUpdateModel (after each base decoder layer):
  h_in = LayerNorm(hidden)                            # shared LN, eps=1e-5
  ht   = Linear2(SiLU(Linear1(h_in)))                 # transform, no sigmoid, dropout mid
  g    = sigmoid(Linear2(SiLU(Linear1(h_in))))        # gate, bias-free everywhere
  shadow' = shadow + g * (ht - shadow)                # == (1-g)*shadow + g*ht
"""
import math

import torch
import torch.nn.functional as F


def inject(hidden, shadow, w_down, w_up, alpha):
    """One ShadowInjectionModel step. hidden/shadow: (b, t, d);
    w_down: (d, k); w_up: (k, d)."""
    delta = hidden - shadow
    delta_t = torch.einsum("btd,dk->btk", delta, w_down)
    delta_t = torch.einsum("btk,kd->btd", delta_t, w_up)
    return hidden + alpha * delta_t


def shadow_update(hidden, shadow, ln_w, ln_b, gate_w1, gate_w2,
                  tf_w1, tf_w2, eps=1e-5):
    """One ShadowUpdateModel step. Weights: gate_w1 (d, gh), gate_w2 (gh, d),
    tf_w1 (d, gh), tf_w2 (gh, d); all bias-free."""
    h_in = F.layer_norm(hidden, (hidden.shape[-1],), ln_w, ln_b, eps)
    g = torch.sigmoid(F.linear(F.silu(F.linear(h_in, gate_w1)), gate_w2))
    ht = F.linear(F.silu(F.linear(h_in, tf_w1)), tf_w2)
    return shadow + g * (ht - shadow)


def gate_values(hidden, ln_w, ln_b, gate_w1, gate_w2, eps=1e-5):
    h_in = F.layer_norm(hidden, (hidden.shape[-1],), ln_w, ln_b, eps)
    return torch.sigmoid(F.linear(F.silu(F.linear(h_in, gate_w1)), gate_w2))


def adapter_param_count(num_layers, hidden_size, injection_hidden_size, gate_hidden_size):
    """Trainable params of injection (downs+ups) and update (gates+transforms+LN)."""
    inject_p = num_layers * (hidden_size * injection_hidden_size
                             + injection_hidden_size * hidden_size)
    update_p = num_layers * 2 * (hidden_size * gate_hidden_size
                                 + gate_hidden_size * hidden_size) + 2 * hidden_size
    return {"injection": inject_p, "update": update_p, "total": inject_p + update_p}


def init_injection(num_layers, hidden_size, injection_hidden_size, seed,
                   initialization_std=0.02):
    """Official init RNG replay: downs ~ N(0, 0.02^2); ups zeros."""
    g = torch.Generator().manual_seed(seed)
    downs = torch.empty(num_layers, hidden_size, injection_hidden_size).normal_(
        mean=0.0, std=initialization_std, generator=g)
    ups = torch.zeros(num_layers, injection_hidden_size, hidden_size)
    return {"downs": downs, "ups": ups}


def solve(mode, **kw):
    if mode == "inject":
        return inject(kw["hidden"], kw["shadow"], kw["w_down"], kw["w_up"],
                      kw.get("alpha", 0.1))
    if mode == "inject_zero_init":
        h = kw["hidden"]
        out = inject(h, kw["shadow"], kw["w_down"],
                     torch.zeros_like(kw["w_up"]), kw.get("alpha", 0.1))
        return bool(torch.equal(out, h))
    if mode == "inject_sequence":
        hidden, shadow = kw["hidden"], kw["shadow"]
        downs, ups = kw["downs"], kw["ups"]
        outs = []
        for layer_idx in range(len(kw.get("layer_indices", range(downs.shape[0])))):
            hidden = inject(hidden, shadow, downs[layer_idx], ups[layer_idx],
                            kw.get("alpha", 0.1))
            outs.append(hidden.clone())
        return outs
    if mode == "update":
        return shadow_update(kw["hidden"], kw["shadow"], kw["ln_w"], kw["ln_b"],
                             kw["gate_w1"], kw["gate_w2"], kw["tf_w1"], kw["tf_w2"],
                             eps=kw.get("eps", 1e-5))
    if mode == "gate":
        return gate_values(kw["hidden"], kw["ln_w"], kw["ln_b"],
                           kw["gate_w1"], kw["gate_w2"], eps=kw.get("eps", 1e-5))
    if mode == "gated_identity":
        s, g, ht = kw["shadow"], kw["gate"], kw["ht"]
        a = s + g * (ht - s)
        b = (1 - g) * s + g * ht
        return bool(torch.allclose(a, b, atol=1e-12, rtol=0))
    if mode == "param_count":
        return adapter_param_count(kw["num_layers"], kw["hidden_size"],
                                   kw["injection_hidden_size"], kw["gate_hidden_size"])
    if mode == "init_injection":
        return init_injection(kw["num_layers"], kw["hidden_size"],
                              kw["injection_hidden_size"], kw["seed"],
                              initialization_std=kw.get("initialization_std", 0.02))
    raise ValueError(f"unknown mode: {mode}")


CHECKS = ["check_inject", "check_update", "check_gated_identity",
          "check_param_count", "check_init_injection"]
