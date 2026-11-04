"""VeLoRA: Memory Efficient Training using Rank-1 Sub-Token Projections
(NeurIPS 2024, official roymiles/VeLoRA, velora/velora.py).

VeLoRA compresses the intermediate activations that backpropagation must
keep in memory onto a FIXED rank-1 subspace: the forward pass splits every
token along the feature dim into num_groups sub-tokens and projects each
onto one frozen unit vector, storing a single scalar per sub-token; the
backward pass coarsely rebuilds the activations from the same vector.
Only lora_A's gradient flows through the RECONSTRUCTED activations;
lora_B's gradient uses the real lora_A outputs. The projection vector is
initialized once from the first batch (batch_average_once) and frozen.

Accounts (official, verbatim):
- project: x (B, N, C) -> reshape (B*N*G, C//G) -> einsum('h s, r s -> h')
- init_embed: normalize(data.mean(0), p=2, dim=-1), rank-1 only
- reconstruct: einsum('h, r s -> h s', zp, v).view(input_shape)
- grads: grad_input = grad_output @ weight; grad_weight = velora_scale *
  sum_b(grad_output[b]^T @ input_hat[b]); grad_bias = grad_output.sum(1)
- lora_block: velora_scale multiplies grad_a ONLY; result is plain LoRA
"""
import torch
import torch.nn.functional as F


def project(x, embed, num_groups):
    raise NotImplementedError


def init_embed(data):
    raise NotImplementedError


def reconstruct(zp, v, shape):
    raise NotImplementedError


def grads(input_hat, weight, grad_output, velora_scale, use_bias):
    raise NotImplementedError


def lora_block(x, weight, bias, lora_a, lora_b, scaling, num_groups, embed,
               velora_scale, grad_out_m, grad_out_r):
    raise NotImplementedError


def solve(mode=None, **kw):
    if mode == "project":
        return project(kw["x"], kw["embed"], kw["num_groups"])
    if mode == "init_embed":
        return init_embed(kw["data"])
    if mode == "reconstruct":
        return reconstruct(kw["zp"], kw["v"], kw["shape"])
    if mode == "grads":
        return grads(kw["input_hat"], kw["weight"], kw["grad_output"],
                     kw["velora_scale"], kw["use_bias"])
    if mode == "lora_block":
        return lora_block(kw["x"], kw["weight"], kw["bias"], kw["lora_a"],
                          kw["lora_b"], kw["scaling"], kw["num_groups"],
                          kw["embed"], kw["velora_scale"],
                          kw["grad_out_m"], kw["grad_out_r"])
    raise ValueError(f"unknown mode {mode!r}")
