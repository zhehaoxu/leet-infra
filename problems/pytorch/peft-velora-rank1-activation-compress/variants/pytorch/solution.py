"""PEFT: VeLoRA rank-1 sub-token activation compression (official repo
roymiles/VeLoRA, velora/velora.py, NeurIPS 2024, arXiv:2405.17991).

Official account:
  1. Sub-token projection (forward): x (B, N, C) is split along the feature
     dim into num_groups chunks; each chunk is flattened to a row and projected
     onto a FIXED rank-1 vector: zp = einsum('h s, r s -> h', flat, embed).
     embed is L2-normalized (unit row). zp (B*N*G,) is all that is stored.
  2. batch_average_once init: embed = normalize(data.mean(0), p=2, dim=-1)
     from the FIRST batch, then frozen behind the initted gate.
  3. Backward reconstruction: input_hat = einsum('h, r s -> h s', zp, v)
     .view(input_shape) — a coarse rank-1 rebuild of the dropped activations.
  4. Gradients: grad_input = grad_output @ weight; grad_weight is the
     per-sample (B, M, D) term grad_output^T @ input_hat whose leading batch
     dim autograd sums, times velora_scale; grad_bias = grad_output.sum(1)
     when a bias exists.
  5. LoRA block: ONLY lora_A's gradient flows through the RECONSTRUCTED
     activations (input_hat); lora_B's gradient uses the REAL lora_A outputs;
     velora_scale multiplies grad_A only; the forward is the plain LoRA path.
"""
import torch
import torch.nn.functional as F


def project(x, embed, num_groups):
    """Official forward: split tokens into sub-token chunks, project each on
    the fixed rank-1 vector -> zp (B*N*G,) is the compressed activation."""
    B, N, C = x.shape
    flat = x.reshape(B * N * num_groups, C // num_groups)
    return torch.einsum("h s, r s -> h", flat, embed)


def init_embed(data):
    """batch_average_once: mean of the first batch, then L2-normalized, frozen."""
    return F.normalize(data.mean(0), p=2, dim=-1)


def reconstruct(zp, v, shape):
    """Official backward: coarse rank-1 rebuild of the dropped activations."""
    return torch.einsum("h, r s -> h s", zp, v).view(shape)


def grads(input_hat, weight, grad_output, velora_scale, use_bias):
    """Official VeLoRA.backward; autograd sums the leading batch dim of the
    per-sample (B, M, D) grad_weight term."""
    grad_input = torch.matmul(grad_output, weight)
    per_batch = torch.matmul(grad_output.transpose(1, 2), input_hat)
    grad_weight = velora_scale * per_batch.sum(0)
    grad_bias = grad_output.sum(1) if use_bias else None
    return {"grad_input": grad_input, "grad_weight": grad_weight,
            "grad_bias": grad_bias}


def lora_block(x, weight, bias, lora_a, lora_b, scaling, num_groups, embed,
               velora_scale, grad_out_m, grad_out_r):
    """Full training-mode branch: only lora_A's gradient flows through the
    reconstructed activations; lora_B uses the real lora_A outputs."""
    base = F.linear(x, weight, bias)
    zp = project(x, embed, num_groups)
    a_out = F.linear(x, lora_a, None)
    input_hat = reconstruct(zp, embed, x.shape)
    grad_a = velora_scale * torch.einsum("b n r, b n c -> r c", grad_out_r, input_hat)
    grad_b = scaling * torch.einsum("b n m, b n r -> m r", grad_out_m, a_out)
    result = base + (a_out @ lora_b.t()) * scaling
    grad_input = grad_out_m @ weight + scaling * ((grad_out_m @ lora_b) @ lora_a)
    return {"result": result, "grad_a": grad_a, "grad_b": grad_b,
            "grad_input": grad_input}


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
