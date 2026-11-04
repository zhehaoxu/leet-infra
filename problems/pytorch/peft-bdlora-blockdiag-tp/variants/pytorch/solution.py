"""PEFT: BD-LoRA block-diagonal factors for tensor-parallel serving (official
PEFT BdLoraLinearVariant + BlockDiagonalLinear, src/peft/tuners/lora/variants.py;
paper arXiv:2510.23346, NeurIPS 2025, Amazon).

Official account:
  1. BlockDiagonalLinear stores nblocks stacked shards as ONE tensor of shape
     (out_features, in_features // nblocks); forward reshapes x to (B, nb, m)
     and the weight to (nb, n, m), computes einsum("bim,inm->bin") per shard
     and concatenates — equivalent to x @ block_diag(W)^T with zero off-diagonal.
  2. weight_as_blockdiagonal_matrix: torch.block_diag(*torch.chunk(weight,
     nblocks, dim=0)) expands the shards back to the vanilla (out, in) layout.
  3. Target classification: substring match against target_modules_bd_a /
     target_modules_bd_b; a target matching BOTH raises ValueError; under
     match_strict a target matching NEITHER raises ValueError (else plain LoRA).
  4. Init: divisibility of in_features and out_features by nblocks is required
     (ValueError); the block-diagonal lora_A is kaiming-uniform (a=sqrt(5))
     while lora_B is zero-initialized (delta W = 0 at start).
  5. Merge: delta = (B_bd @ A_bd) * scaling on the EXPANDED block-diagonal
     matrices — for linear layers (fan_in_fan_out=False) no transpose; the
     merged base weight reproduces the unmerged two-step forward exactly.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def bd_forward(x, weight, nblocks):
    """Official BlockDiagonalLinear.forward: per-shard einsum, output concat."""
    in_features = x.size(-1)
    out_features = nblocks * weight.size(0)
    if in_features % nblocks != 0 or weight.size(0) % nblocks != 0:
        raise ValueError(
            f"self.in_features={in_features} or self.out_features={weight.size(0)} not divisible by {nblocks}"
        )
    first_dims = x.shape[:-1]
    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])
    b = x.shape[0]
    m = x.shape[-1] // nblocks
    n = weight.size(0) // nblocks
    x = x.reshape(b, nblocks, m)
    w = weight.view(nblocks, n, m)
    out = torch.einsum("bim,inm->bin", x, w)
    return out.reshape(*first_dims, -1)


def block_diag_matrix(weight, nblocks):
    """Official weight_as_blockdiagonal_matrix: blocks stacked on the diagonal,
    off-diagonal entries zero."""
    return torch.block_diag(*torch.chunk(weight, nblocks, dim=0))


def classify_target(target_name, patterns_a, patterns_b, match_strict):
    """Official BdLoraLinearVariant.init classification and validation."""
    has_a = any(pattern in target_name for pattern in patterns_a)
    has_b = any(pattern in target_name for pattern in patterns_b)
    if has_a and has_b:
        raise ValueError(f"Target {target_name} matches both A and B block-diagonal patterns")
    if match_strict and not (has_a or has_b):
        raise ValueError(
            f"Target {target_name} matches neither A nor B block-diagonal patterns."
            "If this is intentional, set match_strict=False in BdLoraConfig during initialization. "
        )
    return "a" if has_a else ("b" if has_b else "plain")


def init_factors(in_features, out_features, r, nblocks, role, seed=0):
    """Official init shapes and gate: A kaiming-uniform (a=sqrt(5)), B zeros.
    Returns the compact (stacked-shard) weight."""
    if in_features % nblocks != 0 or out_features % nblocks != 0:
        raise ValueError(
            f"self.in_features={in_features} or self.out_features={out_features} not divisible by {nblocks}"
        )
    if role == "a":
        weight = torch.empty(r, in_features // nblocks, dtype=torch.float64)
        torch.manual_seed(seed)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    elif role == "b":
        weight = torch.zeros(out_features, r // nblocks, dtype=torch.float64)
    else:
        raise ValueError(f"unknown role {role!r}")
    return weight


def delta_weight(weight_a, weight_b, nblocks, scaling, fan_in_fan_out=False):
    """Official merge account on the EXPANDED block-diagonal matrices:
    transpose(B_bd @ A_bd, fan_in_fan_out) * scaling."""
    weight_a = block_diag_matrix(weight_a, nblocks)
    weight_b = block_diag_matrix(weight_b, nblocks)
    if fan_in_fan_out:
        return (weight_a.t() @ weight_b.t()) * scaling
    return (weight_b @ weight_a) * scaling


def layer_forward(x, base_weight, base_bias, weight_a, weight_b, nblocks, scaling):
    """Official training-mode forward: base + lora_B(lora_A(x)) * scaling with
    both factors block-diagonal (dropout is identity in the judging domain)."""
    base = F.linear(x, base_weight, base_bias)
    za = bd_forward(x, weight_a, nblocks)
    zb = bd_forward(za, weight_b, nblocks)
    return base + zb * scaling


def solve(mode=None, **kw):
    if mode == "bd_forward":
        return bd_forward(kw["x"], kw["weight"], kw["nblocks"])
    if mode == "block_diag_matrix":
        return block_diag_matrix(kw["weight"], kw["nblocks"])
    if mode == "classify_target":
        return classify_target(kw["target_name"], kw["patterns_a"],
                               kw["patterns_b"], kw["match_strict"])
    if mode == "init_factors":
        return init_factors(kw["in_features"], kw["out_features"], kw["r"],
                            kw["nblocks"], kw["role"], kw.get("seed", 0))
    if mode == "delta_weight":
        return delta_weight(kw["weight_a"], kw["weight_b"], kw["nblocks"],
                            kw["scaling"], kw.get("fan_in_fan_out", False))
    if mode == "layer_forward":
        return layer_forward(kw["x"], kw["base_weight"], kw["base_bias"],
                             kw["weight_a"], kw["weight_b"], kw["nblocks"],
                             kw["scaling"])
    raise ValueError(f"unknown mode {mode!r}")
