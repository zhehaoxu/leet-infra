"""add_weighted_adapter — HuggingFace PEFT weighted adapter merging.

Reproduces the pure-function core of peft/src/peft/tuners/lora/model.py:
LoraModel.add_weighted_adapter and its two helpers. The new adapter combines
several trained LoRA adapters with per-adapter weights:

- resolve_rank: new_rank rules — a single source adapter forces combination_type
  "linear"; the linear family (linear/ties/dare_*/magnitude_prune) requires all
  ranks equal (else ValueError); "cat" sums the ranks; "*_svd" takes svd_rank or
  max(ranks); anything else raises ValueError.
- resolve_scaling: the combined adapter must have scaling exactly 1 (its config
  gets r = lora_alpha = new_rank and use_rslora=False), so each source's scaling
  is folded into the merged factors instead.
- merge_cat: lora_A scaled by weight * scaling[adapter], lora_B unscaled; A is
  concatenated along dim 0 (rank), B along dim 1.
- merge_linear: task arithmetic on A and B separately with sqrt-split weights —
  weight_with_scaling = weight * scaling; sign goes on the A side only:
  w_A = sqrt(|w*s|) * sign, w_B = sqrt(|w*s|); A_new = sum(w_A * A_i),
  B_new = sum(w_B * B_i).
- merge_svd: delta_i = (B_i @ A_i) * scaling_i (get_delta_weight applies the
  scaling), combined = sum(w_i * delta_i) with the RAW weights; SVD
  (kohya-ss sd-scripts convention) U = U[:, :rank] @ diag(S[:rank]),
  Vh = Vh[:rank, :], optional svd_clamp quantile on the concatenated
  U/Vh values, returns (A, B) = (Vh, U) — the singular values are absorbed
  into B, not split as sqrt.
- merge_bias: new_bias = sum(w_i * scaling_i * b_i) over sources that have
  lora_bias (others contribute nothing).

Linear is a rough approximation: (sum w_A A_i)(sum w_B B_i) has cross terms, it
equals the scaled sum of deltas only for a single source. cat and svd are exact
in delta space. Embedding/Conv branches and the TIES/DARE families are out of
scope (TIES/DARE live in peft-ties-dare-merge).
"""

import math

import torch


def solve(mode, **kw):
    if mode == "effective_rank":
        # per-adapter allocation rank: config.r, or max(r, *rank_pattern.values())
        r, rank_pattern = kw["r"], kw.get("rank_pattern") or {}
        return r if not rank_pattern else max(r, *rank_pattern.values())

    if mode == "resolve_rank":
        ranks, combination_type = kw["ranks"], kw["combination_type"]
        svd_rank = kw.get("svd_rank")
        # a single source adapter can only merge linearly
        combination_type = "linear" if len(ranks) == 1 else combination_type
        if combination_type in ("linear", "ties", "dare_ties", "dare_linear", "magnitude_prune"):
            if len(set(ranks)) != 1:
                raise ValueError(
                    "All adapters must have the same r value when using combination_type linear, ties, dare_ties or "
                    "dare_linear."
                )
            new_rank = ranks[0]
        elif combination_type == "cat":
            new_rank = sum(ranks)
        elif combination_type.endswith("svd"):
            new_rank = svd_rank or max(ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")
        return {"combination_type": combination_type, "new_rank": new_rank}

    if mode == "resolve_scaling":
        # combined config: r = lora_alpha = new_rank, use_rslora=False => scaling == 1
        alpha, r = kw["alpha"], kw["r"]
        if kw.get("use_rslora"):
            return alpha / math.sqrt(r)
        return alpha / r

    if mode == "merge_cat":
        A_list, B_list = kw["A_list"], kw["B_list"]
        weights, scalings = kw["weights"], kw["scalings"]
        loras_A = [A.data * w * s for A, w, s in zip(A_list, weights, scalings)]
        loras_B = [B.data for B in B_list]
        if len(loras_A) == 0:
            raise ValueError("No matching LoRAs found. Please raise an issue on GitHub.")
        return {"A": torch.cat(loras_A, dim=0), "B": torch.cat(loras_B, dim=1)}

    if mode == "merge_linear":
        A_list, B_list = kw["A_list"], kw["B_list"]
        weights, scalings = kw["weights"], kw["scalings"]
        w_a = [math.sqrt(abs(w * s)) * (1 if w * s >= 0 else -1) for w, s in zip(weights, scalings)]
        w_b = [math.sqrt(abs(w * s)) for w, s in zip(weights, scalings)]
        A_new = sum(w * A for w, A in zip(w_a, A_list))
        B_new = sum(w * B for w, B in zip(w_b, B_list))
        return {"A": A_new, "B": B_new}

    if mode == "merge_svd":
        A_list, B_list = kw["A_list"], kw["B_list"]
        weights, scalings = kw["weights"], kw["scalings"]
        new_rank = kw["new_rank"]
        clamp = kw.get("clamp")
        full_matrices = kw.get("full_matrices", False)
        if len(A_list) == 0:
            raise ValueError("No matching LoRAs found. Please raise an issue on GitHub.")
        # get_delta_weight applies the scaling, no need to handle it explicitly
        delta_weight = [(B @ A) * s for A, B, s in zip(A_list, B_list, scalings)]
        combined = sum(w * d for w, d in zip(weights, delta_weight))
        U, S, Vh = torch.linalg.svd(combined, full_matrices=full_matrices)
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:new_rank, :]
        if clamp is not None:
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp)
            U = U.clamp(-hi_val, hi_val)
            Vh = Vh.clamp(-hi_val, hi_val)
        return {"A": Vh, "B": U}

    if mode == "merge_bias":
        biases, weights, scalings = kw["biases"], kw["weights"], kw["scalings"]
        known = [(b, w, s) for b, w, s in zip(biases, weights, scalings) if b is not None]
        if not known:
            raise ValueError("No matching LoRAs found. Please raise an issue on GitHub.")
        new_bias = torch.zeros_like(known[0][0])
        for b, w, s in known:
            new_bias = new_bias + w * s * b
        return {"bias": new_bias}

    raise ValueError(f"unknown mode: {mode}")


CHECKS = [
    "check_resolve",
    "check_cat",
    "check_linear",
    "check_svd",
    "check_bias",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
