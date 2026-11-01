import torch
import torch.nn.functional as F


def solve(enc_out, img_out, t_emb, pooled_proj, weights, attn_output):
    """FLUX single-stream 拼流与 vec 共享账（见 starter 契约）。"""
    tensors = {
        "enc_out": enc_out, "img_out": img_out, "t_emb": t_emb,
        "pooled_proj": pooled_proj, "attn_output": attn_output,
    }
    dtype = None
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if t.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} dtype must be float32/float64")
        if not torch.isfinite(t).all():
            raise ValueError(f"{name} contains non-finite values")
        if dtype is None:
            dtype = t.dtype
        elif t.dtype != dtype:
            raise ValueError("all tensors must share one dtype")
    if enc_out.dim() != 3 or img_out.dim() != 3:
        raise ValueError("enc_out/img_out must be 3-D")
    if t_emb.dim() != 2 or pooled_proj.dim() != 2:
        raise ValueError("t_emb/pooled_proj must be 2-D")
    B, Lt, D = enc_out.shape
    if img_out.shape != (B, img_out.shape[1], D) or img_out.shape[0] != B:
        raise ValueError("enc_out/img_out batch/dim mismatch")
    if img_out.dim() != 3 or img_out.shape[2] != D:
        raise ValueError("img_out channel dim must equal enc_out's")
    if t_emb.shape != (B, D) or pooled_proj.shape != (B, D):
        raise ValueError("t_emb/pooled_proj must be [B, D]")
    if not (1 <= B <= 3) or not (1 <= Lt <= 8) or not (1 <= img_out.shape[1] <= 8) or not (4 <= D <= 16):
        raise ValueError("sizes out of range")
    Li = img_out.shape[1]

    if not isinstance(weights, dict):
        raise ValueError("weights must be a dict")
    for k in ("single_mod_w", "single_mod_b", "proj_mlp_w", "proj_mlp_b",
              "proj_out_w", "proj_out_b"):
        if k not in weights:
            raise ValueError(f"weights missing key {k}")
        w = weights[k]
        if not isinstance(w, torch.Tensor) or w.dtype != dtype:
            raise ValueError(f"weights[{k}] must share the input dtype")
        if not torch.isfinite(w).all():
            raise ValueError(f"weights[{k}] contains non-finite values")
    mod_w, mod_b = weights["single_mod_w"], weights["single_mod_b"]
    mlp_w, mlp_b = weights["proj_mlp_w"], weights["proj_mlp_b"]
    out_w, out_b = weights["proj_out_w"], weights["proj_out_b"]
    if mod_w.dim() != 2 or mod_w.shape != (3 * D, D) or mod_b.shape != (3 * D,):
        raise ValueError("single_mod weights must be [3D, D] / [3D]")
    M = mlp_w.shape[0]
    if mlp_w.dim() != 2 or mlp_w.shape[1] != D or mlp_b.shape != (M,):
        raise ValueError("proj_mlp weights must be [M, D] / [M]")
    if out_w.dim() != 2 or out_w.shape != (D, D + M) or out_b.shape != (D,):
        raise ValueError("proj_out weights must be [D, D+M] / [D]")

    if attn_output.shape != (B, Lt + Li, D):
        raise ValueError("attn_output must be [B, Lt+Li, D]")

    # vec 装配：逐元素加法，不是拼接
    vec = t_emb + pooled_proj

    # 拼流：文本流在前、图像流在后
    hidden = torch.cat([enc_out, img_out], dim=1)
    residual = hidden

    mod = F.linear(F.silu(vec), mod_w, mod_b)
    shift_msa, scale_msa, gate_msa = mod.chunk(3, dim=1)

    norm_x = F.layer_norm(hidden, (D,), None, None, 1e-6)
    norm_x = norm_x * (1 + scale_msa)[:, None] + shift_msa[:, None]

    mlp = F.gelu(F.linear(norm_x, mlp_w, mlp_b), approximate="tanh")
    fused = torch.cat([attn_output, mlp], dim=2)
    out = gate_msa[:, None] * F.linear(fused, out_w, out_b)
    hidden = residual + out

    hidden_final = hidden[:, Lt:, :]
    return vec, shift_msa, scale_msa, gate_msa, hidden_final
