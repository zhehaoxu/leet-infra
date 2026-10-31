import torch
import torch.nn.functional as F


def _adalon_norm_zero(x, temb, w, b):
    """AdaLayerNormZero：silu(temb) 过 linear 后 chunk(6)，返回调制后的 x 与六参数。"""
    emb = F.linear(F.silu(temb), w, b)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
    x = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
    x = x * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
    return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


def _ff(x, w1, b1, w2, b2):
    h = F.linear(x, w1, b1)
    # GELU tanh 近似（FeedForward activation_fn="gelu-approximate"）
    h = F.gelu(h, approximate="tanh")
    return F.linear(h, w2, b2)


def _run_stream(x, temb, mod_w, mod_b, ff1_w, ff1_b, ff2_w, ff2_b, attn):
    x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = _adalon_norm_zero(x, temb, mod_w, mod_b)
    x = x + gate_msa[:, None, :] * attn
    h = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
    h = h * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
    x = x + gate_mlp[:, None, :] * _ff(h, ff1_w, ff1_b, ff2_w, ff2_b)
    return x, x_norm[0]


def solve(hidden_states, encoder_hidden_states, temb, weights, attn_img, attn_txt):
    """按 starter 注释的 5 步跑 FLUX double block 的两条流。"""
    for name, t in (("hidden_states", hidden_states), ("encoder_hidden_states", encoder_hidden_states),
                    ("temb", temb), ("attn_img", attn_img), ("attn_txt", attn_txt)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if t.shape[0] != 1:
            raise ValueError("batch size must be 1")

    w = weights
    img_out, img_norm_in = _run_stream(hidden_states, temb, w["img_mod_w"], w["img_mod_b"],
                          w["img_ff1_w"], w["img_ff1_b"], w["img_ff2_w"], w["img_ff2_b"], attn_img)
    txt_out, txt_norm_in = _run_stream(encoder_hidden_states, temb, w["txt_mod_w"], w["txt_mod_b"],
                          w["txt_ff1_w"], w["txt_ff1_b"], w["txt_ff2_w"], w["txt_ff2_b"], attn_txt)

    img_emb = F.linear(F.silu(temb), w["img_mod_w"], w["img_mod_b"])
    txt_emb = F.linear(F.silu(temb), w["txt_mod_w"], w["txt_mod_b"])

    D = hidden_states.shape[-1]
    return {
        "hidden_states": img_out,
        "encoder_hidden_states": txt_out,
        "img_mod": img_emb[0].reshape(6, D),   # chunk(6) 顺序即行序
        "txt_mod": txt_emb[0].reshape(6, D),
        "img_norm_input": img_norm_in,         # 调制后送入 attention 的值
        "txt_norm_input": txt_norm_in,
    }
