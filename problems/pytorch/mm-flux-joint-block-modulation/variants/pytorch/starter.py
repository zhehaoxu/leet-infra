import torch


def solve(hidden_states, encoder_hidden_states, temb, weights, attn_img, attn_txt):
    """转写 diffusers FluxTransformerBlock（double block）的调制与门控残差账。

    attention 内部（QKV 投影 + softmax + RoPE）不在本题范围——两路 attention
    的输出由调用方给定为 attn_img / attn_txt（形状与输入相同）。本题只考：
    AdaLN-Zero 的六参数生成、调制后的 LayerNorm、门控残差、以及两路的对称性。

    参数
    ----
    hidden_states         : [B, Li, D] 图像流
    encoder_hidden_states : [B, Lt, D] 文本流
    temb                  : [B, D] 时间/池化向量（SiLU 后过调制 Linear）
    weights               : dict，键与形状：
        img_mod_w/b   [6D, D] / [6D]   图像流 AdaLayerNormZero 的 linear
        txt_mod_w/b   [6D, D] / [6D]   文本流 AdaLayerNormZero 的 linear
        img_ff1_w/b   [4D, D] / [4D]   图像流 FeedForward 第一层
        img_ff2_w/b   [D, 4D] / [D]    图像流 FeedForward 第二层
        txt_ff1_w/b / txt_ff2_w/b      文本流同构
    attn_img, attn_txt    : [B, Li, D] / [B, Lt, D]，attention 输出

    返回
    ----
    dict：
        hidden_states         [B, Li, D]  图像流块输出
        encoder_hidden_states [B, Lt, D]  文本流块输出
        img_mod               [6, D]      图像流六参数，顺序
                                          [shift_msa, scale_msa, gate_msa,
                                           shift_mlp, scale_mlp, gate_mlp]
        txt_mod               [6, D]      文本流同序
        img_norm_input        [Li, D]     调制后送入 attention 的图像流值
        txt_norm_input        [Lt, D]     文本流同
        (全部取 batch 0；要求 B==1)

    步骤（与上游逐句对应，两条流共用 temb 但各有一套调制权重）：
      1. emb = Linear(silu(temb))，chunk(6) 成六参数
      2. x_norm = LayerNorm(x, eps=1e-6, 无仿射) * (1 + scale_msa) + shift_msa
      3. x = x + gate_msa * attn
      4. h = LayerNorm(x, eps=1e-6) * (1 + scale_mlp) + shift_mlp
      5. ff = ff2(GELU_tanh(ff1(h)))；x = x + gate_mlp * ff
      文本流以 encoder_hidden_states / txt_* / attn_txt 重复 1–5。

    校验：三条输入的第一维必须为 1（batch 维），否则 ValueError。
    """
    raise NotImplementedError
