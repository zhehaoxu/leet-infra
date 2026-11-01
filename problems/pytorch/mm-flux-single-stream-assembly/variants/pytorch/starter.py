import torch


def solve(enc_out, img_out, t_emb, pooled_proj, weights, attn_output):
    """FLUX single-stream 拼流与 vec 共享账 / FLUX single-stream assembly and shared vec.

    中文契约：
    - 转写 diffusers FluxTransformer2DModel 的 single-stream 阶段（double blocks
      之后的全部接线）。attention 内部（QKV/RoPE/RMSNorm）不考——
      attn_output 由调用方给定。
    - enc_out：[B, Lt, D] float32/float64（double blocks 后的文本流）；
      img_out：[B, Li, D] 同 dtype（double blocks 后的图像流）。B=1..3，
      Lt=1..8，Li=1..8，D=4..16。
    - vec 装配（CombinedTimestepTextProjEmbeddings 的输出语义）：
      t_emb [B, D] 与 pooled_proj [B, D] 同 dtype；vec = t_emb + pooled_proj
      ——是**逐元素加法**，不是拼接（官方 `conditioning = timesteps_emb +
      pooled_projections`）。vec 不再经过任何变换直接喂给每个 single block
      的调制层。
    - 拼流顺序：hidden = cat([enc_out, img_out], dim=1)——**文本流在前、
      图像流在后**（官方 `torch.cat([encoder_hidden_states, hidden_states], dim=1)`）。
    - weights：dict，键与形状：
        single_mod_w/b   [3D, D] / [3D]   AdaLayerNormZeroSingle 的 linear
        proj_mlp_w/b     [M, D]  / [M]    proj_mlp（M = mlp_hidden_dim）
        proj_out_w/b     [D, D+M]/ [D]    proj_out
      3D 与 M 均从权重形状推出，拒绝 D 不一致。
    - 每个 single block（本题只跑一个，两流共享同一次调制账）：
      * residual = hidden（拼流后的整体，文本与图像一起参与残差）；
      * mod = single_mod(silu(vec)) → chunk(3, dim=1) → shift_msa, scale_msa,
        gate_msa（**3 参数**，区别于 double block 的 6 参数）；
      * norm_x = layer_norm(hidden, eps=1e-6, 无仿射) * (1 + scale_msa)[:, None]
        + shift_msa[:, None]；
      * norm_x **同源**喂 attention 与 MLP：mlp = gelu_tanh(proj_mlp(norm_x))；
      * fused = cat([attn_output, mlp], dim=2)（通道维拼接，attn 在前）；
      * out = gate_msa[:, None] * proj_out(fused)；hidden = residual + out。
      attn_output 的形状必须等于 norm_x（[B, Lt+Li, D]）。
    - 切回：hidden_final = hidden[:, Lt:, :]——**按文本流长度切片**取回图像流。
    - 返回四元组（顺序固定）：
      vec：[B, D]；shift_msa / scale_msa / gate_msa 各 [B, D]（三个中间账本）；
      hidden_final：[B, Li, D]。
      实际返回 (vec, shift_msa, scale_msa, gate_msa, hidden_final) 五元组。
    - ValueError：任一输入 dtype 不是 float32/float64、各张量 dtype 不一致、
      含非有限值、形状不合法（含 batch 维不一致、enc_out/img_out 维数不对）、
      weights 缺键或形状与 D/M 推导不符、attn_output 形状不为 [B, Lt+Li, D]。
      接受非连续输入，不修改任何输入。禁止 F.scaled_dot_product_attention；
      F.linear/F.layer_norm/gelu 均可用。

    English contract mirrors the Chinese one: enc_out/img_out [B,Lt,Li dims as
    above] same dtype; vec = t_emb + pooled_proj (elementwise ADD, not concat);
    hidden = cat([enc, img], dim=1) with TEXT first; per single block:
    mod = linear(silu(vec)) chunked into 3 (shift_msa, scale_msa, gate_msa);
    norm_x = layer_norm(hidden, eps=1e-6, no affine) * (1+scale)[:,None] +
    shift[:,None]; the SAME norm_x feeds attention and MLP; fused =
    cat([attn_output, mlp], dim=2); out = gate[:,None] * proj_out(fused);
    hidden = residual + out; finally slice back hidden[:, Lt:]. Returns the
    5-tuple (vec, shift_msa, scale_msa, gate_msa, hidden_final). Raise
    ValueError on every malformed input listed above. No
    scaled_dot_product_attention; F.linear/F.layer_norm/gelu allowed.
    """
    raise NotImplementedError
