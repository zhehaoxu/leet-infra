# 详解：FLUX double block 调制账

## 出处

diffusers `models/transformers/transformer_flux.py` 的 `FluxTransformerBlock.forward`（2026-10-31 经 GitHub contents API 取得，逐句转写）；`AdaLayerNormZero` 在 `models/normalization.py`（`silu → Linear(dim, 6*dim) → chunk(6)`，LayerNorm 无仿射 eps=1e-6）。FeedForward `activation_fn="gelu-approximate"` = GELU tanh 近似。

## 核心与易错点

1. **残差加回原输入**是本题最大陷阱：`hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output`——加的是**进入 block 前的** `hidden_states`，不是调制后的 `norm_hidden_states`（那只是 attention 的输入）。初版纯 Python oracle 把残差加在调制值上，与 torch 中间态对拍时才暴露（oracle 错、solution 对）。
2. **msa 路调制的可观察面**：`shift_msa/scale_msa` 只影响 attention 输入；本题 attention 输出显式入参，若账本不收 `img_norm_input`，这两个参数就完全不可检验（变异 `no-plus-scale` 曾存活正是这个原因）。账本把它纳入后变异全抓——这也是「attention 拆出去的题必须把 attention 的输入记进账」的一般教训。
3. **六参数 chunk 顺序**：`[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]`，`img_mod` 账本行序即 chunk 序。顺序错位会同时破坏输出与账。
4. **gate 归零 = 恒等**：gate_msa 与 gate_mlp 的权重行、偏置段同时清零，输出逐位回到输入（残差恒等）——注意只清 bias 不清 weight 行是无效的（weight 贡献仍在），初版用例踩过。
5. **eps 的可观察面**：D=4、值 O(1) 时 LN eps 1e-6 vs 1e-5 的差异 ~1e-6，落在 1e-5 容差内——用近常数行（方差 ~1e-6，eps 主导分母）放大到 % 级才可判。
6. **两路独立**：两流共用 temb 但调制权重独立；图像流权重扰动不影响文本流输出（`atol=1e-9` 级隔离断言）。

## 变异测试（9/9 全抓）

`1+scale` 丢加法、去 SiLU、残差加到调制值、FF 门控丢失、GELU 改 exact、eps 1e-5、chunk 顺序错位、FF 去激活、两路共用调制权重、batch 校验删除。无等价变异。

## 复杂度

D=4、Li=2、Lt=3，微秒级。
