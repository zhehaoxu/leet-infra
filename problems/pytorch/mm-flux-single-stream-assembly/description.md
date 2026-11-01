# 多模态：FLUX single-stream 拼流与 vec 共享账

## 题面

FLUX 的 transformer 分两段：前面是 double blocks（双流 MMDiT），后面是 single blocks（单流 DiT）。本题转写**后一段的全部接线**：给定 double blocks 之后的文本流与图像流、条件嵌入的两路输入、权重字典与 attention 输出（attention 内部不考），返回 single-stream 阶段的全部中间账本与最终图像流。

输入：
- `enc_out` [B, Lt, D]：double blocks 后的文本流；
- `img_out` [B, Li, D]：double blocks 后的图像流；
- `t_emb` [B, D] / `pooled_proj` [B, D]：timestep 嵌入与池化文本投影；
- `weights`：`single_mod_w/b` [3D,D]/[3D]、`proj_mlp_w/b` [M,D]/[M]、`proj_out_w/b` [D,D+M]/[D]；
- `attn_output` [B, Lt+Li, D]：拼流后的 attention 输出（由调用方给定）。

返回五元组 `(vec, shift_msa, scale_msa, gate_msa, hidden_final)`。

## 规则（全部对应 diffusers v0.32.1 原文）

1. **vec 装配**：`vec = t_emb + pooled_proj`——**逐元素加法**，不是拼接。
2. **拼流**：`hidden = cat([enc_out, img_out], dim=1)`——**文本流在前**。
3. **3 参数调制**：`mod = single_mod(silu(vec))`，chunk(3) 得 shift_msa / scale_msa / gate_msa（single block 是 3 参数，不是 double block 的 6 参数）。
4. **调制归一化**：`norm_x = layer_norm(hidden, eps=1e-6, 无仿射) * (1 + scale_msa) + shift_msa`（[:, None] 广播）。
5. **同源输入**：`norm_x` 同时喂 attention 与 MLP；`mlp = gelu_tanh(linear(norm_x))`——**linear 在前、GELU 在后**（官方 `act_mlp(proj_mlp(x))`）。
6. **通道拼接**：`fused = cat([attn_output, mlp], dim=2)`（attn 在前），`out = gate_msa * proj_out(fused)`，`hidden = hidden + out`（残差作用于拼流后的整体）。
7. **切回**：`hidden_final = hidden[:, Lt:, :]`——按文本流长度切片取回图像流。

## 样例

```text
输入:  B=1, Lt=1, Li=1, D=6（其余见 selftest seed=4 用例）
输出:  vec [1,6]；shift/scale/gate 各 [1,6]（= linear(silu(vec)) 的三等分）；
       hidden_final [1,1,6]（= img_out + gate * proj_out(cat([attn, mlp]))）
（gate 段权重/偏置清零时 hidden_final 逐元素等于 img_out——残差退化）
```

## 数据范围与边界

- B=1..3，Lt/Li=1..8，D=4..16；全部张量 float32/float64 同 dtype，含非有限值即 ValueError。
- weights 缺键、形状与 D/M 推导不符、attn_output 形状不是 [B, Lt+Li, D] 均 ValueError。
- 接受非连续输入，不修改任何输入；禁止 F.scaled_dot_product_attention。

## 提示

- 两个最容易写反的点：vec 是加法不是拼接；拼流文本在前、切片按 Lt。
- MLP 的激活位置：`gelu(linear(x))`，不是 `linear(gelu(x))`。
- gate 只乘 proj_out 的输出，不乘残差。
