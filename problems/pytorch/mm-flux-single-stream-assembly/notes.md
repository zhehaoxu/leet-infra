# 解析：FLUX single-stream 拼流与 vec 共享账

## 实现思路

本题转写 diffusers `FluxTransformer2DModel` 的 single-stream 阶段：double blocks 之后到输出切片之前的全部接线。solution 按官方顺序五步走：vec 装配（`t_emb + pooled_proj` 逐元素加法）、`cat([enc, img], dim=1)` 拼流（文本在前）、AdaLayerNormZeroSingle 的 3 参数调制、attn/MLP 同源归一化输入与通道维拼接、gate 乘法 + 残差 + 按文本长度切片。全程只有张量操作，无任何标量账，所以验证的重点全在「接线的方向与顺序」上。

## 与官方代码的刻意差异

- **batch 维**：官方代码支持任意 batch；本题显式限定 B=1..3 并校验各张量 batch 一致。
- **fp16 clip**：官方 `FluxSingleTransformerBlock` 在 fp16 下做 `clip(-65504, 65504)`；本题只收 float32/float64，fp32 下该 clip 是 no-op，不出题（notes 记账，非法 dtype 走 ValueError）。
- **多 block**：官方跑 38 个 single block，全部共享同一个 `temb`（vec）。本题跑一个 block，但把三个调制参数作为账本返回，足以验证「共享同一 vec、每 block 独立调制」的语义；多 block 只是同一函数的重复应用，不产生新账。
- **controlnet residual**：官方在 single blocks 里把 controlnet 样本只加到 `hidden[:, encoder_len:]`（图像部分）；本题不含 controlnet，留作候选。
- **attention 内部**（RMSNorm qk、RoPE、`pre_only=True` 的无输出投影）不考，`attn_output` 由调用方给定，与 `mm-flux-joint-block-modulation`（order 649）的边界处理一致。

## 五个钉死的账点

1. **vec 是加法不是拼接**。官方 `conditioning = timesteps_emb + pooled_projections`——timestep 嵌入与池化文本投影做**逐元素加法**（两者都是 [B, D]），最容易凭直觉写成 `cat`。
2. **拼流顺序文本在前**。官方 `torch.cat([encoder_hidden_states, hidden_states], dim=1)`，且收尾切片 `hidden[:, encoder_len:]` 与之配对；顺序写反则 LayerNorm 的统计量、gate 的作用对象全错。
3. **single block 是 3 参数调制**。AdaLayerNormZeroSingle 的 linear 输出 `3*emb`，chunk(3) 得 shift_msa / scale_msa / gate_msa——区别于 double block（order 649）的 6 参数（shift/scale ×2 + gate ×2）。归一化形式是 `norm(x) * (1 + scale) + shift`。
4. **norm_x 同源喂 attention 与 MLP**。官方只做一次 `self.norm(hidden_states, emb=temb)`，`norm_hidden_states` 同时进 `self.attn` 与 `self.proj_mlp`——把调制后的归一化输入喂给 MLP 是与「MLP 直接吃原始 hidden」划界的关键。
5. **attn 与 mlp 通道维拼接后过 proj_out**。官方 `torch.cat([attn_output, mlp_hidden_states], dim=2)`（attn 在前），`proj_out` 的输入维度是 `dim + mlp_hidden_dim`，输出 `gate * proj_out(fused)` 再加残差。

## 手算坑（对拍定位）

- **GELU 的位置**：官方 `self.act_mlp(self.proj_mlp(norm_hidden_states))` 是 **gelu(linear(x))**——linear 在前、GELU 在后。oracle 初版写成 `linear(gelu(x))`，导致所有 block 输出对不上；与 solution 逐中间量对比后才定位。这正是「proj_mlp 的输出过激活」与「对归一化输入先激活再投影」两种直觉的分歧点，`gelu-before-linear` 变异专钉。
- **置零 ≠ 清零**：想把 gate 调成 0，只把 `single_mod_b` 的 gate 段设成大负数不够——权重项 `W·silu(vec)` 仍可能把它顶回去；必须把 W 与 b 的整段同时清零。`check_gate_residual` / `check_scale_one_plus` 按此构造。
- **切片非连续**：`big[:, :2]` 在第 0 维切片仍是连续张量；要构造非连续视图得先转置再切（`big.transpose(1,2)[:, :2]`）。

## 独立验证

oracle 不用任何 torch 线性代数：`_linear_row` 用标量双重循环做 matmul，`_silu` / `_gelu_tanh` / `_layer_norm_row`（有偏方差，eps=1e-6）逐元素手写，拼流用列表按 token 组装，最后转回 torch 张量与 solution 对拍（allclose 1e-5 + dtype 一致）。12 个 `check_*` 覆盖：vec 加法语义（含 `torch.equal(vec, te+pp)` 钉死）、拼流顺序（交换 enc/img 后结果必须不同）、3 参数 chunk 边界、MLP 同源输入（用 norm_x 顶替 attn_output 复算）、(1+scale) 退化、gate 归零退化为残差、按 Lt 切片、float64、非连续视图、输入不变性、九组非法输入。

变异 12 个可观察变异全部 CAUGHT（`vec-sub` 减法、`img-first-concat` 顺序、`no-one-plus-scale`、`mlp-raw-input`、`gelu-before-linear`、`mlp-first-fused`、`gate-broadcast-skip`、`no-residual`、`slice-from-zero`、`eps-1e-5`、`silu-to-gelu-mod`、`six-chunk`）。一个教训记入 notes：初版 `vec-concat` 变异写成「cat 后拆开再相加」，与加法数学等价，属无效变异，换成减法后立即被抓。

## 复杂度

设 L = Lt + Li、M = mlp_hidden_dim。单 block：调制 O(D²)，MLP O(L·D·M)，proj_out O(L·D·(D+M))，总计 O(L·D·(D+M))；除输出与三个 [B,D] 账本外无额外中间量保留。

## 来源与范围

依据已回源的 diffusers v0.32.1 三处源码（GitHub API contents + base64）：`transformer_flux.py::FluxSingleTransformerBlock.forward`（残差/调制/拼接/gate/clip）、`transformer_flux.py::FluxTransformer2DModel.forward`（`cat([encoder, hidden])` 拼流与 `hidden[:, encoder_hidden_states.shape[1]:]` 切回）、`normalization.py::AdaLayerNormZeroSingle`（3 参数 chunk 与 `(1+scale)` 形式）、`embeddings.py::CombinedTimestepTextProjEmbeddings`（`conditioning = timesteps_emb + pooled_projections`）。vec 的上游（Timesteps 正弦编码 ×1000、PixArtAlphaTextProjection 两层投影）不考——它们是纯 DSP/MLP，账点在本题的装配语义上。与 `mm-flux-joint-block-modulation`（order 649，double block 六参数）划界：本题是 single stream 的装配与 3 参数共享账。

```text
.venv-eval/bin/python scripts/add_problem.py --selftest mm-flux-single-stream-assembly --mode submit
```
