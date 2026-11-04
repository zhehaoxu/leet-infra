# VeLoRA 实现笔记

## 出处与结构

- 官方仓 `roymiles/VeLoRA`（Apache-2.0），核心 `velora/velora.py`；NeurIPS 2024，arXiv:2405.17991（华为诺亚）。
- `VeLoRAWrapper.forward`：x 转 fp16 → `init_embed_`（首批门控）→ `zp = einsum('h s, r s -> h', x, embed)`；rank=1 时输出 (H,)。
- `VeLoRA(torch.autograd.Function)`：forward 存 (zp, weight, v)；backward 重建 `input_hat = einsum('h, r s -> h s', zp, v).view(ctx.input_shape)`，`grad_input = grad_output @ weight`、`grad_weight = transpose(1,2) @ input_hat`、`grad_bias = grad_output.sum(1)`，返回的 grad_weight 乘 `ctx.scale`。
- `functional_velora_linear_fn`：zp 投影的是 **dropout 之后的 lora 输入**（判题域 dropout 恒等即 x 本身）；`VeLoRA.apply(lora_result, lora_A, None, zp, embed, velora_scale)` 把 scale 送进 backward 只乘 lora_A 的梯度；B 侧走普通 autograd。

## 判题口径与怪癖

- 官方 backward 返回 per-sample (B,M,D)，**不求和**——真实 autograd 消费时对前导维自动求和；判题取等效账 `sum(0)`（文档与测试都写明）。
- `init_embed_` 的 `batch_average_once` 有 `assert rank == 1`；mean(0) 后 `F.normalize(dim=-1)`；initted 门控保证只算一次。
- 重建有损但残差与 embed 正交：hat·v = zp·|v|² = zp = x·v ⇒ (hat−x)·v = 0；round trip `project(reconstruct(zp)) == zp`。
- **不对称性是官方语义**：grad_a 走重建激活、grad_b 走真实 `a_out`；把 grad_b 也换成重建激活是可抓变异（m-lb-ares）。
- `velora_scale` 只乘 lora_A 的梯度；前向 result 与 grad_input、grad_b 都不含它。
- einsum 与 mul-sum 累加序不同（float 上 allclose 而非 equal）；锚值全部 float64、容差 1e-9。
- reshape 行序 (B, N, G)：子令牌组序变异（transpose(1,2)）会让 zp 全变——m-pr-order。
- grad_bias 沿**序列维** dim=1 求和（B,L,M → B,M）；沿 batch（sum(0)）是可抓变异。
