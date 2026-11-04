# PEFT：VeLoRA 秩 1 子令牌激活压缩

## 题目描述

**VeLoRA**（NeurIPS 2024，arXiv:2405.17991，官方 `roymiles/VeLoRA`，`velora/velora.py`）把反向传播必须驻留内存的**中间激活**压缩到一个**秩 1 子空间**：前向把每个 token 沿特征维切成 `num_groups` 个子令牌，各自投影到一条**固定的单位向量**（rank=1 embed）上，每个子令牌只存一个标量 `zp`；反向用同一条向量把激活**粗粒度重建**出来算梯度。投影向量由第一批统计量一次性初始化后冻结（`batch_average_once`）。关键不对称：**只有 lora_A 的梯度流经重建激活**，lora_B 的梯度用真实 lora_A 输出。

本题复刻官方实现的确定性账：

- **子令牌投影**：x (B,N,C) → reshape (B·N·G, C/G) → `zp = einsum('h s, r s -> h', flat, embed)`；embed 是 L2 归一化的单位行向量。
- **batch_average_once 初始化**：`embed = normalize(data.mean(0), p=2, dim=-1)`——首批均值、L2 归一、随后冻结。
- **反向重建**：`input_hat = einsum('h, r s -> h s', zp, v).view(input_shape)`——秩 1 粗重建，有损。
- **梯度账**：`grad_input = grad_output @ weight`；`grad_weight = velora_scale · Σ_b grad_output[b]ᵀ @ input_hat[b]`（官方 backward 返回 per-sample (B,M,D) 项，前导 batch 维由 autograd 求和，等效账即 sum(0)）；`grad_bias = grad_output.sum(1)`（沿序列维，有 bias 时才返回）。
- **LoRA 块**：`velora_scale` 只乘 grad_a；grad_b 用**真实** `a_out = x @ lora_a.T` 乘 lora `scaling`；前向就是普通 LoRA 路径 `base + (a_out @ lora_b.T)·scaling`；`grad_input = grad_out_m @ weight + scaling·((grad_out_m @ lora_b) @ lora_a)`。

## 任务

- `mode="project"`：x、embed、num_groups 入参，返回 zp (B·N·G,)。
- `mode="init_embed"`：data (M,S) 入参，返回单位向量 embed (S,)。
- `mode="reconstruct"`：zp、v (1,S)、shape 入参，返回 shape 形状的重建激活。
- `mode="grads"`：input_hat、weight、grad_output、velora_scale、use_bias 入参，返回 dict(grad_input, grad_weight, grad_bias)；无 bias 时 grad_bias=None。
- `mode="lora_block"`：x、weight、bias、lora_a、lora_b、scaling、num_groups、embed、velora_scale、grad_out_m、grad_out_r 入参，返回 dict(result, grad_a, grad_b, grad_input)。

## 示例

seed 0/2/3/4，float64：

```
project(x (2,3,8), embed (1,2), num_groups=4) -> zp (24,)
  zp[0] = -2.3036448720441434, zp[-1] = -1.3527679700105146
init_embed(data (7,5)) -> embed (5,), embed[0] = -0.7252327797613423, norm = 1
reconstruct(zp, v, (2,4,8)) -> hat (2,4,8), hat[0,0,0] = -0.6753214360032626
grads(input_hat (2,5,6), weight (4,6), grad_output (2,5,4), velora_scale=0.1)
  -> grad_weight[0,0] = 0.22305672787004252, grad_input[0,0,0] = -0.0783169401206513
lora_block(...) -> grad_a[0,0] = -0.3717755681072157, grad_b[0,0] = 1.490616682369623
```

## 约束

- embed 是 L2 归一化的单位行向量；官方 `batch_average_once` 仅支持 rank=1。
- reshape 行序是 (B, N, G)——子令牌组序错了 zp 全变；C 整除 num_groups。
- `velora_scale` 只作用于 grad_weight（lora 块中的 grad_a），不影响 grad_input 与前向输出。
- 重建有损：`hat ≠ x`，但残差在每个子令牌块内与 embed 正交（round trip：`project(reconstruct(zp)) == zp`）。
