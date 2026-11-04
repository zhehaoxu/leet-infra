# 详解：Astra 激活空间尾特征向量初始化

## 出处

- 论文：**Astra: Activation-Space Tail-Eigenvector Low-Rank Adaptation of Large Language Models**（arXiv:2602.19111，ACL 2026 Findings，Ping An Technology）。
- 官方实现：已合入 **HF PEFT**（PR #3725，commit 73f9a1a，`init_lora_weights="astra"`；`tuners/lora/astra.py` + `layer.py` 的 `astra_init`；官方仓 LyoAI/Astra 是入口脚本）。PEFT 注释明说 "This implementation builds on `corda.py`"。
- 关键差异（vs CorDA 一系）：CorDA 用**输入**协方差乘 W 定向 SVD + \(C^{-1}\) 乘回；Astra 用**输出**协方差 `eigh`（无逆、无 damp），README 强调 "no weight SVD or covariance inverse"。

## 官方实现的逐行翻译

**协方差收集**（astra.py 的 hook，逐行）：

```python
output = output[0] if isinstance(output, (tuple, list)) else output
output = output.detach().reshape(-1, output.size(-1)).data   # (num_tokens, dim)
output = output.float()                       # 除非 use_float16_for_covariance
output = output / torch.max(output).abs()     # 带符号 max 的绝对值！
if torch.isnan(output).any() or torch.isinf(output).any():
    raise ValueError("Invalid value found in output...")
covariance = output.t().matmul(output)
module.sample_count += 1
module.covariance_matrix += covariance
# 结束后
module.covariance_matrix /= module.sample_count
```

```python
# torch.max(out) 是带符号最大值（2.83），再 .abs() —— 与 out.abs().max()（3.15）不同；
# 负激活占主导时除数会异常小（放大协方差）——官方行为如此，判题以代码为准。
```

**特征分解**（collect_eigens_for_layer，逐行）：

```python
covariance_matrix = linear.covariance_matrix.float()
S, V = torch.linalg.eigh(covariance_matrix)   # eigh 升序：S[0] 最小
# 形状校验 S (dim,), V (dim, dim)
module.eigens = AstraEigens(S=S.cpu(), V=V.cpu())
```

**crop**（crop_astra_eigens，逐行）：

```python
if module.rank > weight_out_dim:
    raise ValueError(f"Astra requires `r` <= out_features ...")
module.eigens.S = module.eigens.S.clone()               # S 全长保留
module.eigens.V = module.eigens.V[:, -module.rank:].clone().to(device)
```

**astra_init**（layer.py，逐行）：

```python
if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
    raise TypeError("Please initialize Astra under float32, float16, or bfloat16.")
weight = weight.to(torch.float32)
# r > out_dim / V (out_dim, r) / nan-inf 校验
lora_A = (V.t() @ linear_weight).contiguous()     # (r, in)
lora_B = V.contiguous()                            # (out, r)
delta = self.scaling[adapter_name] * lora_B @ lora_A
weight = weight.data - delta                       # 残差化（零增量）
```

## 判题域剔除与红线

- **剔除**：模型级 preprocess 遍历/cache_file/covariance_file、Conv1D 分支、hook 注册机制（激活矩阵显式入参）、use_float16_for_covariance、CPU offload。
- **保留**：hook 账（flatten/float/归一/累加/计数）、eigh + crop（S 全长、V 尾 r 列）、astra_init（A/B/delta/残差化）、dtype 门与全部校验。
- **锚定**：S/V_r/A/W_new fixtures 锚定（float32 管线）；零增量恒等式。

## 关键怪癖（回源逐行确认）

- **归一化除数是带符号 max 的 abs**：`torch.max(out).abs()` ≠ `out.abs().max()`（沙箱实测 2.83 vs 3.15）——m-ca-absmax 可抓变异；全负 batch 时差异最大（除数变小、协方差放大）。
- **eigh 升序 vs SVD 降序**：`V[:, -r:]` 在 eigh 下是 **top-r 特征值方向**（论文 "tail" 措辞与代码方向相反——判题以代码为准）；若误用 SVD 降序再取尾部就变成最小方向（m-ae-head/m-ae-eigh 可抓）。
- **S 全长保留、V 只裁 r 列**：crop 只切 V（m-ae-cropS 可抓）。
- **delta 含 scaling**（lora_alpha/r 业务因子）：m-ai-scaling 可抓。
- **B@A = V(VᵀW) 恰是投影**：零增量恒等式的数学根源。
- **dtype 门**：float64 权重直接 TypeError（判题域默认 float64，测试显式构造 float32）。

## 变异测试（18/18，0 等价剔除）

- 18 个变异覆盖：归一化除数（absmax/减均值）、flatten、累加、样本除法、float32 cast（dtype 契约断言抓）、eigh→svd、头尾裁剪、S 裁剪、r 校验×2、dtype 门、A 的转置、B 的转置、scaling、残差加减号、delta 乘序、V nan 检查。

## 与已有题的划界

- **CorDA（810）**：输入协方差 + W@fix SVD + \(C^{-1}\) 乘回 + damp 循环 + KPA/IPA 双模式；Astra：输出协方差 + eigh + 无逆 + 单一 top-r 子空间。
- **LoRA-GA（809）/LoRA-SB（808）**：SVD 源是梯度/符号梯度；Astra 源是输出激活的二阶统计。
- **EVA（807）**：解释方差驱动**秩预算分配**（跨层）；Astra 是单层初始化，无分配账。
- 题库无「输出激活协方差/eigh」账（grep 已查重）。
