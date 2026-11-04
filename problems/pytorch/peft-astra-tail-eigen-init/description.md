# PEFT：Astra 激活空间尾特征向量初始化

## 题目描述

**Astra**（arXiv:2602.19111，已合入 HF PEFT PR #3725，`init_lora_weights="astra"`）用**输出激活的协方差**决定 LoRA 初始化方向：校准前向收集每层输出的协方差矩阵，做一次**特征分解**（`eigh`，升序），把基座权重投影到特征向量子空间——可训练适配器承载 \(V\) 子空间分量，冻结基座吸收残差。初始化处模型输出与原模型**逐位一致**（零增量），训练只作用于任务相关子空间。官方 README 特别强调：一次特征分解、**无权重 SVD、无协方差逆**（与 CorDA 划界）。

本题复刻官方实现（`tuners/lora/astra.py` + `layer.py` 的 `astra_init`）的确定性账：

- **协方差收集**（hook 账）：每次前向把激活 flatten 成 \((N, dim)\)、转 float，用**该次调用的全局 \(|max|\)** 归一（`torch.max(out).abs()`——先取**带符号** max 再 abs，负激活占主导时与 `abs().max()` 不同！），\(cov \mathrel{+}= Y^\top Y\) 计数，最后除以样本数。
- **特征分解**：`S, V = eigh(cov.float())`——**升序**（\(S[0]\) 最小）；\(S\) 全长保留，\(V\) 只裁 `[:, -r:]`（eigh 升序下即 top-\(r\) 特征值方向）。
- **astra_init**：\(A = V^\top W\)（\(r, in\)）、\(B = V\)（\(out, r\)）；\(\Delta = scaling\cdot B@A\)，\(W \leftarrow W-\Delta\)。\(B@A = V(V^\top W)\) 恰是 \(W\) 在 \(V\) 子空间上的投影。
- **校验**：dtype ∈ {float32, float16, bfloat16}（否则 TypeError）；\(r \le out\_features\)、\(V\) 形状 \((out\_dim, r)\)、NaN/Inf（均 ValueError）。

## 任务

- `mode="cov_accumulate"`：batches（激活张量列表）入参，返回 (cov, sample_count)；NaN/Inf 抛 ValueError。
- `mode="eigens"`：cov、r、out_dim 入参，返回 (S 全长, V 裁剪)；r 非法抛 ValueError。
- `mode="init"`：weight、v_r、scaling 入参，返回 (lora_A, lora_B, W_new)；dtype/形状/NaN 校验如上。
- `mode="forward"`：x、lora_A、lora_B、W_new、scaling 入参，返回 \(scaling\cdot x\,A^\top B^\top + x\,W_{new}^\top\)。
- 协方差管线按官方 float32（`.float()`）。

## 示例

\(out\_dim=6, in\_dim=5, r=2\)，W = randn(6,5)、X = randn(32,6)（seed 2，float32 管线）：

```
cov_accumulate([X]) -> cov (6,6), sample_count = 1
eigens(cov, 2, out_dim=6) -> S 全长 (6,)，S[0]=1.806457757949829、S[-1]=8.876955032348633
                           V_r (6,2)，V_r[0,0]=0.7613240480422974
init(W, V_r, scaling=2) -> A[0,0]=-0.8271405696868896、W_new[0,0]=0.5683993101119995
                           W_new + 2·B@A == W（float32 域）
forward(x, A, B, W_new, scaling=3) == x @ W^T（零增量）
```

## 约束

- `1 <= r <= out_features`；r 非法抛 ValueError。
- 归一化除数是 `torch.max(out).abs()`（带符号 max 的绝对值），不是 `out.abs().max()`。
- 协方差返回 float32；A/B/W_new 返回 weight 的原 dtype。
