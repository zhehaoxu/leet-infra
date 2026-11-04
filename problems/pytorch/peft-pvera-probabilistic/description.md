# PEFT：PVeRA 概率向量随机矩阵自适应

## 题目描述

**PVeRA**（Probabilistic Vector-based Random Matrix Adaptation, arXiv:2512.07703，已入 HF PEFT v0.20.0 `tuners/pvera`）是 VeRA 的概率版：保留「跨层共享冻结随机投影 \(A,B\) + 每层只训两个小向量」的极限参数预算，但把每层的潜在增量从定值升级为**分布**——前向走重参数化技巧采样潜在表示，训练学出 \(\mu/\log\sigma^2\)，推理可选采样出蒙特卡洛置信区间。

官方 `layer.py` 的确定性账（锚点全部可离线重算）：

```
pre = lambda_d * (x @ A_s.T)                # (…, 2r)：lambda_d 前半缩放 μ 通路、后半 logvar 通路
mu, logvar = pre.chunk(2, dim=-1)           # 各 (…, r)
z = mu  （eval）|  mu + eps·exp(½·logvar)  （sampling）
delta = lambda_b * (z @ B_s.T)              # 无 scaling 项！
```

三个关键口径：

1. **\(A\) 是双份银行**：共享 `pvera_A` 形状 \((2r, in_{max})\)（kaiming_uniform 默认 \(a=0\)、CPU generator `projection_prng_key` 播种），\(B\) 形状 \((out_{max}, r)\)。每层切片 `A[:, :in]`、`B[:out]`。μ 与 logvar 共享同一个线性映射的**前后半段**（\(A\) 上半产生 μ、下半产生 logvar），不是两个独立线性层。
2. **merge 只取 μ 半段**：`get_delta_weight` 用 `lambda_d[:r]` 与 `A[:r]`——合并语义是「期望增量」：\(\Delta W = (\lambda_b \cdot B_s)\,(\lambda_{d_\mu} \cdot A_\mu)\)。logvar 半段在 merge 与 eval 前向中都被忽略。
3. **初始化怪癖**：`reset_pvera_parameters` 把 `lambda_d` 全填 `d_initial`（默认 0.1）、`lambda_b` **清零**——`update_layer` 里虽以 `torch.ones` 创建，但 init_weights=True 时被 reset 的 `zeros_` 覆盖，训练起点 \(\Delta W=0\)（与 VeRA 同零起点语义）；init_weights=False 跳过 reset，lambda_b 保持全 1、lambda_d 保持 randn。

## 规则

- `mode="shared_projections"`：`r, in_max, out_max, prng_key` 入参，返回 float64 化的 \(A,B\) 与 float32 原件（官方 RNG 账：kaiming_uniform（\(a=0\)）逐位复现）。
- `mode="slice_pair"`：从共享银行切 `A_s (2r, in)`、`B_s (out, r)`。
- `mode="forward_delta"`：`eps=None` 走 eval（z=μ，确定性）；`eps` 给定走采样（z = μ + eps·exp(½logvar)）。返回 adapter 增量（不含 base）。
- `mode="merge_delta"`：μ 半段合并 \(\Delta W\ (out,in)\)。
- `mode="layer_setup"`：返回 `{"lambda_d": (2r,) 全 d_initial, "lambda_b": (out,) 全零, "lambda_b_uninit": (out,) 全一}`（reset 后账 + init_weights=False 分支的 λ_b）。
- 全程 torch float64（\(A/B\) 原件 float32 锚定后转 float64，对应层内 `.to(dtype)` 语义）。

## 示例

\(r=4, in=8, out=6\)（共享银行 \(A(8,10), B(12,4)\)）：

```
forward_delta(x, lambda_d, lambda_b, A_s, B_s)          # eval: (2,5,6)
forward_delta(x, lambda_d, lambda_b, A_s, B_s, eps)     # sample: z = mu + eps·exp(½logvar)
merge_delta(lambda_d, lambda_b, A_s, B_s) -> (6,8)      # 只用 lambda_d[:4] 与 A_s[:4]
```

## 约束

- eval 前向与 merge 自洽：`forward_delta(x, …) == x @ dW.T`（base 外恒等）。
- 改 logvar 半段（`lambda_d[r:]`）不影响 eval 前向与 merge——可观测分支只在采样路径。
- μ 半段全零 ⇒ \(\Delta W \equiv 0\)（零起点语义）。
- 采样路径的 eps 显式入参（判题不跑随机数）。
