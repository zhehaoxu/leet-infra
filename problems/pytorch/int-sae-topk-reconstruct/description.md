# SAE top-k 编码账：稀疏激活、重构与 L0 稀疏度

## 题目描述

稀疏自编码器（SAE）把模型隐态分解为高维稀疏特征：编码端取 pre-activation 最大的 k 个特征、其余清零，解码端用这些稀疏激活重构隐态。TopK SAE（SAELens 架构）与普通 ReLU SAE 的分界是**不做 ReLU**——负 pre-act 照样入选。本题要求把 encode/decode 账与稀疏度指标逐项算清。

## 规则

实现 `solve(x, W_enc, b_enc, W_dec, b_dec, k)`，返回：

| 字段 | 形状 | 含义 |
|---|---|---|
| `feature_acts` | `[B, F]` | top-k 稀疏激活（其余位置严格 0，负值保留） |
| `recon` | `[B, D_in]` | `acts @ W_dec + b_dec` |
| `recon_error` | `[B, D_in]` | `x − recon` |
| `l0` | `[B]` | 非零激活计数 |
| `mse` | `[B]` | `recon_error` 逐行均方（对 D_in 求均值） |
| `max_feat` | `[B]` | 绝对值最大的激活 index（并列取较小） |

步骤（float 化）：

```
pre_acts = (x − b_dec) @ W_enc.T + b_enc
每行取值最大的 k 个（并列取较小 index），其余清零
recon = acts @ W_dec + b_dec
```

`k` 不满足 `1 ≤ k ≤ F` 时抛 `ValueError`。

## 示例

```
x=[1,2]，W_enc 三行 e0/e1/(e0+e1)/2，b_enc=[0,0,1]，W_dec 两行 e0/e1 + 零行，b_dec=0，k=2
pre = [1, 2, 2.5] → 保留 feat2(2.5)、feat1(2.0)
recon = [0, 2]，mse = (1² + 0²)/2 = 0.5，max_feat = 2
```

## 约束

- `F ≤ 16`、`D_in ≤ 8` 小规模，oracle 纯 Python 双循环
- `k = F` 时激活 = 完整 pre_acts（无清零）
