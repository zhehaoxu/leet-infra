# MonteCLoRA：用贝叶斯重参数化降低低秩估计方差

LoRA 对超参敏感、微调输出方差大。MonteCLoRA（arXiv:2411.04358，官方 LCS2-IIITD/MonteCLoRA）把低秩因子参数化为分布：每步从后验采样、以 Monte Carlo 平均前向，用重参数化技巧让梯度无偏流动，仅 O(m+r) 额外参数就能显著降低估计方差。

## 重参数化链（官方 `MonteCLoRASampler`）

以 `std_prior: (m, r)`（log 尺度）与三个外部随机源 `z_wishart: (m,r,r)`、`z_mvn: (m,n,in,r)`、`z_dirichlet: (m,)` 记：

```python
std = diag_embed(exp(std_prior))            # (m, r, r)
gaussian_var = diag(clip(std @ Z_w @ std.T))  # 对角 Wishart 收缩，clip 下限 1e-6
L = cholesky(gaussian_var)
var = scaler * mean_n(einsum('mnir,mrk->mnik', z_mvn, L))   # (m, in, r)
mu  = log(alpha) - mean(log(alpha))          # alpha = exp(prior)
sigma = diag((1 - 2/M)/alpha + (1/M^2)·sum(1/alpha))
d   = cholesky(sigma) @ z_dirichlet + mu     # Dirichlet 的 log-normal 近似
w   = sigmoid(d); w = w / w.sum()            # 专家权重：sigmoid 后归一，不是 softmax
```

组合回 LoRA 因子（官方 `layer.py`）：

```python
updated = nan_to_num(A.T + scaler * var)    # 逐专家扰动
A_new   = einsum('i,ijk->jk', w, updated)   # 专家加权平均
# 之后仍是标准 LoRA 前向：B(A_new(dropout(x))) * scaling
```

## 变分正则（get_variational_loss）

`kl_loss_weight·(kl_dir + kl_wishart + kl_mv)/(m·n)`，外加一个"熵"项——**官方的熵是权重平方和，不是香农熵**，这是个容易踩的坑：

- `multivariate_kl = n·0.5·(trace(var) − logdet(var) − d)` 对 m 取均值；
- `wishart_kl = 0.5·(−log_trace·d + var_trace·d − d²)`，只用 `std@stdᵀ` 的对角；
- `dirichlet_kl` 是相对对称先验 `dirichlet_prior` 的标准 gamma/digamma 公式。

## 判题域

采样分布本身（Wishart Bartlett 分解、异步预生成线程）不在域内——三个 z 全部由调用方给定，整条重参数化链变为确定性纯函数：std→sigma、Wishart 收缩、Cholesky 噪声收缩、Dirichlet 近似、专家权重归一、因子组合与三项 KL。
