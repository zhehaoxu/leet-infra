# 对齐训练：KTO 前景理论损失

## 题目描述

**KTO**（Kahneman-Tversky Optimization, Ethayarajh et al., arXiv:2402.01306）借鉴前景理论：人把结果相对一个**参考点** \(z_0\) 评成「得/失」，且对损失更敏感。它只需二元好坏标签（不必成对偏好）。

给定每条样本的**隐式对数比** \(z = \log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\)（序列级）与标签（desirable / undesirable），价值函数以 sigmoid 形式相对参考点 \(z_\text{ref}\) 度量：

\[ v = \begin{cases} \sigma\bigl(\beta(z - z_\text{ref})\bigr), & y\ \text{desirable}\\[4pt] \sigma\bigl(\beta(z_\text{ref} - z)\bigr), & y\ \text{undesirable}\end{cases} \]

最小化「未实现的价值」，并对好/坏样本各加权 \(\lambda_D,\lambda_U\)：

\[ \mathcal{L}_i = \begin{cases} \lambda_D\,\bigl(1 - \sigma(\beta(z - z_\text{ref}))\bigr), & \text{desirable}\\ \lambda_U\,\bigl(1 - \sigma(\beta(z_\text{ref} - z))\bigr), & \text{undesirable}\end{cases},\qquad \mathcal{L}=\frac1N\sum_i \mathcal{L}_i \]

参考点 \(z_\text{ref}\)（论文里由 KL 估计、detach 为常量）本题作显式入参给定。

## 规则

输入 `log_ratios`（\((N,)\)，即 \(z\)）、`labels`（\((N,)\)，1=desirable、0=undesirable）、标量 `z_ref`、`beta`、`lambda_D`、`lambda_U`。返回 0 维 float32 标量＝平均 KTO 损失。禁止现成对齐库。

## 示例

`log_ratios=[0.5,-0.3,0.2,-0.8]`、`labels=[1,0,1,0]`、`z_ref=0.1`、`beta=0.1`、`lambda_D=lambda_U=1.0`：

```
i0 (好): 1-σ(0.1*(0.5-0.1))=1-σ(0.04)=0.490000
i1 (坏): 1-σ(0.1*(0.1-(-0.3)))=1-σ(0.04)=0.490000
i2 (好): 1-σ(0.1*(0.2-0.1))=1-σ(0.01)=0.497500
i3 (坏): 1-σ(0.1*(0.1-(-0.8)))=1-σ(0.09)=0.477502
loss = mean = 0.488754
```

## 约束

- CPU float32；\(\beta,\lambda_D,\lambda_U>0\)；`labels` 取 0/1。
- desirable 用 \(\sigma(\beta(z-z_\text{ref}))\)、undesirable 用 \(\sigma(\beta(z_\text{ref}-z))\)，方向相反（损失厌恶的体现）。
- \(z=z_\text{ref}\) 时价值=\(\sigma(0)=0.5\)、逐样本损失 \(=\lambda/2\)。
