# 多模态：SigLIP Sigmoid 对比损失

## 题目描述

SigLIP（Zhai et al., *Sigmoid Loss for Language-Image Pre-Training*, arXiv:2303.15343）把 CLIP 的 softmax 归一化换成**逐对 sigmoid**：每个 (图像 \(i\), 文本 \(j\)) 对独立地被判为「正/负」，无需跨 GPU 收集整批做全局归一化，小 batch 也稳、且易扩到超大 batch。

对相似度矩阵 \(s_{ij}=x_i^\top y_j\)，先做仿射 \(z_{ij}=t\,s_{ij}+b\)（\(t=\) `logit_scale`、\(b=\) `logit_bias`），标签 \(y_{ij}=+1\)（对角正对）否则 \(-1\)：

\[ \mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{B}\log\sigma\bigl(y_{ij}\,z_{ij}\bigr) \]

注意分母是 batch 大小 \(B\)（对全部 \(B^2\) 个 pair 求和后除以 \(B\)），不是 \(B^2\)。

## 规则

实现 `solve(sim, logit_scale, logit_bias)`。输入 `sim`（float32，方阵 \((B,B)\)，已是图像×文本点积）、标量 `logit_scale`、`logit_bias`。按上式返回 0 维 float32 损失标量。`logsigmoid` 需数值稳定。禁止调对比学习库。

## 示例

`sim=[[1.0,0.2],[0.3,0.9]]`、`logit_scale=5.0`、`logit_bias=-1.0`：

```
z = 5*sim - 1 = [[4.0, 0.0],[0.5, 3.5]]
y = [[+1,-1],[-1,+1]]
loss = -(1/2)*Σ log σ(y⊙z) = 0.857562
```

## 约束

- CPU float32；`sim` 为 \(B\times B\) 方阵，对角为正样本。
- 与 CLIP 的关键差别：逐对 sigmoid、无 softmax 全局归一，分母取 \(B\)。
