# AWQ：激活感知逐通道缩放

## 题目描述

**AWQ**（Activation-aware Weight Quantization，Lin et al., 2024, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)）用于 W4A16 之类的**权重量化**：权重里并非所有通道同等重要——**连接到大激活通道的权重**对输出影响更大，量化误差也更伤。AWQ 给每个输入通道配一个缩放因子 $s_j$（正比于该通道激活幅值的 $\alpha$ 次幂），量化前把重要通道的权重"放大"、从而在固定比特下分到更多量化级、误差更小；反量化时再除回 $s_j$。$\alpha$ 通过小网格搜索选最优。

## 规则

给定权重 $W$ 形状 $(O, C)$、每输入通道的平均激活幅值 `x_bar` 形状 $(C,)$、候选指数网格 `alphas`、量化位宽 `bits`。对某个 $\alpha$：

1. 原始重要性 $r_j = \text{x\_bar}_j^{\alpha}$；**归一化** $s_j = r_j / \operatorname{mean}_k r_k$（令缩放整体不改变平均量级）。
2. 放大后量化：$\tilde{W} = W \odot s$（按列乘），对 $\tilde{W}$ 做 **per-tensor 对称 RTN**（$q_{max}=2^{b-1}-1$、scale $=\max|\tilde{W}|$，就近取整再反量化）得 $\widetilde{W}_q$。
3. 缩回：$\hat{W} = \widetilde{W}_q / s$（按列除）。
4. **输出误差（对角近似）**：$\mathcal{L}(\alpha)=\sum_{o,j} \text{x\_bar}_j^2\,(W_{oj}-\hat W_{oj})^2$。

在 `alphas` 上取使 $\mathcal{L}$ 最小的 $\alpha^\*$（并列取网格中**先出现**者）。返回 `(best_alpha, W_hat)`，`W_hat` 形状 $(O,C)$。

## 示例

```
W = [[2.0, 1.0, 0.10, 3.0],
     [1.0, 2.0, 0.12, 2.0],
     [0.5, 1.0, 0.08, 1.0]]
x_bar = [0.2, 0.3, 9.0, 0.1]   # 通道2激活巨大但权重很小 → 需保护
alphas = [0.0, 0.25, 0.5, 0.75, 1.0],  bits = 4

best_alpha = 1.0
s = x_bar / mean(x_bar) = [0.0833, 0.1250, 3.7500, 0.0417]
W_hat ≈ [[2.3143, 1.0286, 0.1029, 3.0857],
         [0.7714, 2.0571, 0.1200, 1.5429],
         [0.7714, 1.0286, 0.0857, 1.5429]]
```

（通道 2 被 $\times 3.75$ 放大后量化，反缩放后其权重量化误差显著低于 $\alpha=0$。）

## 约束

- CPU float32；`best_alpha` 用 `float`，`W_hat` 用 `allclose(atol=1e-4)`。
- per-tensor RTN 的取整用**就近、`.5` 取偶**（与 `torch.round` 一致）。
- `x_bar` 各元素 $>0$；归一化分母用 $r$ 的算术平均。
