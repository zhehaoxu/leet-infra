# 大模型组件：SwiGLU MLP

## 题目描述

Transformer 每层里做非线性变换的前馈网络（FFN / MLP）。现代大模型（LLaMA / Qwen / Mistral / DeepSeek）不用经典「两线性层 + ReLU」，而用 **SwiGLU**——一个带门控的变体：

$$ \text{FFN}(x) = W_{down}\,\big(\ \text{SiLU}(W_{gate}\,x)\ \odot\ (W_{up}\,x)\ \big) $$

其中 $\text{SiLU}(z) = z\cdot\sigma(z)$（即 swish），$\odot$ 是逐元素乘。三条投影都**不带 bias**。

## 规则

输入 $x$（形状 $(\dots, \text{hidden})$）与三个线性权重（均无 bias，采用 `nn.Linear.weight` 的 $[\text{out},\text{in}]$ 布局）：`gate_weight`、`up_weight` 形状 $(\text{inter}, \text{hidden})$，`down_weight` 形状 $(\text{hidden}, \text{inter})$。返回形状 $(\dots, \text{hidden})$：

1. 门路 $g = x\,W_{gate}^\top$，$u = x\,W_{up}^\top$（各 $(\dots,\text{inter})$）。
2. $\text{SiLU}(g) \odot u$。
3. 输出 $= (\cdot)\,W_{down}^\top$。

> 判题用 `transformers` 的 `Qwen2MLP`（同权重）作 oracle 对拍，`hidden_act="silu"`。

## 示例

```
x = [1, 2]，gate_weight = up_weight = down_weight = I(2×2 单位阵)
g = [1, 2]，SiLU(g) = [1·σ(1), 2·σ(2)] = [0.7310586, 1.7615941]
SiLU(g) ⊙ u = [0.7310586·1, 1.7615941·2] = [0.7310586, 3.5231881]
输出 = [0.731059, 3.523188]
```

## 约束

- $\text{inter} \ge \text{hidden}$（真实模型约 $8/3\cdot\text{hidden}$，本判题用小尺寸即可）
- 权重无 bias；线性用 $W$ 的 $[\text{out},\text{in}]$ 布局（等价 `F.linear(x, W)`）
- dtype float32，`torch.allclose` 相对容差 $10^{-5}$ 对拍 `transformers`；不修改输入
