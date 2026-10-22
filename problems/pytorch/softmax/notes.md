# Softmax 详解

## 它到底在算什么

Softmax 把一整行实数 $z$ 映射成概率分布，保持顺序且和为 1：

$$ \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}} $$

它是「指数族 + 归一化」的组合，是交叉熵、注意力权重、温度采样的公共内核。理解 softmax 的关键不在公式，而在它为什么**必须减最大值**。

## 数值稳定性：减 max 不是优化，是必需

`float32` 的 $e^{x}$ 在 $x \gtrsim 88$ 时上溢成 `inf`，$x \lesssim -104$ 时下溢成 0。若直接对 `exp` 求和：

- 大 logits（如分类网络最后一层、注意力 $QK^\top$）→ `inf/inf = NaN`；
- 全负 logits → 分母趋 0。

利用平移不变性：$\text{softmax}(z) = \text{softmax}(z - c)$，取 $c = \max_i z_i$，则指数的参数都在 $(-\infty, 0]$，$e^{\cdot} \in (0,1]$，永不上溢。分母至少含 $\max$ 那一项 $=1$，不会为 0。

> 减 max 后即便仍有极负项下溢成 0 也无妨：它们本就对应概率 ~0，不影响和与归一化精度。

## 与温度、Top-k/Top-p 的关系

温度缩放即先除以 $T$ 再 softmax：$\text{softmax}(z/T)$。$T\to 0$ 逼近 one-hot(argmax)，$T\to\infty$ 逼近均匀分布。本合集的「温度缩放与 Top-p」题就是这条链路的延伸。

## 梯度（供理解，不要求实现）

设 $p = \text{softmax}(z)$，则雅可比为

$$ \frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j) $$

与交叉熵组合时梯度极干净：$\partial \mathcal{L}/\partial z = p - y$（预测减 one-hot 标签），这正是所有分类头最后一层反传的形态。

## 常见坑

1. **沿错维度**：按行归一用 `dim=-1`；批注意力里是沿 key 轴，别误用 `dim=0`。
2. **整行相同值**：$z=[a,a,a]$ 无论 $a$ 多大，减 max 后应得均匀 $[1/3,\dots]$；未减 max 则可能 `inf/inf`。
3. **全 -inf 行**（被 mask 死）：减 max 得 `NaN`，需单独处理（torch 注意力里约定输出 0），本关不涉及但联动题要注意。
