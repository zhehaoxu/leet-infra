# Dropout（train / eval）

## 题目描述

Inverted Dropout：训练时以概率 $p$ 随机把元素置 0，并把**保留**的元素除以 $(1-p)$ 放大，使输出的期望等于输入。推理时不做任何变化（恒等）。

$$ y_i = \frac{m_i}{1-p}\,x_i,\qquad m_i \sim \text{Bernoulli}(1-p) $$

## 规则

- 实现 `solve(x, p=0.5, training=True, seed=1234)`。
- `training=False` 或 `p <= 0` 时返回**恒等**（原值，包括 `p=0.9` 的 eval 也必须恒等）。
- 训练时输出只能是 $0$ 或 $x/(1-p)$（保留值精确放大 $1/(1-p)$，期望保持不变）。
- **必须严格遵循以下随机数约定**（否则与参考实现对拍不上）：
  1. 用局部生成器：`g = torch.Generator(device=x.device).manual_seed(seed)`；
  2. 掩码：`mask = torch.rand(x.shape, generator=g, device=x.device, dtype=x.dtype) >= p`（保留 `>= p` 的位置）；
  3. 输出：`x * mask / (1 - p)`。
- **不得消费全局 RNG**：只能使用上面的局部 `Generator`，测试会用固定基准序列验证全局随机状态未被改动。
- 同 `seed` 必须得到同掩码；不同 `seed` 得到不同掩码。输出 dtype 跟随输入（支持 float64）。

## 示例

```
输入：x=ones(1,8), p=0.5, training=True, seed=1234
   按上述 Generator 约定生成掩码，仅第 5 个位置被保留
输出：[[0,0,0,0,0,2,0,0]]    # 保留值 = 1/(1-0.5) = 2.0
```

## 约束

- float32/float64 输入
- `0 <= p < 1`；结果由 `seed` 完全确定（可复现）
