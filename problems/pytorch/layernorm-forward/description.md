# LayerNorm 前向传播

不使用 `torch.nn.LayerNorm` 或 `F.layer_norm`，手写 LayerNorm 的前向计算。LayerNorm 是 Transformer 每个子层的标配，也是理解"归约维度"的经典练习。

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

其中 $\mu$、$\sigma^2$ 沿**最后一维**逐行计算，$\sigma^2$ 使用**有偏**方差（除以 $H$ 而不是 $H-1$，与 `F.layer_norm` 保持一致）。

## 规则

1. 输出与 `x` 形状、dtype 相同。
2. 方差必须是有偏的（除 `H`），否则无法与 `F.layer_norm` 对齐。
3. 常数列（方差为 0）输出应趋近 `bias`（`weight=None` 时为全 0），不允许 `nan`/`inf`。
4. `weight`/`bias` 不为 `None` 时通过广播完成仿射变换，**不得**修改输入张量（无 in-place）；`None` 分别等价于全 1 / 全 0。

## 示例

```python
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
solve(x)
# mu = 2.5, var = 1.25
# tensor([[-1.3416, -0.4472, 0.4472, 1.3416]])
```

## 提示

- `x.mean(dim=-1, keepdim=True)` 保持维度才能广播
- 有偏方差可以直接 `((x - mu) ** 2).mean(dim=-1, keepdim=True)`，避免 `var` 的 `unbiased` 参数差异坑
- 归一化后再乘除 `eps`：先 `x_hat = (x - mu) / torch.sqrt(var + eps)`

## 约束

- `2 <= H <= 4096`，batch 元素总数 `<= 2^20`
