# 局部响应归一化（LRN）

## 题目描述

局部响应归一化（Local Response Normalization）是 AlexNet 时代引入的「跨通道」归一化：
对每个空间位置，沿**通道维**取一个大小为 `size` 的窗口，用窗口内各通道激活的平方和来
抑制当前通道的响应（数值上类似侧抑制）。它在卷积网络早期用于稳定大激活、增强泛化，
后来被 BatchNorm / LayerNorm 取代，但仍是理解归一化谱系的重要一环。

## 规则

- 实现 `solve(x, size, alpha=1e-4, beta=0.75, k=1.0)`，`x` 形状 `(N, C, H, W)`。
- 对每个元素，沿通道维取以它为中心、宽度为 `size` 的窗口（越界的通道按 **补零** 处理），
  记窗口内平方和为 $\sum_{j} x_{j}^{2}$，则
  $$ y_{i} = \frac{x_{i}}{\left(k + \dfrac{\alpha}{\text{size}} \sum_{j=\max(0,\,i-\lfloor \text{size}/2\rfloor)}^{\min(C-1,\,i+\lfloor \text{size}/2\rfloor)} x_{j}^{2}\right)^{\beta}} $$
- 归一化只在**通道维**滑动，对每个 $(n, h, w)$ 独立进行；$H, W$ 不参与窗口。
- `size` 须为奇数（这样窗口才关于中心对称）。结果须与
  `F.local_response_norm(x, size, alpha=alpha, beta=beta, k=k)` 一致。

## 示例

```
输入：x 形状 (1,3,1,1)，三个通道值 = [1, 2, 3]
     size=3, alpha=0.5, beta=1.0, k=1.0
   平方向量 = [1, 4, 9]，通道窗口 half=1，边界补零 → [0,1,4,9,0]
   各通道窗口平方和 Σ = [0+1+4, 1+4+9, 4+9+0] = [5, 14, 13]
   分母 = (1 + 0.5/3 · Σ)^1 = [1.8333, 3.3333, 3.1667]
   y = x / 分母 = [1/1.8333, 2/3.3333, 3/3.1667]
输出：[0.545455, 0.6, 0.947368]
```

（每个通道的窗口不同：通道 0 只看 [ch0,ch1] 加左补零，通道 2 只看 [ch1,ch2] 加右补零，
故三通道分母互不相同。）

## 约束

- float32 / float64 输入；`size` 为正奇数；`alpha, beta, k > 0`
- 边界通道用**补零**（等价于把缺失通道的平方当作 0），不做循环/复制
- 不使用 `F.local_response_norm` / `torch.nn.LocalResponseNorm`
