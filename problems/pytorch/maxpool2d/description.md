# 2D 最大池化

## 题目描述

Max Pooling 在每个 $k_h \times k_w$ 窗口内取最大值，是 CNN 下采样的主力（保留最强响应、平移不变性）：

$$
\text{out}[n,c,i,j] = \max_{0 \le u < k_h,\ 0 \le v < k_w} x\!\left[n,c,\, i\cdot s_h + u,\, j\cdot s_w + v\right]
$$

输出尺寸 $\left\lfloor \frac{H + 2p_h - k_h}{s_h} \right\rfloor + 1$（宽同理）。

## 规则

- 实现 `solve(x, kernel_size, stride=None, padding=0)`，x 形状 `(N, C, H, W)`；三个参数支持 int 或 `(h, w)` 元组，`stride=None` 时等于 `kernel_size`。
- **padding 陷阱**：最大池化的填充值必须是 $-\infty$。填 0 在全负输入时会把边界输出的最大值错误地抬成 0。
- 禁止直接调用 `F.max_pool2d` / `nn.MaxPool2d`；可用 `unfold` 展开窗口后 `amax`，或循环实现。

## 示例

```
输入：x = 1x1 通道 4x4 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], kernel=2
输出：[[[6, 8], [14, 16]]]
```

## 约束

- float32/float64；输出尺寸按 floor 公式（不支持 ceil_mode）
- 结果与 `torch.nn.functional.max_pool2d` 一致
