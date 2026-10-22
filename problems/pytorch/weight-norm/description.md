# WeightNorm 重参数化

## 题目描述

Weight Normalization 把权重 $w$ 拆成**方向** $v$ 与**长度** $g$ 两个独立参数：

$$
w = g \cdot \frac{v}{\|v\|_2},\qquad \|v\|_2 = \sqrt{\textstyle\sum_{\text{slice}} v_i^2}
$$

沿某一维（`dim`）把张量切成若干「切片」（比如卷积核每个输出通道），每个切片独立分解：切片范数恰好是 $|g|$，方向保持 $v$ 的方向。优化器在 $\log$-尺度 $g$ 上条件数更好。

## 规则

- 实现 `solve(weight_g: torch.Tensor, weight_v: torch.Tensor, dim: int = 0) -> torch.Tensor`，返回展开后的等效权重 $w$，形状与 `weight_v` 相同。
- 范数沿 `dim` 求且 **`keepdim=True`**（`g` 的形状是 `v` 在 `dim` 上坍缩为 1 后的形状），保证逐切片广播。
- **除零陷阱**：某切片全零时 $\|v\|=0$。PyTorch 的实现把分母 `clamp_min(1e-12)`，全零切片输出为 0，不得出现 NaN。
- 输入张量不可原地修改。

## 示例

```
输入：g = [[1.0], [2.0]], v = [[3,4], [3,4]], dim = 1
每行范数：||(3,4)|| = 5
输出：[[0.6, 0.8], [1.2, 1.6]]     # g * v / 5
```

## 约束

- float32/float64，`dim` 为合法维
- 与 `torch.nn.utils.weight_norm` 展开后的 weight 一致；非零切片满足 $\|w_{\text{slice}}\| = |g_{\text{slice}}|$
