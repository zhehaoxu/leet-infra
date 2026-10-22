# torch.where / 三目选择详解

## 定义

逐元素条件选择，且**支持广播**：

$$ \text{out}_i=\begin{cases}x_i,& c_i\ \text{真}\\ y_i,& c_i\ \text{假}\end{cases} $$

`c`（条件）、`x`、`y` 三者按 broadcasting 统一到同一形状再逐位选。等价于 NumPy `np.where`。

## 方向（本关易反）

`torch.where(cond, x, y)`：**cond 为 True 取 x，False 取 y**。手写掩码式实现要把分量对齐——本参考解：

$$ \text{out}=x\cdot m + y\cdot(1-m),\quad m=\text{broadcast}(\text{cond}.\text{bool}()) $$

$m$ 为真时留 $x$ 分量、置 $y$ 为 0；写成 $x(1-m)+y\,m$ 就全反。

## 实现与坑

1. **先广播再算**：`m, xt, yt = torch.broadcast_tensors(cond.bool(), x, y)`，否则 `x*mask` 形状不匹配、或漏掉 x/y 之间的广播。
2. `cond` 可能是任意数值，需 `.bool()`；掩码转 dtype 用 `m.to(dtype)` 参与乘加，保持与 x/y 同 dtype（半精度尤其注意）。
3. 纯 `where` 不改变 dtype/shape，只搬运；与布尔索引 `x[cond]`（会**压平**、丢形状）不同——要保形用 `where`，要筛选元素用布尔索引。

## 常见用途

mask 后置零（注意力把屏蔽位填 0）、数值裁剪、`where(isnan, 0, x)` 修 NaN、梯度里按符号分支。是比 for-else 快得多的向量化三目。
