# Adam 全分支更新账

`torch.optim.Adam` 的公式人人会背，坑都在细节里：偏差修正的 `t` 只数**真正更新过**
的步（没梯度的参数根本不建 state，`step` 也就不涨）、`eps` 加在 `sqrt(v)/sqrt(1-β₂ᵗ)`
**之后**（没乘 `sqrt(1-β₂ᵗ)`，和论文后期的修正式不同）、`amsgrad` 的最大值**先更新
再进分母**且 state 里初始化为全零、`maximize` 的取负发生在 `weight_decay` **之前**、
而 Adam 的 `weight_decay` 是 **L2 式**（λp 并进梯度后同时污染两个矩，被 `1/√v` 打折，
与 AdamW 的解耦衰减是两条路）。

本题按 torch 2.14.1 的 `torch/optim/adam.py` 复刻 `Adam.__init__` 的五道校验与
`_single_tensor_adam` 的标量路径，一次跑多步，返回可逐步对拍的账本。

实现 `solve(params, grads, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
amsgrad=False, maximize=False, state=None)`。

## 输入

- `params`：任意形状的实数张量（float32 / float64）。
- `grads`：长度 `T` 的序列，每项是与 `params` 同形状的张量，或 `None`（该步无梯度）。
- `lr` / `eps` / `weight_decay`：标量；`betas`：两个 float 的元组。
- `amsgrad` / `maximize`：布尔。
- `state`：`None`（全新开始）或一个 dict，四个键都要有：`exp_avg`、`exp_avg_sq`、
  `max_exp_avg_sq`（只在 `amsgrad=True` 时读）、`step`（**已经更新过的步数**，int）。

## 参数校验

五道门，按下列**顺序**抛 `ValueError`：

```
Invalid learning rate: {}
Invalid epsilon value: {}
Invalid beta parameter at index 0: {}
Invalid beta parameter at index 1: {}
Invalid weight_decay value: {}
```

判据是 `not 0.0 <= x`（`betas` 再加 `< 1.0`）。这个写法有两个后果：**NaN 也会被拒**
（`0.0 <= nan` 为假，文案里就打出 `nan`），而 `beta == 1.0` 非法。`betas` 不是两个
元素时抛 `ValueError`。

## 单步更新

对第 `t` 步：

1. `grads[t] is None`：**整步跳过**——不建 state、不动参数、**`step` 计数不涨**。
2. `step += 1`（计数器在**使用前**自增，所以第一次更新用的是 `t = 1`）。
3. `maximize` 为真：`g ← -grads[t]`（**先取负**）。
4. `weight_decay != 0`：`g ← g + weight_decay · params`（**本步更新前**的参数值）。
5. `m ← β₁·m + (1-β₁)·g`；`v ← β₂·v + (1-β₂)·g²`。
6. `bc1 = 1 - β₁^step`，`bc2 = 1 - β₂^step`，`step_size = lr / bc1`。
7. `denom = sqrt(v) / sqrt(bc2) + eps`；`amsgrad` 为真时先把
   `mx ← max(mx, v)`，再用 `sqrt(mx)/sqrt(bc2) + eps` 当分母。
8. `params ← params + (m / denom) · (-step_size)`。

## 返回

一个 dict，七个键：

- `params`：更新后的参数。
- `exp_avg` / `exp_avg_sq`：两个矩的最终值。
- `max_exp_avg_sq`：`amsgrad=True` 时的滑动最大值；否则全零。
- `path`：长度 `T + 1` 的参数快照，`path[0]` 是入参的拷贝。
- `steps`：长度 `T`，每步实际用到的 `step` 计数（跳过步为 `None`）——偏差修正是
  从它算出来的。
- `deltas`：长度 `T`，每步实际位移 `params_after - params_before`（跳过步为 `None`）。

入参一律不原地修改；返回字段各自独立，就地改 `deltas` 不会动到 `params`。

## 约束

- `T` 可以是 0：`path` 只有一段，`steps` / `deltas` 为空，两个矩全零。
- 与 `torch.optim.Adam` 的逐步结果在 float64 下按 `1e-12` 对拍（`eps=1e-8` 时
  第一步的位移是 `0.099999998` 而不是 `0.1`——eps 不加权就是这么显形）。
- 只考实数张量：官方 `is_complex` 的 `view_as_real` 路径与 `capturable` /
  `differentiable` / `foreach` / `fused` 分支不在本题范围内。

## 示例

```
输入：params=[1.0, -2.0], grads=[[0.5, -1.0]], lr=0.1
输出：params=[0.900000002, -1.900000001], steps=[1], deltas=[[-0.099999998, 0.099999999]]
```

单步时 `m = 0.1g`、`v = 0.001g²`、`bc1 = 0.1`、`bc2 = 0.001`，于是
`denom = |g| + 1e-8`、`step_size = 1`，位移就是 `-g/(|g|+1e-8)`。

```
输入：params=[1.0, -2.0], grads=[null, [0.5, -1.0]], lr=0.1
输出：steps=[null, 1], path=[[1.0, -2.0], [1.0, -2.0], [0.900000002, -1.900000001]]
```

第一步没有梯度，`state` 根本不会创建，`step` 也不涨；第二步才是第一次更新，偏差修正
仍按 `t = 1` 算。
