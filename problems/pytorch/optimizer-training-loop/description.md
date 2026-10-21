# SGD 训练循环与数值复现

## 题目描述

训练循环的每一行都在决定数值结果：梯度何时清零、loss 何时记录、`momentum` 缓冲区如何累积。少写一个 `zero_grad()` 或记录时机差一步，loss 轨迹就完全不同。本题要求精确复现一条标准训练循环的 loss 轨迹。

## 规则

实现 `solve() -> list[float]`，严格按以下流程执行并返回 loss 序列：

1. 参数：$w = \texttt{torch.tensor([2.0], requires\_grad=True)}$
2. 优化器：`torch.optim.SGD([w], lr=0.1, momentum=0.9)`
3. 循环 20 步，每步依次为：
   - `opt.zero_grad()`
   - 计算目标 $loss = (w^2).\mathrm{sum}()$
   - **记录当前 loss 数值**（`loss.item()`，在 `backward()` 之前）
   - `loss.backward()`
   - `opt.step()`
4. 返回长度 20 的 `list[float]`。

注意：

- 记录时机在 `backward`/`step` 之前，因此第 0 项是初始 $w=2.0$ 的 loss $=4.0$。
- **不要**动全局 seed，本流程无随机性，结果是确定性浮点序列。
- momentum 会令 $w$ 越过极小点 $0$ 来回振荡——观察第 4~7 项的回升即是动量过冲的直观体现。

## 示例

```
输入：无
输出（前 5 项）：[4.0, 2.56, 0.8464, 0.0154, 0.3809, ...]
```

推演第 0→1 步：$loss_0 = 2.0^2 = 4.0$；$\nabla w = 2w = 4.0$；momentum buffer $b = 0.9 \cdot 0 + 4.0 = 4.0$；$w \leftarrow 2.0 - 0.1 \cdot 4.0 = 1.6$；$loss_1 = 1.6^2 = 2.56$。

## 约束

- 返回 `list[float]`（非张量），长度恰好 20
- 评测按 `atol=1e-3` 逐元素对拍全序列
