# 机制：指数滑动平均（EMA）权重与升温

## 题目描述

很多训练配方（扩散模型、Mean Teacher 自训练、Model Soup、评测期权重平均）会额外维护一份**指数滑动平均**（Exponential Moving Average, EMA）权重：它不参与求导，只是每步把「真实训练权重」以很慢的速度吸进来，得到一条更平滑、泛化更好的权重轨迹，常拿来评估或生成。为了不让 EMA 在最初几步被随机初始化的坏权重「定死」，普遍加一个**升温**（warmup）：早期用更小的 decay（更快跟随新权重），随后 decay 逐步爬到设定值。

本题要求你实现一个 EMA 影子权重维护器：它持有与训练参数一一对应的影子副本，每次调用 `update()` 就地把当前参数平滑进影子。**注意**：EMA 不更新训练参数、不用梯度，是一个独立于 `Optimizer` 的机制对象。

## 规则

- 工厂函数 `solve(params, decay=0.9, warmup=True)`，返回一个 EMA 对象，须暴露：
  - `.shadow`：**列表**，与 `params` 一一对应；构造时每个影子张量初始化为**对应参数的当前值副本**（`shadow_i \leftarrow p_i`）。
  - `.update()`：执行一次滑动平均（读**当前**的 `params` 值，就地更新 `.shadow`）。
- 记第 \(n\) 次 `update()`（\(n=1,2,3,\dots\)），本步的有效衰减：
  \[d_n=\begin{cases}\min\!\left(decay,\ \dfrac{1+n}{10+n}\right), & \texttt{warmup}\\[6pt] decay, & \text{\texttt{warmup}=False}\end{cases}\]
  然后对每个参数：\[shadow_i \leftarrow d_n\cdot shadow_i + (1-d_n)\cdot p_i\quad(\text{就地})\]
- `update()` **不修改** `params`、不清零任何东西；只推进内部计数 \(n\) 并更新 `.shadow`。影子张量与训练参数**分离**（是 clone，不是同一块内存）。
- 全程 float32；禁止用现成的 EMA 库。

## 示例

单参数 \(p=[1.0,\,-2.0,\,0.5]\)，`decay=0.9`、`warmup=True`。模拟训练：每步把 \(p\) 增加 \([0.1,\,-0.05,\,0.2]\) 后调用一次 `update()`，共 7 次。升温使早期 \(d_n\) 很小（\(d_1=2/11\approx0.18\)）、影子快速跟随，随 \(n\) 增大 \(d_n\) 逼近上界但 7 步内仍小于 0.9：

```
7 次 update 后 shadow ≈ [1.620004, -2.310002, 1.740008]
```

## 约束

- \(n\) 从 \(1\) 起（构造时 \(n=0\)、\(shadow=p_0\)）。
- 须与参考实现逐元素一致（对拍口径见 notes.md）。
- 支持多个、不同形状的参数；升温对每个参数共享同一 \(d_n\)。
