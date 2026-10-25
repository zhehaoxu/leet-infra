# 调度器：ReduceLROnPlateau 指标驱动平台期降速

## 题目描述

`LRScheduler` 家族大多有闭式 \(lr(t)\)：给定总步数就能提前算出每一步的学习率。但**平台期降速**（reduce on plateau）这一支不看未来，只观测一条**验证指标流**：指标连续足够多次不再显著改善，就把学习率乘一次 `factor`。于是它不是一个函数，而是一台**状态机**——状态只有三个量 \((best, num\_bad, cooldown\_counter)\)，学习率是一串台阶而非曲线。PyTorch 自 0.2 起内置该调度器，它是「显式喂指标」这一交互范式的唯一代表：`step()` 必须带参数。

本题要求实现这台状态机，语义与原生 `torch.optim.lr_scheduler.ReduceLROnPlateau` **逐位一致**。评测不喂梯度，而是喂一串固定指标数值给 `step(metrics)`，每步之后读取 `param_groups[0]["lr"]`，与原生参照以及独立重算的台阶序列逐元素对拍。

## 规则

- 工厂函数 `solve(params, base_lr=0.1, factor=0.5, patience=2, min_lr=0.0, mode="min", threshold=1e-4, threshold_mode="rel", cooldown=0, eps=1e-8)`：内部用 `torch.optim.SGD(params, lr=base_lr)` 造优化器，返回其上的 `LRScheduler` 子类实例；`step(metrics)` 只改学习率，绝不动参数张量。
- 状态初值：\(best=+\infty\)（`mode="min"`）或 \(-\infty\)（`mode="max"`），\(num\_bad=0\)，\(cooldown\_counter=0\)。构造时学习率保持 `base_lr` 不变。
- 判定「显著改善」\(\text{is\_better}(cur, best)\)（`threshold` 用来忽略不痛不痒的进步）：
  - `min` + `rel`：\(cur < best\cdot(1-threshold)\)；`min` + `abs`：\(cur < best - threshold\)
  - `max` + `rel`：\(cur > best\cdot(1+threshold)\)；`max` + `abs`：\(cur > best + threshold\)
- 每次 `step(metrics)` 严格按下面顺序执行（顺序换一步结果就不同）：
  1. \(cur \leftarrow\) `float(metrics)`（0 维张量也要能吃）
  2. 若 \(\text{is\_better}(cur,best)\)：\(best\leftarrow cur,\ num\_bad\leftarrow 0\)；否则 \(num\_bad \leftarrow num\_bad+1\)
  3. 若 \(cooldown\_counter>0\)：\(cooldown\_counter\leftarrow cooldown\_counter-1,\ num\_bad\leftarrow 0\)（冷却期内的变差全部作废）
  4. 若 \(num\_bad > patience\)（**严格大于**，等于不降）：执行一次降速，随后 \(cooldown\_counter\leftarrow cooldown,\ num\_bad\leftarrow 0\)
- 降速对每个参数组：\(new = \max(old\cdot factor,\ min\_lr)\)，**仅当** \(old-new > eps\) 才把 \(new\) 写回；否则视作无效降速、保持原值（第 4 步的 \(num\_bad/cooldown\) 记账照常进行）。
- **禁止**调用 `torch.optim.lr_scheduler.ReduceLROnPlateau` 或其它现成调度器。

## 示例

`base_lr=0.1`、`factor=0.5`、`patience=2`、`mode="min"`、`threshold_mode="rel"`、`cooldown=0`，依次喂入指标
`[1.0, 0.9, 0.95, 0.9, 0.895, 0.7, 0.75, 0.76, 0.77, 0.77, 0.2, 0.3, 0.3, 0.3, 0.3]`，每步之后的 lr：

```
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025]
 # 0.9 与上一 best 相等不算改善；0.895 相对 best=0.9 是改善 → 复位计数
 # 第 9 步 num_bad=3>2 才降第一次；第 11 步 0.2 破纪录又把计数清零
```

## 约束

- 指标为 float 或 0 维张量；全程按 Python float 运算
- \(0<factor<1\)，`patience`/`cooldown` 为非负整数
- 学习率序列必须单调不增、恒 \(\ge min\_lr\)，且与原生参照逐步一致（对拍口径见 notes.md）
