# 调度器：Noam 逆平方根学习率（inverse square root）

## 题目描述

Transformer 原论文（Vaswani et al., arXiv:1706.03762，式 (5)）给出的调度只用了两个量——模型宽度 \(d_{model}\) 和 warmup 步数 \(w\)：

\[ lr(t)=d_{model}^{-1/2}\cdot\min\!\big(t^{-1/2},\; t\cdot w^{-3/2}\big) \]

前一支按平方根倒数衰减，后一支线性上升，二者在 \(t=w\) 处相交，于是曲线是「线性爬到峰值 \(\Rightarrow\) 之后一路 \(t^{-1/2}\) 长尾下降」。它**没有总步数**、也**没有下限夹紧**，所以可以从任意步续训，是"无需知道训练长度"这一流派的原型（AdaFactor 的相对步长、Big Vision 的 inverse-sqrt 都是它的改写）。

本题要求实现这个调度器。你要**继承 `torch.optim.lr_scheduler.LRScheduler`** 返回一个调度器实例，约定「构造完成即第 0 步、每调用一次 `step()` 才前进」；评测读出每一步写回 `param_groups[0]["lr"]` 的学习率序列，与独立重算（论文 min 闭式）以及 `transformers.get_inverse_sqrt_schedule` 逐元素对拍。

## 规则

- 工厂函数 `solve(params, model_dim=512, warmup_steps=4000)`：内部用 `torch.optim.SGD(params, lr=model_dim**-0.5)` 造优化器，返回其上的 `LRScheduler` 子类实例；`step()` 只改学习率，不动参数。
- 记 \(t\) 为当前步（从 0 计）、\(d=\) `model_dim`、\(w=\) `warmup_steps`，分段定义（与上式在 \(t\ge1\) 时等价，但 \(t=0\) 必须走线性支取 0，**不许**真的去算 \(0^{-1/2}\)）：
  \[ lr(t)=\begin{cases} d^{-1/2}\cdot t\cdot w^{-3/2}, & t<w\\[4pt] d^{-1/2}\cdot t^{-1/2}, & t\ge w \end{cases} \]
- 于是 \(lr(0)=0\)，峰值 \(lr(w)=d^{-1/2}w^{-1/2}\)，且在 \(t=w\) 处连续（相邻差仍等于 warmup 斜率 \(d^{-1/2}w^{-3/2}\)）。
- 学习率是**绝对值**，不乘各参数组的 base_lr；多个参数组写入同一个 \(lr(t)\)。
- 无总步数、无 `min_lr`：\(t\) 再大也只是继续按 \(t^{-1/2}\) 下降，永远为正，不夹紧、不停止。
- `warmup_steps` 为正整数（可为 1，此时第 1 步即达峰、之后纯衰减）；`model_dim` 为正整数。
- **禁止**调用 `get_inverse_sqrt_schedule`、`LambdaLR`、`LinearLR` 等现成调度器，需自己写分段式。

## 示例

`model_dim=64`、`warmup_steps=5`（峰值 \(=64^{-1/2}\cdot5^{-1/2}\approx0.0559017\)），读出 \(t=0..12\)：

```
[0.0, 0.01118034, 0.02236068, 0.03354102, 0.04472136,   # 线性 warmup，公差 d^-0.5·w^-1.5
 0.0559017,                                             # t=w=5 处峰值
 0.05103104, 0.04724556, 0.04419417, 0.04166667,        # t^-0.5 长尾
 0.03952847, 0.03768892, 0.03608439]
```

## 约束

- 全程 float；\(0\le t\le 60\) 内与论文 min 闭式逐点一致（\(10^{-12}\) 量级），并与 `transformers.get_inverse_sqrt_schedule`（取 `timescale=warmup_steps`、优化器 lr 设为峰值）在相对误差 \(10^{-12}\) 内吻合
- 序列必须满足：warmup 段等差、峰值恰好落在 \(t=\) `warmup_steps`、其后严格单调下降且恒正
- `warmup_steps>=1`、`model_dim>=1`；不得在 \(t=0\) 处触发除零或返回 `inf`/`nan`
