# 调度器：WSD 预热-恒定-多项式退火（Warmup-Stable-Decay）

## 题目描述

余弦退火要求训练一开始就知道总步数，中途无法续训；MiniCPM（arXiv:2404.06395）把调度改成三段式 **WSD**：线性 warmup → **恒定**段（stable，lr 全程不动，可无限续训、可从同一检查点分出多条退火支路）→ 末端**多项式退火** \(lr\cdot(1-t)^p\) 把 lr 快速压到地板。Wen et al.（arXiv:2410.05192）进一步分析了它在退火支路上的掉 loss 行为。本题要求实现这个三段式调度器。

你要**继承 `torch.optim.lr_scheduler.LRScheduler`** 返回一个调度器实例：约定「构造完成即第 0 步、每调用一次 `step()` 才前进到下一步」，评测按该约定读出每一步写回 `param_groups[0]["lr"]` 的学习率序列，与独立重算（以及 \(p=1\) 时的 `transformers.get_wsd_schedule`）逐元素对拍。

## 规则

- 工厂函数 `solve(params, base_lr=0.1, warmup_steps=4, stable_steps=6, decay_steps=5, decay_p=1.0, min_lr_ratio=0.0)`：内部用 `torch.optim.SGD(params, lr=base_lr)` 造优化器，返回其上的 `LRScheduler` 子类实例；`step()` 只改学习率，不动参数。
- 记 \(f(t)\) 为第 \(t\) 步（\(t\) 从 0 计）相对 `base_lr` 的倍率，\(lr(t)=\text{base\_lr}\cdot f(t)\)，分段如下：
  1. **warmup**（`warmup_steps`>0 且 \(t<\) `warmup_steps`）：\(f=\dfrac{t}{\text{warmup\_steps}}\)
  2. **stable**（\(t<\) `warmup_steps`+`stable_steps`）：\(f=1\)
  3. **decay**（`decay_steps`>0 且 \(t<\) `warmup_steps`+`stable_steps`+`decay_steps`）：
     \[ t_{\text{decay}}=\frac{t-\text{warmup\_steps}-\text{stable\_steps}}{\text{decay\_steps}},\qquad f=(1-t_{\text{decay}})^{\text{decay\_p}} \]
  4. **其余**（含退火段结束后、`decay_steps`≤0 时恒定段之后）：\(f=\) `min_lr_ratio`
- 倍率要按地板 `min_lr_ratio` \(m\) 线性下移，并夹紧非负（只对 warmup 段与 decay 段生效，另外两段本来就满足）：
  \[ f\leftarrow\max\big(0,\;f\cdot(1-m)+m\big) \]
- 全程 float；`decay_p` 可为任意正实数（1 为线性退火、2 为平方退火、0.5 为 \(1-\sqrt{\cdot}\) 之外的根式族）。
- **禁止**调用 `transformers.get_wsd_schedule`、`LambdaLR`、`LinearLR` 等现成调度器，需自己写分段函数。

## 示例

`base_lr=0.5`、`warmup_steps=4`、`stable_steps=6`、`decay_steps=5`、`decay_p=2.0`、`min_lr_ratio=0.1`，读出 \(t=0..14\)：

```
[0.05, 0.1625, 0.275, 0.3875,      # warmup（地板下移后从 0.1·base 起步）
 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,     # stable 恒定段
 0.5, 0.338, 0.212, 0.122, 0.068]  # decay 段 (1-t)^2 与之后的地板值
```

## 约束

- 各段步数为非负整数；`decay_steps` 可为 0（恒定段之后直接落到地板）
- 学习率序列须与独立重算逐点一致（\(10^{-9}\) 量级），对拍口径见 notes.md
- 超出 `warmup+stable+decay` 之后 lr 恒为 `base_lr·min_lr_ratio`
