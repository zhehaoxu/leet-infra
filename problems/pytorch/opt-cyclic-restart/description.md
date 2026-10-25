# 调度器：SGDR 余弦暖重启与周期账本（Warm Restarts）

## 题目描述

单条余弦退火只有一次机会：lr 一路滑到谷底，训练就结束了。SGDR（Loshchilov & Hutter，arXiv:1608.03983）把余弦**周期化**——每个周期结束不是停住，而是把 lr **跳回峰值**重新退火一次（"暖重启"，让优化器一次次跳过尖锐极小值），并且周期长度按 \(T_i=T_0\cdot T_{mult}^{\,i}\) 逐次拉长，前期探索、后期精修。它没有总步数概念，可以无限续训，代价是你必须自己维护一份**周期账本** \((T_{cur}, T_i)\)。

本题要求实现这个调度器，并与原生 `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` 语义逐位一致。你要**继承 `torch.optim.lr_scheduler.LRScheduler`** 返回一个调度器实例，约定「构造完成即第 0 步、每调用一次 `step()` 才前进」；评测按该约定读出每一步写回 `param_groups[0]["lr"]` 的学习率序列，与原生参照以及闭式重算逐元素对拍，并单独核对重启点账本。

## 规则

- 工厂函数 `solve(params, base_lr=0.1, T_0=3, T_mult=2, eta_min=0.0)`：内部用 `torch.optim.SGD(params, lr=base_lr)` 造优化器，返回其上的 `LRScheduler` 子类实例；`step()` 只改学习率。
- 状态量 \(T_{cur}\)（本周期内已走步数）与 \(T_i\)（本周期长度），初值 \(T_{cur}=0,\ T_i=T_0\)；构造即第 0 步，故 \(lr(0)=\) `base_lr`（\(T_{cur}=0\) 时余弦取到峰值）。
- 记 \(B\) 为该参数组的 base_lr、\(\eta=\) `eta_min`，第 \(t\) 步的学习率：
  \[ lr = \eta + (B-\eta)\cdot\frac{1+\cos\!\big(\pi\, T_{cur}/T_i\big)}{2} \]
- `step()` 的**顺序不可颠倒**（先推进账本，再写 lr）：
  1. \(T_{cur}\leftarrow T_{cur}+1\)
  2. 若 \(T_{cur}\ge T_i\)：\(T_{cur}\leftarrow T_{cur}\bmod T_i\)，且 \(T_i\leftarrow T_i\cdot T_{mult}\)（发生一次重启）
  3. 按上式写回 lr
- 由这条顺序直接推出三个必须成立的事实：**重启点恰好落在 \(T_{cur}=0\) 处、lr 精确回到 `base_lr`**；第 \(n\) 个周期（从 0 计）长度 \(T_0\cdot T_{mult}^n\)，重启发生在 \(t=T_0(T_{mult}^n-1)/(T_{mult}-1)\)（\(T_{mult}=1\) 时为 \(t=nT_0\)）；而 \(\cos\) 取到 \(-1\) 的**谷底 \(\eta\) 永远取不到**（那一步在写 lr 之前就被归零了）。
- `eta_min` 是下界而非终点值；调度器无终点，步数再多也不会夹紧或停止。
- **禁止**调用 `CosineAnnealingWarmRestarts`、`CosineAnnealingLR`、`SequentialLR` 等现成调度器。

## 示例

`base_lr=0.1`、`T_0=3`、`T_mult=2`、`eta_min=0.0`，读出 \(t=0..13\)：

```
[0.1,      0.075,    0.025,      # 周期0：长度 3，T_cur=0,1,2
 0.1,      0.093301, 0.075,      0.05,      0.025,      0.006699,   # 周期1：长度 6，t=3 跳回峰值
 0.1,      0.098296, 0.093301,   0.085355,  0.075]      # 周期2：长度 12，t=9 再次跳回峰值
```

重启点是 `t = 0, 3, 9, ...`（间隔 \(3,6,12=T_0\cdot2^n\)）；\(t=2\) 的 0.025 之后不是 0 而是 0.1。

## 约束

- `T_0` 为正整数、`T_mult` 为不小于 1 的整数、`eta_min` 为非负实数
- 学习率序列须与原生参照以及闭式重算逐步一致（\(10^{-12}\) 量级），对拍口径见 notes.md
- 周期数不设上限：读到第 40 步、第 60 步都必须继续按 \(T_0\cdot T_{mult}^n\) 拉长
