# 详解：LR 调度器在训练循环中的位置

## 出处

PyTorch 优化文档 *How to adjust Learning Rate* 与 `torch.optim.lr_scheduler`：自 1.1.0 起官方约定 `optimizer.step()` 在 `lr_scheduler.step()` **之前**；先调 `scheduler.step()` 会触发 UserWarning「Failure to do this will result in PyTorch skipping the first value of the learning rate schedule」。机理：`LRScheduler.__init__` 已把 `last_epoch=0` 的 lr「预告」过一遍（`__init__` 里调了 `self.step()`），所以第 \(i\) 步实际生效 lr 对应 \(i\) 次衰减后的调度值；`StepLR.get_lr = base_lr * gamma ** (last_epoch // step_size)`。`ReduceLROnPlateau` 不是 epoch 驱动而是指标驱动：mode/threshold(rel)/patience/cooldown/factor/min_lr 全在 `lr_scheduler.py` 的状态机里，`best` 只在指标更好时更新，`min_lr` 是硬下限。

## 核心与易错点

1. **「记录时机」就是判题时机**：本题在 `opt.step()` 前记录 lr——变体 A 第 0 项 \(=0.1\)（完整 base_lr），变体 B 第 0 项 \(=0.05\)（调度表第一格 0.1 被跳过）。两种写法 `last_epoch` 都走到 4，计数器相同≠轨迹相同。
2. **权重终值是位置差异的可观测后果**：\(w_{k+1}=w_k(1-2\,lr_k)\)，A 多迈了大 lr 的第一步，终值 0.667 < B 的 0.823。只 lr 账不够，判题还会用权重账反推。
3. **Plateau 的 patience 是「坏步数超过 patience 才降」**：patience=1 → 连续 2 个坏指标降一次；降完 `num_bad` 清零重新攒。
4. **min_lr 触底不豁免降档动作**：第二次降档 \(0.5\to0.25\) 被抬回 0.3（第三次想降 \(0.3\to0.15\) 同样压在 0.3），但每次 `num_bad` 依然清零——从外观看就是「每个攒满 patience 的坏步都'降'了一次，只是不再变」。
5. **threshold 判定用相对式** \(m < best\cdot(1-10^{-4})\)：指标 1.0→1.1 全程单调变差，第一次的 best 一旦记为 1.0 就不再更新。

## 对拍口径

solution 跑真调度器；`_oracle` 纯 Python：StepLR 用幂次闭式公式 \(0.1\cdot0.5^i\) 直接生成两串 lr（乘 0.5 在 IEEE 下与框架逐位一致），权重用 \(w(1-2lr)\) 连乘；Plateau 手写 improved/num_bad/reduce 状态机逐指标重放。另设「B[0]==A[1]」「两变体 last_epoch 相等但权重不等」「plateau 触底序列」等断言。

## 复杂度

\(O(\text{步数})\) 的纯记账，瞬时。
