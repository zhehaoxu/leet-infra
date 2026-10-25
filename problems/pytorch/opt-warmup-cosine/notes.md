# 详解：线性 warmup + 余弦退火学习率

## 出处
「warmup + cosine」是 Transformer/ViT 训练的默认学习率配方：Loshchilov & Hutter 的 SGDR（arXiv:1608.03983）提出余弦退火；Goyal et al.（arXiv:1706.02677）提出线性 warmup。二者组合被 `torch.optim.lr_scheduler.LinearLR` 与 `CosineAnnealingLR` 通过 `SequentialLR` 拼接实现——本题要求你**自己按闭式公式**算出每一步的 \(lr\)，不用现成调度器拼装。

## 核心思想
- **线性 warmup**：开头若干步把 \(lr\) 从 \(base\_lr/\text{warmup}\) 线性抬到 \(base\_lr\)。作用是让 Adam 类优化器的二阶矩估计先积累、避免初期大 \(lr\) 放大噪声梯度。
- **余弦退火**：随后按 \(\tfrac12(1+\cos(\pi\cdot\text{progress}))\) 从 \(base\_lr\) 平滑降到 \(min\_lr\)；`progress` 从 0 到 1。相比阶梯式下降，余弦更平滑、末期收敛更细。

## 为什么容易写错（易错点）
1. **\(t\) 的基准**：构造完成后就应处于第 \(0\) 步（\(lr=lr(0)=base\_lr/\text{warmup}\)），而不是 \(lr(0)=base\_lr\)。第 \(k\) 次 `step()` 后进入第 \(k\) 步（\(k=1,2,\dots\)）。
2. **warmup 用 \(t+1\) 而非 \(t\)**：若用 \(t/\text{warmup}\) 则第 \(0\) 步 \(lr=0\)，与参考不符。本式第 \(0\) 步是 \(base\_lr/\text{warmup}>0\)。
3. **峰值处衔接**：warmup 末（\(t=\text{warmup}-1\)）与余弦起点（\(t=\text{warmup}\)）都等于 \(base\_lr\)，示例里出现两个 0.1 是正常的，不是 bug。
4. **`progress` 的分母与夹紧**：分母是 `total_steps - warmup_steps`（余弦段长度），不是 `total_steps`；且需 `clamp` 到 \([0,1]\) 防止 \(t\) 超出 \(lr\) 反弹。`denom<=0` 退化情形直接取 \(lr=min\_lr\)。
5. **不调用 `LRScheduler.__init__` 的记账**：官方 `LRScheduler` 构造时会自动 `step()` 一次并维护 `_step_count`，语义与本题「构造即第 0 步、step 才前进」不同。参考解自管 `_t` 计数、构造时直接写入 \(lr(0)\)，`step()` 内 `_t+=1\) 后写回，避免与父类记账打架。

## 对拍口径
判题用独立纯 Python 重算 `lr(t)` 闭式（`test_cases.py` 里的 `_ref_lr`，逐段用标准 `math` 计算，与参考解的结构不同），驱动调度器对象逐步读 `param_groups[0]["lr"]`，对 `atol=1e-6, rtol=1e-6` 比对整条序列。覆盖：示例参数、`warmup_steps=0` 纯余弦、不同超参组合、下界夹紧（末步应≈`min_lr` 且余弦段单调不增）。

## 复杂度
\(O(1)\) 每步；无额外张量状态，仅维护计数字段。
