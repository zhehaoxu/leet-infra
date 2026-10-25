# 详解：OneCycle 单周期学习率与动量

## 出处
- Smith & Topin, *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates*, arXiv:1708.07120（1cycle 策略：单个 lr 峰 + 动量反对称 + 远低于起点的下限）。
- 参考实现：PyTorch 原生 `torch/optim/lr_scheduler.py::OneCycleLR`（`_schedule_phases` / `_annealing_cos` / `_annealing_linear` / `get_lr`），fastai 的 `Learner.one_cycle` 是同一策略的工程版。
- todos.md 里写的 arXiv:1708.07123 有误，PyTorch 官方文档链接给的是 **1708.07120**，本题按后者与原生实现口径。

## 核心与易错点
1. **升-降两段而不是单调降**：`pct_start` 控制峰的位置（上段长度约为 `pct_start*total`），下段要一路降到 `floor = initial/final_div_factor`。`final_div_factor` 让**下限可以远低于 base_lr**，这是 1cycle 与 warmup+cosine 的本质差别。
2. **动量与 lr 反对称**：上段 lr 从 `initial` 升到 `max_lr` 时，momentum 必须从 `max_momentum` 降到 `base_momentum`；下段反过来。二者共用**同一个 pct** 与同一个 anneal 函数，只是端点互换方向。评测的不变量断言就是"lr 峰与动量谷同址"。
3. **段边界是浮点**：`pct_start*total_steps - 1` 一般不是整数，写成整除或取整会让后半段整体错位（PyTorch 源码就是 `float(pct_start * total_steps) - 1`）。
4. **分段的判定是 `t <= end_i`（左闭右开到最后一段的兜底）**，首段的左端点是 `0.0`；三阶段时第二/三段的左端点分别是第一/第二段的上界。
5. **构造即第 0 步**：`OneCycleLR` 在构造时就会写入 \(t=0\) 的 lr/momentum（此时 lr 恰好等于 `initial`、momentum 等于 `max_momentum`），本题沿用同一约定，别多走一步。
6. `cycle_momentum=False` 时**绝不能**触碰 `group["momentum"]`（评测检查它恒为 `max_momentum`）。

## 对拍口径
- 外部参照：**原生 `torch.optim.lr_scheduler.OneCycleLR`**，用同一个 `SGD(lr=max_lr/div_factor, momentum=max_momentum)` 起头，按同一约定读 \(t=0..\text{total}-1\) 的 `lr` 与 `momentum`，逐点绝对误差 < 1e-12（覆盖 cos/linear × 两/三阶段 × cycle_momentum 开关共 8 组配置）。
- 独立重算：`_oracle` 用 `a+(b-a)*(1-cos(pi*pct))/2` 这一数学等价但浮点落点不同的写法，容差 1e-9，用来卡住"照抄原生源码但仍写错分段"的情况。
- 退化/不变量：终点 lr 低于起点 lr、动量恒在 `[base_momentum, max_momentum]` 带内、`cycle_momentum=False` 时动量不变。

## 复杂度
每步 \(O(\text{param groups})\) 的闭式计算，无状态张量；段定位是常数个小段比较。
