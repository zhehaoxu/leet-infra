# RNG 可复现账：seed 包装、显式 generator 与状态快照

## 题意

可复现性的全部账目都落在**生成器对象**上：`torch.manual_seed` 播的是全局默认生成器，`torch.Generator` 是显式随机源，`get_state/set_state` 是快照原语。本题手写 seed 分类与包装、状态快照的 fork 语义、多生成器消费路由——不调 `torch.manual_seed` / `torch.initial_seed` / `torch.random`（含 `fork_rng`）与 `Generator.manual_seed` / `Generator.initial_seed`，只允许 `torch.rand/randn/randint/randperm` 配显式 `generator=...` 消费、`get_state/set_state` 做状态账。

## 求解

官方口径（torch 2.14.1 CPU 实测钉死）：

- **seed 包装**：合法域 `[-2**63, 2**64-1]`，越界 `ValueError`；`initial_seed` 返回 `s mod 2**64` 的**无符号**值——`-5 → 2**64-5`、`2**63 → 2**63`（不是 int64 重解释的负数）；`bool/float/str → RuntimeError`（bool 是 int 子类，须先于 int 分支显式拒）；
- **fresh Generator**：未播种的 `initial_seed = 67280421310721`（MT19937 缺省种子常数），且两个 fresh Generator 序列逐元素相同；
- **状态快照**：`get_state` 返回 `uint8[5056]`；`set_state` 精确恢复到快照时刻——中流恢复后重复消费得到相同值；fork 语义：快照 → 消费 → 旁路消费 → 恢复 → 再消费 == 快照时刻的直接续消费；
- **同 seed 不同 Generator 对象** → 序列逐元素相同；`initial_seed` 不推进状态；
- **dtype 进入消费路径**：同 seed 的 float32/float64 消耗不同流；
- **消费顺序敏感**：`rand → randn → randint → randperm` 依次消耗同一生成器。

判题面提供七键 generator 包（`"a"/"b"` 同 seed 异对象、`"mid"` 中流态、`"f32"/"f64"`、`"chain"`、`"default"` 即 `torch.default_generator` 本身）。

## 数据规模

- seed 账 13 个（合法域边界 `±2**63`、`2**64-1`、负数、0、普通值；越界 `2**64`/`2**65+7`/`-2**63-1`；`True/False/1.5/"7"` 类型拒）。
- 消费账 4 组：同 seed 对偶、中流快照恢复、default fork（快照→rand(2)→rand(4)→恢复→rand(2)）、dtype 双流与四原语消费链。
- 随机对拍 60 组：随机合法 seed + 随机中流消耗量。

## 评分

账逐项与官方 `Generator.manual_seed/initial_seed` 直调结果全等（含无符号包装与异常分类）；fork 恢复精确到 default 生成器状态逐字节一致（快照恢复后恰一次 `rand(2)` 消耗）。

## 红线

- 不得调用 `torch.manual_seed` / `torch.seed` / `torch.initial_seed` / `torch.random`（AST 调用节点级检查）；
- 不得调用 `Generator.manual_seed` / `Generator.initial_seed`；
- 显式 generator 的播种全部由判题面完成，solve 只做路由、快照、恢复与消费账。
