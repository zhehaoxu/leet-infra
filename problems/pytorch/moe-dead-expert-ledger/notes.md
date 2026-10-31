# 参考做法：把滞回状态机写成「外层循环 step、内层向量化」

## 实现

1. EMA 递推按字面写：`keep = beta`、`rise = 1 - beta`、`ema` 初值是 `zeros(E)`，然后**每个 step 都递推** `ema = keep * ema + rise * load[step]`，包括第一步——这就是 `ema[0] == (1-beta)*load[0]` 的来源。
2. 步内先算 `alive = ~is_dead`、`below = ema < death`、`above = ema >= revive`，然后只推进当前状态那个计数器：`miss = where(alive & below, miss+1, where(alive, 0, miss))`，`back` 对称。
3. `dies = alive & (miss >= patience)`、`wakes = is_dead & (back >= revive_patience)`。两个掩码因为 `alive`/`is_dead` 互补而天然互斥，可以在同一条 `where` 链里一起应用。
4. 翻转时把 `miss` 和 `back` 一起清零，再更新 `is_dead`，最后一起写 `dead[step]` 与 `streak[step] = where(is_dead, back, miss)`。
5. 三份账在同一个循环里累加：`first_dead = where(dies & (first_dead < 0), step, first_dead)`（`< 0` 的闩锁保证只记第一次），`changes += dies + wakes`，`dead_steps += is_dead`。

四个常见错误：

- **把比较写成 `<=` / `>`**。判死用严格小于、复活用大于等于，这是题面的核心口径；改一个符号，滞回带就变成闭区间，边界值上的行为整体错位。
- **用「上一步的 EMA」判状态**，或者把状态机放在 EMA 递推之前。差一步看似只挪动一次翻转，实际会让 `first_dead`、`state_changes`、`dead_steps` 连锁错位。
- **把 `dead[s]` 记成翻转前的状态**。`dead[s]` 的语义是「这一步推完之后」，判死的那一步就必须是 True。
- **`first_dead` 不加 `< 0` 的闩锁**，于是每次判死都覆盖，最终记到的是最后一次而不是第一次。

## 正确性检验

8 个 `check_*`：手算示例（六个返回项全部核对）、随机对拍（8 组 `beta`/阈值/耐心/负载跨度组合 + 三种非连续视图）、状态机不变量（`dead_steps == dead.sum(0)`、`first_dead` 与 `dead` 列首个 True 一致、`state_changes` 等于从 `dead` 列重数的翻转次数并含第 0 步、翻转那一步 `streak == 0`、同状态相邻步 `streak ∈ {0, 上一个+1}`、活计数器永远达不到自己的阈值、EMA 递推与冷启动口径逐位复核）、滞回带（EMA 落在 `[death, revive)` 内时 alive 不涨 miss、dead 不涨 back）、阈值边界（`death == revive` 且 EMA 恰好等于阈值：alive 侧不算 miss、dead 侧算复活；`beta = 0` 时 `ema == load`）、耐心与冷启动（`patience > S` 永不判死、`patience = 1` 时冷启动第一步就判死、单步历史、全零负载）、非法输入、最大形状 `32×16`。

oracle 是纯 Python：逐专家、逐 step 用标量 `float`/`int` 走同一台状态机，不借用任何张量算子，也不复用参考实现里的掩码写法。

变异测试抓到 10 个：`<=` 判死、`>` 复活、`miss > patience` 的差一、滞后一步的 EMA、EMA 初值取 `load[0]`、`dead[s]` 记翻转前状态、`streak` 永远输出 `miss`、`first_dead` 每次判死都覆盖、`state_changes` 只数判死、`dead_steps` 数翻转前状态。另有 4 个 mutant **判不出来，因为在题面保证下它们与参考写法等价**：题面要求 `death_threshold <= revive_threshold`，于是「低于死亡线」与「不低于复活线」在同一步互斥，「每步两个计数器都推进」「翻转时只清自己那个」都得不到不同的表；`torch.lerp(load, ema, beta)` 在 `2e-11` 容差下也与显式递推不可区分。这不是判题留了洞，而是这条不等式本身就是让状态机不必小心的原因——题面把它写成硬性校验，实现者改口径时才会立刻报错。

## 范围与来源

「用 EMA 统计专家负载、连续若干步低于阈值就判死」是稀疏 MoE 训练里的常见做法，但本仓库判题环境所用的 transformers 5.x 里**没有**对应实现（全库检索 `ema_load`、`dead_expert`、`jitter` 只命中无关词），也没取到可核对的原文公式，所以本题全部口径——EMA 初值 0、每步递推、严格 `<` 判死、`>=` 复活、翻转清零、`streak` 的双含义、三份账的定义——都是**题面自定**，不做归属声明。双阈值滞回借的是控制领域「迟滞比较器」这个通用概念，不是某篇论文的实现。

与合集内其他题的分工：单步的分派与丢弃见 `moe-capacity-dispatch` 与 `moe-capacity-jitter-dispatch`；本题只管**跨 step 的负载账本**，既不分派也不改权重。把负载信息回灌进路由偏置的机制见 `moe-aux-free-routing`（DeepSeek-V3 的 bias 更新），负载均衡损失的族谱见 `moe-switch-aux-loss`、`moe-cv-aux-loss`、`moe-sequence-aux-loss`。
