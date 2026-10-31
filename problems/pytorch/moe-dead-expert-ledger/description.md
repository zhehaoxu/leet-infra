# MoE：负载 EMA 与死专家账本（双阈值滞回）

## 任务

实现 `solve`：给定每个 step、每个专家收到的 token 数，维护一条负载 EMA，并用一台**带滞回的状态机**给每个专家记账——连续多少步饿在死亡线以下就判死，判死之后又要连续多少步站上复活线才允许它回来。返回 EMA、逐步的死活状态、当前连击计数，以及首次判死步、翻转次数、累计死亡步数三份账。

这是 MoE 训练可观测性里最常手搓的一段：专家塌缩（routing collapse）不是「某一步负载为 0」，而是「持续没活干」；而判死/复活如果共用一个阈值，负载在阈值附近抖动时状态会来回翻（flapping），所以工程实现普遍用双阈值。本题把两件事都做成纯函数：没有隐藏状态、没有随机数，输入就是一段历史。不涉及参数重初始化、不涉及 expert reload 的权重拷贝。

## 输入

- `load`：CPU float64 `[S,E]`，非空 2-D、有限、元素非负。第 $s$ 行是 step $s$ 里 $E$ 个专家收到的 token 数。
- `beta`：Python 实数（`bool` 不算），$0\le \beta<1$，EMA 的保留率。
- `death_threshold`、`revive_threshold`：Python 实数（`bool` 不算），均 $\ge 0$ 且有限，并且 $\text{revive}\ge\text{death}$（倒置报 `ValueError`）。
- `patience`、`revive_patience`：Python `int`，$\ge 1$（`bool` 不算，允许大于 $S$）。

接受非连续张量，不修改输入。

## 精确规则

每个专家独立走下面这套流程，初始状态是 **alive**，两个计数器都是 0，EMA 初值是 0。

**EMA**：**每一个 step 都做一次递推**，冷启动不豁免：

$$
\mathit{ema}^{(0)}=0,\qquad
\mathit{ema}^{(s)}=\beta\,\mathit{ema}^{(s-1)}+(1-\beta)\,\mathit{load}[s].
$$

于是第一步的结果是 $\mathit{ema}^{(0)}=(1-\beta)\,\mathit{load}[0]$。

**状态机**：在第 $s$ 步递推完 EMA 之后，按**当前状态**只推进一个计数器：

$$
\begin{aligned}
\text{alive 态：}\ & \mathit{miss}\ \leftarrow\begin{cases}\mathit{miss}+1,&\mathit{ema}^{(s)}<\text{death}\\ 0,&\text{否则}\end{cases}
\qquad \mathit{miss}\ge\text{patience}\ \Rightarrow\ \text{转 dead}\\[4pt]
\text{dead 态：}\ & \mathit{back}\ \leftarrow\begin{cases}\mathit{back}+1,&\mathit{ema}^{(s)}\ge\text{revive}\\ 0,&\text{否则}\end{cases}
\qquad \mathit{back}\ge\text{revive\_patience}\ \Rightarrow\ \text{转 alive}
\end{aligned}
$$

三条字面口径：

- 判死用**严格小于** $\mathit{ema}<\text{death}$，复活用**大于等于** $\mathit{ema}\ge\text{revive}$。因此落在 $[\text{death},\text{revive})$ 里的 EMA 两边都不累计——这就是滞回带。
- 一步之内只走一条分支：判死与复活不会在同一步连续发生。
- 发生翻转时，**两个计数器一起归零**；因此翻转那一步 `streak` 一定是 0。

**账目**：`dead[s]` 记录第 $s$ 步**状态机推完之后**的状态；`streak[s]` 记录该状态下的活计数器（alive 是 `miss`，dead 是 `back`）；`first_dead[e]` 是首次转成 dead 的 step，从未死过为 $-1$；`state_changes[e]` 统计两个方向的翻转次数；`dead_steps[e]` 统计 `dead` 为真的步数。

## 返回

`(ema, dead, streak, first_dead, state_changes, dead_steps)`：

| 项 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `ema` | float64 `[S,E]` | 每步递推后的 EMA 负载 |
| `dead` | bool `[S,E]` | 每步结束时的死活状态 |
| `streak` | int64 `[S,E]` | 当前状态下的连击计数 |
| `first_dead` | int64 `[E]` | 首次判死的 step，从未判死为 `-1` |
| `state_changes` | int64 `[E]` | 状态翻转次数（判死与复活都算） |
| `dead_steps` | int64 `[E]` | 处于 dead 的步数 |

浮点用 `rtol=atol=2e-11`，其余张量逐位比较。

## 示例

$S=8$，$E=3$，`beta=0.5`，`death_threshold=1.0`，`revive_threshold=2.0`，`patience=2`，`revive_patience=2`：

```python
load = [[4, 0, 0],
        [4, 0, 0],
        [4, 0, 0],
        [4, 0, 3],
        [4, 0, 3],
        [4, 0, 3],
        [4, 0, 3],
        [4, 0, 3]]

# ema          s0..s7 = e0: [2, 3, 3.5, 3.75, 3.875, 3.9375, 3.96875, 3.984375]
#                         e1: all 0       e2: [0, 0, 0, 1.5, 2.25, 2.625, 2.8125, 2.90625]
# dead         [[0,0,0], [0,1,1], [0,1,1], [0,1,1], [0,1,1], [0,1,0], [0,1,0], [0,1,0]]
# streak       [[0,1,1], [0,0,0], [0,0,0], [0,0,0], [0,0,1], [0,0,0], [0,0,0], [0,0,0]]
# first_dead [-1, 1, 1]   state_changes [0, 1, 2]   dead_steps [0, 7, 4]
```

逐步说明：

- **e0** 每步 EMA 都远高于死亡线，`miss` 永远是 0，全程 alive。注意 s0 的 EMA 是 $0.5\times0+0.5\times4=2$——初值 0 让第一步天然偏低，但这里还没低过阈值。
- **e1** 从头到尾没有负载：s0 的 EMA 是 0（$<1$）记 1 次 miss，s1 记到 2 达到 `patience` → **判死**，该步 `streak` 归 0。此后 EMA 始终 0，永远够不到复活线 2，于是 s1..s7 共 7 步都是 dead。
- **e2** s0、s1 两次 miss 攒满 → s1 判死。s2 负载仍是 0 → `back` 不涨。s3 负载 3 让 EMA 升到 1.5：它落在滞回带 $[1,2)$ 里，**既不算 miss 也不算 revive**，`back` 保持 0。s4 的 EMA 2.25 才第一次算复活进度（`back=1`），s5 的 2.625 让 `back=2` 达到 `revive_patience` → **复活**，该步 `streak` 又归 0。它一共死了 s1..s4 四步。

`beta=0.5` 配整数负载，EMA 全是精确的二进制分数，可以手算核对。
