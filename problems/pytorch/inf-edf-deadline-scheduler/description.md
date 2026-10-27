# SLO 感知的 EDF 调度与等待老化

## 问题

推理服务给请求配的 SLO 通常是一个首 token 或整条请求的**截止步**。调度器每一轮只能让有限个请求各前进一步，于是选谁就成了核心决策：按最早截止期（EDF）能让整体超期数最少，但它是「绝对优先」的——只要不断有更紧的 deadline 到达，一个 deadline 很松的长请求可能永远轮不到，这就是饿死。

工程上常见的补丁是**老化（aging）**：连续被跳过太多次的请求升级优先级，插到队首。本题把两者叠起来，输出每个请求的开始/完成步、超期计数和被老化提前的次数。

## 任务

实现 `solve(arrival, num_tokens, deadline, max_batch, max_wait, max_steps)`：

- 输入：`arrival` `(N,)` 到达步；`num_tokens` `(N,)` 需要跑的 decode 步数；`deadline` `(N,)` 要求在这一步（含）之前完成；`max_batch` 每步最多同时跑几个；`max_wait` 连续被跳过多少步后升级；`max_steps` 模拟步数上限。
- 每一步：
  1. 就绪集 = 已到达且未完成的请求；
  2. `wait >= max_wait` 的请求整批放到最前面（老化）；
  3. 两个组内部都按 `(deadline 升序, id 升序)` 排序；
  4. 取前 `max_batch` 个各跑一步（`num_tokens` 减一），跑完的记下完成步；
  5. 被选中的 `wait` 归零，就绪但没被选中的 `wait` 加一。
- 输出：`start` `(N,)` 首次运行步（-1 表示从未运行）、`finish` `(N,)` 完成步（-1 表示未完成）、`missed` int64 标量（完成步晚于 deadline，或根本没完成）、`promotions` int64 标量（因老化而被选中的总次数）。

## 示例

```
arrival = [0, 0], num_tokens = [3, 1], deadline = [100, 100]
max_batch = 1, max_wait = 2, max_steps = 6

step 0  wait = [0, 0]，都不老化；同 deadline 取小 id -> 跑 r0。r1 的 wait 变 1
step 1  wait = [0, 1]，都不老化 -> 仍跑 r0。r1 的 wait 变 2
step 2  r1 的 wait = 2 >= max_wait -> 老化插到队首 -> 跑 r1，r1 完成   promotions = 1
step 3  r0 跑完最后一步 -> 完成

start      = [0, 2]
finish     = [3, 2]
missed     = 0
promotions = 1
```

## 约束

- **老化是整批前置，不是插到某个位置**：所有 `wait >= max_wait` 的请求排在最前面，彼此之间仍按 `(deadline, id)` 排。
- **老化阈值是 `>=`**：`max_wait = 2` 表示「已经被跳过 2 步」就升级，不是「跳过 3 步」。
- **`wait` 只在「就绪但没被选中」时加一**：还没到达的请求不累计；被选中的请求直接归零。
- **`promotions` 数的是入选次数**，若一步里 3 个老化请求同时入选就计 3。
- **`missed` 的口径**：`finish > deadline` 或 `finish == -1` 都算超期，包含「因为 `max_steps` 不够而没跑完」的情况。
- **`max_batch >= 1`**：每步至少能跑一个，否则无法保证推进。
