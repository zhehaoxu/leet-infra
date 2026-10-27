# multi-LoRA 热加载：adapter 驻留 LRU 与显存账

## 问题

一个 base model 服务几十上百个 LoRA adapter 时，GPU 上不可能全放得下。S-LoRA / Punica / vLLM 的 adapter manager 的做法是一个**有限容量的驻留池**：请求带着 adapter id 过来，池里有就直接用（命中），没有就换入（缺页），池满时按 LRU 换出最久未用的那个。

两个工程细节决定了它和教科书 LRU 的差别：

- **pin**：少数热门 adapter 被钉住常驻，永不换出。它们占掉固定份额，剩下的容量才由 LRU 支配。
- **拒服务**：如果池子被 pinned 占满，而请求要的 adapter 又不在池里，这一轮只能拒绝——而不是把请求排队等某个 pinned adapter 消失（它永远不会消失）。

本题把这个状态机跑完，输出命中、换出序列、加载次数、拒绝次数与峰值显存。

## 任务

实现 `solve(requests, adapter_bytes, capacity, pinned)`：

- 输入：`requests` `(M,)` 按到达顺序的 adapter id；`adapter_bytes` `(A,)` 各 adapter 的显存占用；`capacity` 同时驻留上限；`pinned` `(P,)` 常驻 adapter（仿真开始时已加载）。
- 处理每个请求 `a`：
  - `a` 在驻留池 → 命中，刷新其最近使用次序；
  - 否则缺页：池未满直接加载；池满则换出最久未用的**非 pinned** adapter 再加载；
  - 池满且驻留者全是 pinned → 该请求计入 `rejected`，状态不变。
- 输出：`hit` `(M,)` bool、`evicted` `(M,)` int64（本步换出的 adapter，-1 表示无）、`loads` int64（动态加载次数，不含初始 pinned）、`rejected` int64、`peak_bytes` int64（驻留 adapter 字节和的峰值，含 pinned）。

## 示例

```
adapter_bytes = [100, 200, 300, 400]
capacity = 2, pinned = [0], requests = [1, 1, 2, 2, 1]

初始驻留 = {0}（pinned），占用 100

r0: 请求 1  缺页 -> 加载，驻留 {0,1}   loads=1  peak=300
r1: 请求 1  命中                        hit
r2: 请求 2  缺页 -> 换出 LRU=1（非 pinned 只有它），加载 2   evicted=1  loads=2  peak=400
r3: 请求 2  命中
r4: 请求 1  缺页 -> 换出 2，加载 1                          evicted=2  loads=3  peak=400

hit     = [F, T, F, T, F]
evicted = [-1, -1, 1, -1, 2]
loads   = 3
rejected = 0
peak_bytes = 400
```

## 约束

- **pinned 只增不减**：它们初始就在池里，全程不参与 LRU 队列，永远不会被换出。
- **LRU 的「最近使用」在命中时也要刷新**：命中不刷新，就等于 FIFO，会在 `1,2,1,3` 这类序列上换错对象。
- **换出目标是「最久未用的非 pinned 者」**，不是「最久未用的者」——若把 pinned 也算进候选，会直接把常驻 adapter 踢掉。
- **拒绝是终态**：被拒绝的请求不会排队，也不会改变任何状态，只让 `rejected` 加一。
- **`peak_bytes` 包含 pinned 的初始占用**，即仿真第 0 步的峰值就不是 0。
- **`pinned` 不能重复、不能越界、数量不能超过 `capacity`**，否则抛 `ValueError`。
