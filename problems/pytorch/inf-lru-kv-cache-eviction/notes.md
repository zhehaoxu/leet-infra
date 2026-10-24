# 解题说明 · LRU 驱逐

## 双向链表 + 哈希表的 O(1) 实现

工业级 LRU 用 `HashMap<key, Node>` 配一条双向链表：命中时 `O(1)` 把节点摘出并头插，未满直接头插，满了 tail 节点即 LRU，`O(1)` 删除。本题规模小，`list` 的 `in / remove / insert(0)` 足够；oracle 用 `OrderedDict`（`move_to_end` + `popitem(last=False)`）走的是另一条实现路径，两者结果必须一致。

## 命中语义的坑

"命中"指**当前仍在缓存里**，而不是"这个 id 以前出现过"。被驱逐后再访问 = 未命中（要重新加载）。这正是 prefix cache 被抢占后需重算 prefill 的原因。

## 与真实 KV 驱逐的差别

真实系统的驱逐单位是 **block**，且带引用计数：正在被活跃请求使用的 block 不可驱逐（`refcount>0`），只有 free 队列里的 block 才在 LRU 候选中；还会区分"可驱逐的空闲 KV"与"仍在 decode 的热 KV"。本题把每个访问当作对某 block 的一次 touch，抽象掉引用计数，聚焦纯粹的 LRU 时序逻辑。

## capacity 边界

`capacity==0` 时缓存恒空，任何访问都无法驻留（全部未命中），`final_order==[]`；`capacity==1` 退化为"只有紧邻重复才命中"。这两个边界最易写错，测试专门覆盖。
