# KV / Prefix Cache 的 LRU 驱逐

## 题目描述

显存有限，但热门前缀（system prompt、多轮对话历史、共享 few-shot 模板）对应的 KV block 值得尽量留住。**LRU（Least-Recently-Used）** 是 serving 里最常见的缓存驱逐策略：容量满时，淘汰**最久没被访问**的那个 block。命中的 block 会被"续命"——刷新为最近使用，从而让热数据长期驻留、冷数据自然沉底被淘汰。vLLM 的 prefix caching、各类推理网关的 KV block pool 回收都以 LRU 家族为骨架。

本题给出一段访问流（每次访问一个 block id），要求模拟一个容量为 `capacity` 的 LRU 缓存，统计命中/未命中次数，并给出结束时缓存内 block 的新旧次序。

## 规则

给定 `capacity`（int，缓存最多容纳的不同 block 数）、`accesses`（一维，按时间顺序的 block id 流）：

维护一个有序缓存，**队首 = 最近使用（MRU）**，队尾 = 最久未用（LRU）。逐个处理访问 `a`：

1. **命中**（`a` 已在缓存中）：`hits += 1`，把 `a` 移到队首（刷新为最近使用）
2. **未命中**（`a` 不在缓存中）：`misses += 1`
   - 若缓存已满（元素数 `== capacity`）：先移除**队尾**（LRU）
   - 把 `a` 插入队首
3. `capacity == 0` 时缓存恒空，每次都是未命中且无处驻留

返回三元组 `(hits, misses, final_order)`：`hits`/`misses` 为 int，`final_order` 为 `list[int]`，表示处理完全部访问后缓存内容、**按 MRU→LRU 排列**（队首在前）。

## 示例

```
capacity = 3
accesses = [1, 2, 3, 4, 1, 2, 5, 1, 2]      （队首 MRU ← → LRU 队尾）

 1: miss [1]
 2: miss [2,1]
 3: miss [3,2,1]
 4: miss 满→驱逐1  [4,3,2]
 1: miss 满→驱逐2  [1,4,3]
 2: miss 满→驱逐3  [2,1,4]
 5: miss 满→驱逐4  [5,2,1]
 1: hit  提到队首  [1,5,2]
 2: hit  提到队首  [2,1,5]

hits = 2, misses = 7, final_order = [2, 1, 5]
```

## 约束

- CPU；`capacity >= 0`；block id 为非负整数。
- 命中判定基于"当前是否在缓存内"，不是总出现次数。
- `final_order` 严格按 MRU→LRU；只含仍在缓存内的 block（至多 `capacity` 个）。
