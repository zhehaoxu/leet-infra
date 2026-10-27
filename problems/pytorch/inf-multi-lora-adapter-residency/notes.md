# 解题说明 · multi-LoRA adapter 驻留

## 两个集合，一个队列

这个状态机里有三种角色，混淆任意两个就会错：

- **resident（集合）**：当前在显存里的 adapter，用来判命中。包含 pinned 与非 pinned。
- **lru（有序表）**：**只有非 pinned 的 resident** 在里面，头部最久未用。pinned 永远不进这个表。
- **pinned（集合）**：初始就在 resident 里，且永远不会被选中换出。

判命中的条件是 `a in resident`；判能否换入的条件是 `lru 非空`。这两件事必须分开写。

## 命中要动队列，但 pinned 不用

命中 pinned adapter 时不需要刷新任何东西（它不在 lru 里）。命中非 pinned 时必须把它从 lru 里摘下来再 append 到尾部——不刷新就退化成 FIFO，`1,2,1,3` 这种序列会换错人。

## 换出必须在加载之前

顺序是「先腾位置再加载」，不是「先加载再腾」。若先加载，中间会短暂超出 `capacity`，峰值 `peak_bytes` 会算高一个 adapter，而且如果先加载时发现超限再回滚，逻辑会变得很绕。

## 拒绝不等于「什么都不做」

被拒绝的请求确实不改任何状态，但 `evicted` 要输出 `-1`、`hit` 要输出 `False`，两者都必须按位对齐请求序列，不能跳过不写（否则后续下标整体错位）。

## `peak_bytes` 的初始值

峰值初值是 pinned 的字节和，不是 0：仿真第 0 步之前 pinned 就已经在显存里了。漏掉这一点，只在 `pinned` 非空且从未发生换入的用例上才会暴露——恰好是最容易漏测的组合。

## 与 KV 侧 LRU 的区别

`inf-lru-kv-cache-eviction` 的对象是 KV 块，容量按块数算、没有 pin；本题的对象是 adapter，容量按个数算、有 pin、并且显存账按字节而不是按个数。两者的状态机骨架一样，但这两处差异足以让「照抄」的实现全错。

## 复杂度

每步最坏 O(capacity)（找 LRU 需要 O(1)，但求和算 bytes 是 O(capacity)）。真实实现给每个 adapter 维护字节数累加器，把每步降到 O(1)。

## 相关题目

- `inf-lru-kv-cache-eviction`：KV 块上的同类驱逐，无 pin。
- `peft-lora-memory`：训练侧的 adapter 参数/显存公式，与运行时的驻留账互补。
- `inf-cache-aware-routing`：adapter 亲和性也可以作为路由代价的一项。
