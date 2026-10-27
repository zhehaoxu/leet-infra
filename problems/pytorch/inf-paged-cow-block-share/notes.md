# 解题说明 · 页表间接寻址与写时复制

## 先把「页」和「token」两套下标分开

最容易写错的是 append 时判断「哪些页被写了」。正确做法是用 token 区间推页号：

```
first = old_tokens // block_size
last  = (old_tokens + arg - 1) // block_size
```

然后遍历 `first..last`。若按「已有页数」或「有没有共享」去判断，就会出现两种典型错误：把不该复制的整页都复制一遍，或者漏掉正好跨页时落在新页上的那一部分。

## 起止页要用旧 token 数算

`first` 用的是**追加前**的 token 数：`old_tokens // block_size`。当 `old_tokens` 恰好是 `block_size` 的整数倍时，`first` 指向的是「下一页」，也就是新页的起点，此时该页还不存在（`first >= len(table)`），走分配分支而不是复制分支。这正好对应「正好写满的页不算被写」。

## 共享判定只有一个条件：引用计数 > 1

物理块被复制的前提是「我要写它」，而它是否被别人共享只看引用计数。不要在序列侧再维护一个 `is_shared` 标记——那需要跟着 fork/free 同步更新，很容易和引用计数不一致。引用计数是唯一真相。

## fork 是纯记账

fork 之后两条序列的页表内容相同但已经是两个独立的 list；后续任一侧 append 会改自己的 list，不会影响对方。所以 `tables[dst] = list(tables[src])` 这一份浅拷贝是必须的，漏了就会两条序列共用一个 list，复制计数与页表全错。

## 最终快照要包含已经释放的序列

`free` 把序列从「存活表」里摘掉，但输出是按序列 id 排列的最终页表，被释放的序列仍占一行。实现上把「存活表」和「最终快照表」分开：append 改的是同一个 list 对象，快照自然同步；free 只删存活表项。

## 复杂度

每个操作 O(被触及的页数)；分配空闲块是 O(num_blocks) 线性扫描，用空闲链表可以做到 O(1)。真实 paged attention 里块数上万，线性扫描仅适用于教学规模。

## 相关题目

- `inf-paged-kv-block-alloc`：同一套块池，但只做「按序取块」的准入，没有页表间接与共享。
- `inf-kv-block-admission`：块预算不足时决定谁能进、谁继续等。
- `inf-lru-kv-cache-eviction`：块回收侧的替换策略。
