# PagedAttention 页表间接寻址与写时复制共享

## 问题

分页 KV cache 把每个序列的 KV 切成固定大小（`block_size` 个 token）的块，序列自己不再持有一整段连续显存，而是持有一张「逻辑页 → 物理块」的页表。物理块由全局块池分配，页表把逻辑页号翻译成块号，寻址变成两次查表。

页表带来两个副产品：

- **共享**：beam search 的多个候选、并行采样、前缀缓存命中，都可以让多条序列指向同一批物理块，每个物理块维护一个引用计数。fork 只是拷贝页表、把引用计数加一，显存一次都不用动。
- **写时复制（copy-on-write）**：一旦某条序列要往共享块里写，就必须先复制出一份私有块（引用计数各减一，新块置 1），再改新块。不复制就会把别的序列的 KV 一起改掉。

本题把这两件事放进同一个整数状态机里，输出页表、引用计数与复制次数。

## 任务

实现 `solve(num_blocks, block_size, ops)`：

- 输入：
  - `num_blocks`：物理块总数，块号 `0..num_blocks-1`；
  - `block_size`：每块能放的 token 数；
  - `ops`：`(K, 3)` int64，每行 `(kind, seq, arg)`：
    - `kind=0` create：新建序列 `seq`，写入 `arg` 个 token（`arg >= 1`）；
    - `kind=1` fork：目标序列 `arg` 复制 `seq` 的页表并共享其物理块；
    - `kind=2` append：给序列 `seq` 追加 `arg` 个 token（`arg >= 1`）；
    - `kind=3` free：释放序列 `seq` 持有的全部块引用（`arg` 忽略）。
- 输出：
  - `page_tables`：`(S, P)` int64，`S` 为最大序列 id + 1，`P` 为所有序列用过的最大页数，短页表用 `-1` 补齐；
  - `refcount`：`(num_blocks,)` int64，仿真结束时的引用计数；
  - `copies`：int64 标量，写时复制发生的次数。
- 分配规则：物理块总是取**编号最小**的空闲块；`free` 只递减引用计数，计数归零即回到空闲池。
- 序列 id 不可复用（析构后再 create 同号要报错）；物理块不足或操作非法时抛 `ValueError`。

## 示例

```
num_blocks = 6, block_size = 4

ops: create seq0 (6 token)  -> 2 页 -> 块 0,1        refcount = [1,1,0,0,0,0]
     create seq1 (4 token)  -> 1 页 -> 块 2          refcount = [1,1,1,0,0,0]
     fork   seq0 -> seq2                              refcount = [2,2,1,0,0,0]
     append seq2 (+3 token)  : 页 1 共享 -> 复制块 3
                               页 2 新开 -> 块 4       refcount = [2,1,1,1,1,0]  copies = 1
     fork   seq2 -> seq3                              refcount = [3,1,1,2,2,0]
     append seq3 (+1 token)  : 落在共享页 2 -> 复制块 5
                                                      refcount = [3,1,1,2,1,1]  copies = 2

page_tables = [[0,1,-1], [2,-1,-1], [0,3,4], [0,3,5]]
refcount    = [3, 1, 1, 2, 1, 1]
copies      = 2
```

## 约束

- **fork 不复制数据**：只把页表按位拷一份、引用计数加一；`copies` 不因 fork 增长。
- **只有被写入的共享页才复制**：判断依据是 `[旧 token 数, 旧 token 数 + arg)` 这段区间落在哪些逻辑页上，而不是「这个序列有没有共享块」。
- **正好写满的页不算被写**：若旧 token 数是 `block_size` 的整数倍，追加的第一个 token 落在**新页**上，此时直接分配新块，不触发写时复制。
- **`free` 之后仍要能看到它的页表**：输出是各序列的最终快照，被释放序列的行保留它释放前的页表。
- **`refcount` 必须回到 0 才算真空闲**：多序列共享时，一次 `free` 只减一。
- 页表用 `-1` 补齐而不是 `0`，因为块号 0 是合法块。
