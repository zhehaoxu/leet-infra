# 缓存分配器：best-fit 块分配、扩段与相邻空闲合并

## 题目描述

深度学习框架不会每来一个张量就向驱动 `cudaMalloc` 一次（那要几十微秒且会同步设备），而是自己当二房东：一次向设备要大段显存（segment），再在段内切块分发。PyTorch 的 caching allocator（`CUDACachingAllocator`）就是这么干的——**块表 + best-fit + 释放后合并**。这套逻辑决定了三个工程上最常被问的量：

- **reserved**：实际向设备要了多少（框架视角的「占用」，`torch.cuda.memory_reserved`）；
- **碎片**：释放后能不能被后续申请原地复用，取决于相邻空闲块有没有合并；
- **扩段次数**：块表里找不到合适块时才向设备再要，扩一次就是一次昂贵的大分配。

本题用整数账把这些量完整模拟出来：给定一串 `alloc/free` 操作、对齐粒度与最小扩段量，输出整段模拟的峰值占用、扩段次数、结束时空闲块布局。

## 规则

整卡视作一条从地址 0 开始的连续地址空间。所有申请先向上取整到 `align` 的倍数记为 `need`（`align` 字节对齐），空闲块集合始终只保存**极大块**（相邻空闲块必须立即合并），并按 `start` 升序排列。

每次 `alloc`：

1. 计算 `need = ceil(nbytes / align) * align`；
2. 在满足 `size >= need` 的空闲块里做 **best-fit**：取 `size` 最小者，并列取 `start` 更小者；
3. 从选中块的头部切出 `need`，若块更大，剩余 `[start+need, start+size)` 仍为空闲块；
4. 若没有块装得下，先**扩段**：`grow = max(need, ceil(grow_unit/align)*align)`，`reserved` 增加 `grow`，新段 `[旧 reserved, 旧 reserved+grow)` 加入空闲集合（与相邻块合并），然后**重新做一次 best-fit**——注意它可能落在旧的小块上而不是新段上；
5. `reserved` 只增不减，`peak_reserved` 是它的历史最大值。

每次 `free`：把该块的 `[start, need)` 还回空闲集合，并与左右相邻的空闲块各合并一次（一个释放最多连成三段）。

输出六元组：

- `peak_reserved`、`final_reserved`：峰值/结束时 reserved 字节数；
- `num_grow`：扩段次数；
- `num_free_blocks`：结束时空闲块个数；
- `largest_free`：结束时最大空闲块字节数（无空闲块记 0）；
- `free_blocks`：结束时空闲块 `[[start, size], ...]`，按 `start` 升序。

## 示例

`align=512`、`grow_unit=4096`，操作序列与每步状态：

```
alloc a 1000 : need=1024，无空闲块 → 扩 4096（reserved=4096）
               a=[0,1024)            free=[[1024,3072]]
alloc b 3000 : need=3072，正好吃掉    b=[1024,3072)   free=[]
alloc c 600  : need=1024，无空闲块 → 扩到 8192
               c=[4096,1024)         free=[[5120,3072]]
free  b      : 两侧都不相邻            free=[[1024,3072],[5120,3072]]
alloc d 2000 : need=2048，两块同为 3072 → 取地址小的
               d=[1024,2048)         free=[[3072,1024],[5120,3072]]
free  c      : 左邻 [3072,1024)、右邻 [5120,3072)，两次合并成一块
                                     free=[[3072,5120]]
alloc e 5000 : need=5120，正好吃掉    e=[3072,5120)   free=[]
alloc f 512  : 扩到 12288             f=[8192,512)    free=[[8704,3584]]
free  a      : 与 [8704,3584) 不相邻  free=[[0,1024],[8704,3584]]
```

```
peak_reserved  = 12288
final_reserved = 12288
num_grow       = 3
num_free_blocks= 2
largest_free   = 3584
free_blocks    = [[0, 1024], [8704, 3584]]
```

## 约束

- 输入操作序列合法：`id` 唯一、`free` 只针对当前存活且未释放的块；`nbytes >= 1`（`free` 的第三位恒为 0）。
- `align >= 1`、`grow_unit >= 1`；两者都不要求是 2 的幂，但扩段量必须向上对齐。
- best-fit 的并列裁决固定为「size 小者优先，其次 start 小者优先」，不要用「第一个装得下的块」。
- 扩容后必须重新做 best-fit（而不是直接在新段上分配）。
- 返回的六个字段类型固定：前五项为 `int`，`free_blocks` 为 `list[list[int]]`。
