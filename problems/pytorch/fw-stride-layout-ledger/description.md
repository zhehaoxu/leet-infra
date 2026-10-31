# 布局账：shape/stride/offset 与 view-or-copy 判定

## 题目描述

PyTorch 的张量不是「一块数据」，而是「一段存储 + 三个元数据」：`shape`、`stride`、`storage_offset`。官方 *View operations* 文档把操作分成三类——一定返回视图（`transpose`/`permute`/`narrow`/`select`/切片/`unsqueeze`/`squeeze`/`expand`/`view`…）、可能返回视图也可能返回新张量（`reshape`/`flatten`）、以及一定复制（`contiguous` 在需要时）。为什么 `t.expand(3,4).reshape(12)` 要复制而 `t.permute(2,0,1).reshape(4,6)` 不用复制，是入门阶段最集中的一片「看不懂就得靠猜」的知识，也是调试显存翻倍、`is_contiguous()` 意外为假、`view()` 报 `at least one dimension spans across two contiguous subspaces` 的直接原因。

本题给定一个基张量的元数据和一串视图操作，要求**逐步**算出每次操作后的 `shape`、`strides`、`storage_offset`、`is_contiguous()`，以及这一步是视图（`view`）还是复制（`copy`）。没有任何张量数值参与。

## 规则

实现 `solve`（参数形状、操作集与非法输入的契约见 starter docstring）。负数维一律先 `+ ndim` 归一化。

1. `transpose`/`permute` 只重排 `shape` 与 `strides` 的分量，`offset` 不变。
2. `unsqueeze(d)` 插入长度 1 的维，其步长取 `size[d] * stride[d]`；`d == ndim`（追加到末尾）时步长为 1。于是 `(3,4)/(4,1)` 在 0/1/2 三个位置插入分别得到 `(12,4,1)`、`(4,4,1)`、`(4,1,1)`。
3. `squeeze` 只是把分量删掉，**不重算**留下的步长；指定的维长度不是 1 时整步为空操作（0 维张量上也是空操作）。
4. `narrow`/`select`/`slice` 都推进 offset：`offset += start_norm * stride[d]`，`start_norm` 先 `+= size[d]`（若为负）再夹紧到 `[0, size[d]]`；`slice` 的 end 先 `+= size[d]` 再夹紧到 `[start_norm, size[d]]`，长度 `= ceil((end - start_norm)/step)`、该维步长 `*= step`；长度为 0 时步长按同一式给出、offset 仍推进。`select` 相当于取长度 1 后删掉该维（所以内维被 `select` 之后很容易不再连续）。
5. `expand(sizes)` 右对齐逐维：长度相等（含 1 对 1）保留原步长；原长 1 而目标 > 1 置 0；其余不合法。多出的前置新维步长为 0，除非其长度为 1，此时取 `size[i+1] * stride[i+1]`（若它已是最后一维则取 0）。
6. `reshape`/`flatten` 的 view 判定（ATen `computeStride`）：把旧形状从最内维向外切成若干**连续子空间**——设本段已累乘的长度之积为 `tensor_numel`、段基步长为 `base`，则维 `i` 与 `i+1` 同段当且仅当 `size[i] == 1` 或 `stride[i] == tensor_numel(i+1 起) * base`；每段必须被目标形状从最内维向外精确吃满，段内第 j 个目标维步长 = 它之前在本段内累乘的长度 × `base`，长度为 1 的目标维不消耗段长度但按同一累乘取步长。任一段吃不满或目标维有剩余即判 `copy`。`copy` 之后步长是目标形状的标准连续步长（长度 ≤ 1 的维不推进累乘）、offset 归 0。总长度为 0 时 `reshape` 恒为 view：与旧形状相同则照抄旧步长，否则给标准连续步长；0 维基张量的 `reshape` 一律给全 1 步长。`flatten(a,b)`：`a == b` 是空操作，0 维时等价于 reshape 到长度 1 的一维，否则按前述判定合并后的形状。
7. `contiguous` 在当前布局已 C 连续时是 `view`（元数据原样），否则是 `copy`。C 连续判定：ndim == 0 或总长度为 0 恒为真；否则从最内维向外，期望步长从 1 起，每维要求 `stride == expected` 或 `size == 1`，然后 `expected *= size`。长度为 1 的维的步长虽然「自由」，torch 仍会按规则 2/5/6 填成确定值，本题要求连这些填充值也一致。

## 示例

每行 `操作 -> shape/strides/offset contig=<bool> <view|copy>`，`shape/strides` 写作 `(d1,d2,...)`，基元数据写在每条链的 `base` 行。

```
base (3,4)/(4,1)/0
  transpose(1,0)    -> (4,3)/(1,4)/0     contig=False view
  narrow(0,1,2)     -> (2,3)/(1,4)/1     contig=False view
  reshape(6)        -> (6)/(1)/0         contig=True  copy
  reshape(2,3)      -> (2,3)/(3,1)/0     contig=True  view

base (2,3,4)/(12,4,1)/0
  permute(2,0,1)    -> (4,2,3)/(1,12,4)/0  contig=False view
  reshape(4,6)      -> (4,6)/(1,4)/0       contig=False view
  contiguous        -> (4,6)/(6,1)/0       contig=True  copy

base (3,4)/(4,1)/0
  unsqueeze(1)      -> (3,1,4)/(4,4,1)/0      contig=True view
  unsqueeze(0)      -> (1,3,1,4)/(12,4,4,1)/0 contig=True view
  squeeze(None)     -> (3,4)/(4,1)/0          contig=True view

base (1,4)/(4,1)/0
  expand(3,4)       -> (3,4)/(0,1)/0     contig=False view
  expand(2,3,4)     -> (2,3,4)/(0,0,1)/0 contig=False view
  reshape(24)       -> (24)/(1)/0        contig=True  copy
  contiguous        -> (24)/(1)/0        contig=True  view

base (2,3,4)/(12,4,1)/0
  slice(1,0,3,2)    -> (2,2,4)/(12,8,1)/0 contig=False view
  flatten(0,1)      -> (4,4)/(4,1)/0      contig=True  copy
  flatten(0,1)      -> (16)/(1)/0         contig=True  view

base (2,3,4)/(12,4,1)/0
  select(1,-1)      -> (2,4)/(12,1)/8       contig=False view
  unsqueeze(0)      -> (1,2,4)/(24,12,1)/8  contig=False view
  narrow(1,1,1)     -> (1,1,4)/(24,12,1)/20 contig=True  view

base (4,3)/(1,4)/0
  reshape(12)       -> (12)/(1)/0      contig=True  copy
  reshape(2,6)      -> (2,6)/(6,1)/0   contig=True view
  contiguous        -> (2,6)/(6,1)/0   contig=True view

base (2,0,4)/(8,4,1)/0
  transpose(0,2)    -> (4,0,2)/(1,4,8)/0 contig=True view
  reshape(0)        -> (0)/(1)/0          contig=True view
  unsqueeze(0)      -> (1,0)/(0,1)/0      contig=True view
```

推演第 2 条链的第 2 步：`(4,2,3)/(1,12,4)` 从最内维向外，`stride[1] = 12` 恰等于 `size[2] * base = 3 * 4`，于是维 1、维 2 同属一段（段长 6、段基步长 4）；而 `stride[0] = 1` 不等于 `6 * 4`，维 0 自成一段。目标 `(4,6)` 的 6 吃满第一段给出步长 `1*4 = 4`，4 吃满第二段给出步长 `1*1 = 1` → 视图 `(4,6)/(1,4)`。第 4 条链的 `reshape(24)`：最内段长 4（段基步长 1），目标只有一个长度 24 的维，一段吃不满即判 copy；随后 `contiguous` 已经连续，因而是 view。

## 约束

- 输入规模：基元数据 `ndim ≤ 5`、每维长度 ≤ 16、`stride ≤ 64`、`offset ≤ 64`，操作序列 ≤ 8 步；输出整数的大小不受限。
- 输出全是整数与布尔，精确对拍，无浮点比较、无随机数。
- 题目保证操作序列每一步都能在真实 torch 上执行成功（越界、尺寸冲突等不会出现），你只需按契约对**非法输入**抛 `ValueError`。
- 不得用 torch（或 numpy）构造张量再读它的 `shape/stride/storage_offset/is_contiguous` 来查表得答案。
