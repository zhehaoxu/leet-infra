# KV Cache：KVLink 段落掩码与 link token 布局

## 题目描述

KVLink（2410.06511）把长上下文拆成若干文档，**各自独立预计算 KV**：既然文档之间不互相注意，每个文档的 KV 就可以离线算好、跨请求复用。为了补回跨文档信息，它在每段末尾插入 `reencode_num` 个训练好的 **link token**（README 默认 5 个），并让这些 token 属于 **segment 0**；最终问题与答案也在 segment 0。于是掩码只需要满足三条：

1. 先按 causal 遮住未来；
2. 只有同 segment 的 token 能互看；
3. **segment 0 的目标**可以看所有合法的 source（link token 因此成为「全局摘要」，只被 segment-0 的目标读到，文档之间仍然隔离）。

pad 位置用 `segment = -1` 标记，整行整列都遮掉。本题实现这套布局与掩码，即上游 `make_segment_mask` 的 CPU 语义。

## 任务

实现 `solve(chunk_lengths, prefix_len=0, reencode_num=5, pad_to=None)`：

- `chunk_lengths`：`list[int]`，每个文档 chunk 的 token 数，均 ≥ 1；
- `prefix_len`：开头的系统提示 token 数，≥ 0，全部属于 segment 0；
- `reencode_num`：每段末尾的 link token 数，≥ 0；
- `pad_to`：可选，给出时把序列在**尾部**补 `-1` 到该长度，必须 ≥ 真实长度。

布局（真实长度 \(L_0=\texttt{prefix\_len}+\sum_j(\ell_j+r)\)）：

```
segment_ids = [0]*prefix_len
for j, len_j in enumerate(chunk_lengths):
    segment_ids += [j+1]*len_j + [0]*reencode_num
if pad_to is not None:
    segment_ids += [-1]*(pad_to - L0)
```

返回 `dict`：

- `segment_ids`：`(L,)` int64；
- `bias`：`(L, L)` float32，`0` 表示可见、`-inf` 表示遮住。规则：
  1. 严格上三角（`j > i`，左上对齐）恒为 `-inf`；
  2. 任一端 `segment == -1` 的位置为 `-inf`（pad 行、pad 列全遮，含对角）；
  3. 目标 `segment == 0` 时，合法 source 全可见；
  4. 其余目标只能看见**同 segment** 的合法 source。

非法输入（非 list/tuple、chunk 长度 < 1、prefix/link 为负、`pad_to` 小于真实长度、bool 冒充 int、总长为 0）抛 `ValueError`。

## 示例

```
chunk_lengths = [2, 1]
prefix_len = 1
reencode_num = 1

segment_ids = [0, 1, 1, 0, 2, 0]
#                 ↑        ↑     ↑
#              prefix   doc0 的  doc1 的
#                        link    link

每行可看的列（bias[row, col] == 0）：
row 0 (seg 0): [0]
row 1 (seg 1): [1]
row 2 (seg 1): [1, 2]
row 3 (seg 0): [0, 1, 2, 3]
row 4 (seg 2): [4]
row 5 (seg 0): [0, 1, 2, 3, 4, 5]
```

第 3 行是 doc0 的 link token：它属于 segment 0，能看到它之前的所有 token（0..3），所以能总结 doc0。第 4 行是 doc1 的第一个 token，只能看自己（col 4）——它看不到 doc0，也看不到任何 link token，这正是「文档 KV 可独立复用」的前提。第 5 行是最后的 segment-0 目标，能看全部 6 个位置。

## 约束

- `bias` 只取 `0.0` 与 `-inf` 两种值，dtype 固定 float32；测试按逐元素相等比较，不做容差。
- 段落编号按输入顺序从 1 开始；prefix 与所有 link token 都是 0，且 padding 只能是尾部 `-1`。
- 掩码的「全局」只给 **segment 0 的目标**，不给 segment 0 的 source：非 0 目标看 link token 必须被遮住。
- 本题不产出 `input_ids`/`position_ids`：上游拼接后的位置就是默认的 `arange`，link token 的 id 是训练好的特殊 token，均不在本题范围内。
- 不含训练、不含 `construct_biased_attention_matrix`（该文件在上游仓库里缺失），只复刻 `make_segment_mask`。
