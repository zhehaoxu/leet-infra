# 上下文压缩：GemFilter 早层注意力输入裁剪

## 题目描述

GemFilter（2409.17422）的思路：与其在每一层都算完整注意力，不如只用**第 13 层**的 QK 内积快速判断「输入里哪些 token 与当前 query 相关」，把选中 token 之外的全部裁掉，后续层只看剩下的 KV。打分函数 `standard_dis_index` 一共四步：GQA 展开后的 QK 内积、只看**最后一个** query 位置、把各头的分数**求和**（不是平均）、再过一维均值池化平滑后取 top-k。这四步里每一步换一种「想当然」的实现都会得到另一份 top-k。本题给定选择层的 K、Q（RoPE 后）与 GQA 分组数，复现这份打分与选择。

## 任务

实现 `solve(key_states, query_states, num_key_value_groups, topk=1024, kernel_size=5, norm=1.0)`：

- `key_states`：`[kv_heads][n][d]` 的数值嵌套列表，选择层的 K（RoPE 后）；`n >= 1`、`d >= 1`；
- `query_states`：`[num_heads][q_len][d]` 的数值嵌套列表，同一层的 Q；**只有最后一个 query 位置的向量参与打分**；
- `num_key_value_groups`：GQA 分组数 $g \ge 1$，须满足 `num_heads == kv_heads * g`；K 先按 `repeat_kv` 复制到每头（**每个 KV 头连续复制 g 份**，即 `repeat_interleave` 语义，不是 `repeat` 平铺）；
- `topk`：$\ge 1$ 的整数，实际取 `k_eff = min(topk, n)`；
- `kernel_size`：正奇数（上游默认 5），均值池化 `padding = kernel_size // 2`、`stride = 1`；
- `norm`：正实数（上游默认 1）；`norm != 1` 时 `distances` 额外除以 `norm`，`indices` 不变。

返回 `dict`，键为 `indices`、`distances`、`pooled`。

## 精确规则

1. **GQA 展开**：`K_expanded[h]`，其中头 $h = c \cdot g + t$（$0 \le t < g$）使用 `key_states[c]`——第 $c$ 个 KV 头连续占据 $g$ 个查询头。
2. **取最后 query**：`q_last[h] = query_states[h][-1]`，形状 `[num_heads][d]`。
3. **内积与头间求和**：$s_j = \sum_{h=0}^{H-1} \langle q_{last}[h],\ K_{expanded}[h][j] \rangle$，得长度 $n$ 的分数序列（**求和**而非平均，头数会改变量纲）。
4. **均值池化**：`pooled[j] = (Σ_{t=j-r}^{j+r} s_t) / kernel_size`，越界下标按 **0 填充**参与求和、但**分母恒为 kernel_size**（`avg_pool1d` 的 `count_include_pad=True` 默认行为）；$r = \lfloor kernel\_size / 2 \rfloor$。
5. **top-k**：按 `pooled` 值降序取前 `k_eff` 个；**同值按下标升序**。`indices` 按该顺序输出；`distances` 为对应的 `pooled` 值（`norm != 1` 时除以 `norm`）；`pooled` 原样返回完整序列。

## 示例

```text
2 个 KV 头（g=2，共 4 个查询头），n=6，d=2：
  K0 = [[1,0],[0,1],[1,1],[2,0],[0,2],[1,-1]]
  K1 = [[0,1],[1,1],[2,0],[0,2],[1,0],[1,1]]
  Q 的最后一个 query 位置（4 头）= [(1,1), (2,1), (0,1), (1,0)]

interleave 展开后：头 0、1 用 K0（配 (1,1)、(2,1)），头 2、3 用 K1（配 (0,1)、(1,0)）
  内积按头：
    头0 (K0,(1,1)): [1, 1, 2, 2, 2, 0]
    头1 (K0,(2,1)): [2, 1, 3, 4, 2, 1]
    头2 (K1,(0,1)): [1, 1, 0, 2, 0, 1]
    头3 (K1,(1,0)): [0, 1, 2, 0, 1, 1]
  头间求和 sums = [4, 4, 7, 8, 5, 3]
kernel_size=3 的均值池化（零填充）：
  pooled = [8/3, 5.0, 19/3, 20/3, 16/3, 8/3]
topk=2 → indices = [3, 2]，distances = [20/3, 19/3]
topk=10 → k_eff=6 全取，indices = [3, 2, 4, 1, 0, 5]（j0 与 j5 同值 8/3，j0 在前）
norm=2 → indices 不变，distances = [10/3, 19/6]
```

若把 interleave 错写成 `repeat`（头 0、2 用 K0，头 1、3 用 K1），sums 会变成 `[3,5,9,8,7,3]`，整份 top-k 都不一样——这是本题的核心陷阱。

## 约束

- 所有数值均为有限实数；嵌套列表各头 token 数一致、向量维度一致且 `Q`、`K` 的 `d` 相同。
- `num_heads == kv_heads * num_key_value_groups`。
- `topk >= 1`；`kernel_size` 为正奇数；`norm > 0`。
- 不修改输入；所有非法输入抛 `ValueError`。

## 与相邻题目的边界

- `ctx-llmlingua2-word-compress` / `ctx-longllmlingua-doc-budget` 压的是 **prompt 里的词与文档**（分类概率 / 文档 PPL）；本题压的是**预填输入 token**（早层注意力打分），且分数只在选择层算一次、全层复用。
- `mm-fastv-visual-token-prune` 是视觉 token 的**末层**逐层裁剪；本题是文本输入 token 的**早层**一次性选择。
- `kv-snapkv-pooling` 池化的是 **KV cache**（解码期、按观察窗口）；本题池化的是**输入 token 打分**（预填期、无 cache）。
