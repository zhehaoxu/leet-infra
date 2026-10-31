# KV Cache：SCOPE 解码期的窗口预算与不连续驱逐

## 题目描述

SCOPE（arXiv 2412.13649，ACL 2025）把 KV 压缩拆成两段：预填充期用 SnapKV 的聚类 topk 选出一段 prompt 缓存，**解码期**再套一套随步数长大的窗口调度。上游 `model/kv_utils.py::SnapKVCluster.update_kv_in_decoding` 按四种 `decoding_metric` 分支决定解码期行为，其中两个分支的预算由**类级计数器**驱动——计数器在「每一层的调用」里推进，而不是在每个 token 里推进，所以一个解码 token 会把它推 `num_hidden_layers` 格。

预填充段的 `pooling` / `topk` / 聚类折叠与 `kv-snapkv-pooling` 逐行重复，本题**只考解码调度**：给定解码期每步的注意力分数，问窗口预算怎么长、哪些步真的发生驱逐、跑完后每个 head 的缓存里留着哪些位置。

令 `dw` = 本步解码窗口预算，`rec` = 必留的近期窗口长度，`mcp` = 冻结的预填充前缀长度：

- `None`：立刻返回，不压，`dw` 记 0。
- `slide`：`dw = decoding_window_size` 恒定，缓存一满就重选。
- `adaptive`：`dw = rec + current_decoding_step // (delta * num_hidden_layers)`，随后 `current_decoding_step += 1`（每层调用一次），于是预算每 `delta` 个 token 才 +1。
- `discontinuous`：预算与 `adaptive` 完全同式，只在「预算门已过」之后再叠一道突发门 `jump_step` / `jump_layer`。`jump_step` 每层调用 +1，攒到 `delta * num_hidden_layers` 才真驱逐；驱逐时每层各动一次，动满 `num_hidden_layers` 次才把 `jump_step` 清零。落到 token 粒度就是「攒 `delta` 个 token 不动，第 `delta + 1` 个 token 每个 head 各驱逐一次」。

## 任务

实现 `solve(init_positions, attn, decoding_metric="discontinuous", num_hidden_layers=32, delta=15, max_capacity_prompt=512, decoding_recent_size=128, decoding_window_size=1024)`：

- `init_positions`：int64 一维张量，长度 `P >= 1`，预填充压缩**之后**缓存各槽位对应的位置。互不相同、都落在 `[0, P + T)`；顺序就是槽位顺序，**不要求升序**。
- `attn`：float 三维张量 `[T, H, P + T]`，`attn[t, h, p]` 是第 `t` 个解码步、第 `h` 个 head 的 query 给位置 `p` 的注意力权重，只有大小关系起作用。第 `t` 个新生成 token 的位置是 `P + t`。
- `decoding_metric`：`"None"` / `"slide"` / `"adaptive"` / `"discontinuous"` 四选一，区分大小写。
- 其余五项都是整数且 `>= 1`，并满足 `max_capacity_prompt <= P`。

每个解码步 `t`（`T` 步，`L` = 追加后的缓存长度）依次：

1. 把位置 `P + t` 追加到**每个** head 的缓存尾部。
2. 定预算：`slide` 取 `decoding_window_size`；`adaptive` / `discontinuous` 取 `rec + cds // (delta * num_hidden_layers)` 并把 `cds` 加 `num_hidden_layers`（`cds` 从 0 起，与上游「每层调用 +1」等价）；`None` 跳过调度、预算记 0 且不驱逐。
3. 判门：`L >= max_capacity_prompt + dw` 才可能驱逐。`discontinuous` 再加突发门——`jump >= delta * num_hidden_layers` 时把 `jump` 清零并允许驱逐，否则 `jump += num_hidden_layers` 且本步不驱逐。`slide` / `adaptive` 无此门。
4. 驱逐：`keep = dw - rec`；候选区 = `slots[max_capacity_prompt : L - rec]`（冻结前缀与尾部 `rec` 个槽位无条件保留、不参与竞争）；按 `attn[t][h][position]` **降序**取 `keep` 个，同分时取候选区里靠前的；新缓存 = 冻结前缀 + 选中的（按分数降序落位，不排回位置升序）+ 尾部。每个 head 用各自的分数行独立选。
5. 记录本步结束后的长度，以及本步是否真的驱逐。

返回 4 键字典，值一律 int64：`window_sizes` 形状 `[T]`（每步的 `dw`）、`cache_lens` 形状 `[T]`（每步结束后的长度，各 head 等长，取 head 0）、`evict_steps` 形状 `[E]`（发生驱逐的步号，升序）、`kept_positions` 形状 `[H, 最终长度]`（按槽位顺序）。

`decoding_metric` 取值非法、五个整数任一不是 `int`（`bool` 也算不合法）或 `< 1`、`max_capacity_prompt > P`、`attn` 不是三维浮点张量、`attn` 的宽不等于 `P + T`、`T < 1`、`init_positions` 不是一维 / 为空 / 有重复 / 越界、`slide` 且 `decoding_window_size <= decoding_recent_size` 时抛 `ValueError`。

## 示例

```
P = 6, T = 6, H = 1, num_hidden_layers = 2, delta = 2
max_capacity_prompt = 4, decoding_recent_size = 3, decoding_window_size = 4
init_positions = [0, 1, 2, 3, 4, 5]
attn[t][0], columns = positions 0..11
  t=0   0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16
  t=1   0.16 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15
  t=2   0.10 0.11 0.12 0.13 0.20 0.50 0.14 0.15 0.16 0.17 0.18 0.19
  t=3   0.19 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.21 0.22
  t=4   0.22 0.19 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.20
  t=5   0.30 0.31 0.32 0.33 0.34 0.60 0.25 0.70 0.10 0.40 0.20 0.15

solve(..., "discontinuous", ...)
  window_sizes   = [3, 3, 4, 4, 5, 5]   # dw = 3 + floor(t / 2), cds adds 2 per token
  cache_lens     = [7, 8, 8, 9, 10, 9]
  evict_steps    = [2, 5]               # t = 0, 1 only feed the jump gate
  kept_positions = [[0, 1, 2, 3, 7, 5, 9, 10, 11]]
  # t=5 keep=2 over [5, 6, 7, 8]: 0.70 then 0.60 -> positions 7, 5 in score order

solve(..., "slide", ...)
  window_sizes = [4, 4, 4, 4, 4, 4]  cache_lens = [7, 8, 8, 8, 8, 8]
  evict_steps  = [1, 2, 3, 4, 5]     kept_positions = [[0, 1, 2, 3, 7, 9, 10, 11]]

solve(..., "adaptive", ...)
  window_sizes = [3, 3, 4, 4, 5, 5]  cache_lens = [7, 7, 8, 8, 9, 9]
  evict_steps  = [0, 1, 2, 3, 4, 5]  kept_positions = [[0, 1, 2, 3, 7, 6, 9, 10, 11]]

solve(..., "None", ...)
  window_sizes = [0, 0, 0, 0, 0, 0]  cache_lens = [7, 8, 9, 10, 11, 12]  evict_steps = []
  kept_positions = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
```

```
tie break: P = 4, T = 3, H = 1, num_hidden_layers = 1, delta = 1
max_capacity_prompt = 2, decoding_recent_size = 1, metric = "adaptive"
attn rows, columns = positions 0..6
  t=0   0.40 0.10 0.30 0.20 0.15 0.25 0.35
  t=1   0.25 0.35 0.30 0.30 0.15 0.20 0.40
  t=2   0.05 0.45 0.20 0.30 0.40 0.40 0.10

window_sizes = [1, 2, 3]  cache_lens = [3, 4, 5]  evict_steps = [0, 1, 2]
kept_positions = [[0, 1, 4, 5, 6]]   # t=2 keep=2 over [4, 5], both score 0.40
```

## 约束

- **计数器按「层调用」而非「token」推进**：`cds` 每个 token 恰好 +`num_hidden_layers`，于是 `dw = rec + floor(t / delta)`；写成每 token +1 会让预算长得快 `num_hidden_layers` 倍。
- **突发门的 `jump` 同样按层调用走**：`delta = 2`、层数 = 2 时是「攒 2 个 token、动 1 个」，`evict_steps` 落在 `t = 2, 5, 8, ...`。
- **预算门与突发门串行**：`jump` 只在 `L >= mcp + dw` 的调用里推进——门没过时既不驱逐也不攒 `jump`，这就是 `discontinuous` 与「无条件每 `delta + 1` 步动一次」的区别。
- **`keep = dw - rec` 可以为 0**：`adaptive` 在 `dw == rec` 时照样触发驱逐，但清空整段候选区，只留冻结前缀 + 尾部 `rec` 个，`cache_lens` 掉到 `mcp + rec`（第二个例子的 `t=0` 就是这种）。
- **`init_positions` 的槽位顺序会被保留**：冻结的是「前 `mcp` 个槽位」而不是「位置小于 `mcp`」，压缩过的缓存里位置本就无序。
- 判分逐元素比列表，不做集合比较；`kept_positions` 允许 `..., 7, 5, 9, ...` 这种分数序排列。`T` 上限 400，`H` 上限 8，`P` 上限 512。
