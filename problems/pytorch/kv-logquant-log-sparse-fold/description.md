# KV Cache：LogQuant 全精度索引的分层折叠

## 题目描述

LogQuant（arXiv 2503.19950，ICLR 2025 SLLM workshop）把 KV 分成两坨：一坨 2-bit 量化，一坨**全精度**。全精度那坨的总量恒为 `3 * window_length`（下称 `W`），再劈成两段——一段是**分层抽样**出来的稀疏位置 `log_sparse_index`，一段是最近 `W`~`2W` 个位置的滑窗 `local_index`。滑窗满了就把**最旧的 `W` 个**并进抽样段并对整段取偶数位，于是越老的位置被抽得越稀，这就是「log-distributed」。

量化本体（quanto qint2/qint4、`q_group_size=64`、axis 同 KIVI）与 `quant-kivi-kv-asym` 重复，本题**只考索引调度**：给定预填充长度、解码步数与 `W`，问各阶段的 `log_sparse_index` / `local_index` 里到底是哪些 token 下标。

上游 `src/LogQuant.py::LogQuantizedCache.update` 把这两段维护在 `layer_idx == 0` 的那次调用里，预填充与解码走两条不同的阈值：

1. **预填充**（一次进 `L > 1` 个 token）：`local = range(L)`，只要 `len(log + local) >= 3W` 就折叠一次——`log = (log + local)[0 : 2W : 2]`、`local = (log + local)[2W :]`，然后重新拼接继续判，循环能跑好几轮，每轮都是对上一轮结果**再抽一次偶数位**；循环停了以后，若 `W < L < 3W` 再覆盖成 `log = tmp[:W]`、`local = tmp[W:]`。
2. **解码**（每次进 1 个 token）：位置追加进 `local`；一旦 `len(local) > 2W`，把 `local` 最旧的 `W` 个并到 `log` 后面整段取偶数位（`log` 为空时直接把这 `W` 个当 `log`，不抽样），`local` 砍掉这 `W` 个。

## 任务

实现 `solve(prefill_len, decode_steps, window_length)`：

- `prefill_len`：非负整数，预填充一次进来的 token 数 `L`。`L == 1` 按上游口径落进解码分支（只算一个解码步），`L == 0` 表示没有预填充调用。
- `decode_steps`：非负整数，预填充之后又进了多少个单 token 解码步。
- `window_length`：正整数 `W`（上游默认 64，README 示例用 4/8）。

返回 5 键字典，值一律 int64 一维张量（位置都是绝对下标，0 起）：

- `prefill_log_index` / `prefill_local_index`：预填充段结束、进入解码步**之前**的两段；
- `final_log_index` / `final_local_index`：全部步数跑完后的两段；
- `final_fp_positions`：两者并集去重升序，即最终仍留全精度的 token 位。

`prefill_len` 与 `decode_steps` 同时为 0、任一为负、任一不是 `int`（`bool` 也算不合法）、`window_length < 1` 时抛 `ValueError`。

## 示例

```
W = 4, 3W = 12, 2W = 8

solve(20, 0, 4)     # prefill only, L = 20
  local = [0..19], tmp = log + local = [0..19]                 len 20 >= 12
  fold1: log = tmp[0:8:2] = [0,2,4,6],   local = tmp[8:] = [8..19],   tmp len 16 >= 12
  fold2: log = tmp[0:8:2] = [0,4,8,10],  local = tmp[8:] = [12..19],  tmp len 12 >= 12
  fold3: log = tmp[0:8:2] = [0,8,12,14], local = tmp[8:] = [16..19],  tmp len 8  <  12
  W < L < 3W is false (L = 20), so the post-loop override does not run
  prefill_log_index   = [0, 8, 12, 14]
  prefill_local_index = [16, 17, 18, 19]
  final_log_index     = [0, 8, 12, 14]        final_local_index = [16, 17, 18, 19]
  final_fp_positions  = [0, 8, 12, 14, 16, 17, 18, 19]

solve(12, 5, 4)     # prefill L = 12, then 5 decode steps (12,13,14,15,16)
  after prefill: log = [0,2,4,6], local = [8,9,10,11]          # one fold, no override
  step 12 -> len(local)=5, step 13 -> 6, step 14 -> 7, step 15 -> 8   # 8 > 8 is false
  step 16 -> len(local)=9 > 8:
      log = ([0,2,4,6] + [8,9,10,11])[0::2] = [0,4,8,10]
      local = [8,9,10,11,12,13,14,15,16][4:] = [12,13,14,15,16]
  final_log_index   = [0, 4, 8, 10]
  final_local_index = [12, 13, 14, 15, 16]
  final_fp_positions = [0, 4, 8, 10, 12, 13, 14, 15, 16]

solve(7, 0, 4)  -> log = [0, 1, 2, 3],  local = [4, 5, 6]      # 4 < 7 < 12: override branch
solve(4, 0, 4)  -> log = [],            local = [0, 1, 2, 3]   # L == W: no fold, no override
```

## 约束

- **折叠是「整段前 2W 个取偶数位」**：`tmp = log + local` 先拼接再 `tmp[0:2W:2]`，等价于对上一轮的抽样结果再抽一次；写成「log 和 local 各抽一半」会得到不同的位置。
- **两侧阈值不对称且开闭不同**：预填充是 `len(tmp) >= 3W` 才折，解码是 `len(local) > 2W` 才并——第 `2W+1` 个解码 token 才触发，正好 `2W` 个时不动。
- **后置覆盖只看原始 `L`**：判定用的是 `W < L < 3W` 里的 `L`，不是循环结束后的 `len(tmp)`；`L >= 3W` 时哪怕折完只剩 `2W` 个也不覆盖。
- **`L == W` 两头都落空**：既进不了 `>= 3W` 的循环，也进不了 `W < L` 的覆盖，`log` 保持空、`local` 就是 `range(W)`。
- **`log` 为空时解码合并不抽样**：第一次触发直接把 `local[:W]` 整段当 `log`，第二次起才走 `(log + local[:W])[0::2]`。
- **每步最多触发一次**：一次解码只进一个 token，砍完 `W` 个以后 `len(local)` 必然回到 `W + 1`，不存在连续两次合并。
- **只判索引**：不重建 KV、不算量化误差；`final_fp_positions` 是两段并集，段内保持各自的抽取顺序即可，输出前统一升序。
