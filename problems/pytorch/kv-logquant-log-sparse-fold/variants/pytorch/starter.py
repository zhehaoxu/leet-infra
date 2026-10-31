import torch


def solve(prefill_len, decode_steps, window_length):
    """LogQuant 的全精度索引调度：预填充 + 解码步跑完后，哪些 token 位还留着全精度。

    只维护两个**位置列表**（都是绝对 token 下标，0 起）：
      log_sparse_index : 分层抽样出来的一段
      local_index      : 最近的一段滑窗
    初始都为空，`seen`（已进缓存的 token 数）从 0 起。

    1. 预填充段（一次进 `prefill_len = L > 1` 个 token，`seen` 先加 L）：
       a. `local_index = list(range(L))`，`tmp = log_sparse_index + local_index`。
       b. 只要 `len(tmp) >= 3 * window_length` 就折叠一次：
          `log_sparse_index = tmp[0 : 2*window_length : 2]`、`local_index = tmp[2*window_length :]`，
          再 `tmp = log_sparse_index + local_index`，回到本步继续判。**整段前 2W 个取偶数位**，
          不是各取一半；循环可以跑很多轮，每轮的偶数位都是对上一轮结果再抽一次。
       c. 循环停下后，若 `window_length < L < 3 * window_length`（严格不等号）则**覆盖**：
          `log_sparse_index = tmp[:window_length]`、`local_index = tmp[window_length:]`。
          `L == window_length` 落不进这个分支，`log_sparse_index` 保持空。
    2. 预填充只进 1 个 token（`L == 1`）时走的是解码分支，一次都不算预填充：
       直接执行第 3 步一次，位置 0 进 `local_index`。`L == 0` 表示没有预填充调用，
       位置从第 0 个解码 token 开始。整段流程覆盖的 token 数恒为 `L + decode_steps`。
    3. 每个解码步（共 `decode_steps` 步，每步 `seen += 1`）：
       a. `local_index.append(seen - 1)`。
       b. 若 `len(local_index) > 2 * window_length`（严格大于，第 2W+1 个才触发）：
          `log_sparse_index` 非空则 `= (log_sparse_index + local_index[:W])[0::2]`，
          为空则 `= local_index[:W]`；随后 `local_index = local_index[W:]`。每步最多触发一次。

    返回 5 键 dict，值一律 int64 一维张量：
      prefill_log_index   : 预填充段结束时（进入解码步之前）的 log_sparse_index
      prefill_local_index : 预填充段结束时的 local_index
      final_log_index     : 全部步数跑完后的 log_sparse_index
      final_local_index   : 全部步数跑完后的 local_index
      final_fp_positions  : 两者并集去重升序（还留着全精度的 token 位）

    非法输入抛 ValueError：三个参数任一不是 int（bool 也算不合法）；`window_length < 1`
    （为 0 时第 1.b 步的循环不会收敛）；`prefill_len` 或 `decode_steps` 为负；
    两者同时为 0。
    """
    # TODO
    raise NotImplementedError
