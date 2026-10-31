import torch


def solve(keys, values, quantized_len, sink_id, sink_scores, group_size=128,
          residual_length=128, sink_num=5, sink_max_size=32):
    """OTT 的离群 token 追踪：一次量化步里，哪些 token 被拉进全精度追踪集、K/V 怎么回填。

    keys / values : float32/float64 的 (1, H, F, D)，本层「尚未量化」的 fp 缓冲区，batch 恒为 1
    quantized_len : int，此前已经量化掉的 token 数（= 本次 chunk 的绝对下标偏移）
    sink_id       : int64 (n,) 已追踪 token 的**绝对**下标；n == 0 表示还没追踪过任何 token
    sink_scores   : 与 keys 同 dtype 的 (n,) 已追踪 token 的分数，已按升序排好
    group_size / residual_length / sink_num / sink_max_size : 上游默认 128 / 128 / 5 / 32

    1. `quant_len = max(0, (F - residual_length) // group_size * group_size)`：只量化整数组，
       并且至少留 `residual_length` 个最新 token 不动。`quant_len == 0` 时本次什么都不做
       （上游 L361 的门槛 `F >= residual_length + group_size` 就是这个意思），状态原样回吐。
    2. 本次候选段 `chunk = keys[:, :, :quant_len, :]`（values 同），chunk 下标 t 的绝对下标是
       `t + quantized_len`。
    3. 分数 `scores[t] = chunk[0].abs().mean(dim=(0, 2))[t]`，即该 token 在**所有 KV head 与
       所有通道**上 |K| 的平均。value 不参与打分（上游 `metric(k, v)` 只用到 k）。
    4. `sink_id` 为空时：取 scores 最小的 `sink_num` 个，同分取**下标小**的，
       `new_indices` 按分数升序。
    5. `sink_id` 非空时：门槛是 `sink_scores[sink_num - 1]`（用配置的 sink_num，不是 n-1）。
       候选 = `scores < 门槛` 的位置（严格小于）；若候选为空，或 `n >= sink_max_size`，本次
       不新增；否则在候选里取最小的 `min(sink_num, 候选数)` 个。
    6. 追加后整表重排：`sink_id = cat(旧, 新绝对下标)`、`sink_scores = cat(旧分数, 新分数)`，
       再按分数升序稳定排序（同分时旧条目排在新条目前面），`sink_id` 跟着同一份置换。
    7. 回填：`keys_out` / `values_out` 是 chunk 的副本，本次新增追踪的那批 token 在**每个 head
       上共享同一个代表向量**——它们在该 head 上的均值 `(H, m, D)` 沿 m 求平均，再广播写回这批
       token 的全部位置（是跨 token 取均值，不是跨 head！）。广播的是同一个向量，所以取数按分数序
       （上游写法）还是按下标序都同结果。未追踪的位置原样；只动本次 chunk。

    返回 7 键 dict：
      quant_len    : int64 零维张量
      scores       : (quant_len,)，与 keys 同 dtype
      new_indices  : int64 (m,)，本次新增的**chunk 内**下标，按分数升序
      sink_id      : int64 (n+m,)，绝对下标，按 sink_scores 升序
      sink_scores  : (n+m,)，与 keys 同 dtype，升序
      keys_out / values_out : (1, H, quant_len, D)，与输入同 dtype

    非法输入抛 ValueError：keys/values 不是四维、不是 float32/float64、形状或 dtype 不一致、
    batch 不等于 1；sink_id / sink_scores 不是一维、sink_id 不是 int64、sink_scores dtype 与
    keys 不一致、两者长度不一致；`sink_id` 非空但条数少于 `sink_num`（上游那行会 IndexError）；
    整数参数不是 int（bool 也算不合法）或越界（`quantized_len`/`residual_length` 为负，
    `group_size`/`sink_num`/`sink_max_size` 小于 1）；`quant_len > 0` 时 `sink_num > quant_len`。
    """
    # TODO
    raise NotImplementedError
