def solve(input_ids, special_mask, sequence_ids, max_length,
          strategy="longest_first", keep="head"):
    """保护特殊位置的截断 / Truncation with protected special positions.

    中文契约：
    - 三个等长 Python 列表，长度 0..200。input_ids 是 0..10^6 整数；
      special_mask 是 bool；特殊位置的 sequence_ids=-1，其余为 0 或 1。
      标记按位置生效，不由 token 值推断；各序列可交错。不得修改输入。
    - max_length 是非负 Python int（不含 bool）；非法时 ValueError。
      strategy 为 only_first/only_second/longest_first；keep 为 head/tail/middle；
      未知取值 ValueError。其他输入合法。
    - 全部特殊位置必须保留。需要删除 r=max(0,L-max_length) 个普通 token。
      only_first/second 只删序列 0/1，数量不足时 ValueError。
      longest_first 每步从剩余普通 token 较多的序列删一个，相同则删序列 1；
      特殊位置本身超过预算时 ValueError。
    - 确定各序列保留数 q 后，head 留前 q，tail 留后 q；middle 删除中部，
      留前 ceil(q/2) 和后 floor(q/2)。q=0 留空。最后按原始位置合并。
    - 返回恰五键字典，均为 Python 列表：input_ids、special_mask、sequence_ids
      为保留子序列；kept_indices 是升序原始下标；removed_counts=[序列0删数,序列1删数]。

    English contract:
    - Three equal-length Python lists, length 0..200; token IDs are integers
      in 0..10^6, special_mask is bool, and sequence_ids is -1 exactly at
      special positions, otherwise 0 or 1. Sequences may interleave.
      Protection is positional, not inferred from token values. Do not mutate inputs.
    - max_length is a nonnegative Python int, excluding bool; otherwise ValueError.
      strategy: only_first/only_second/longest_first; keep: head/tail/middle.
      Unknown options raise ValueError. Other inputs are valid.
    - Preserve all special positions. Delete r=max(0,L-max_length) ordinary tokens.
      only_first/second deletes only from sequence 0/1; insufficient tokens raise
      ValueError. longest_first repeatedly decrements the larger remaining ordinary
      count, choosing sequence 1 on ties. Too many protected positions raise ValueError.
    - For each resulting quota q, head retains its first q tokens, tail its last q,
      and middle retains the first ceil(q/2) plus last floor(q/2), deleting the center.
      Zero quota retains nothing. Merge retained positions in original order.
    - Return exactly five list-valued keys: input_ids, special_mask, sequence_ids,
      kept_indices (ascending original positions), removed_counts ([removed0,removed1]).
    """
    raise NotImplementedError
