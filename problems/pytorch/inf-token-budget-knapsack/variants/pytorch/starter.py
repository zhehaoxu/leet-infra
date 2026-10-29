def solve(prompt_lens, values, budget, decode_count):
    """一步 forward 的 token 预算怎么花：decode 保底，剩下的在 prefill 里做准入背包。

    prompt_lens : list[int]，等待队列里每个请求的 prompt 长度（>=1），队列顺序即下标顺序（FCFS）
    values      : list[int]，每个请求的优先级权重（>=0），完整 prefill 一个请求就得到它的权重
    budget      : int，这一步允许处理的 token 总数（>=1）
    decode_count: int，这一步要服务的 decode 请求数（每个占 1 个 token 名额，>=0），恒有 budget >= decode_count

    规则（一步之内，全部整数）：
      1) decode 是保底负载：先扣掉 decode_count 个名额，剩 remaining = budget - decode_count 个
         名额留给 prefill。
      2) 0/1 背包：选一个下标集合 S，要求 Σ_{i∈S} prompt_lens[i] <= remaining，且被选中的请求
         这一步「整段 prefill 完成」。目标按字典序依次为：
           a. Σ_{i∈S} values[i] 最大；
           b. 若并列，Σ_{i∈S} prompt_lens[i] 最小（省下的名额给后一个）；
           c. 再并列，S 的升序下标列表按字典序最小。
      3) 余量分块：leftover = remaining - Σ_{i∈S} prompt_lens[i]。若 leftover > 0 且存在未入选的
         请求，取下标最小（FCFS 最靠前）的那个 j，给它分 min(leftover, prompt_lens[j]) 个 token 的
         分块——它本步不算完成、也不算入选；分块之后剩下的 token 记作 unused。

    返回 (served, total_value, chunk_index, chunk_tokens, unused)：
      served       : list[int]，入选集合 S 的升序下标
      total_value  : int，Σ_{i∈S} values[i]
      chunk_index  : int，拿到余量分块的下标；没有分块时为 -1
      chunk_tokens : int，分块 token 数；没有分块时为 0
      unused       : int，分块之后仍未用掉的 token 数
    """
    # TODO
    raise NotImplementedError
