def solve(kv, prompt, generated, priority, need, bytes_per_token, swap_bw, tick_per_token, freezable):
    """显存不够时的抢占决策：按优先级挑牺牲集，并逐个比较「换出」与「重算」的开销。

    kv            : list[int]，第 i 个在跑请求当前占用的 KV token 数（>=1）
    prompt        : list[int]，第 i 个请求的 prompt 长度（>=1）
    generated     : list[int]，第 i 个请求已经产出的 token 数（>=0）
    priority      : list[int]，第 i 个请求的优先级，数值越大越重要
    need          : int，本次需要腾出的 KV 名额（token 槽位）数（>=1）
    bytes_per_token : int，每个 KV token 占的字节数（>=1）
    swap_bw       : int，KV 换出/换入的带宽，单位「字节/拍」（>=1）
    tick_per_token  : int，重新 prefill 一个 token 需要的拍数（>=1）
    freezable     : list[bool]，第 i 个请求是否可以被抢占（False 表示受保护，不参与牺牲集）

    抢占规则（按固定次序贪心，够用即停）：
      * 候选只含 freezable 为 True 的请求，按优先级从低到高分层处理；同一层内先挑
        kv 更大的（一次腾出更多），kv 相同则下标小的先挑。
      * 一旦已腾出的槽位 >= need 就立刻停止，不再动后续请求。
      * 每个被选中的请求二选一：
          swap      换出到主机内存、稍后再换回来，耗时 = ceil(2 * kv[i] * bytes_per_token / swap_bw)
          recompute 丢掉 KV、之后整段重新 prefill，耗时 = (prompt[i] + generated[i]) * tick_per_token
        两者相等时选 swap。
      * 若遍历完所有可用优先级仍腾不够 need，判定不可行，不做任何抢占。

    返回 (victims, modes, freed, ticks, replay_tokens)：
      victims       : list[int]，被抢占请求的下标，按选择先后排列
      modes         : list[str]，与 victims 一一对应，取值 "swap" 或 "recompute"
      freed         : int，实际腾出的槽位数（可能大于 need）
      ticks         : int，所有被抢占请求的开销之和（swap 与 recompute 各自的口径见上）；
                      不可行时为 -1
      replay_tokens : int，走 recompute 的请求需要重新 prefill 的 token 总数；不可行时为 0
    """
    # TODO
    raise NotImplementedError
