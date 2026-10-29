def solve(prompt_lens, gen_lens, kv_bytes_per_token, caps, bws, tps, prefill_tps):
    """PD 分离（prefill/decode disaggregation）的放置账。

    服务拓扑：一个共享的 prefill 池 + 若干 decode 副本；prefill 算完的 KV 通过
    KV 传输链路搬到某个 decode 副本上继续解码。传输链路不与 prefill 计算重叠，
    但每条链路同一时刻只搬一个请求。

    prompt_lens     : list[int]，第 i 个请求的 prompt 长度（>=1）
    gen_lens        : list[int]，第 i 个请求还要生成的 token 数（>=0）
    kv_bytes_per_token : int，每 token 的 KV 字节数（>=1）
    caps            : list[int]，每个 decode 副本能容纳的 KV 字节数（>=0）
    bws             : list[int]，每条 KV 传输链路带宽，单位「字节/拍」（>=1）
    tps             : list[int]，每个 decode 副本的张量并行度（>=1）
    prefill_tps     : int，prefill 池的吞吐，单位「prompt token/拍」（>=1）

    时间以整数拍计，所有请求在下标顺序上依次 prefill（FCFS，互不重叠）。
    记 cum_prompt[k] = prompt_lens[0..k] 之和，第 i 个请求 prefill 完成的拍数为
        prefill_done[i] = ceil(cum_prompt[i] / prefill_tps)
    （用累计量整体上取整，而不是每个请求各自取整。）

    请求 i 需要落地的 KV 字节数（峰值，prompt 与将要生成的 token 都算）：
        kv[i] = (prompt_lens[i] + gen_lens[i]) * kv_bytes_per_token

    decode 副本 d 的链路上要搬的字节数分两段：
        handoff = kv[i]                                          # prefill 池交给该副本
        ring    = ceil(kv[i] * (tps[d] - 1) / tps[d])            # 副本内 TP 组环形 all-gather
        moved   = handoff + ring                                 # tps[d]==1 时 ring==0
    即环形交换按「每条链路分到 (C-1)/C 的份额」计入同一条链路的带宽（与
    distributed-sequence-parallel 的环系数口径一致；此处把节点内交换也记在这条
    链路上，属简化假设）。

    放置规则（逐请求按 i 从小到大决定，决定即生效）：
      1. 先做显存准入：只考虑 resident[d] + kv[i] <= caps[d] 的副本。
      2. 对每个可用副本 d 预估完成拍：
             start  = max(prefill_done[i], link_free[d])
             finish = start + ceil(moved / bws[d])
         取 finish 最小的副本（相同则取下标小的）；此处的 link_free[d] 是本请求
         之前该链路上最后一个传输的完成拍。
      3. 一个副本都不满足准入时，请求被拒绝（drop）：placement 与 ttft 记 -1，
         不占显存、不占链路；但它的 prefill 份额已经消耗，后续请求的
         prefill_done 仍按全部请求的累计量计算。
      4. 放置成功后：resident[d] += kv[i]，wire[d] += moved，link_free[d] = finish。

    返回 (placement, ttft, wire, resident, rejected, makespan)：
      placement : list[int]，每请求落到哪个 decode 副本；被拒绝为 -1
      ttft      : list[int]，每请求的 KV 全部就位拍（即首个 token 可出的时刻）；
                  被拒绝为 -1
      wire      : list[int]，每个副本的链路累计搬运字节数（handoff+ring）
      resident  : list[int]，每个副本上被接纳请求的 KV 字节总数（峰值口径）
      rejected  : int，被拒绝的请求数
      makespan  : int，所有被接纳请求 ttft 的最大值；无请求被接纳时为 0
    """
    # TODO
    raise NotImplementedError
