def solve(capacity, ops):
    """Radix-tree 前缀缓存（token 级、边压缩字典树）：按请求流输出前缀命中与驱逐账。

    capacity : int，缓存最多保留多少个 token 的 KV（>=1）
    ops      : list[tuple]，按时间顺序排列的操作，元素是下面两种之一：
                 ("request", rid, tokens)  请求 rid 带着 prompt tokens（list[int]，长度 >= 1）到达
                 ("finish", rid)           请求 rid 结束，释放它对缓存的占用
               rid 是整数且每个 rid 只 request 一次、finish 至多一次。

    时间戳就是 ops 的下标（第 0 个操作发生在时刻 0）。缓存是一棵「边压缩」的 token 字典树：
    每条边存一段连续 token，边上的节点本身不占容量；根节点是空路径。缓存里 token 总数 =
    所有边长度之和。

    语义：
      * request：从根开始沿与 tokens 的最长公共前缀下行。若某条边只匹配了它的前半段，
        就在匹配处把这条边劈成两段（劈出来的中间节点是新建节点）；若走完 tokens 时仍停在
        某条边的中间，也照劈不误。下行经过的每个节点（含劈出的新节点）都把「最近使用时刻」
        刷新为当前时刻。命中的 token 数记为 hit；若 tokens 还有剩余，剩下的整段作为一条新边
        挂到匹配终点下（新节点的最近使用时刻也是当前时刻；没有剩余则不新建节点）。该请求
        进入「活跃」集，占用整条 tokens 路径上的 KV，直到被 finish。
      * finish：请求离开活跃集，它的 KV 继续留在缓存里供后来的请求复用；它此前钉住的节点
        从这一步起变成可驱逐的（同一步的容量检查照常执行）。finish 本身不刷新最近使用时刻。
      * 容量：每个操作处理完后，只要缓存 token 总数 > capacity 就反复驱逐 —— 在「是叶子节点
        且没有任何活跃请求的 prompt 以它的路径为前缀」的节点中，取 (最近使用时刻, 节点编号)
        字典序最小的一个，把它的整条边连同节点一起删掉；若一个新的叶子因此空出来，下一轮再算。
        找不到可驱逐的叶子就停手，所以活跃请求占满时缓存允许暂时超过 capacity。

    节点编号：根为 0，其余按创建先后从 1 开始递增。

    返回 (hits, evicted_tokens, cached_tokens)：
      hits           : list[int]，按 request 出现顺序，每个请求命中的 token 数
      evicted_tokens : int，全程被驱逐的 token 总数
      cached_tokens  : int，全部操作结束后缓存里剩余的 token 数
    """
    # TODO
    raise NotImplementedError
