def solve(cu_seqlens, cp_degree):
    """中文契约 / English contract
    cu_seqlens 是长度 B+1 >= 2 的 python list/tuple，元素是非 bool 的 python int；
    cu_seqlens[0] == 0，严格递增，末项（总 token 数）<= 65536，因此每段长度 >= 1。
    cp_degree 是 python int（bool 不算），1 <= cp_degree <= 256。
    非法入参（类型不符、起点非 0、非严格递增、负数、越界）一律抛 ValueError /
        raise ValueError on invalid input; mutate nothing.

    记 C = cp_degree，对齐粒度 align = 2C。对第 b 段（长度 n_b），补齐长度
    p_b = ceil(n_b / align) * align；全部段首尾相接得到补齐后的打包流，总长 P = Σ p_b，
    每个 rank 的局部长度 L = P / C（P 必是 2C 的倍数，故整除）。
    段内切成 2C 个等长块，块长 chunk_b = p_b / (2C)，第 cid 块覆盖段内偏移
    [cid*chunk_b, (cid+1)*chunk_b)。rank r 持有第 r 块与第 2C-1-r 块，
    局部顺序是「前块（cid = r）+ 后块（cid = 2C-1-r）」/
        each sequence is padded to a multiple of 2C, cut into 2C equal chunks; rank r owns
        chunk r and chunk 2C-1-r, front chunk first.

    局部位置：段 b 在补齐流里的起点为 q_b（前缀和），前块放在 q_b/C 起的 chunk_b 个槽，
    后块紧接着再放 chunk_b 个槽 / local base is q_b // C; the back chunk follows the front one.
    槽里写该 token 在整个 cu_seqlens 打包流里的原始下标（从 0 起，跨段连续）；
    段尾补齐的位置写 -1 / slots hold global original indices, padding is -1.

    返回四元组 / Returns a 4-tuple:
      (cu_seqlens_padded: list[int] 长度 B+1，补齐后的段边界，首项 0,
       rank_order: list[list[int]] 形状 C x L，每 rank 的局部 token 下标，padding 为 -1,
       pad_total: int，补齐总量 P - N,
       pad_between_seqs: bool，只要有任何一段被补齐就为 True)
    """
    raise NotImplementedError
