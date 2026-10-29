def solve(intervals, align):
    """离线显存复用规划：给一组有生命周期的张量分配地址偏移，让峰值尽量小。

    intervals : 列表，每项 (start, end, nbytes)
                start/end 是逻辑步编号（大小为半开区间 [start, end)，要求 end > start），
                nbytes >= 1；两个张量「生命周期重叠」指 max(s1,s2) < min(e1,e2)
    align     : 对齐粒度（>=1），每个张量实际占用 size = ceil(nbytes/align)*align，
                起始偏移也必须是 align 的整数倍

    规划算法（务必按此定义，判题按它逐项对拍）：
      1. 排序键 = (size 降序, start 升序, end 升序, 原始下标升序)；
      2. 依序为每个张量选**最小**的可放置偏移 o（沿 align 递增）：
         要求 [o, o+size) 与「所有已放置且生命周期重叠的张量」的字节区间都不相交；
      3. 峰值 peak = max(o + size)。

    lower_bound：任一时刻同时存活的张量字节和的下界，即
      lower_bound = max over t of Σ size(存活于 t 的张量)，t 取所有 start 值。
      贪心按大小放置并不保证达到下界（可能因碎片超出），peak >= lower_bound 恒成立。

    返回 (offsets, peak, lower_bound)：
      offsets   : list[int]，按输入顺序给出每个张量的偏移
      peak      : int，规划后的峰值字节数
      lower_bound: int，上述同时存活字节和的最大值
    """
    # TODO
    raise NotImplementedError
