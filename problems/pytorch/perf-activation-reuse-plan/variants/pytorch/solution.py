"""参考答案：按大小降序 + 候选偏移集合（0 与各已放置块末端）取最小可行者。"""


def solve(intervals, align):
    align = int(align)

    def round_up(n):
        return -(-int(n) // align) * align

    n = len(intervals)
    starts = [int(iv[0]) for iv in intervals]
    ends = [int(iv[1]) for iv in intervals]
    sizes = [round_up(iv[2]) for iv in intervals]
    order = sorted(range(n), key=lambda i: (-sizes[i], starts[i], ends[i], i))

    placed = []            # (start, end, offset, size) 已放置
    offsets = [0] * n
    peak = 0

    def conflict(a_start, a_end, a_off, a_size, b):
        b_start, b_end, b_off, b_size = b
        if max(a_start, b_start) >= min(a_end, b_end):
            return False   # 生命周期不重叠，字节可以重叠
        return max(a_off, b_off) < min(a_off + a_size, b_off + b_size)

    for i in order:
        s, e, size = starts[i], ends[i], sizes[i]
        cands = sorted({0} | {b[2] + b[3] for b in placed if max(s, b[0]) < min(e, b[1])})
        chosen = cands[-1]
        for off in cands:
            if not any(conflict(s, e, off, size, b) for b in placed):
                chosen = off
                break
        offsets[i] = chosen
        placed.append((s, e, chosen, size))
        peak = max(peak, chosen + size)

    lower_bound = 0
    for t in set(starts):
        live = sum(sizes[i] for i in range(n) if starts[i] <= t < ends[i])
        lower_bound = max(lower_bound, live)

    return ([int(o) for o in offsets], int(peak), int(lower_bound))
