"""参考答案：有序空闲块表 + best-fit 线性扫描，释放时向左右各合并一次。"""


def solve(ops, align, grow_unit):
    align = int(align)
    grow_unit = int(grow_unit)

    def round_up(n):
        return -(-int(n) // align) * align

    free = []          # 按 start 升序的 [start, size]，保持极大块
    live = {}
    reserved = 0
    peak_reserved = 0
    num_grow = 0

    def best_fit(need):
        pick = -1
        for i, (_, size) in enumerate(free):
            if size >= need and (pick < 0 or size < free[pick][1]):
                pick = i
        return pick

    def insert_free(start, size):
        i = 0
        while i < len(free) and free[i][0] < start:
            i += 1
        free.insert(i, [start, size])
        if i + 1 < len(free) and free[i][0] + free[i][1] == free[i + 1][0]:
            free[i][1] += free[i + 1][1]
            free.pop(i + 1)
        if i > 0 and free[i - 1][0] + free[i - 1][1] == free[i][0]:
            free[i - 1][1] += free[i][1]
            free.pop(i)

    for op in ops:
        ident = int(op[1])
        if op[0] == "alloc":
            need = round_up(op[2])
            pick = best_fit(need)
            if pick < 0:
                grow = max(need, round_up(grow_unit))
                insert_free(reserved, grow)
                reserved += grow
                peak_reserved = max(peak_reserved, reserved)
                num_grow += 1
                pick = best_fit(need)
            start, size = free.pop(pick)
            if size > need:
                insert_free(start + need, size - need)
            live[ident] = (start, need)
        else:
            start, size = live.pop(ident)
            insert_free(start, size)

    return (
        int(peak_reserved),
        int(reserved),
        int(num_grow),
        len(free),
        int(max((size for _, size in free), default=0)),
        [[int(start), int(size)] for start, size in free],
    )
