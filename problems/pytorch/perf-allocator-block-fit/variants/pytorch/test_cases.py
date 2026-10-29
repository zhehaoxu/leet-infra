# 独立重算（对拍口径）：不维护空闲块表，而是把整段地址按 align 切成「页」布尔位图，
# best-fit 用「扫描连续空闲页段、取满足 need 的最短段（并列取地址小者）」重新求出；
# 扩段就是往位图尾部追加 True 页；释放就是把页翻回 True。极大块/合并都由位图自然涌现，
# 与参考解的「有序块表 + 左右合并」结构完全不同。
def _oracle(ops, align, grow_unit):
    align = int(align)
    grow_unit = int(grow_unit)

    def round_up(n):
        return -(-int(n) // align) * align

    pages = []
    reserved = 0
    peak = 0
    num_grow = 0
    live = {}

    def runs():
        out = []
        i = 0
        while i < len(pages):
            if pages[i]:
                j = i
                while j < len(pages) and pages[j]:
                    j += 1
                out.append([i * align, (j - i) * align])
                i = j
            else:
                i += 1
        return out

    def best_fit(need):
        pick = None
        for start, size in runs():
            if size >= need and (pick is None or size < pick[1]):
                pick = (start, size)
        return pick

    for op in ops:
        if op[0] == "alloc":
            need = round_up(op[2])
            hit = best_fit(need)
            if hit is None:
                grow = max(need, round_up(grow_unit))
                pages.extend([True] * (grow // align))
                reserved += grow
                peak = max(peak, reserved)
                num_grow += 1
                hit = best_fit(need)
            start = hit[0]
            for k in range(start // align, (start + need) // align):
                pages[k] = False
            live[int(op[1])] = (start, need)
        else:
            start, size = live.pop(int(op[1]))
            for k in range(start // align, (start + size) // align):
                pages[k] = True

    blocks = runs()
    return (
        peak,
        reserved,
        num_grow,
        len(blocks),
        max([size for _, size in blocks], default=0),
        blocks,
    )


def _assert_tuple(got, want):
    assert len(got) == 6, got
    for k in (0, 1, 2, 3, 4):
        assert int(got[k]) == int(want[k]), (k, got, want)
    assert [[int(a), int(b)] for a, b in got[5]] == [[int(a), int(b)] for a, b in want[5]], (got[5], want[5])


def _rand_ops(rng, n_ops, max_bytes):
    ops = []
    live = []
    next_id = 0
    for _ in range(n_ops):
        if live and (len(live) > 3 or rng.random() < 0.35):
            i = rng.randrange(len(live))
            ops.append(("free", live.pop(i), 0))
        else:
            ops.append(("alloc", next_id, rng.randrange(1, max_bytes)))
            live.append(next_id)
            next_id += 1
    return ops


# 样例：align=512、grow_unit=4096
#   alloc a1000 -> 需 1024，空 -> 扩 4096，a=[0,1024)     free=[1024,3072)
#   alloc b3000 -> 需 3072，正好吃掉               b=[1024,3072)  free=[]
#   alloc c600  -> 需 1024，空 -> 扩到 8192         c=[4096,1024)  free=[5120,3072)
#   free b      -> 采不到相邻块                     free=[1024,3072),[5120,3072)
#   alloc d2000 -> 需 2048，两块同为 3072 -> 取小地址 d=[1024,2048)  free=[3072,1024),[5120,3072)
#   free c      -> 与左右各相邻一次 -> 合并成 3072 起 5120 长     free=[3072,5120)
#   alloc e5000 -> 需 5120，正好吃掉                 e=[3072,5120)  free=[]
#   alloc f512  -> 扩到 12288                       f=[8192,512)   free=[8704,3584)
#   free a      -> 与 [8704,3584) 不相邻            free=[0,1024),[8704,3584)
def check_example(solve):
    ops = [
        ("alloc", 0, 1000),
        ("alloc", 1, 3000),
        ("alloc", 2, 600),
        ("free", 1, 0),
        ("alloc", 3, 2000),
        ("free", 2, 0),
        ("alloc", 4, 5000),
        ("alloc", 5, 512),
        ("free", 0, 0),
    ]
    got = solve(ops, 512, 4096)
    _assert_tuple(got, (12288, 12288, 3, 2, 3584, [[0, 1024], [8704, 3584]]))


check_example.example = True


def check_matches_bitmap_oracle(solve):
    import random

    rng = random.Random(2026)
    for align, grow, max_bytes in [(512, 4096, 6000), (1, 7, 40), (256, 1024, 3000), (64, 64, 500), (128, 100, 900)]:
        for _ in range(12):
            ops = _rand_ops(rng, rng.randrange(6, 22), max_bytes)
            _assert_tuple(solve(ops, align, grow), _oracle(ops, align, grow))


def check_grow_is_min_segment(solve):
    # 单次申请超过 grow_unit 时，扩段量就是 need 本身，不多要
    got = solve([("alloc", 0, 9000)], 512, 4096)
    assert int(got[1]) == 9216, got
    assert int(got[2]) == 1, got
    # 申请小于 grow_unit 时，扩段量是 grow_unit（向上对齐）
    got = solve([("alloc", 0, 1000)], 512, 4096)
    assert int(got[1]) == 4096, got
    assert [[int(a), int(b)] for a, b in got[5]] == [[1024, 3072]], got
    # grow_unit 本身不对齐 -> 也要向上对齐
    got = solve([("alloc", 0, 10)], 64, 100)
    assert int(got[1]) == 128, got


def check_coalescing_reuses_without_growth(solve):
    ops = [
        ("alloc", 0, 1024),
        ("alloc", 1, 1024),
        ("free", 0, 0),
        ("free", 1, 0),
    ]
    got = solve(ops, 512, 4096)
    assert int(got[2]) == 1, got
    assert [[int(a), int(b)] for a, b in got[5]] == [[0, 4096]], got
    # 合并后一个 4096 的申请可以原地复用，不再扩段
    ops.append(("alloc", 2, 4096))
    got = solve(ops, 512, 4096)
    assert int(got[2]) == 1 and int(got[1]) == 4096, got
    assert int(got[3]) == 0, got


def check_free_all_leaves_one_block(solve):
    import random

    rng = random.Random(7)
    for align, grow in [(512, 4096), (1, 5), (64, 256)]:
        ops = _rand_ops(rng, 24, 2000)
        ids = [op[1] for op in ops if op[0] == "alloc"]
        held = set()
        cleaned = []
        for op in ops:
            if op[0] == "alloc":
                held.add(op[1])
                cleaned.append(op)
            elif op[1] in held:
                held.discard(op[1])
                cleaned.append(op)
        for ident in sorted(held):
            cleaned.append(("free", ident, 0))
        got = solve(cleaned, align, grow)
        assert int(got[3]) == 1, (align, got)
        assert [[int(a), int(b)] for a, b in got[5]] == [[0, int(got[1])]], got
        assert int(got[4]) == int(got[1]), got


def check_free_blocks_valid(solve):
    import random

    rng = random.Random(99)
    for align, grow in [(512, 4096), (1, 7), (128, 384)]:
        ops = _rand_ops(rng, 20, 3000)
        got = solve(ops, align, grow)
        blocks = [[int(a), int(b)] for a, b in got[5]]
        prev_end = 0
        for start, size in blocks:
            assert size > 0 and start >= prev_end, blocks
            assert start % align == 0 and size % align == 0, (align, blocks)
            prev_end = start + size
        assert prev_end <= int(got[1]), (blocks, got[1])
        assert int(got[0]) == int(got[1]), got   # reserved 只增不减
        assert int(got[4]) == max([s for _, s in blocks], default=0), got


def check_empty_ops(solve):
    got = solve([], 512, 4096)
    _assert_tuple(got, (0, 0, 0, 0, 0, []))
