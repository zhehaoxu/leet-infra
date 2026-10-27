import random

import torch


def _oracle(num_blocks, block_size, ops):
    """独立 oracle：用「每个物理块记录 (seq, page) 归属集合」代替引用计数整数。"""
    owners = [set() for _ in range(num_blocks)]
    tables, n_tok, final = {}, {}, {}
    copies = 0

    def take():
        for b, own in enumerate(owners):
            if not own:
                return b
        raise ValueError("out of blocks")

    for kind, seq, arg in ops:
        if kind == 0:
            if seq in final or arg < 1:
                raise ValueError("bad create")
            pages = (arg + block_size - 1) // block_size
            blocks = []
            for page in range(pages):
                b = take()
                owners[b].add((seq, page))
                blocks.append(b)
            tables[seq] = blocks
            final[seq] = blocks
            n_tok[seq] = arg
        elif kind == 1:
            if seq not in tables or arg in final:
                raise ValueError("bad fork")
            blocks = list(tables[seq])
            for page, b in enumerate(blocks):
                owners[b].add((arg, page))
            tables[arg] = blocks
            final[arg] = blocks
            n_tok[arg] = n_tok[seq]
        elif kind == 2:
            if seq not in tables or arg < 1:
                raise ValueError("bad append")
            blocks = tables[seq]
            total = n_tok[seq] + arg
            for page in range(n_tok[seq] // block_size, (total - 1) // block_size + 1):
                if page >= len(blocks):
                    b = take()
                    owners[b].add((seq, page))
                    blocks.append(b)
                elif len(owners[blocks[page]]) > 1:
                    old = blocks[page]
                    owners[old].discard((seq, page))
                    b = take()
                    owners[b].add((seq, page))
                    blocks[page] = b
                    copies += 1
            n_tok[seq] = total
        elif kind == 3:
            if seq not in tables:
                raise ValueError("bad free")
            for page, b in enumerate(tables[seq]):
                owners[b].discard((seq, page))
            del tables[seq], n_tok[seq]
        else:
            raise ValueError("bad kind")

    n_seq = max(final) + 1 if final else 0
    n_page = max((len(t) for t in final.values()), default=0)
    page_tables = torch.full((n_seq, n_page), -1, dtype=torch.int64)
    for s, blocks in final.items():
        if blocks:
            page_tables[s, : len(blocks)] = torch.tensor(blocks, dtype=torch.int64)
    return page_tables, torch.tensor([len(o) for o in owners], dtype=torch.int64), copies


def check_example(solve):
    torch.manual_seed(0)
    ops = torch.tensor(
        [
            [0, 0, 6],  # seq0: 6 token -> 2 页 -> 块 0,1
            [0, 1, 4],  # seq1: 4 token -> 1 页 -> 块 2
            [1, 0, 2],  # fork seq0 -> seq2，块 0,1 引用计数变 2
            [2, 2, 3],  # seq2 追加 3 token：页 1 共享 -> 复制块 3；页 2 新开块 4
            [1, 2, 3],  # fork seq2 -> seq3
            [2, 3, 1],  # seq3 追加 1 token：落在共享页 2 -> 复制块 5
        ],
        dtype=torch.int64,
    )
    tables, refcount, copies = solve(6, 4, ops)
    want = torch.tensor(
        [[0, 1, -1], [2, -1, -1], [0, 3, 4], [0, 3, 5]], dtype=torch.int64
    )
    assert torch.equal(tables, want), f"页表错误：{tables.tolist()}"
    assert torch.equal(
        refcount, torch.tensor([3, 1, 1, 2, 1, 1], dtype=torch.int64)
    ), f"引用计数错误：{refcount.tolist()}"
    assert int(copies) == 2, f"写时复制次数应为 2，got {int(copies)}"


check_example.example = True


def check_fork_only_shares(solve):
    """fork 只加引用计数，不发生任何数据复制。"""
    torch.manual_seed(1)
    tables, refcount, copies = solve(4, 2, torch.tensor([[0, 0, 5], [1, 0, 1]]))
    assert int(copies) == 0, "fork 不应触发复制"
    assert torch.equal(tables[0], tables[1]), "fork 后两侧页表应完全一致"
    used = tables[0][tables[0] >= 0]
    assert used.numel() == 3, f"5 个 token、块大小 2 应占 3 页，got {used.numel()}"
    assert torch.equal(
        refcount[used], torch.full((3,), 2, dtype=torch.int64)
    ), f"共享块引用计数应为 2：{refcount.tolist()}"


def check_cow_only_on_written_pages(solve):
    """只有真正被写入的共享页才复制；写满的页继续追加不触发复制。"""
    torch.manual_seed(2)
    # 正好写满一页再 fork，追加落在新页上 -> 0 次复制
    tables, refcount, copies = solve(
        4, 4, torch.tensor([[0, 0, 4], [1, 0, 1], [2, 1, 4]])
    )
    assert int(copies) == 0, f"写满页追加不该复制，got {int(copies)}"
    assert torch.equal(
        tables[1], torch.tensor([0, 1], dtype=torch.int64)
    ), f"seq1 页表应为 [0,1]，got {tables[1].tolist()}"
    assert int(refcount[0]) == 2 and int(refcount[1]) == 1, "首块仍共享、新块独占"

    # 半满页被追加 -> 该页复制
    tables, refcount, copies = solve(
        3, 4, torch.tensor([[0, 0, 3], [1, 0, 1], [2, 1, 1]])
    )
    assert int(copies) == 1, f"半满共享页写入应复制 1 次，got {int(copies)}"
    assert int(tables[0, 0]) == 0 and int(tables[1, 0]) == 1, "两侧应指向不同物理块"
    assert bool((refcount[[0, 1]] == 1).all()), "复制后两侧各自独占"


def check_random_against_oracle(solve):
    """随机操作序列与「按归属集合」实现的 oracle 对拍（含块耗尽）。"""
    rng = random.Random(1234)
    for trial in range(40):
        num_blocks = rng.randint(2, 8)
        block_size = rng.randint(1, 4)
        ops, live, next_id = [], [], 0
        for _ in range(rng.randint(3, 14)):
            r = rng.random()
            if (not live) or (r < 0.3 and next_id < 5):
                ops.append((0, next_id, rng.randint(1, 2 * block_size)))
                live.append(next_id)
                next_id += 1
            elif r < 0.55 and next_id < 5:
                ops.append((1, rng.choice(live), next_id))
                live.append(next_id)
                next_id += 1
            elif r < 0.85:
                ops.append((2, rng.choice(live), rng.randint(1, block_size)))
            else:
                victim = rng.choice(live)
                ops.append((3, victim, 0))
                live.remove(victim)

        try:
            got = solve(num_blocks, block_size, ops)
        except ValueError:
            try:
                _oracle(num_blocks, block_size, ops)
            except ValueError:
                continue
            raise AssertionError(f"oracle 不报错但实现报错：blocks={num_blocks} ops={ops}")
        want = _oracle(num_blocks, block_size, ops)
        assert torch.equal(got[0], want[0]), f"trial {trial} 页表不一致：{got[0].tolist()} vs {want[0].tolist()}"
        assert torch.equal(got[1], want[1]), f"trial {trial} 引用计数不一致"
        assert int(got[2]) == int(want[2]), f"trial {trial} 复制次数不一致"


def check_invalid_inputs(solve):
    good = torch.tensor([[0, 0, 1]], dtype=torch.int64)
    for bad in (
        lambda: solve(0, 4, good),                                  # 块数为 0
        lambda: solve(2, 0, good),                                  # 块大小为 0
        lambda: solve(2, 2, torch.tensor([0, 0, 1])),               # ops 不是二维
        lambda: solve(2, 2, torch.zeros((3, 2), dtype=torch.int64)),  # 列数不是 3
        lambda: solve(1, 2, torch.tensor([[0, 0, 4]])),             # 2 页但只有 1 块
        lambda: solve(1, 2, torch.tensor([[0, 0, 1], [2, 0, 2]])),  # 追加跨页但已无空块
        lambda: solve(4, 2, torch.tensor([[5, 0, 1]])),             # 未知操作码
        lambda: solve(4, 2, torch.tensor([[2, 0, 1]])),             # 操作不存在的序列
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
