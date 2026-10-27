import torch


def solve(num_blocks, block_size, ops):
    if num_blocks < 1 or block_size < 1:
        raise ValueError("num_blocks / block_size 必须为正")
    if not isinstance(ops, torch.Tensor):
        ops = torch.tensor(ops, dtype=torch.int64)
    if ops.dim() != 2 or ops.shape[1] != 3:
        raise ValueError("ops 必须是 (K, 3) 的 int64 张量")

    refcount = [0] * num_blocks
    live = {}       # seq -> 当前页表（list of block id）
    final = {}      # seq -> 最终页表快照（free 之后仍保留）
    n_tok = {}      # seq -> 已写入的 token 数
    copies = 0

    def alloc_one():
        for b in range(num_blocks):
            if refcount[b] == 0:
                refcount[b] = 1
                return b
        raise ValueError("物理块耗尽")

    for kind, seq, arg in ops.to(torch.int64).tolist():
        if kind == 0:
            if seq < 0 or arg < 1:
                raise ValueError("create 需要 seq >= 0 且 n_tokens >= 1")
            if seq in final:
                raise ValueError("序列 id 不可复用")
            table = [alloc_one() for _ in range((arg + block_size - 1) // block_size)]
            live[seq] = table
            final[seq] = table
            n_tok[seq] = arg
        elif kind == 1:
            if seq not in live or arg < 0:
                raise ValueError("fork 的源序列不存在")
            if arg in final:
                raise ValueError("fork 的目标序列 id 不可复用")
            table = list(live[seq])
            for b in table:
                refcount[b] += 1
            live[arg] = table
            final[arg] = table
            n_tok[arg] = n_tok[seq]
        elif kind == 2:
            if seq not in live or arg < 1:
                raise ValueError("append 需要存活序列且追加数 >= 1")
            table = live[seq]
            total = n_tok[seq] + arg
            # 只有 [旧 token 数, 新 token 数) 覆盖到的逻辑页会被写
            for page in range(n_tok[seq] // block_size, (total - 1) // block_size + 1):
                if page >= len(table):
                    table.append(alloc_one())
                elif refcount[table[page]] > 1:
                    # 共享页要写 -> 复制一份私有块，引用计数各减一
                    refcount[table[page]] -= 1
                    table[page] = alloc_one()
                    copies += 1
            n_tok[seq] = total
        elif kind == 3:
            if seq not in live:
                raise ValueError("free 的序列不存在")
            for b in live[seq]:
                refcount[b] -= 1
            del live[seq], n_tok[seq]
        else:
            raise ValueError("未知操作码")

    n_seq = max(final) + 1 if final else 0
    n_page = max((len(t) for t in final.values()), default=0)
    page_tables = torch.full((n_seq, n_page), -1, dtype=torch.int64)
    for s, table in final.items():
        if table:
            page_tables[s, : len(table)] = torch.tensor(table, dtype=torch.int64)
    return (
        page_tables,
        torch.tensor(refcount, dtype=torch.int64),
        torch.tensor(copies, dtype=torch.int64),
    )
