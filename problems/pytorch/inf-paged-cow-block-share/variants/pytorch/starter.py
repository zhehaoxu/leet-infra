import torch


def solve(num_blocks, block_size, ops):
    """模拟 PagedAttention 的页表间接寻址、fork 共享与写时复制。

    num_blocks:物理块总数（块号从 0 开始）;
    block_size:每块能放的 token 数;
    ops:(K, 3) int64，每行 (kind, seq, arg)：
      kind=0 create —— 新建序列 seq，写入 arg 个 token（arg >= 1）;
      kind=1 fork   —— 目标序列 arg 复制 seq 的页表，共享同一批物理块;
      kind=2 append —— 给序列 seq 追加 arg 个 token（arg >= 1）;
      kind=3 free   —— 释放序列 seq 的全部块引用（arg 忽略）。

    返回 (page_tables:(S, P) int64（-1 补齐）、refcount:(num_blocks,) int64、
          copies:int64 标量（写时复制次数）)。物理块不足或操作非法时抛 ValueError。
    """
    # TODO
    raise NotImplementedError
