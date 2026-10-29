import torch


def solve(token_emb, attention_mask, chunk_bounds, hit_chunks, parent_size,
          normalize=True):
    """Late chunking 池化 + 父块扩展（CPU、确定性）。

    参数：
    - token_emb: (L, D) 浮点张量，整篇文档**已上下文化**的 token 向量
      （本题不跑编码器，输入即编码结果）。
    - attention_mask: (L,) bool 或整型，非 0 表示有效 token。
    - chunk_bounds: (C, 2) int64，每行 [start, end) 是一个子块的 token
      区间；按 start 非降序给出，块之间允许重叠（滑窗）或留空隙，也
      允许零长度块与尾块不足。
    - hit_chunks: (H,) int64，检索命中的子块下标；允许重复、允许为空。
    - parent_size: int >= 1。子块 c 属于父块 p = c // parent_size，即
      每 parent_size 个相邻子块合成一个父块。
    - normalize: bool，默认 True。True 时对块向量与父块向量逐行做 L2
      归一化（零向量保持全 0）；False 时返回原始 mask 均值。

    池化口径：块向量 = 块区间内 attention_mask 非 0 的 token 向量的
    算术平均；有效 token 数为 0（含零长度块、全被 mask 掉的块）时该
    块向量记全 0。父块区间 = 组内首个子块的 start 到末个子块的 end
    （闭开区间）；父块向量在同一区间上按同样的 mask 均值口径**从
    token 重算**，不是子块向量的平均，因此子块重叠时父块区间也可以
    重叠。

    返回四元组：
    - chunk_vecs: (C, D) 与 token_emb 同 dtype，全部子块向量（与命中
      无关，始终完整返回）。
    - parent_bounds: (P, 2) int64，被命中父块的 token 区间，按父块
      下标升序去重输出；P 为不同父块个数。
    - parent_vecs: (P, D) 与 parent_bounds 对应的父块向量。
    - hit_to_parent: (H,) int64，按 hit_chunks 原顺序给出每个命中子块
      对应的 parent_bounds 行号（0..P-1）；同一父块内的重复命中映射
      到同一行。

    空命中：H = 0 时 parent_bounds 形状 (0, 2)、parent_vecs (0, D)、
    hit_to_parent (0,)，chunk_vecs 照常返回。

    错误约定：token_emb 非二维或非浮点、attention_mask 长度不符、
    chunk_bounds 形状不是 (C, 2) 或 dtype 非整型、块区间满足
    start > end 或越出 [0, L]、chunk_bounds 的 start 乱序、hit_chunks
    非一维/非整型或其值越出 [0, C)、parent_size < 1，均抛 ValueError。
    """
    # TODO
    raise NotImplementedError
