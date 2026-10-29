import torch


def solve(weight):
    """把 2:4 稀疏矩阵编码成 Tensor Core 使用的「压缩值 + 元数据」两段式布局。
    weight float32 (R,K)，K%4==0，且每一组 4 个连续元素中恰好有 2 个非零（非零 = != 0.0）。
    返回 (values, meta)：
      values float32 (R, K//2)：每组按索引升序存放 2 个保留值；
      meta   uint8   (R, P//2)：每组一个 4-bit 描述符 nib = i0 | ((i1-1) << 2)（i0<i1 为组内索引），
             低 nibble 在前（第 2j 组在 byte j 的低 4 位、第 2j+1 组在高 4 位），
             每行 nibble 数补零到 P = ceil((K//4)/8)*8。
    禁止调用 torch.sparse / cuSPARSELt。"""
    # TODO
    raise NotImplementedError
