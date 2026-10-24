import torch


def solve(seqs: list, block_size: int, pad_token_id: int) -> dict:
    """把多条变长序列首尾拼接成一条流，切成定长 block_size 的块，仅最后一块补 pad。

    返回 dict，四个 torch.long 张量，形状 [num_blocks, block_size]：
      input_ids      : 打包 token，末尾补 pad_token_id
      position_ids   : 每 token 在其原子序列内的下标（每条子序列从 0 起，跨块继续），pad=0
      segment_ids    : 每 token 所属子序列序号（0,1,2...，跨块连续），pad=-1
      attention_mask : 真实=1, pad=0
    num_blocks = ceil(sum(len(s)) / block_size)。
    """
    # TODO: 实现序列打包
    raise NotImplementedError
