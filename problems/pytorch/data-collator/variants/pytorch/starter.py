import torch


def solve(seqs: list, pad_token_id: int, ignore_index: int = -100) -> dict:
    """把变长 token 序列右侧补齐成 batch，产出 input_ids / attention_mask / labels。

    seqs : list[list[int]]，每条长度可不同（均 >= 1）
    返回 dict，三个 torch.long 张量，形状 [B, L]，L = max(len(s))：
      input_ids      : 右侧补 pad_token_id
      attention_mask : 真实=1, pad=0
      labels         : 同 input_ids，但 pad 位置填 ignore_index
    """
    # TODO: 实现批整理（padding + mask + labels）
    raise NotImplementedError
