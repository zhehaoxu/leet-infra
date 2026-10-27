import torch


def solve(pool, task_ids, E):
    """Prompt-tuning：按任务行索引 gather 软提示池，前缀在前拼接原嵌入。
    pool:(P,d)；task_ids:(T,L) 池行索引；E:(S0,d)。
    返回 (X:(T,L+S0,d) float32, n_params:int=T*L*d)。禁止现成 PEFT/transformers 库。"""
    # TODO
    raise NotImplementedError
