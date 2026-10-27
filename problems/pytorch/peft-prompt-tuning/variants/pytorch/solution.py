import torch


def solve(pool, task_ids, E):
    """Prompt-tuning：按任务行索引 gather 软提示池，前缀在前拼接原嵌入。
    pool:(P,d)；task_ids:(T,L) 池行索引；E:(S0,d)。
    返回 (X:(T,L+S0,d) float32, n_params:int=T*L*d)。禁止现成 PEFT/transformers 库。"""
    T, L = task_ids.shape
    d = E.shape[1]
    rows = pool.to(torch.float64)[task_ids.to(torch.long)]          # (T,L,d)
    emb = E.to(torch.float64).unsqueeze(0).expand(T, -1, -1)        # (T,S0,d)
    X = torch.cat([rows, emb], dim=1)
    return X.to(torch.float32), int(T * L * d)
