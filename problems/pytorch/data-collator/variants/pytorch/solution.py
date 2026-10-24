import torch


def solve(seqs: list, pad_token_id: int, ignore_index: int = -100) -> dict:
    B = len(seqs)
    L = max(len(s) for s in seqs)
    input_ids = torch.full((B, L), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)
    labels = torch.full((B, L), ignore_index, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = len(s)
        row = torch.tensor(s, dtype=torch.long)
        input_ids[i, :n] = row
        attention_mask[i, :n] = 1
        labels[i, :n] = row
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
