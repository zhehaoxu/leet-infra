"""参考答案：token 预算下的 few-shot 演示截断（自带玩具分词器，不依赖 HF tokenizer）。"""

import torch

VOCAB = {
    "<unk>": 0,
    "the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5, "a": 6, "dog": 7, "is": 8,
    "nice": 9, "Question:": 10, "Answer:": 11, "True": 12, "False": 13,
    "yes": 14, "no": 15, "here": 16, "there": 17,
}


def tokenize(text):
    """按空白切词；词在 VOCAB 内 -> 1 个 token，否则展开成逐字符 token（每字符 1 个 <unk>）。"""
    ids = []
    for word in text.split():
        if word in VOCAB:
            ids.append(VOCAB[word])
        else:
            ids.extend([0] * len(word))
    return ids


def solve(demos, query, tail, budget):
    used = len(tokenize(query)) + len(tokenize(tail))   # 查询与标签尾巴永远保留
    kept = []
    for idx, demo in enumerate(demos):
        if used + len(tokenize(demo)) <= budget:        # 恰好用满也允许
            kept.append(idx)
            used += len(tokenize(demo))
        else:
            break                                       # 硬截断：后面的更短也不许回头捡
    return torch.tensor(kept, dtype=torch.long), used
