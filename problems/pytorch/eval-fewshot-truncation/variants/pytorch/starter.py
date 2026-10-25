import torch

# 玩具分词器：判题环境没有 HF tokenizer，本题自带一个确定性的字符/词混合分词器。
VOCAB = {
    "<unk>": 0,
    "the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5, "a": 6, "dog": 7, "is": 8,
    "nice": 9, "Question:": 10, "Answer:": 11, "True": 12, "False": 13,
    "yes": 14, "no": 15, "here": 16, "there": 17,
}


def tokenize(text):
    """按空白切词；词在 VOCAB 内 -> 1 个 token；不在 -> 展开为逐字符 token（每字符记 id 0）。
    返回 token id 列表，**它的长度就是该串的 token 数**。"""
    ids = []
    for word in text.split():
        if word in VOCAB:
            ids.append(VOCAB[word])
        else:
            ids.extend([0] * len(word))
    return ids


def solve(demos, query, tail, budget):
    """few-shot 提示组装 + 预算截断。demos: list[str]（固定顺序的演示文本）、query: str、
    tail: str（必须存活到最后的标签尾巴，如 "Answer:"）、budget: int（整条 prompt 的 token 上限）。

    组装顺序：选中的演示（按给定顺序）→ query → tail；预算先被 query 与 tail 占住，它们**永不裁剪**。
    贪心规则：依次尝试第 0、1、2… 条演示，若「已用 token + 该演示 token 数 <= budget」就收下并累加，
    否则**立刻停止**（硬截断，不许跳过长的去捡后面的短的）。
    返回二元组 (kept, final_len)：kept 是 int64 张量 (m,)（被保留演示的下标，升序），
    final_len 是 Python int（组装后 prompt 的总 token 数，等于 len(tokenize(拼好的串))）。"""
    # TODO
    raise NotImplementedError
