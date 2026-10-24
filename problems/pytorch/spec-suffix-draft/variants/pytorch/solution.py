import torch


def solve(datastore, context, gamma, min_match):
    """REST / Suffix Decoding 最长后缀检索草稿 (arXiv:2311.08252 / 2411.04975)。

    对 context 求最长的、能作为 datastore 某条 entry 连续子串出现的后缀（长度 >= min_match），
    草稿即该匹配点紧随其后的至多 gamma 个 token。tie-break：更长后缀 -> 更早 entry -> 更早位置。
    """
    context = list(context)
    best_len = -1
    best = None
    for entry in datastore:
        entry = list(entry)
        # 后缀从长到短；一旦本 entry 已有不短于当前 n 的匹配即可停（更短的 n 不可能翻盘）
        for n in range(len(context), min_match - 1, -1):
            if n > len(entry):
                continue
            suffix = context[-n:]
            for i in range(len(entry) - n + 1):
                if entry[i:i + n] == suffix:
                    if n > best_len:
                        best_len = n
                        best = entry[i + n:i + n + gamma]
                    break  # 同 entry 同 n 取最早位置，之后不再看更后位置
            if best_len >= n:
                break
    if best is None:
        return torch.tensor([], dtype=torch.long)
    return torch.tensor(best, dtype=torch.long)
