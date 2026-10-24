def _merge_pair(pair, syms):
    first, second = pair
    out = []
    i = 0
    n = len(syms)
    while i < n:
        if i < n - 1 and syms[i] == first and syms[i + 1] == second:
            out.append(first + second)
            i += 2
        else:
            out.append(syms[i])
            i += 1
    return out


def solve(word: str, merges: list) -> list:
    rank = {tuple(m): i for i, m in enumerate(merges)}
    syms = list(word) + ["</w>"]
    while True:
        pairs = [(syms[i], syms[i + 1]) for i in range(len(syms) - 1)]
        candidates = [p for p in pairs if p in rank]
        if not candidates:
            break
        best = min(candidates, key=lambda p: rank[p])
        syms = _merge_pair(best, syms)
    return syms
