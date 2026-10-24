from collections import defaultdict


def _count_pairs(words):
    pairs = defaultdict(int)
    for syms, freq in words.items():
        for a, b in zip(syms, syms[1:]):
            pairs[(a, b)] += freq
    return pairs


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
    return tuple(out)


def solve(word_freqs, num_merges: int) -> list:
    words = defaultdict(int)
    for w, f in word_freqs:
        words[tuple(list(w) + ["</w>"])] += f

    merges = []
    for _ in range(num_merges):
        pairs = _count_pairs(words)
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        merges.append(best)
        new_words = defaultdict(int)
        for syms, freq in words.items():
            new_words[_merge_pair(best, syms)] += freq
        words = new_words
    return merges
