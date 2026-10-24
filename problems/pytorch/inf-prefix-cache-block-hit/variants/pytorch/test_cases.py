def _oracle(block_size, requests):
    """独立 trie 实现：每个满块作为一层，共享前缀命中到对应深度。"""
    bs = int(block_size)
    root = {}
    hits = []
    for req in requests:
        n_full = len(req) // bs
        path = [tuple(req[j * bs:(j + 1) * bs]) for j in range(n_full)]
        node = root
        h = 0
        still = True
        for block in path:
            if still and block in node:
                h += 1
            else:
                still = False
            if block not in node:
                node[block] = {}
            node = node[block]
        hits.append(h)
    return hits


def check_example(solve):
    bs = 4
    reqs = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14],
        [7, 7, 7, 7, 8, 8, 8, 8],
        [1, 2, 3, 4, 9, 9, 9, 9],
        [1, 2, 3, 4, 5, 6, 7],
    ]
    assert solve(bs, reqs) == [0, 2, 0, 1, 1], solve(bs, reqs)


def check_partial_tail_ignored(solve):
    # 长度不足一块 → 无满块 → 命中 0
    assert solve(4, [[1, 2, 3]]) == [0]
    # 完全相同前缀的重复请求，第二次整段命中
    assert solve(2, [[1, 2, 3, 4], [1, 2, 3, 4]]) == [0, 2]


def check_position_sensitive(solve):
    # 同内容但顺序不同 → 前缀链不同，第 2 请求首块即 [3,4] 未命中
    assert solve(2, [[1, 2, 3, 4], [3, 4, 1, 2]]) == [0, 0]


def check_matches_oracle(solve):
    import random
    random.seed(0)
    for _ in range(300):
        bs = random.randint(1, 4)
        nreq = random.randint(1, 6)
        reqs = [[random.randint(0, 3) for _ in range(random.randint(0, 12))] for _ in range(nreq)]
        assert solve(bs, reqs) == _oracle(bs, reqs), (bs, reqs)


check_example.example = True
