def solve(block_size, requests):
    """Automatic Prefix Caching 命中判定。"""
    bs = int(block_size)
    cache = set()
    hits = []
    for req in requests:
        n_full = len(req) // bs
        prefixes = []
        acc = []
        for j in range(n_full):
            acc = acc + list(req[j * bs:(j + 1) * bs])
            prefixes.append(tuple(acc))
        h = 0
        for key in prefixes:
            if key in cache:
                h += 1
            else:
                break
        hits.append(h)
        for key in prefixes:
            cache.add(key)
    return hits
