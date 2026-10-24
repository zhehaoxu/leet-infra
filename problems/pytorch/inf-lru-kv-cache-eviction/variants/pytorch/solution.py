def solve(capacity, accesses):
    """LRU 驱逐：list 队首=MRU。"""
    capacity = int(capacity)
    cache = []  # front = MRU, back = LRU
    hits = 0
    misses = 0
    for a in accesses.tolist():
        a = int(a)
        if capacity == 0:
            misses += 1
            continue
        if a in cache:
            hits += 1
            cache.remove(a)
            cache.insert(0, a)
        else:
            misses += 1
            if len(cache) >= capacity:
                cache.pop()  # evict LRU
            cache.insert(0, a)
    return hits, misses, cache
