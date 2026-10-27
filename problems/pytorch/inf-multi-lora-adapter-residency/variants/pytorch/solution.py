import torch


def solve(requests, adapter_bytes, capacity, pinned):
    if capacity < 1:
        raise ValueError("capacity 必须为正")
    if not isinstance(requests, torch.Tensor) or requests.dim() != 1:
        raise ValueError("requests 必须是一维张量")
    if not isinstance(adapter_bytes, torch.Tensor) or adapter_bytes.dim() != 1:
        raise ValueError("adapter_bytes 必须是一维张量")
    if not isinstance(pinned, torch.Tensor) or pinned.dim() != 1:
        raise ValueError("pinned 必须是一维张量")
    n_adapter = int(adapter_bytes.numel())
    if n_adapter < 1:
        raise ValueError("至少需要一个 adapter")
    sizes = adapter_bytes.to(torch.int64).tolist()
    if min(sizes) < 1:
        raise ValueError("adapter_bytes 必须为正")
    reqs = requests.to(torch.int64).tolist()
    pin = pinned.to(torch.int64).tolist()
    if len(set(pin)) != len(pin):
        raise ValueError("pinned 不能重复")
    if len(pin) > capacity:
        raise ValueError("pinned 数量已超过 capacity")
    for a in pin + reqs:
        if a < 0 or a >= n_adapter:
            raise ValueError("adapter id 越界")

    resident = set(pin)
    lru = []          # 非 pinned 驻留者，头部最久未使用
    hits, evicted = [], []
    loads = 0
    rejected = 0
    peak = sum(sizes[a] for a in pin)

    for a in reqs:
        if a in resident:
            hits.append(True)
            evicted.append(-1)
            if a in lru:
                lru.remove(a)
                lru.append(a)
            continue

        hits.append(False)
        if len(resident) >= capacity:
            if not lru:
                rejected += 1
                evicted.append(-1)
                continue
            victim = lru.pop(0)
            resident.discard(victim)
            evicted.append(victim)
        else:
            evicted.append(-1)
        resident.add(a)
        lru.append(a)
        loads += 1
        peak = max(peak, sum(sizes[x] for x in resident))

    return (
        torch.tensor(hits, dtype=torch.bool),
        torch.tensor(evicted, dtype=torch.int64),
        torch.tensor(loads, dtype=torch.int64),
        torch.tensor(rejected, dtype=torch.int64),
        torch.tensor(peak, dtype=torch.int64),
    )
