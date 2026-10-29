def solve(input_ids, special_mask, sequence_ids, max_length,
          strategy="longest_first", keep="head"):
    if type(max_length) is not int or max_length < 0:
        raise ValueError("max_length must be a nonnegative integer")
    if strategy not in ("only_first", "only_second", "longest_first"):
        raise ValueError("unknown strategy")
    if keep not in ("head", "tail", "middle"):
        raise ValueError("unknown retention policy")
    if sum(special_mask) > max_length:
        raise ValueError("protected positions exceed the budget")
    positions = [[i for i, sid in enumerate(sequence_ids) if sid == s] for s in (0, 1)]
    remaining = [len(p) for p in positions]
    remove = max(0, len(input_ids) - max_length)
    if strategy != "longest_first":
        target = int(strategy == "only_second")
        if remaining[target] < remove:
            raise ValueError("selected sequence has insufficient removable tokens")
        remaining[target] -= remove
    else:
        for _ in range(remove):
            target = 0 if remaining[0] > remaining[1] else 1
            remaining[target] -= 1
    retained = set(i for i, special in enumerate(special_mask) if special)
    for indices, quota in zip(positions, remaining):
        if keep == "head":
            retained.update(indices[:quota])
        elif keep == "tail":
            retained.update(indices[len(indices) - quota:])
        else:
            front, back = (quota + 1) // 2, quota // 2
            retained.update(indices[:front])
            retained.update(indices[len(indices) - back:])
    kept = sorted(retained)
    return {
        "input_ids": [input_ids[i] for i in kept],
        "special_mask": [special_mask[i] for i in kept],
        "sequence_ids": [sequence_ids[i] for i in kept],
        "kept_indices": kept,
        "removed_counts": [len(positions[s]) - remaining[s] for s in (0, 1)],
    }
