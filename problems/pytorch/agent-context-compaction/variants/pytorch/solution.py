def _check(segments, budget, pinned_priority):
    if not isinstance(segments, (list, tuple)):
        raise ValueError("segments 必须是 list")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError("budget 必须是非负整数")
    if not isinstance(pinned_priority, int) or isinstance(pinned_priority, bool):
        raise ValueError("pinned_priority 必须是整数")
    for item in segments:
        if len(item) != 3:
            raise ValueError("每个片段必须是 (name, tokens, priority)")
        _name, tokens, priority = item
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise ValueError("tokens 必须是非负整数")
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise ValueError("priority 必须是整数")


def solve(segments, budget, pinned_priority=100):
    _check(segments, budget, pinned_priority)

    pinned = [i for i, (_, _, priority) in enumerate(segments) if priority >= pinned_priority]
    pinned_tokens = sum(segments[i][1] for i in pinned)
    if pinned_tokens > budget:
        raise ValueError(f"钉住片段已需 {pinned_tokens} token，超出预算 {budget}")

    # 优先级降序、同优先级按原始下标升序，保证确定性
    order = sorted(range(len(segments)), key=lambda i: (-segments[i][2], i))

    used = 0
    kept = []
    for i in order:
        tokens = segments[i][1]
        if used + tokens > budget:
            continue  # 装不下就跳过，继续看后面的小块
        used += tokens
        kept.append(i)

    kept.sort()  # 回到时间顺序
    return kept, used
