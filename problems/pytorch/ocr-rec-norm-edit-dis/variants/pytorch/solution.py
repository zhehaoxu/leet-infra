import string


def _normalize_text(text):
    # 官方形状：filter(lambda x: x in (digits + letters), text) 再 lower
    keep = string.digits + string.ascii_letters
    return "".join(filter(lambda x: x in keep, text)).lower()


def _levenshtein(a, b):
    # 经典三操作 DP（insert/delete/substitute 权重均为 1），滚动数组
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def solve(pairs, ignore_space=True, is_filter=False):
    if not isinstance(pairs, (list, tuple)):
        raise ValueError("pairs must be a list of [pred, target] pairs")
    if not isinstance(ignore_space, bool) or not isinstance(is_filter, bool):
        raise ValueError("ignore_space and is_filter must be bools")
    for pr in pairs:
        if not isinstance(pr, (list, tuple)) or len(pr) != 2:
            raise ValueError("each pair must be [pred, target]")
        for s in pr:
            if not isinstance(s, str):
                raise ValueError("pair entries must be strings")

    eps = 1e-5
    rows = []
    sum_ned = 0.0
    exact_cnt = 0
    for pred_raw, target_raw in pairs:
        pred = pred_raw.replace(" ", "") if ignore_space else pred_raw
        target = target_raw.replace(" ", "") if ignore_space else target_raw
        if is_filter:
            pred = _normalize_text(pred)
            target = _normalize_text(target)
        dist = _levenshtein(pred, target)
        maximum = max(len(pred), len(target))
        ned = dist / maximum if maximum else 0.0
        exact = pred == target
        if exact:
            exact_cnt += 1
        sum_ned += ned
        rows.append(
            {"pred": pred, "target": target, "dist": dist, "ned": ned, "exact": exact}
        )
    n = len(rows)
    return {
        "rows": rows,
        "acc": exact_cnt / (n + eps),
        "norm_edit_dis": 1 - sum_ned / (n + eps),
    }
