def solve(spec, masks, axis="time", replace_with_zero=True):
    """FunASR `mask_along_axis` 段生成与布尔并合并账（段显式入参）。

    返回 {"masked": 谱, "segments": [[start, end)...]}（区段左闭右开、升序）。
    """
    T = len(spec)
    F = len(spec[0]) if T else 0
    D = T if axis == "time" else F

    # 官方：mask = (mask_pos <= aran) * (aran < mask_pos + mask_length)，
    # 再 any(dim=1) —— 这里直接构造布尔并
    mask = [False] * D
    for pos, width in masks:
        for d in range(D):
            if pos <= d < pos + width:
                mask[d] = True

    # 布尔并后的连续区段（any 的确定性呈现：相邻段连成一节）
    segments = []
    i = 0
    while i < D:
        if mask[i]:
            j = i
            while j < D and mask[j]:
                j += 1
            segments.append([i, j])
            i = j
        else:
            i += 1

    # 填充值：官方 replace_with_zero=False 时 value = spec.mean()，
    # 对原始全谱（含将被 mask 的元素）计算
    if replace_with_zero:
        value = 0.0
    else:
        value = sum(v for row in spec for v in row) / (T * F)

    masked = [row[:] for row in spec]
    for d in range(D):
        if mask[d]:
            if axis == "time":
                masked[d] = [value] * F
            else:
                for t in range(T):
                    masked[t][d] = value

    return {"masked": masked, "segments": segments}
