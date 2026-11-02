import torch


def solve(scores, rois, k):
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k 必须是 >=1 的整数")
    if scores.dim() != 4:
        raise ValueError("scores 形状必须是 (C, k*k, H, W)")
    c, kk, h, w = scores.shape
    if kk != k * k:
        raise ValueError(f"scores 第二维应为 {k * k}，实际 {kk}")
    if rois.dim() != 2 or rois.shape[1] != 4:
        raise ValueError("rois 形状必须是 (N, 4)")
    if rois.numel() and not bool((rois == rois.round()).all()):
        raise ValueError("本题要求 rois 为整数特征坐标")

    n = int(rois.shape[0])
    s = scores.to(torch.float64)
    bounds, bins, sizes, empty = [], [], [], []
    pooled = torch.zeros(n, c, dtype=torch.float64)

    for r in range(n):
        x1 = int(min(max(rois[r, 0].item(), 0), w))
        y1 = int(min(max(rois[r, 1].item(), 0), h))
        x2 = int(min(max(rois[r, 2].item(), 0), w))
        y2 = int(min(max(rois[r, 3].item(), 0), h))
        rw = max(x2 - x1, 0)
        rh = max(y2 - y1, 0)
        bounds.append([x1, y1, x2, y2])
        row = []
        sz = []
        emp = []
        for i in range(k):
            rs = y1 + (i * rh) // k
            re = y1 + ((i + 1) * rh) // k
            for j in range(k):
                cs = x1 + (j * rw) // k
                ce = x1 + ((j + 1) * rw) // k
                cnt = (re - rs) * (ce - cs)
                ch = i * k + j
                if cnt == 0:
                    val = torch.zeros(c, dtype=torch.float64)
                else:
                    val = s[:, ch, rs:re, cs:ce].mean(dim=(1, 2))
                row.append(val)
                sz.append(cnt)
                emp.append(cnt == 0)
        block = torch.stack(row, dim=1)  # (C, k*k)
        bins.append(block)
        sizes.append(sz)
        empty.append(emp)
        pooled[r] = block.mean(dim=1)

    return {
        "pooled": pooled,
        "bins": torch.stack(bins) if n else torch.zeros((0, c, k * k), dtype=torch.float64),
        "bin_sizes": torch.tensor(sizes, dtype=torch.int64).reshape(n, k * k),
        "bounds": torch.tensor(bounds, dtype=torch.int64).reshape(n, 4),
        "empty": torch.tensor(empty, dtype=torch.bool).reshape(n, k * k),
    }
