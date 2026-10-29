import torch


def solve(theta, alpha, variant, group_size, out_features, x):
    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError("group_size 必须是正偶数")
    if out_features <= 0 or out_features % group_size != 0:
        raise ValueError("out_features 必须能被 group_size 整除")
    if variant not in ("road_1", "road_2", "road_4"):
        raise ValueError("variant 只能是 road_1 / road_2 / road_4")

    mult = {"road_1": 1, "road_2": 2, "road_4": 4}[variant]
    want = out_features * mult // 2
    th = theta.to(torch.float64)
    al = alpha.to(torch.float64)
    if th.ndim != 1 or al.ndim != 1:
        raise ValueError("theta 与 alpha 都必须是一维")
    if th.numel() != want or al.numel() != want:
        raise ValueError(f"variant={variant} 时 theta/alpha 的长度必须是 {want}")
    if x.ndim < 1 or x.shape[-1] != out_features:
        raise ValueError("x 的最后一维必须等于 out_features")

    half = group_size // 2
    if variant == "road_1":
        th = th.reshape(-1, half).repeat_interleave(2, dim=0).flatten()
        al = al.reshape(-1, half).repeat_interleave(2, dim=0).flatten()
        first_col = al * th.cos()
        second_col = al * th.sin()
    elif variant == "road_2":
        first_col = al * th.cos()
        second_col = al * th.sin()
    else:
        th = th.reshape(-1, 2, group_size)
        al = al.reshape(-1, 2, group_size)
        first_col = al[:, 0, :].flatten() * th[:, 0, :].cos().flatten()
        second_col = al[:, 1, :].flatten() * th[:, 1, :].sin().flatten()

    size = out_features
    R = torch.diag(first_col)
    swapped = second_col.reshape(-1, 2, half)[:, [1, 0], :].flatten()
    rot = torch.diag(swapped).reshape(-1, 2, half, size)[:, [1, 0], :, :]
    rot[:, 0, :, :] *= -1
    R = R + rot.reshape(size, size)

    xf = x.to(torch.float64).reshape(-1, size)
    grouped = xf.reshape(-1, 2, half)
    rotate_half = torch.stack((-grouped[:, 1, :], grouped[:, 0, :]), dim=1)
    y = xf * first_col + rotate_half.reshape(xf.shape) * second_col
    return R.to(torch.float32), y.reshape(x.shape).to(torch.float32)
