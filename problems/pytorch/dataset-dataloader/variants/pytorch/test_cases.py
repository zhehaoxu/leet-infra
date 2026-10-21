import torch
from torch.utils.data import DataLoader


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_len(solve):
    ds = solve()
    assert isinstance(ds, torch.utils.data.Dataset), "必须返回 Dataset 实例"
    assert len(ds) == 100, f"len 应为 100, got {len(ds)}"


def check_getitem_values(solve):
    ds = solve()
    x, y = ds[7]
    assert torch.equal(x, torch.tensor([7.0, 14.0, 21.0, 28.0])), f"x 错误: got {x.tolist()}"
    assert y.item() == 1, f"y 应为 7%2=1, got {y.item()}"
    x0, y0 = ds[0]
    assert torch.equal(x0, torch.zeros(4)), f"ds[0] 的 x 应全 0, got {x0.tolist()}"
    assert y0.item() == 0, f"ds[0] 的 y 应为 0, got {y0.item()}"


def check_dtypes(solve):
    ds = solve()
    x, y = ds[3]
    assert x.dtype == torch.float32, f"x 应为 float32, got {x.dtype}"
    assert x.shape == (4,), f"x 形状应为 (4,), got {tuple(x.shape)}"
    assert y.dtype == torch.int64, f"y 应为 int64, got {y.dtype}"
    assert y.dim() == 0, f"y 应为 0 维标量张量, got dim={y.dim()}"


def check_dataloader_first_batch(solve):
    ds = solve()
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    x, y = next(iter(dl))
    assert x.shape == (8, 4), f"首批 x 形状应为 (8,4), got {tuple(x.shape)}"
    assert y.shape == (8,), f"首批 y 形状应为 (8,), got {tuple(y.shape)}"
    assert torch.equal(x[0], ds[0][0]), "shuffle=False 时首批第一行应为 ds[0]"
    assert torch.equal(x[1], ds[1][0]), "shuffle=False 时首批第二行应为 ds[1]"
    assert y.tolist() == [0, 1, 0, 1, 0, 1, 0, 1], f"标签列错误: got {y.tolist()}"


def check_index_bounds(solve):
    ds = solve()
    for bad in (100, -1):
        try:
            ds[bad]
        except IndexError:
            continue
        raise AssertionError(f"索引 {bad} 应抛 IndexError（负索引也不放行）")


check_getitem_values.example = True
