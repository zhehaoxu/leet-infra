import torch


def _oracle(mm_type, grid_thw):
    L = len(mm_type)
    pos = [[0] * L for _ in range(3)]
    running = 0
    img_ptr = 0
    i = 0
    mt = [int(v) for v in mm_type.tolist()]
    while i < L:
        if mt[i] == 0:
            pos[0][i] = pos[1][i] = pos[2][i] = running
            running += 1
            i += 1
        else:
            h, w = int(grid_thw[img_ptr][1]), int(grid_thw[img_ptr][2])
            for k in range(h * w):
                r, c = divmod(k, w)
                pos[0][i + k] = running
                pos[1][i + k] = running + r
                pos[2][i + k] = running + c
            running += max(h, w)
            i += h * w
            img_ptr += 1
    return pos


def _build(spec):
    mt, grids = [], []
    for kind, *p in spec:
        if kind == 't':
            for _ in range(p[0]):
                mt.append(0)
        else:
            h, w = p
            grids.append([1, h, w])
            mt.extend([1] * (h * w))
    grid = torch.tensor(grids, dtype=torch.long).reshape(-1, 3)
    return torch.tensor(mt), grid


def check_example(solve):
    mt = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 0])
    grid = torch.tensor([[1, 2, 3]])
    y = solve(mt, grid)
    exp = [[0, 1, 2, 2, 2, 2, 2, 2, 5],
           [0, 1, 2, 2, 2, 3, 3, 3, 5],
           [0, 1, 2, 3, 4, 2, 3, 4, 5]]
    assert y.tolist() == exp, f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    specs = [
        [('t', 2)],
        [('img', 2, 3), ('t', 1)],
        [('t', 1), ('img', 3, 2), ('t', 2), ('img', 1, 4)],
        [('img', 2, 2)],
        [('t', 1), ('img', 2, 2), ('img', 3, 3), ('t', 2)],
    ]
    for spec in specs:
        mt, grid = _build(spec)
        y = solve(mt, grid)
        assert y.tolist() == _oracle(mt, grid), (spec, y.tolist())


def check_pure_text_monotonic(solve):
    mt = torch.tensor([0, 0, 0, 0])
    grid = torch.zeros(0, 3, dtype=torch.long)
    y = solve(mt, grid)
    exp = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    assert y.tolist() == exp, y.tolist()


def check_image_t_constant(solve):
    # 单个 3x2 图像块：t 行应全等于起始 running(=0)
    mt = torch.tensor([1, 1, 1, 1, 1, 1])
    grid = torch.tensor([[1, 3, 2]])
    y = solve(mt, grid)
    assert (y[0] == 0).all(), y[0].tolist()


def check_shape_dtype(solve):
    mt = torch.tensor([0, 1, 1, 1, 1])
    grid = torch.tensor([[1, 2, 2]])
    y = solve(mt, grid)
    assert y.shape == (3, 5) and y.dtype == torch.int64, (y.shape, y.dtype)
