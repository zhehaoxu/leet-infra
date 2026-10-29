import math


def solve(world_size, dims, order):
    if len(order) != len(dims) or set(order) != set(dims):
        raise ValueError("order must contain each axis exactly once")
    if math.prod(dims.values()) != world_size:
        raise ValueError("world_size must equal the product of axis sizes")
    strides = {}
    stride = 1
    for axis in order:
        strides[axis] = stride
        stride *= dims[axis]
    infos = []
    for rank in range(world_size):
        coord = {a: rank // strides[a] % dims[a] for a in order}
        groups = {}
        group_ids = {}
        for a in order:
            step = strides[a]
            base = rank - coord[a] * step
            groups[a] = [base + k * step for k in range(dims[a])]
            group_ids[a] = rank % step + rank // (step * dims[a]) * step
        infos.append({"rank": rank, "coord": coord, "groups": groups,
                      "local_ranks": dict(coord), "group_ids": group_ids})
    return infos
