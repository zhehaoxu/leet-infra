from itertools import permutations, product


def _oracle(dims, order):
    coords = [dict(zip(reversed(order), values))
              for values in product(*(range(dims[a]) for a in reversed(order)))]
    all_groups = {}
    for a in order:
        unique = {tuple(j for j, candidate in enumerate(coords)
                        if all(candidate[b] == coord[b] for b in order if b != a))
                  for coord in coords}
        all_groups[a] = sorted(unique)
    result = []
    for rank, coord in enumerate(coords):
        groups, local, ids = {}, {}, {}
        for a in order:
            gid, members = next((i, members) for i, members in enumerate(all_groups[a]) if rank in members)
            groups[a] = list(members)
            local[a] = members.index(rank)
            ids[a] = gid
        result.append({"rank": rank, "coord": coord, "groups": groups,
                       "local_ranks": local, "group_ids": ids})
    return result


def check_example(solve):
    infos = solve(8, {"tp": 2, "dp": 4}, ["tp", "dp"])
    assert infos[0] == {"rank": 0, "coord": {"tp": 0, "dp": 0},
                        "groups": {"tp": [0, 1], "dp": [0, 2, 4, 6]},
                        "local_ranks": {"tp": 0, "dp": 0}, "group_ids": {"tp": 0, "dp": 0}}
    assert infos[1]["groups"] == {"tp": [0, 1], "dp": [1, 3, 5, 7]}
    assert infos[1]["group_ids"] == {"tp": 0, "dp": 1}


check_example.example = True


def check_permuted_axes(solve):
    dims = {"tp": 2, "cp": 3, "dp": 2, "pp": 2}
    for order in permutations(dims):
        before = dims.copy()
        assert solve(24, dims, list(order)) == _oracle(dims, list(order))
        assert dims == before


def check_group_partition_and_dense_ids(solve):
    dims, order = {"tp": 2, "cp": 2, "dp": 4}, ["dp", "tp", "cp"]
    infos = solve(16, dims, order)
    for axis in order:
        groups = {tuple(info["groups"][axis]) for info in infos}
        assert sorted(r for members in groups for r in members) == list(range(16))
        assert {info["group_ids"][axis] for info in infos} == set(range(16 // dims[axis]))
        for info in infos:
            members = info["groups"][axis]
            assert members == sorted(members)
            assert members[info["local_ranks"][axis]] == info["rank"]
            for other in members:
                assert infos[other]["groups"][axis] == members
                assert infos[other]["group_ids"][axis] == info["group_ids"][axis]


def check_unit_axes_and_single_axis(solve):
    for dims, order in [({"tp": 1, "dp": 1}, ["dp", "tp"]),
                        ({"pp": 4}, ["pp"]),
                        ({"tp": 1, "cp": 3, "dp": 1}, ["dp", "tp", "cp"])]:
        size = 1
        for value in dims.values():
            size *= value
        assert solve(size, dims, order) == _oracle(dims, order)


def check_invalid_layout(solve):
    for world, dims, order in [(8, {"tp": 2, "dp": 3}, ["tp", "dp"]),
                               (8, {"tp": 2, "dp": 4}, ["tp"]),
                               (2, {"tp": 2, "dp": 1}, ["tp", "dp", "dp"]),
                               (8, {"tp": 2, "dp": 4}, ["tp", "cp"])]:
        try:
            solve(world, dims, order)
        except ValueError:
            continue
        raise AssertionError((world, dims, order))
