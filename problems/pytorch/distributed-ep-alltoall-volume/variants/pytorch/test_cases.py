import math

import torch


def _oracle(routing, bytes_per_token, bandwidth):
    world_size, experts = routing.shape
    send = [[0] * world_size for _ in range(world_size)]
    recv = [[0] * world_size for _ in range(world_size)]
    network = [0] * world_size
    for source in range(world_size):
        for expert in range(experts):
            destination = expert % world_size
            tokens = int(routing[source, expert])
            send[source][destination] += tokens
            recv[destination][source] += tokens
            if source != destination:
                network[source] += tokens * bytes_per_token
                network[destination] += tokens * bytes_per_token
    maximum = max(network)
    mean = sum(network) / world_size
    return (
        torch.tensor(send, dtype=torch.int64, device=routing.device),
        torch.tensor(recv, dtype=torch.int64, device=routing.device),
        torch.tensor(network, dtype=torch.int64, device=routing.device),
        maximum,
        maximum / mean - 1.0 if mean else 0.0,
        maximum / bandwidth,
    )


def _assert_result(got, expected, device):
    assert isinstance(got, tuple) and len(got) == 6
    for actual, target in zip(got[:3], expected[:3]):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == torch.int64 and actual.device == device
        assert torch.equal(actual, target), (actual, target)
    assert type(got[3]) is int and got[3] == expected[3]
    for actual, target in zip(got[4:], expected[4:]):
        assert type(actual) is float
        assert math.isclose(actual, target, rel_tol=1e-12, abs_tol=1e-15), (actual, target)


def check_example(solve):
    routing = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    expected = (torch.tensor([[4, 2], [10, 5]], dtype=torch.int64),
                torch.tensor([[4, 10], [2, 5]], dtype=torch.int64),
                torch.tensor([48, 48], dtype=torch.int64), 48, 0.0, 4.8e-8)
    _assert_result(solve(routing, 4, 1e9), expected, routing.device)


check_example.example = True


def check_expert_loop_oracle(solve):
    generator = torch.Generator().manual_seed(76)
    for world_size, experts in ((2, 3), (3, 7), (4, 5), (5, 3), (4, 8), (8, 17)):
        for _ in range(4):
            routing = torch.randint(0, 20, (world_size, experts), generator=generator,
                                    dtype=torch.int64)
            before = routing.clone()
            _assert_result(solve(routing, 7, 3e8), _oracle(routing, 7, 3e8), routing.device)
            assert torch.equal(routing, before)


def check_single_rank(solve):
    routing = torch.tensor([[3, 5, 7]], dtype=torch.int64)
    got = solve(routing, 8, 1e9)
    _assert_result(got, _oracle(routing, 8, 1e9), routing.device)
    assert got[0].tolist() == [[15]] and got[2].tolist() == [0]
    assert got[3:] == (0, 0.0, 0.0)


def check_all_local(solve):
    routing = torch.zeros(4, 9, dtype=torch.int64)
    for expert in range(9):
        routing[expert % 4, expert] = expert + 1
    got = solve(routing, 16, 2e9)
    _assert_result(got, _oracle(routing, 16, 2e9), routing.device)
    assert got[0].diagonal().sum() == routing.sum()
    assert got[2].tolist() == [0, 0, 0, 0] and got[3:] == (0, 0.0, 0.0)


def check_zero_and_no_experts(solve):
    for shape in ((3, 5), (1, 0), (4, 0)):
        routing = torch.zeros(shape, dtype=torch.int64)
        got = solve(routing, 4, 1e9)
        _assert_result(got, _oracle(routing, 4, 1e9), routing.device)
        assert got[3:] == (0, 0.0, 0.0)


def check_asymmetric_network(solve):
    routing = torch.tensor([[9, 2, 1, 4, 3], [7, 11, 0, 2, 6],
                            [0, 5, 13, 4, 1]], dtype=torch.int64)
    got = solve(routing, 4, 100.0)
    _assert_result(got, _oracle(routing, 4, 100.0), routing.device)
    assert not torch.equal(got[0], got[1])
    assert got[2].tolist() == [76, 80, 44]
    assert got[3] == 80 and got[5] == 0.8


def check_off_diagonal_straggler(solve):
    routing = torch.zeros(4, 8, dtype=torch.int64)
    routing[0, 1] = 1000
    got = solve(routing, 4, 1e9)
    _assert_result(got, _oracle(routing, 4, 1e9), routing.device)
    assert got[2].tolist() == [4000, 4000, 0, 0]
    assert got[3:] == (4000, 1.0, 4e-6)


def check_uniform_network(solve):
    routing = torch.full((4, 8), 10, dtype=torch.int64)
    got = solve(routing, 2, 1e9)
    _assert_result(got, _oracle(routing, 2, 1e9), routing.device)
    assert got[2].tolist() == [240] * 4 and got[4] == 0.0


def check_large_counts_without_float32_mean(solve):
    for count in (2**24 + 1, 2**40 + 7):
        routing = torch.tensor([[0, count], [0, 0]], dtype=torch.int64)
        got = solve(routing, 1, 1e9)
        _assert_result(got, _oracle(routing, 1, 1e9), routing.device)
        assert got[4] == 0.0
    routing = torch.tensor([[0, 2**24 + 1, 0], [0, 0, 3], [7, 0, 0]], dtype=torch.int64)
    _assert_result(solve(routing, 3, 1e9), _oracle(routing, 3, 1e9), routing.device)


def check_scaling_and_local_invariance(solve):
    routing = torch.tensor([[1, 4, 3], [7, 2, 0], [5, 6, 9]], dtype=torch.int64)
    base = solve(routing, 2, 1e6)
    for routes, token_bytes, bandwidth in ((routing, 14, 1e6), (routing, 2, 5e6),
                                          (routing * 3, 2, 1e6)):
        _assert_result(solve(routes, token_bytes, bandwidth),
                       _oracle(routes, token_bytes, bandwidth), routes.device)
    byte_scaled = solve(routing, 14, 1e6)
    assert torch.equal(byte_scaled[2], base[2] * 7)
    assert math.isclose(byte_scaled[5], base[5] * 7, rel_tol=1e-12)
    assert byte_scaled[4] == base[4]
    assert math.isclose(solve(routing, 2, 5e6)[5], base[5] / 5, rel_tol=1e-12)
    with_local = routing + torch.eye(3, dtype=torch.int64) * 100000000
    local_result = solve(with_local, 2, 1e6)
    assert torch.equal(local_result[2], base[2]) and local_result[3:] == base[3:]
    assert int(local_result[0].sum()) == int(base[0].sum()) + 300000000


def check_noncontiguous_routing(solve):
    routing = torch.arange(15, dtype=torch.int64).reshape(5, 3).T
    assert not routing.is_contiguous()
    _assert_result(solve(routing, 3, 7e8), _oracle(routing, 3, 7e8), routing.device)
