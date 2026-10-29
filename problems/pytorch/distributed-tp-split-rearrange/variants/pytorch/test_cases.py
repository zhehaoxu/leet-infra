import torch
import torch.nn.functional as F


def _assert_case(solve, W, tp_size, mode, x, targets=None):
    # Dense PyTorch is an independent numerical reference, not a TP algorithm.
    original_W = W.clone()
    original_x = x.clone()
    original_targets = None if targets is None else targets.clone()
    expected_output = original_x @ original_W.T
    expected_losses = (
        F.cross_entropy(expected_output, original_targets.long(), reduction='none')
        if mode == 'vocab' else None
    )
    got = solve(W, tp_size, mode, x, targets)
    assert isinstance(got, dict)
    assert set(got) == {'shards', 'label_map', 'output', 'losses'}
    assert torch.equal(W, original_W), 'W was mutated'
    assert torch.equal(x, original_x), 'x was mutated'
    if targets is not None:
        assert torch.equal(targets, original_targets), 'targets were mutated'

    axis = 1 if mode == 'row' else 0
    width = W.shape[axis] // tp_size
    assert isinstance(got['shards'], list) and len(got['shards']) == tp_size
    for rank, shard in enumerate(got['shards']):
        begin, end = rank * width, (rank + 1) * width
        expected = original_W[:, begin:end] if axis == 1 else original_W[begin:end, :]
        assert isinstance(shard, torch.Tensor)
        assert shard.shape == expected.shape and shard.dtype == W.dtype
        assert shard.device == W.device and torch.equal(shard, expected)

    expected_map = {}
    if mode == 'vocab':
        for rank in range(tp_size):
            for local in range(width):
                expected_map[rank * width + local] = (rank, local)
    assert isinstance(got['label_map'], dict) and got['label_map'] == expected_map

    tolerance = 1e-10 if W.dtype == torch.float64 else 1e-5
    output = got['output']
    assert isinstance(output, torch.Tensor) and output.shape == expected_output.shape
    assert output.dtype == W.dtype and output.device == W.device
    torch.testing.assert_close(output, expected_output, atol=tolerance, rtol=tolerance)
    if mode == 'vocab':
        losses = got['losses']
        assert isinstance(losses, torch.Tensor) and losses.shape == (x.shape[0],)
        assert losses.dtype == W.dtype and losses.device == W.device
        assert torch.isfinite(losses).all(), 'CE must remain finite for extreme logits'
        torch.testing.assert_close(losses, expected_losses, atol=tolerance, rtol=tolerance)
    else:
        assert got['losses'] is None
    return got


def check_example(solve):
    W = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]], dtype=torch.float64)
    x = torch.tensor([[2., 1.], [-1., 3.]], dtype=torch.float64)
    targets = torch.tensor([2, 0])
    got = _assert_case(solve, W, 2, 'vocab', x, targets)
    print('W =', W.tolist())
    print('x =', x.tolist())
    print("tp_size = 2, mode = 'vocab', targets =", targets.tolist())
    print({
        'shards': [shard.tolist() for shard in got['shards']],
        'label_map': got['label_map'],
        'output': got['output'].tolist(),
        'losses': [round(value, 6) for value in got['losses'].tolist()],
    })


check_example.example = True


def check_column_nonsquare(solve):
    generator = torch.Generator().manual_seed(12)
    W = torch.randn(6, 5, generator=generator, dtype=torch.float64)
    x = torch.randn(4, 5, generator=generator, dtype=torch.float64)
    for tp_size in (1, 2, 3, 6):
        _assert_case(solve, W, tp_size, 'column', x)
    # The optional targets default must work for non-vocab modes.
    got = solve(W, 2, 'column', x)
    torch.testing.assert_close(got['output'], x @ W.T)
    assert got['losses'] is None and got['label_map'] == {}


def check_row_nonsquare_reduction(solve):
    generator = torch.Generator().manual_seed(23)
    W = torch.randn(5, 12, generator=generator, dtype=torch.float64)
    x = torch.randn(4, 12, generator=generator, dtype=torch.float64)
    for tp_size in (1, 2, 3, 4, 6, 12):
        _assert_case(solve, W, tp_size, 'row', x)
    got = solve(W, 3, 'row', x)
    torch.testing.assert_close(got['output'], x @ W.T)
    assert got['losses'] is None and got['label_map'] == {}


def check_vocab_target_shard_boundaries(solve):
    generator = torch.Generator().manual_seed(34)
    W = torch.randn(12, 5, generator=generator, dtype=torch.float64)
    for tp_size in (1, 2, 3, 4, 6, 12):
        width = 12 // tp_size
        targets = torch.tensor([
            label for rank in range(tp_size)
            for label in (rank * width, (rank + 1) * width - 1)
        ])
        x = torch.randn(len(targets), 5, generator=generator, dtype=torch.float64)
        _assert_case(solve, W, tp_size, 'vocab', x, targets)


def check_vocab_label_permutation(solve):
    generator = torch.Generator().manual_seed(45)
    W = torch.randn(12, 5, generator=generator, dtype=torch.float64)
    x = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    targets = torch.tensor([0, 3, 4, 7, 8, 11, 4])
    permutation = torch.tensor([8, 0, 5, 11, 2, 7, 4, 10, 1, 9, 6, 3])
    inverse = torch.argsort(permutation)
    original = _assert_case(solve, W, 3, 'vocab', x, targets)
    permuted = _assert_case(solve, W[permutation], 3, 'vocab', x, inverse[targets])
    torch.testing.assert_close(permuted['output'], original['output'][:, permutation])
    torch.testing.assert_close(permuted['losses'], original['losses'])


def check_vocab_extreme_positive_negative_logits(solve):
    x = torch.tensor([[1.], [-1.], [0.]], dtype=torch.float64)
    cases = [
        # All-positive / all-negative samples: naive exp over/underflows.
        ([10000., 9999., 10002., 9998., 10001., 10003.], [0, 5, 2]),
        # Targets far below the maximum on different owner ranks.
        ([10000., -10000., 9999., -9999., 0., -5000.], [1, 0, 4]),
        # Restoring this common offset before subtraction loses log(6).
        ([1e16] * 6, [0, 5, 3]),
    ]
    for values, labels in cases:
        W = torch.tensor(values, dtype=torch.float64).reshape(6, 1)
        for tp_size in (1, 2, 3, 6):
            _assert_case(solve, W, tp_size, 'vocab', x, torch.tensor(labels))


def check_dtype_strides_and_input_preservation(solve):
    generator = torch.Generator().manual_seed(56)
    for dtype in (torch.float32, torch.float64):
        W_storage = torch.randn(6, 16, generator=generator, dtype=dtype)
        x_storage = torch.randn(5, 16, generator=generator, dtype=dtype)
        target_storage = torch.tensor([0, 1, 2, 3, 5, 0, 1, 2, 4, 3], dtype=torch.int32)
        W, x, targets = W_storage[:, ::2], x_storage[:, ::2], target_storage[::2]
        saved = (W_storage.clone(), x_storage.clone(), target_storage.clone())
        for mode in ('column', 'row', 'vocab'):
            _assert_case(solve, W, 2, mode, x, targets)
        for storage, before in zip((W_storage, x_storage, target_storage), saved):
            assert torch.equal(storage, before), 'input backing storage was mutated'


def check_single_label_and_uniform_logits(solve):
    _assert_case(solve, torch.tensor([[2., -3.]], dtype=torch.float64), 1,
                 'vocab', torch.tensor([[4., 5.]], dtype=torch.float64), torch.tensor([0]))
    W = torch.zeros(6, 4, dtype=torch.float64)
    x = torch.ones(3, 4, dtype=torch.float64)
    for mode in ('column', 'row', 'vocab'):
        _assert_case(solve, W, 2, mode, x, torch.tensor([0, 2, 5]))


def check_declared_value_errors(solve):
    cases = [
        (torch.ones(4, 6, dtype=torch.float64), 2, 'invalid'),
        (torch.ones(5, 6, dtype=torch.float64), 2, 'column'),
        (torch.ones(4, 5, dtype=torch.float64), 2, 'row'),
        (torch.ones(5, 4, dtype=torch.float64), 2, 'vocab'),
    ]
    for W, tp_size, mode in cases:
        try:
            solve(W, tp_size, mode, torch.ones(2, W.shape[1], dtype=W.dtype),
                  torch.tensor([0, 1]))
        except ValueError:
            continue
        raise AssertionError(f'{mode}: expected ValueError')
