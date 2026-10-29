import torch
import torch.nn.functional as F


# Recorded from solution.py and independently confirmed with native bicubic.
EXAMPLE_OUTPUT = [[[99.0], [-0.84375], [1.8125], [6.1875], [8.84375]]]


def _native(pos_embed, old_hw, new_hw, prefix_count):
    """Independent oracle: native size-based non-AA bicubic, not the explicit kernel."""
    dim = pos_embed.shape[-1]
    spatial = pos_embed[:, prefix_count:].reshape(1, *old_hw, dim).permute(0, 3, 1, 2)
    resized = F.interpolate(spatial.to(torch.float64), size=new_hw, mode="bicubic",
                            align_corners=False, antialias=False).to(pos_embed.dtype)
    return torch.cat((pos_embed[:, :prefix_count],
                      resized.permute(0, 2, 3, 1).reshape(1, -1, dim)), dim=1)


def _bits(tensor):
    integer_dtype = torch.int32 if tensor.dtype == torch.float32 else torch.int64
    return tensor.detach().contiguous().view(integer_dtype)


def _check(solve, pos_embed, old_hw, new_hw, prefix_count=1, use_default=False):
    before = pos_embed.clone()
    stride = pos_embed.stride()
    expected = _native(before, old_hw, new_hw, prefix_count)
    if use_default:
        out = solve(pos_embed, old_hw, new_hw)
    else:
        out = solve(pos_embed, old_hw, new_hw, num_prefix_tokens=prefix_count)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, prefix_count + new_hw[0] * new_hw[1], pos_embed.shape[-1])
    assert out.dtype == pos_embed.dtype and out.device.type == "cpu"
    assert torch.equal(_bits(pos_embed), _bits(before)), "input changed"
    assert pos_embed.stride() == stride
    assert torch.equal(_bits(out[:, :prefix_count]), _bits(before[:, :prefix_count])), \
        "prefix tokens must be bit-exact, not interpolated or downcast"
    tolerance = 1e-5 if pos_embed.dtype == torch.float32 else 1e-10
    torch.testing.assert_close(out[:, prefix_count:], expected[:, prefix_count:],
                               atol=tolerance, rtol=tolerance)
    if tuple(old_hw) == tuple(new_hw):
        assert torch.equal(_bits(out), _bits(before)), "identity must preserve all bits"
        assert out.data_ptr() != pos_embed.data_ptr(), "identity must clone"
        out.fill_(123.0)
        assert torch.equal(_bits(pos_embed), _bits(before)), "identity output aliases input"
    return out


def _positions(old_hw, prefix_count, dim, dtype, generator):
    length = prefix_count + old_hw[0] * old_hw[1]
    return torch.randn((1, length, dim), dtype=dtype, generator=generator)


def check_example(solve):
    pos = torch.tensor([[[99.0], [0.0], [8.0]]], dtype=torch.float32)
    out = _check(solve, pos, (1, 2), (1, 4), use_default=True)
    assert torch.equal(out, torch.tensor(EXAMPLE_OUTPUT, dtype=torch.float32))


check_example.example = True


def check_prefix_sentinels(solve):
    generator = torch.Generator().manual_seed(8610)
    for dtype, huge in ((torch.float32, 1e20), (torch.float64, 1e200)):
        pos = _positions((3, 5), 4, 4, dtype, generator)
        pos[:, :4] = torch.tensor([huge, -huge, -0.0, 0.0], dtype=dtype)
        out = _check(solve, pos, (3, 5), (7, 2), 4)
        no_prefix = _check(solve, pos[:, 4:], (3, 5), (7, 2), 0)
        assert torch.equal(out[:, 4:], no_prefix), "prefix leaked into spatial grid"


def check_height_width_and_row_major(solve):
    y = torch.arange(3, dtype=torch.float64)[:, None].expand(3, 5)
    x = torch.arange(5, dtype=torch.float64)[None, :].expand(3, 5)
    grid = torch.stack((y, x, y * 10 + x), dim=-1).reshape(1, 15, 3)
    prefix = torch.tensor([[[123.0, -456.0, 789.0]]], dtype=torch.float64)
    pos = torch.cat((prefix, grid), dim=1)
    for size in ((7, 2), (2, 7), (5, 3)):
        _check(solve, pos, (3, 5), size)


def check_zero_and_multiple_prefixes(solve):
    generator = torch.Generator().manual_seed(8611)
    for prefix_count in (0, 1, 2, 3, 4):
        for dtype in (torch.float32, torch.float64):
            pos = _positions((2, 7), prefix_count, 5, dtype, generator)
            _check(solve, pos, (2, 7), (5, 3), prefix_count)


def check_single_axis_and_single_cell(solve):
    generator = torch.Generator().manual_seed(8612)
    for dtype in (torch.float32, torch.float64):
        for old, new in (((1, 1), (4, 7)), ((1, 7), (5, 3)), ((7, 1), (3, 5)),
                         ((4, 7), (1, 3)), ((7, 4), (3, 1)), ((4, 6), (1, 1)),
                         ((1, 9), (1, 4)), ((9, 1), (4, 1))):
            _check(solve, _positions(old, 1, 3, dtype, generator), old, new)


def check_upsampling(solve):
    generator = torch.Generator().manual_seed(8613)
    for dtype in (torch.float32, torch.float64):
        for old, new in (((2, 3), (7, 11)), ((3, 5), (9, 8)), ((4, 4), (16, 16))):
            _check(solve, _positions(old, 1, 6, dtype, generator), old, new)


def check_downsampling_without_antialias(solve):
    generator = torch.Generator().manual_seed(8614)
    for dtype in (torch.float32, torch.float64):
        for old, new in (((13, 11), (4, 3)), ((16, 16), (3, 5)), ((9, 7), (2, 2))):
            _check(solve, _positions(old, 2, 4, dtype, generator), old, new, 2)


def check_mixed_and_same_area(solve):
    generator = torch.Generator().manual_seed(8615)
    for old, new in (((2, 8), (4, 4)), ((8, 2), (4, 4)), ((7, 3), (2, 11)),
                     ((4, 9), (4, 5)), ((9, 4), (5, 4))):
        for dtype in (torch.float32, torch.float64):
            _check(solve, _positions(old, 1, 7, dtype, generator), old, new)


def check_constant_fields(solve):
    for dtype in (torch.float32, torch.float64):
        constant = torch.tensor([-3.0, 0.0, 0.25, 2.0], dtype=dtype)
        for old, new in (((2, 3), (7, 11)), ((11, 7), (3, 2)), ((1, 1), (8, 5))):
            pos = constant.expand(1, old[0] * old[1], 4).clone()
            out = _check(solve, pos, old, new, 0)
            tolerance = 1e-5 if dtype == torch.float32 else 1e-10
            torch.testing.assert_close(out, constant.expand_as(out),
                                       atol=tolerance, rtol=tolerance)


def check_identity_clone_bitwise(solve):
    generator = torch.Generator().manual_seed(8616)
    for dtype in (torch.float32, torch.float64):
        for prefix_count in (0, 1, 4):
            pos = _positions((3, 5), prefix_count, 6, dtype, generator)
            pos[0, -1, 0] = -0.0
            _check(solve, pos, (3, 5), (3, 5), prefix_count)
        pos = torch.tensor([[[-0.0]]], dtype=dtype)
        _check(solve, pos, (1, 1), (1, 1), 0)


def check_noncontiguous_and_input_immutability(solve):
    generator = torch.Generator().manual_seed(8617)
    for dtype in (torch.float32, torch.float64):
        base = torch.randn((1, 8, 19), dtype=dtype, generator=generator)
        saved = base.clone()
        pos = base.transpose(1, 2)
        assert not pos.is_contiguous()
        _check(solve, pos, (3, 5), (6, 2), 4)
        _check(solve, pos, (3, 5), (3, 5), 4)
        assert torch.equal(_bits(base), _bits(saved))
        base = torch.randn((1, 34, 12), dtype=dtype, generator=generator)
        saved = base.clone()
        pos = base[:, 1::2, ::2]
        assert not pos.is_contiguous()
        _check(solve, pos, (3, 5), (4, 7), 2)
        assert torch.equal(_bits(base), _bits(saved))


def check_invalid_old_grid_length(solve):
    pos = torch.arange(14, dtype=torch.float32).reshape(1, 7, 2)
    for old, new, prefix_count in (((2, 2), (3, 4), 1), ((2, 4), (2, 4), 1),
                                   ((2, 3), (2, 3), 0), ((1, 7), (1, 7), 4)):
        before = pos.clone()
        try:
            solve(pos, old, new, prefix_count)
        except ValueError:
            pass
        else:
            raise AssertionError("old_hw/sequence length mismatch must raise ValueError")
        assert torch.equal(_bits(pos), _bits(before))


def check_maximum_dimensions_and_values(solve):
    generator = torch.Generator().manual_seed(8618)
    for dtype in (torch.float32, torch.float64):
        for old, new in (((16, 16), (15, 13)), ((7, 9), (16, 16))):
            _check(solve, _positions(old, 4, 16, dtype, generator), old, new, 4)
        for _ in range(8):
            pos = (torch.rand((1, 16, 4), dtype=dtype, generator=generator) * 2 - 1) * 10000
            _check(solve, pos, (3, 5), (7, 11))


def check_seeded_random(solve):
    generator = torch.Generator().manual_seed(8619)
    for dtype in (torch.float32, torch.float64):
        for _ in range(20):
            h0, w0, h1, w1 = torch.randint(1, 17, (4,), generator=generator).tolist()
            dim = int(torch.randint(1, 17, (), generator=generator))
            prefix_count = int(torch.randint(0, 5, (), generator=generator))
            old, new = (h0, w0), (h1, w1)
            _check(solve, _positions(old, prefix_count, dim, dtype, generator),
                   old, new, prefix_count)
