import struct

import torch


# Recorded by running solution.py in the local eval environment; not hand-derived.
EXAMPLE_IMAGE = [
    [[0, 255, 0], [255, 0, 255], [51, 102, 153]],
    [[255, 255, 255], [0, 0, 0], [204, 153, 102]],
]
EXAMPLE_OUTPUT = [
    [[-1.0, 1.0], [1.0, -1.0]],
    [[1.0, 0.0], [1.0, 0.0]],
    [[-2.0, 2.0], [2.0, -2.0]],
]


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _pixel_oracle(image, crop_size, mean, std):
    """Independent Python pixel loops, explicitly rounding scalar stages to float32."""
    source = image.tolist()
    h, w = crop_size
    top = (len(source) - h) // 2
    left = (len(source[0]) - w) // 2
    output = []
    for c in range(3):
        channel = []
        for y in range(h):
            row = []
            for x in range(w):
                scaled = _f32(float(source[top + y][left + x][c]) / 255.0)
                centered = _f32(scaled - _f32(mean[c]))
                row.append(_f32(centered / _f32(std[c])))
            channel.append(row)
        output.append(channel)
    return torch.tensor(output, dtype=torch.float32)


def _check(solve, image, crop_size, mean, std):
    before = image.clone()
    stride = image.stride()
    means, stds = mean[:], std[:]
    expected = _pixel_oracle(image, crop_size, mean, std)
    out = solve(image, crop_size, mean, std)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, *crop_size), out.shape
    assert out.dtype == torch.float32 and out.device.type == "cpu"
    assert out.is_contiguous(), "output must be contiguous CHW"
    assert torch.equal(image, before) and image.stride() == stride, "input changed"
    assert mean == means and std == stds, "normalization parameters changed"
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
    return out


def _image(h, w, generator):
    return torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, generator=generator)


def check_example(solve):
    image = torch.tensor(EXAMPLE_IMAGE, dtype=torch.uint8)
    out = _check(solve, image, (2, 2), [0.5, 0.0, 0.5], [0.5, 1.0, 0.25])
    assert torch.equal(out, torch.tensor(EXAMPLE_OUTPUT, dtype=torch.float32))


check_example.example = True


def check_odd_difference_three(solve):
    image = torch.arange(7 * 8 * 3).remainder(256).to(torch.uint8).reshape(7, 8, 3)
    for size in ((4, 5), (4, 8), (7, 5)):
        _check(solve, image, size, [0.1, 0.3, 0.7], [0.2, 0.5, 0.9])


def check_rectangular_crops(solve):
    generator = torch.Generator().manual_seed(8601)
    for h, w, size in ((4, 11, (2, 7)), (11, 4, (7, 2)), (8, 13, (3, 10))):
        _check(solve, _image(h, w, generator), size, [0.2, 0.5, 0.8], [0.3, 0.4, 0.5])


def check_identity_crop(solve):
    generator = torch.Generator().manual_seed(8602)
    for h, w in ((1, 1), (3, 5), (5, 3)):
        _check(solve, _image(h, w, generator), (h, w), [0.0] * 3, [1.0] * 3)


def check_distinct_rgb_channels(solve):
    image = torch.empty((6, 9, 3), dtype=torch.uint8)
    image[..., 0] = 17
    image[..., 1] = 123
    image[..., 2] = 241
    _check(solve, image, (3, 4), [-0.4, 0.5, 1.2], [0.1, 0.7, 2.0])


def check_zero_and_255(solve):
    image = torch.tensor([[[0, 255, 0], [255, 0, 255]]], dtype=torch.uint8)
    _check(solve, image, (1, 2), [0.5, 0.2, 0.8], [0.1, 0.4, 0.25])
    for value in (0, 255):
        _check(solve, torch.full((3, 2, 3), value, dtype=torch.uint8),
               (2, 1), [0.3, 0.4, 0.5], [0.2, 0.3, 0.4])


def check_noncontiguous_input(solve):
    generator = torch.Generator().manual_seed(8603)
    base = _image(12, 18, generator)
    for image in (base[::2, 1::2], base.transpose(0, 1)):
        assert not image.is_contiguous()
        saved = base.clone()
        _check(solve, image, (3, 4), [0.48, 0.45, 0.41], [0.27, 0.26, 0.28])
        assert torch.equal(base, saved), "underlying storage changed"
    channels = torch.randint(0, 256, (5, 7, 6), dtype=torch.uint8, generator=generator)
    saved = channels.clone()
    _check(solve, channels[..., ::2], (4, 5), [0.1] * 3, [0.3] * 3)
    assert torch.equal(channels, saved)


def check_single_axis_and_pixel(solve):
    generator = torch.Generator().manual_seed(8604)
    for h, w, size in ((1, 9, (1, 4)), (9, 1, (4, 1)), (6, 8, (1, 1))):
        _check(solve, _image(h, w, generator), size, [0.6, 0.2, 0.9], [0.3, 0.2, 0.1])


def check_float32_rounding(solve):
    image = torch.tensor([[[128, 64, 192], [127, 63, 191]]], dtype=torch.uint8)
    _check(solve, image, (1, 2), [128 / 255, 64 / 255, 192 / 255], [0.1, 0.2, 0.3])


def check_invalid_crop(solve):
    image = torch.zeros((4, 5, 3), dtype=torch.uint8)
    invalid = [(0, 2), (2, 0), (-1, 2), (2, -1), (5, 2), (2, 6),
               (5, 6), (), (2,), (1, 2, 3), (2.0, 2), (True, 2),
               (2, False), ("2", 2), None, 2, "22"]
    for size in invalid:
        before = image.clone()
        try:
            solve(image, size, [0.0] * 3, [1.0] * 3)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for crop {size!r}")
        assert torch.equal(image, before)


def check_maximum_sizes(solve):
    generator = torch.Generator().manual_seed(8605)
    image = _image(128, 128, generator)
    for size in ((128, 128), (125, 127)):
        _check(solve, image, size, [0.4, 0.5, 0.6], [0.2, 0.3, 0.4])


def check_seeded_random(solve):
    generator = torch.Generator().manual_seed(8606)
    for _ in range(16):
        h, w = torch.randint(1, 25, (2,), generator=generator).tolist()
        crop_h = int(torch.randint(1, h + 1, (), generator=generator))
        crop_w = int(torch.randint(1, w + 1, (), generator=generator))
        mean = (torch.rand(3, generator=generator) * 2 - 0.5).tolist()
        std = (torch.rand(3, generator=generator) + 0.1).tolist()
        _check(solve, _image(h, w, generator), (crop_h, crop_w), mean, std)
