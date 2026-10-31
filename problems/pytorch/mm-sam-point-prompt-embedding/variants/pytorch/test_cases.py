"""Pure-Python coordinate, sin/cos and grid-cell oracle; shares no code with the solution."""

import math

import torch


def _oracle(points, labels, boxes, grid_size_in, pe_matrix, point_embeddings,
            not_a_point_embed, image_size=1024, dense_size=None):
    dtype = points.dtype
    old_h = int(grid_size_in[0])
    old_w = int(grid_size_in[1])
    scale = float(image_size) * 1.0 / max(old_h, old_w)
    new_h = int(old_h * scale + 0.5)
    new_w = int(old_w * scale + 0.5)
    factor_x = new_w / old_w
    factor_y = new_h / old_h

    matrix = [[float(value) for value in row] for row in pe_matrix.tolist()]
    half = len(matrix[0])
    width = 2 * half
    embeds = [[float(value) for value in row] for row in point_embeddings.tolist()]
    padding_value = [float(value) for value in not_a_point_embed.tolist()]
    two_pi = 2.0 * math.pi

    def encode(cx, cy):
        # cx, cy are already normalised to [0, 1]; sin block first, then cos block.
        ux = 2.0 * cx - 1.0
        uy = 2.0 * cy - 1.0
        angles = [two_pi * (ux * matrix[0][j] + uy * matrix[1][j]) for j in range(half)]
        return ([math.sin(angle) for angle in angles]
                + [math.cos(angle) for angle in angles])

    coord_rows = [[x * factor_x + 0.5, y * factor_y + 0.5] for x, y in points.tolist()]
    effective = [int(value) for value in labels.tolist()]
    corner_rows = []
    for x1, y1, x2, y2 in boxes.tolist():
        corner_rows.append([x1 * factor_x + 0.5, y1 * factor_y + 0.5])
        corner_rows.append([x2 * factor_x + 0.5, y2 * factor_y + 0.5])
    if not corner_rows:
        coord_rows.append([0.0, 0.0])
        effective.append(-1)
    coord_rows.extend(corner_rows)

    raw = [encode(row[0] / float(image_size), row[1] / float(image_size)) for row in coord_rows]
    point_count = len(coord_rows) - len(corner_rows)
    sparse = []
    for index in range(point_count):
        label = effective[index]
        row = raw[index]
        if label == -1:
            # Upstream computes the encoding for the padding row and only then wipes it
            # to zero, so the row's embedding is not_a_point_embed whatever its coordinate.
            zeroed = [0.0] * width
            sparse.append([value + pad for value, pad in zip(zeroed, padding_value)])
        else:
            chosen = embeds[label]
            sparse.append([row[k] + chosen[k] for k in range(width)])
    for index in range(point_count, len(raw)):
        chosen = embeds[2 + ((index - point_count) % 2)]
        sparse.append([raw[index][k] + chosen[k] for k in range(width)])

    if dense_size is None:
        dense = torch.empty((0,), dtype=dtype)
    else:
        height = int(dense_size[0])
        length = int(dense_size[1])
        cells = [[encode((j + 0.5) / length, (i + 0.5) / height) for j in range(length)]
                 for i in range(height)]
        planes = [[[cells[i][j][channel] for j in range(length)] for i in range(height)]
                  for channel in range(width)]
        dense = torch.tensor(planes, dtype=dtype)

    def as_rows(values, columns):
        if values:
            return torch.tensor(values, dtype=dtype)
        return torch.empty((0, columns), dtype=dtype)

    return (as_rows(sparse, width), as_rows(coord_rows, 2),
            torch.tensor((new_h, new_w), dtype=torch.int64), as_rows(raw, width), dense)


def _case(n=3, b=2, p=4, hw=(37, 53), image_size=1024, dense_hw=None,
          dtype=torch.float64, seed=7):
    generator = torch.Generator().manual_seed(seed)
    old_h, old_w = hw
    axes = torch.tensor((old_w, old_h), dtype=dtype)

    def rand(*shape):
        return torch.randn(shape, dtype=dtype, generator=generator) * 0.6

    points = torch.rand((n, 2), dtype=dtype, generator=generator) * axes
    labels = (torch.rand((n,), generator=generator) > 0.5).to(torch.int64)
    if b:
        low = torch.rand((b, 2), dtype=dtype, generator=generator) * (axes * 0.5)
        high = low + 1.0 + torch.rand((b, 2), dtype=dtype, generator=generator)
        boxes = torch.cat((low, high), dim=1)
    else:
        boxes = torch.empty((0, 4), dtype=dtype)
    dense = None if dense_hw is None else torch.tensor(dense_hw, dtype=torch.int64)
    return (points, labels, boxes, torch.tensor((old_h, old_w), dtype=torch.int64),
            rand(2, p), rand(4, 2 * p), rand(2 * p), image_size, dense)


def _example():
    return (
        torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float64),
        torch.tensor([1, 0], dtype=torch.int64),
        torch.empty((0, 4), dtype=torch.float64),
        torch.tensor((48, 64), dtype=torch.int64),
        torch.tensor([[0.5, -1.0, 0.25], [1.0, 0.5, -0.75]], dtype=torch.float64),
        torch.tensor([[0.5, -0.5, 1.0, 0.0, -1.0, 0.25],
                      [-0.5, 1.0, 0.5, -1.0, 0.0, 0.25],
                      [1.0, 1.0, 0.0, 0.0, -1.0, -1.0],
                      [0.0, -1.0, 1.0, 0.5, 0.5, -0.5]], dtype=torch.float64),
        torch.tensor([2.0, -2.0, 0.5, -0.5, 1.5, -1.5], dtype=torch.float64),
        1024,
        torch.tensor((2, 3), dtype=torch.int64),
    )


def _box_example():
    return (
        torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([[2.0, 3.0, 6.0, 9.0]], dtype=torch.float64),
        torch.tensor((30, 50), dtype=torch.int64),
        torch.tensor([[0.5, -1.0], [1.0, 0.5]], dtype=torch.float64),
        torch.tensor([[0.5, -0.5, 1.0, 0.0],
                      [-0.5, 1.0, 0.5, -1.0],
                      [1.0, 1.0, 0.0, 0.0],
                      [0.0, -1.0, 1.0, 0.5]], dtype=torch.float64),
        torch.tensor([2.0, -2.0, 0.5, -0.5], dtype=torch.float64),
        1024,
        None,
    )


def _close(actual, expected):
    tolerance = 3e-5 if expected.dtype == torch.float32 else 2e-10
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def _expect_error(solve, args):
    try:
        solve(*args)
    except ValueError:
        return
    raise AssertionError("expected ValueError for the patched argument set")


def _verify(solve, args):
    expected = _oracle(*args)
    result = solve(*args)
    assert isinstance(result, tuple) and len(result) == 5
    points, labels, boxes, _, pe_matrix, point_embeddings, not_a_point = args[:7]
    dense_size = args[8]
    dtype = points.dtype
    width = 2 * pe_matrix.shape[1]
    n, b = points.shape[0], boxes.shape[0]
    m = n + b * 2 + (1 if b == 0 else 0)

    for actual, target in zip(result, expected):
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == target.shape
        assert actual.device == points.device
    assert result[0].dtype == dtype and result[1].dtype == dtype
    assert result[2].dtype == torch.int64
    assert result[3].dtype == dtype and result[4].dtype == dtype
    assert result[0].shape == (m, width) and result[1].shape == (m, 2)
    assert result[3].shape == (m, width)
    assert torch.isfinite(result[0]).all() and torch.isfinite(result[3]).all()
    if dense_size is None:
        assert result[4].shape == (0,) and result[4].numel() == 0
    else:
        assert result[4].shape == (width, int(dense_size[0]), int(dense_size[1]))
    for index in range(5):
        if index == 2:
            assert torch.equal(result[index], expected[index])
        else:
            _close(result[index], expected[index])

    sparse, _, _, raw = result[0], result[1], result[2], result[3]
    point_count = n + (1 if b == 0 else 0)
    effective = labels.tolist() + ([-1] if b == 0 else [])
    for row, label in enumerate(effective):
        if label == -1:
            _close(sparse[row], not_a_point)
            assert torch.count_nonzero(raw[row]) > 0, "raw padding row stays observable"
        else:
            _close(sparse[row], raw[row] + point_embeddings[label])
    for row in range(point_count, m):
        corner = point_embeddings[2 + ((row - point_count) % 2)]
        _close(sparse[row], raw[row] + corner)
    return result


def check_example_point_padding(solve):
    _verify(solve, _example())


def check_example_box_corners(solve):
    _verify(solve, _box_example())


def check_padding_only_when_boxes_empty(solve):
    empty = list(_case(n=2, b=0, p=3, seed=5))
    sparse, coords, _, raw = _verify(solve, empty)[:4]
    assert sparse.shape[0] == 3 and coords.shape[0] == 3
    _close(coords[2], torch.zeros(2, dtype=torch.float64))
    assert not torch.allclose(sparse[2], raw[2])

    filled = list(_case(n=2, b=1, p=3, seed=5))
    sparse_two, coords_two = _verify(solve, filled)[:2]
    assert sparse_two.shape[0] == 4 and coords_two.shape[0] == 4
    _close(coords_two[:2], coords[:2])


def check_padding_row_wipes_positional_encoding(solve):
    # A -1 row is encoded and only then zeroed, so its embedding cannot depend on the
    # coordinate, the gaussian matrix, the axes or the image size. Every configuration
    # below keeps n/b/p fixed, which also keeps not_a_point_embed bitwise identical.
    anchor = None
    variants = ((1.0, 1024, (30, 50), [[1.0, 2.0], [37.0, 41.0]]),
                (37.0, 64, (30, 50), [[1.0, 2.0], [37.0, 41.0]]),
                (-53.5, 4096, (7, 4096), [[4095.0, 3.0], [0.0, 0.5]]),
                (0.125, 33, (1, 1), [[0.75, 0.25], [0.5, 0.5]]))
    for scale, image_size, hw, rows in variants:
        args = list(_case(n=2, b=0, p=3, hw=hw, image_size=image_size, seed=61))
        args[0] = torch.tensor(rows, dtype=torch.float64)
        args[1] = torch.tensor([0, 1], dtype=torch.int64)
        args[4] = args[4] * scale
        sparse, coords, _, raw = _verify(solve, args)[:4]
        pad_row = sparse.shape[0] - 1
        assert coords[pad_row].tolist() == [0.0, 0.0]
        _close(sparse[pad_row], args[6])
        assert torch.count_nonzero(raw[pad_row]) > 0, "padding row encoding stays observable"
        assert not torch.allclose(sparse[pad_row], raw[pad_row] + args[6],
                                  rtol=1e-6, atol=1e-6), "padding row must be wiped"
        assert not torch.allclose(sparse[0], args[6], rtol=1e-6, atol=1e-6)
        assert not torch.allclose(sparse[1], sparse[0], rtol=1e-6, atol=1e-6)
        if anchor is None:
            anchor = sparse[pad_row].clone()
        else:
            assert torch.equal(sparse[pad_row], anchor)


def check_only_background_labels(solve):
    args = list(_case(n=4, b=1, p=3, seed=17))
    args[1] = torch.zeros(4, dtype=torch.int64)
    sparse, _, _, raw = _verify(solve, args)[:4]
    _close(sparse[:4], raw[:4] + args[5][0])
    assert not torch.allclose(sparse[0], raw[0])
    _close(sparse[4], raw[4] + args[5][2])
    _close(sparse[5], raw[5] + args[5][3])


def check_no_point_prompt_cases(solve):
    only_boxes = list(_case(n=0, b=2, p=3, seed=23))
    sparse, coords, _, raw = _verify(solve, only_boxes)[:4]
    assert sparse.shape == (4, 6) and coords.shape[0] == 4
    _close(sparse[0] - raw[0], only_boxes[5][2])
    _close(sparse[1] - raw[1], only_boxes[5][3])
    _close(sparse[2] - raw[2], only_boxes[5][2])
    _close(sparse[3] - raw[3], only_boxes[5][3])

    nothing = list(_case(n=0, b=0, p=2, seed=29))
    sparse = _verify(solve, nothing)[0]
    assert sparse.shape == (1, 4)
    _close(sparse[0], nothing[6])


def check_dense_grid_axes_and_sin_order(solve):
    _verify(solve, _case(n=2, b=1, p=3, dense_hw=(1, 4)))
    _verify(solve, _case(n=2, b=1, p=3, dense_hw=(4, 1)))
    _verify(solve, _case(n=3, b=2, p=2, dense_hw=(2, 3), dtype=torch.float32))

    args = list(_case(n=1, b=1, p=2, seed=3))
    args[4] = torch.tensor([[1.0, 0.5], [0.0, 1.0]], dtype=torch.float64)
    args[8] = torch.tensor((1, 2), dtype=torch.int64)
    dense = _verify(solve, args)[4]
    assert dense.shape == (4, 1, 2)
    # Cell (0, 0): x = 0.25, y = 0.5 -> angles (-pi, -pi/2): sin (0, -1), cos (-1, 0).
    _close(dense[:, 0, 0], torch.tensor([0.0, -1.0, -1.0, 0.0], dtype=torch.float64))
    # Cell (0, 1): x = 0.75, y = 0.5 -> angles (pi, pi/2): sin (0, 1), cos (-1, 0).
    _close(dense[:, 0, 1], torch.tensor([0.0, 1.0, -1.0, 0.0], dtype=torch.float64))

    single = list(args)
    single[8] = torch.tensor((1, 1), dtype=torch.int64)
    # x = y = 0.5 -> zero angle -> the sin block is all zero and the cos block all one.
    _close(_verify(solve, single)[4][:, 0, 0],
           torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64))

    swapped = list(args)
    swapped[8] = torch.tensor((2, 1), dtype=torch.int64)
    assert _verify(solve, swapped)[4].shape == (4, 2, 1)


def check_dense_absent_returns_empty_tensor(solve):
    for dtype in (torch.float32, torch.float64):
        result = _verify(solve, _case(n=2, b=1, p=3, dense_hw=None, dtype=dtype))
        assert result[4].shape == (0,) and result[4].numel() == 0
        assert result[4].dtype == dtype
        assert result[4].device.type == "cpu"


def check_preprocess_rounding_is_half_up(solve):
    for hw in ((1, 2048), (3, 2048), (48, 64), (1023, 1024), (7, 33), (1, 1), (1024, 1)):
        args = list(_case(n=2, b=1, p=3, hw=hw, seed=41))
        old_h, old_w = hw
        scale = 1024.0 * 1.0 / max(old_h, old_w)
        expected = torch.tensor((int(old_h * scale + 0.5), int(old_w * scale + 0.5)),
                                dtype=torch.int64)
        assert torch.equal(_verify(solve, args)[2], expected)
    # A naive truncation (or Python round) would drop the height below its half value.
    args = list(_case(n=1, b=0, p=2, hw=(1, 2048), seed=43))
    assert torch.equal(_verify(solve, args)[2], torch.tensor((1, 1024), dtype=torch.int64))


def check_axis_factors_are_not_swapped(solve):
    args = list(_case(n=2, b=0, p=3, hw=(30, 50), seed=31))
    args[0] = torch.tensor([[1.0, 29.0], [49.0, 3.0]], dtype=torch.float64)
    coords = _verify(solve, args)[1]
    correct = torch.stack((args[0][:, 0] * (1024 / 50) + 0.5,
                           args[0][:, 1] * (614 / 30) + 0.5), dim=1)
    _close(coords[:2], correct)
    swapped = torch.stack((args[0][:, 1] * (614 / 30) + 0.5,
                           args[0][:, 0] * (1024 / 50) + 0.5), dim=1)
    assert not torch.allclose(coords[:2], swapped, rtol=1e-3, atol=1e-3)


def check_row_order_points_then_corners(solve):
    args = list(_case(n=2, b=2, p=3, hw=(64, 48), seed=13))
    sparse, coords, size, raw = _verify(solve, args)[:4]
    assert sparse.shape[0] == 6 and coords.shape[0] == 6
    _close(sparse[2] - raw[2], args[5][2])
    _close(sparse[3] - raw[3], args[5][3])
    _close(sparse[4] - raw[4], args[5][2])
    _close(sparse[5] - raw[5], args[5][3])
    assert not torch.allclose(sparse[4], sparse[5])
    factor_x = float(size[1]) / 48.0
    factor_y = float(size[0]) / 64.0
    last_box = args[2][1]
    _close(coords[4], torch.tensor([float(last_box[0]) * factor_x + 0.5,
                                    float(last_box[1]) * factor_y + 0.5],
                                   dtype=torch.float64))
    _close(coords[5], torch.tensor([float(last_box[2]) * factor_x + 0.5,
                                    float(last_box[3]) * factor_y + 0.5],
                                   dtype=torch.float64))


def check_alternative_image_size(solve):
    for image_size in (33, 64, 512):
        args = list(_case(n=3, b=1, p=3, hw=(20, 45), image_size=image_size,
                          dense_hw=(2, 2), seed=53))
        sparse, _, size, raw = _verify(solve, args)[:4]
        scale = float(image_size) * 1.0 / 45.0
        assert torch.equal(size, torch.tensor((int(20 * scale + 0.5), image_size),
                                              dtype=torch.int64))
        assert torch.isfinite(sparse).all() and torch.isfinite(raw).all()
        default = list(args)
        default[7] = 1024
        other = _verify(solve, default)[3]
        assert not torch.allclose(other, raw, rtol=1e-3, atol=1e-3)


def check_float32_and_float64(solve):
    for dtype in (torch.float32, torch.float64):
        _verify(solve, _case(n=6, b=3, p=5, hw=(101, 37), dense_hw=(3, 4), dtype=dtype))
        _verify(solve, _case(n=1, b=0, p=1, hw=(7, 7), dense_hw=None, dtype=dtype))


def check_noncontiguous_inputs_are_not_mutated(solve):
    for dtype in (torch.float32, torch.float64):
        args = list(_case(n=4, b=2, p=3, dense_hw=(2, 2), dtype=dtype))
        for index in (0, 2, 4, 5, 6):
            tensor = args[index]
            args[index] = torch.stack((tensor, torch.zeros_like(tensor)), dim=-1)[..., 0]
            assert not args[index].is_contiguous()
        copies = [value.clone() if isinstance(value, torch.Tensor) else value
                  for value in args]
        _verify(solve, args)
        for value, copy in zip(args, copies):
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, copy), "input mutated"


def check_seeded_shape_sweep(solve):
    shapes = ((1, 0, 1, (1, 1), (1, 1)), (2, 1, 2, (48, 64), None),
              (5, 2, 3, (30, 50), (2, 3)), (12, 6, 8, (1023, 1024), (8, 8)),
              (0, 4, 5, (5, 4096), (3, 1)), (4, 0, 2, (600, 12), (2, 5)),
              (7, 3, 6, (1, 4096), None), (9, 5, 4, (33, 129), (4, 2)),
              (3, 2, 7, (128, 640), (2, 2)), (11, 1, 3, (77, 31), (5, 3)))
    for index, (n, b, p, hw, dense_hw) in enumerate(shapes):
        dtype = torch.float32 if index % 2 else torch.float64
        _verify(solve, _case(n=n, b=b, p=p, hw=hw, dense_hw=dense_hw,
                             dtype=dtype, seed=200 + index))
        _verify(solve, _case(n=max(n - 1, 0), b=b, p=p, hw=(hw[1], hw[0]),
                             dense_hw=dense_hw, image_size=64, seed=300 + index))


def check_invalid_inputs(solve):
    for labels, rows in ((torch.tensor([-1], dtype=torch.int64), 1),
                         (torch.tensor([2], dtype=torch.int64), 1),
                         (torch.tensor([0, 1], dtype=torch.bool), 2),
                         (torch.tensor([0, 1, 0], dtype=torch.int64), 2)):
        args = list(_case(n=rows, b=1, p=3))
        args[1] = labels
        _expect_error(solve, args)

    replacements = (
        (0, torch.zeros((2, 3), dtype=torch.float64)),          # points need 2 columns
        (2, torch.zeros((1, 3), dtype=torch.float64)),          # boxes need 4 columns
        (3, torch.tensor((0, 64), dtype=torch.int64)),          # non-positive grid height
        (3, torch.tensor((48, 0, 1), dtype=torch.int64)),       # wrong grid size length
        (4, torch.zeros((3, 4), dtype=torch.float64)),          # pe_matrix needs (2, P)
        (5, torch.zeros((4, 7), dtype=torch.float64)),          # point_embeddings (4, 2P)
        (6, torch.zeros((7,), dtype=torch.float64)),            # not_a_point_embed (2P,)
        (8, torch.tensor((0, 2), dtype=torch.int64)),           # dense height must be positive
        (8, torch.tensor((2,), dtype=torch.int64)),             # dense_size needs a pair
        (8, torch.tensor([2, 3], dtype=torch.bool)),            # bool dense_size is invalid
    )
    for index, replacement in replacements:
        args = list(_case(n=2, b=1, p=3, dense_hw=(2, 3)))
        args[index] = replacement
        _expect_error(solve, args)

    for image_size in (0, -8, 1024.0, True, None):
        args = list(_case(n=2, b=1, p=3))
        args[7] = image_size
        _expect_error(solve, args)

    mixed = list(_case(n=2, b=1, p=3))
    mixed[2] = mixed[2].to(torch.float32)
    _expect_error(solve, mixed)


check_example_point_padding.example = True
check_example_box_corners.example = True
