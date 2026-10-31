"""SAM sparse prompt embedding (points + boxes) and the dense random-PE grid."""

import math

import torch

_FLOAT_DTYPES = (torch.float32, torch.float64)
_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)


def _pair(value, name):
    """A length-2 positive integer pair, given as an integer tensor or a sequence."""
    if isinstance(value, torch.Tensor):
        if value.dtype not in _INT_DTYPES or value.numel() != 2:
            raise ValueError(f"{name} must be a length-2 integer pair")
        items = [int(v) for v in value.reshape(-1)]
    else:
        try:
            items = list(value)
        except TypeError:
            raise ValueError(f"{name} must be a length-2 integer pair")
        if len(items) != 2 or any(type(v) is bool or type(v) is not int for v in items):
            raise ValueError(f"{name} entries must be plain Python ints")
    if len(items) != 2 or any(v <= 0 for v in items):
        raise ValueError(f"{name} entries must be positive")
    return items[0], items[1]


def _pe_encoding(coords, pe_matrix):
    """Upstream PositionEmbeddingRandom._pe_encoding; coords are already in [0, 1]."""
    centered = 2.0 * coords - 1.0
    angles = (2.0 * math.pi) * (centered @ pe_matrix)
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


def solve(points, labels, boxes, grid_size_in, pe_matrix, point_embeddings,
          not_a_point_embed, image_size=1024, dense_size=None):
    tensors = (points, labels, boxes, grid_size_in, pe_matrix, point_embeddings,
               not_a_point_embed)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("all tensor arguments must be torch.Tensor")
    if type(image_size) is not int or image_size <= 0:
        raise ValueError("image_size must be a positive Python int")

    floats = (points, boxes, pe_matrix, point_embeddings, not_a_point_embed)
    if any(tensor.dtype not in _FLOAT_DTYPES for tensor in floats):
        raise ValueError("coordinate and embedding tensors must be floating point")
    dtype = points.dtype
    if any(tensor.dtype != dtype for tensor in floats):
        raise ValueError("all floating tensors must share one dtype")
    if labels.dtype not in _INT_DTYPES:
        raise ValueError("labels must be an integer tensor")

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
        raise ValueError("labels must have shape (N,) matching points")
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("point labels must be 0 (background) or 1 (foreground)")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (B, 4)")
    if pe_matrix.ndim != 2 or pe_matrix.shape[0] != 2:
        raise ValueError("pe_matrix must have shape (2, P)")
    width = 2 * pe_matrix.shape[1]
    if point_embeddings.ndim != 2 or tuple(point_embeddings.shape) != (4, width):
        raise ValueError("point_embeddings must have shape (4, 2P)")
    if not_a_point_embed.ndim != 1 or not_a_point_embed.shape[0] != width:
        raise ValueError("not_a_point_embed must have shape (2P,)")

    old_h, old_w = _pair(grid_size_in, "grid_size_in")
    dense_hw = None if dense_size is None else _pair(dense_size, "dense_size")

    # ResizeLongestSide.get_preprocess_shape: Python float rounding, dtype independent.
    scale = float(image_size) * 1.0 / float(max(old_h, old_w))
    new_h = int(old_h * scale + 0.5)
    new_w = int(old_w * scale + 0.5)
    factors = torch.tensor((new_w / old_w, new_h / old_h), dtype=dtype, device=points.device)

    n_points = points.shape[0]
    n_boxes = boxes.shape[0]
    pad = n_boxes == 0
    point_coords = points * factors + 0.5
    if pad:
        point_coords = torch.cat((point_coords, point_coords.new_zeros((1, 2))), dim=0)
    box_coords = (boxes * factors.repeat(2) + 0.5).reshape(2 * n_boxes, 2)
    prompt_coords = torch.cat((point_coords, box_coords), dim=0)

    # forward_with_coords divides x and y by the (square) input image size.
    raw = _pe_encoding(prompt_coords / float(image_size), pe_matrix)
    n_point_rows = point_coords.shape[0]

    effective = labels
    if pad:
        effective = torch.cat((labels, labels.new_full((1,), -1)), dim=0)
    is_padding = effective == -1
    lookup = torch.cat((point_embeddings[:2], not_a_point_embed.unsqueeze(0)), dim=0)
    rows = torch.where(is_padding, torch.full_like(effective, 2), effective)
    additions = lookup.index_select(0, rows)
    sparse_points = torch.where(
        is_padding.unsqueeze(1), additions, raw[:n_point_rows] + additions
    )

    corner_index = torch.arange(2 * n_boxes, device=points.device) % 2 + 2
    sparse_boxes = raw[n_point_rows:] + point_embeddings.index_select(0, corner_index)
    sparse_embeddings = torch.cat((sparse_points, sparse_boxes), dim=0)

    if dense_hw is None:
        dense_pe = prompt_coords.new_empty((0,))
    else:
        h, w = dense_hw
        grid = prompt_coords.new_ones((h, w))
        y_embed = (grid.cumsum(dim=0) - 0.5) / float(h)
        x_embed = (grid.cumsum(dim=1) - 0.5) / float(w)
        dense_pe = _pe_encoding(torch.stack((x_embed, y_embed), dim=-1), pe_matrix)
        dense_pe = dense_pe.permute(2, 0, 1)

    preprocess_size = torch.tensor((new_h, new_w), dtype=torch.int64, device=points.device)
    return sparse_embeddings, prompt_coords, preprocess_size, raw, dense_pe
