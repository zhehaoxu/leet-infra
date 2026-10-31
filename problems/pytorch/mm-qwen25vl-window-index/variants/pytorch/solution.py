"""Reference solution: Qwen2.5-VL vision window index and cu-window seqlens.

A line-by-line transcription of transformers' `vision_utils.get_vision_window_index`
(the exact function imported by `modeling_qwen2_5_vl.py`), plus the reverse
permutation the model computes right after it.
"""

import torch
import torch.nn.functional as F


def solve(grid_thw, spatial_merge_size, patch_size, window_size):
    if not isinstance(grid_thw, torch.Tensor):
        raise ValueError("grid_thw must be a torch tensor")
    if grid_thw.dtype is not torch.int64:
        raise ValueError("grid_thw must be int64")
    if grid_thw.dim() != 2 or grid_thw.shape[1] != 3:
        raise ValueError("grid_thw must be [G,3]")
    if grid_thw.shape[0] < 1:
        raise ValueError("grid_thw needs at least one row")
    for name, value in (
        ("spatial_merge_size", spatial_merge_size),
        ("patch_size", patch_size),
        ("window_size", window_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a plain Python int")
        if value < 1:
            raise ValueError(f"{name} must be >= 1")

    grids = grid_thw.tolist()
    vit_merger_window_size = window_size // spatial_merge_size // patch_size
    if vit_merger_window_size < 1:
        raise ValueError("vit_merger_window_size must be >= 1")
    spatial_merge_unit = spatial_merge_size**2

    window_index = []
    cu_window_seqlens = [0]
    window_index_id = 0

    for grid_t, grid_h, grid_w in grids:
        if grid_t < 1 or grid_h < 1 or grid_w < 1:
            raise ValueError("every t/h/w must be >= 1")
        if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
            raise ValueError("h and w must be divisible by spatial_merge_size")
        grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w
        )
        # NOTE: pad equals vws when the edge is divisible -> an empty window row
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += grid_t * llm_grid_h * llm_grid_w

    window_index = torch.cat(window_index, dim=0)
    cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32)
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    reverse_indices = torch.argsort(window_index)
    return window_index, cu_window_seqlens, reverse_indices
