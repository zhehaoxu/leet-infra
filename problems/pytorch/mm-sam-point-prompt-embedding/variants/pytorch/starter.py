import torch


def solve(points, labels, boxes, grid_size_in, pe_matrix, point_embeddings,
          not_a_point_embed, image_size=1024, dense_size=None):
    """中文契约 / English contract
    points: float32/float64 (N,2)，原始图像像素坐标，最后一维是 (x,y) /
        original-image pixel coordinates, last axis is (x, y).
    labels: int64 (N,)，只能是 0（背景）或 1（前景），否则 ValueError /
        int64 (N,), only 0 or 1, else ValueError (-1 is internal padding only).
    boxes: 同 dtype (B,4) xyxy，原始像素坐标；无框时传空张量 (0,4)，等价上游 pad=True /
        same dtype (B, 4) xyxy; an empty (0, 4) tensor means no boxes (pad path).
    grid_size_in: int64 (2,) = (H,W) 原始图像尺寸，每项 >=1，非正 → ValueError /
        int64 (2,) = (H, W), each entry >= 1, else ValueError.
    pe_matrix: 同 dtype (2,P) 高斯位置编码矩阵（embed_dim = 2P） /
        same dtype (2, P) gaussian matrix; embedding width is 2P.
    point_embeddings: (4,2P)，下标依次是背景点、前景点、box 左上、box 右下 /
        (4, 2P): background point, foreground point, box top-left, box bottom-right.
    not_a_point_embed: (2P,) 一维，padding 行的唯一取值 / 1-D (2P,) padding-row value.
    image_size: 正 Python int（默认 1024），prompt encoder 输入边长；bool/非 int/<=0 →
        ValueError / positive Python int (bool excluded), else ValueError.
    dense_size: None 或 int64 (2,) = (h,w)，每项 >=1，否则 ValueError /
        None or a positive int64 (2,) pair, else ValueError.
    N=0..12, B=0..6, P=1..8, 1<=H,W<=4096, 1<=h,w<=8；全部浮点值有限 / finite floats.
    五个浮点张量同 dtype、同 device，否则 ValueError；其余输入保证合法 /
        the five float tensors share dtype/device, else ValueError.
    返回 / Returns (sparse_embeddings, prompt_coords, preprocess_size, positional_raw,
        dense_pe)：(M,2P)、(M,2)、int64 (2,)=(new_h,new_w)、(M,2P)、(2P,h,w) 或 (0,) 空张量；
        M = N + (B==0 时为 1) + 2B；浮点输出保持输入 dtype/device / floats preserve dtype.
    行序 / row order: 输入点顺序（padding 行在所有点之后）→ 每个 box 左上、右下交替 /
        input points, then the padding row, then per box top-left then bottom-right.
    positional_raw 是掩码/相加之前的 _pe_encoding 输出 / raw output before label masking.
    支持非连续输入，不修改任何输入；输出 dtype 与输入一致 /
        accept noncontiguous inputs; mutate no inputs; outputs reuse the input dtype.
    禁止 import segment_anything/transformers；只用基础 torch 算子 / no SAM libraries.
    """
    raise NotImplementedError
