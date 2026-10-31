import torch


def solve(query, image_grid, text_len, axes_dim=(16, 56, 56), theta=10000.0):
    """FLUX.1 二维 RoPE：三轴频率表 + 相邻对旋转 / 2-D RoPE tables and adjacent-pair rotation.

    参数 / Args
    query: CPU float32 或 float64 张量 (B, S, head_dim)，head_dim == sum(axes_dim)，
        B 在 1..3，S == text_len + image_grid[0]*image_grid[1]；可非连续，不得修改。
        CPU float32/float64 tensor (B, S, head_dim) with head_dim == sum(axes_dim);
        B in 1..3 and S == text_len + h*w; may be noncontiguous; must not be mutated.
    image_grid: int (2,) = (latent_h, latent_w)，逐项 >= 1，张量或整数序列均可。
        int tensor/sequence (2,) = (latent_h, latent_w); every entry >= 1.
    text_len: 非负 Python int（bool 非法）/ non-negative Python int (bool rejected).
    axes_dim: 长度 3 的正偶数整数序列 / length-3 sequence of positive even ints.
    theta: Python 实数标量且 >= 1.0（bool 与张量非法）/ finite Python real >= 1.0
        (bool and tensors rejected).

    构造 / Construction
    img_ids 按上游 _prepare_latent_image_ids 布局：轴 0 恒为 0（帧轴），轴 1 是行、
    轴 2 是列，行主序展平成 (h*w, 3)；txt_ids 是 (text_len, 3) 全零；
    ids = cat([txt_ids, img_ids], dim=0)（文本在前、图像在后，整段联合序列一张表，
    不做 img/text 拆表）。上游给 q 与 k 的是同一张表，本题只算一次。
    pos = ids.float()（上游口径，位置先转 float32）。
    对每个轴 a：inv_freq_a = theta ** (-(0:2:axes_dim[a]) / axes_dim[a])，指数分母是
    **该轴自己的 dim**，不是 head_dim；在 float64 里算 angles_a = outer(pos[:, a], inv_freq_a)，
    cos/sin 也在 float64 里取，再沿末维 repeat_interleave(2) 复制成 2 倍宽，最后 .float() 落表；
    三轴沿末维拼接。
    For axis a: inv_freq_a = theta ** (-(0:2:axes_dim[a]) / axes_dim[a]) — the exponent
    denominator is THAT AXIS's own dim, not head_dim; angles = outer(pos[:, a], inv_freq_a)
    and cos/sin are evaluated in float64, duplicated with repeat_interleave(2) on the last
    dim, then cast to float32 and concatenated over the three axes.

    精度口径 / Precision policy
    freqs_cos、freqs_sin 恒为 float32，与 query 的 dtype 无关（上游 .float() 落表）。
    ids、angles、inv_freqs 恒为 float64。旋转在 query 自身的 dtype 下进行：float32 query
    全程 float32；float64 query 把 float32 表提升为 float64 后相乘（本题有意不做上游
    apply_rotary_emb 的 x.float() 降精度），rotated 的 dtype 与 query 一致。
    Tables are always float32 whatever the query dtype; ids/angles/inv_freqs are float64.
    The rotation happens in the query dtype, so a float64 query promotes the float32 tables.

    旋转 / Rotation
    因 repeat_interleave 有 cos[2m] == cos[2m+1]，每对角度取 m 位即可，用相邻对 2x2 旋转：
    out[2m] = q[2m]*c[m] - q[2m+1]*s[m]，out[2m+1] = q[2m+1]*c[m] + q[2m]*s[m]，
    其中 c[m] = freqs_cos[..., 2m]（等价 freqs_cos.reshape(..., -1, 2)[..., 0]）。
    Adjacent pairs (0,1), (2,3), ... each rotate by its own angle.

    返回 / Returns (rotated (B,S,head_dim) 同 query dtype/device, freqs_cos (S,head_dim) float32,
    freqs_sin (S,head_dim) float32, ids (S,3) float64, angles (S,head_dim//2) float64
    （复制前的逐对角度）/ per-pair angles before duplication, inv_freqs (head_dim//2,) float64)。

    异常 / Errors —— 以下情形抛 ValueError：axes_dim 项非正偶数 int（bool 除外非法）或长度不为 3；
    theta < 1.0 或非实数；image_grid 项 < 1 或非整数；text_len 为负、bool 或浮点；
    query 不是三维、末维不等于 sum(axes_dim)、或 S 与 text_len + h*w 不一致。
    ValueError for a non-positive-even axes_dim entry (bools excluded) or wrong length;
    theta < 1.0 or not real; image_grid entry < 1 or non-integral; negative/bool/float text_len;
    query with ndim != 3, last dim != sum(axes_dim), or S != text_len + h*w.

    判题容差 / Tolerances: float32 输出 atol=rtol=4e-5，float64 输出 atol=rtol=2e-12；
    ids 要求逐位相等。无梯度需求。/ No backward pass required.
    """
    raise NotImplementedError
