def solve(resolution, image_sizes, divisibility=16, base_resolutions=None):
    """生成分辨率分桶，并把每张图的 (h, w) 分配到最近的一个桶。

    resolution:目标分辨率（正整数），像素预算为 resolution ** 2；
    image_sizes:形状 (N, 2) 的整型张量，每行是 (height, width)，均为正；
    divisibility:桶的每条边都对齐到该值的倍数且不小于它（正整数，默认 16）；
    base_resolutions:可选，[(h, w), ...] 形式的基准分辨率列表；给出时用它们的长宽比
        替换默认的 11 种比例（内部换算成 (w, h)，aspect = w / h）。

    桶的构造：对每个长宽比 a = w / h，先令 h = sqrt(resolution^2 / a)、w = h * a，
    再分别 round(x / divisibility) * divisibility、夹到 >= divisibility，按顺序去重。
    分配：取使 |h * bw - w * bh| 最小的桶（(bh, bw) 为桶的边长）；并列时取列表里更靠后的桶。

    返回 (buckets, assignments)：buckets 是 (B, 2) 的 int64 张量（每行 (h, w)），
    assignments 是 (N,) 的 int64 张量（每张图命中的桶下标）。
    非法输入（维度/类型不对、尺寸非正、base_resolutions 为空或格式错误）抛 ValueError。
    """
    # TODO
    raise NotImplementedError
