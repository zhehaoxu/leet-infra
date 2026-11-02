import math


def _target_ratios(min_num, max_num):
    """官方 set comprehension 原样转录（i/j 上界是 n 不是 max_num），
    按乘积升序稳定排序。"""
    ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    return sorted(ratios, key=lambda x: x[0] * x[1])


def solve(width, height, base_size=1024, image_size=640, cropping=True,
          min_crops=2, max_crops=6, patch_size=16, downsample_ratio=4):
    """DeepSeek-OCR 五档动态分辨率与 tile 账（deepseek-ai/DeepSeek-OCR
    DeepSeek-OCR-vllm process/image_process.py 的 dynamic_preprocess +
    tokenize_with_images 视觉 token 序列账的标量无状态化；五档配置：
    Tiny 512/512、Small 640/640、Base 1024/1024、Large 1280/1280 均
    crop_mode=False，Gundam 1024/640 crop_mode=True）。

    官方口径（逐条钉死）：

      小图门   cropping 且 width <= image_size 且 height <= image_size
               （**and**，一维越界即切）-> crop_ratio (1,1)；
               cropping=False 直接 (1,1)
      ratio 集 {(i,j): n∈[min,max], 1<=i,j<=n, min <= i*j <= max}
               ——i/j 上界是循环变量 n 而非 max_num（min=2 排除 (1,1)）；
               sorted 按乘积升序（稳定）
      最近匹配 aspect = w/h；diff = |aspect - i/j| 严格 < 更新（先到
               先得）；**diff 相等时 area 抢位**：area > 0.5*image_size^2
               * i*j 即改选当前 ratio（原图面积恒定，阈值随 tile 数增大）
      tile 切分 resize 到 (image_size*i, image_size*j)——**直接拉伸**
               非等比；box 行优先：x0=(k%nw)*image_size,
               y0=(k//nw)*image_size，k∈[0, i*j)
      token 账 num_queries q = ceil((image_size//patch)/down)；全局
               qb 同 base_size；序列 = 全局 ([qb 个 img + 1 sep]) * qb
               再 +1 sep（**(qb+1)*qb + 1**）；nw>1 或 nh>1 时追加局部
               **([q*nw 个 img + 1 sep]) * (q*nh)**（= (q*nw+1)*q*nh，
               行内跨 tile 连排、每行一个 sep）；README 的"vision
               tokens"（64/100/256/400）是 qb^2/q^2 纯编码器口径，
               序列账多出行 sep——两口径并存
      五档     q: Tiny 8 / Small 10 / Base 与 Gundam 全局 16 /
               Large 20；Gundam 局部 q=10

    参数均为标量；width/height 为原图尺寸。
    返回 dict（键固定）：crop_ratio [nw, nh] / tiles / tile_boxes
    （行优先 [x0,y0,x1,y1] 列表，单 tile 时仍给 1 个）/ global_tokens /
    local_tokens / num_image_tokens。

    红线：不出 PIL 图像；纯标量账。
    """
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    if base_size <= 0 or image_size <= 0:
        raise ValueError("base_size/image_size must be positive")

    if not cropping:
        crop_ratio = (1, 1)
        tiles = 1
        tile_boxes = [[0, 0, image_size, image_size]]
    elif width <= image_size and height <= image_size:
        crop_ratio = (1, 1)
        tiles = 1
        tile_boxes = [[0, 0, image_size, image_size]]
    else:
        aspect = width / height
        ratios = _target_ratios(min_crops, max_crops)
        best_diff = float("inf")
        best = (1, 1)
        area = width * height
        for r in ratios:
            diff = abs(aspect - r[0] / r[1])
            if diff < best_diff:
                best_diff = diff
                best = r
            elif diff == best_diff:
                if area > 0.5 * image_size * image_size * r[0] * r[1]:
                    best = r
        crop_ratio = best
        nw, nh = best
        tiles = nw * nh
        tw, th = image_size * nw, image_size * nh
        tile_boxes = []
        for k in range(tiles):
            x0 = (k % (tw // image_size)) * image_size
            y0 = (k // (tw // image_size)) * image_size
            tile_boxes.append([x0, y0, x0 + image_size, y0 + image_size])

    nw, nh = crop_ratio
    q = math.ceil((image_size // patch_size) / downsample_ratio)
    qb = math.ceil((base_size // patch_size) / downsample_ratio)
    global_tokens = (qb + 1) * qb + 1
    if nw > 1 or nh > 1:
        local_tokens = (q * nw + 1) * (q * nh)
    else:
        local_tokens = 0
    return {"crop_ratio": [nw, nh], "tiles": tiles,
            "tile_boxes": tile_boxes, "global_tokens": global_tokens,
            "local_tokens": local_tokens,
            "num_image_tokens": global_tokens + local_tokens}
