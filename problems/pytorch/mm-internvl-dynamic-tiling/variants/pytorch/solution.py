import math


def solve(width, height, min_num, max_num, image_size, use_thumbnail,
          patch_size=14, scale_factor=0.5):
    """按 starter 注释的规则选 InternVL tile 网格并数 token 账。"""
    for name, v in (("width", width), ("height", height), ("image_size", image_size)):
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, v in (("min_num", min_num), ("max_num", max_num)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"{name} must be a positive integer")
    if min_num > max_num:
        raise ValueError("min_num must not exceed max_num")
    if not isinstance(use_thumbnail, bool):
        raise ValueError("use_thumbnail must be a bool")

    aspect_ratio = width / height

    # 规则 1：候选网格，按 (i*j, i, j) 升序去重（上游 set 序不可判，题面钉死）
    ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    ratios.add((i, j))
    target_ratios = sorted(ratios, key=lambda x: (x[0] * x[1], x[0], x[1]))

    # 规则 2：closest aspect ratio（上游 find_closest_aspect_ratio，diff 相等走面积规则）
    best_ratio_diff = math.inf
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    gi, gj = best_ratio
    target_width = image_size * gi
    target_height = image_size * gj
    blocks = gi * gj

    # 规则 3：行主序切块（此处只记账不裁剪）
    thumbnails = 1 if (use_thumbnail and blocks != 1) else 0
    total_tiles = blocks + thumbnails

    per_tile_tokens = (image_size // patch_size) ** 2 * scale_factor ** 2
    if per_tile_tokens != int(per_tile_tokens):
        raise ValueError("per-tile token count must be an integer")
    per_tile_tokens = int(per_tile_tokens)
    total_visual_tokens = per_tile_tokens * total_tiles

    return {
        "aspect_ratio": aspect_ratio,
        "target_ratio": [gi, gj],
        "target_width": target_width,
        "target_height": target_height,
        "blocks": blocks,
        "thumbnails": thumbnails,
        "total_tiles": total_tiles,
        "per_tile_tokens": per_tile_tokens,
        "total_visual_tokens": total_visual_tokens,
        "image_token_count": 2 + total_visual_tokens,
    }
