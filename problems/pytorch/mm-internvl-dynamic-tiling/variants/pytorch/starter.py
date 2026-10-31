def solve(width, height, min_num, max_num, image_size, use_thumbnail,
          patch_size=14, scale_factor=0.5):
    """按 InternVL dynamic_preprocess 的口径，为一张图选择 tile 网格并数 token 账。

    参数
    ----
    width, height : 正整数，原图尺寸。
    min_num, max_num : tile 总块数的下/上界（1 <= min_num <= max_num）。
    image_size : 正整数，单块 tile 的边长（如 448）。
    use_thumbnail : bool。块数 != 1 时是否追加一张整图缩略 tile。
    patch_size : ViT patch 边长（InternVL2 用 14）。
    scale_factor : pixel shuffle 的下采样因子（InternVL2 用 0.5，token 数乘 scale**2）。

    返回
    ----
    dict：
        aspect_ratio        float  width / height
        target_ratio        [i, j] 选中的 tile 网格（i 列 j 行，i*j ∈ [min, max]）
        target_width        int    image_size * i
        target_height       int    image_size * j
        blocks              int    i * j
        thumbnails          int    追加的缩略 tile 数（use_thumbnail 且 blocks != 1 时为 1，否则 0）
        total_tiles         int    blocks + thumbnails
        per_tile_tokens     int    (image_size // patch_size)**2 * scale_factor**2
        total_visual_tokens int    per_tile_tokens * total_tiles
        image_token_count   int    2 + total_visual_tokens（<img> </img> 各 1）

    规则（与上游逐句对应，tie-break 已钉死为确定性版本）：
    1. 候选网格：所有 (i, j) 满足 i, j >= 1 且 min_num <= i*j <= max_num，
       按 (i*j, i, j) 升序去重排列（上游用 set，迭代序不可判，此处钉死字典序）。
    2. aspect_ratio = width / height。遍历候选：
       diff = |aspect_ratio - i/j|；
       - diff < best_diff：更新 best（best_diff 初始为正无穷，best 初始 (1,1)）
       - diff == best_diff：仅当 width*height > 0.5 * image_size**2 * i * j 时覆盖 best
         （上游原式；含义是「原图面积超过该网格拼合面积的一半才选更大的网格」）
       - diff > best_diff：跳过
    3. tiles 切分：target_width x target_height 的网格按行主序出 blocks 块。
    4. 校验：min_num/max_num/width/height/image_size 任一非法（非正整数、
       min_num > max_num）抛 ValueError。
    """
    raise NotImplementedError
