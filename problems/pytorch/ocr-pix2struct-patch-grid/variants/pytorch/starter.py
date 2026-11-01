import math


def solve(image_height, image_width, max_patches, patch_height=16, patch_width=16):
    """Pix2Struct patch 网格账（transformers `extract_flattened_patches` 口径）。

    输入
    ----
    image_height / image_width : 原始图像高宽（正 int）
    max_patches                : patch 预算 M（正 int）
    patch_height / patch_width : patch 尺寸（正 int，默认 16×16）

    记账（题面钉死，与官方一致）：
      1. scale = sqrt(M * (ph / H) * (pw / W))——在保持宽高比的前提下
         最大化缩放，使总 patch 数逼近 M；
      2. R = max(min(floor(scale * H / ph), M), 1)
         C = max(min(floor(scale * W / pw), M), 1)
         ——先对 M clamp 上限，再对 1 clamp 下限；
      3. resized_height = max(R * ph, 1)、resized_width = max(C * pw, 1)；
      4. total_patches = R * C。记 x = scale * H / ph，则 scale * W / pw = M / x，
         由 floor(x) * floor(M / x) ≤ x * (M / x) = M 可证 R * C ≤ M 恒成立，
         故 padded = M - R * C ≥ 0；
      5. patch_ids 按**行优先**展开，每项 [row_id, col_id]，且
         **偏移 +1**——0 被保留给 padding 位（官方注释原话）。

    返回
    ----
    dict：
      num_rows        int            R
      num_cols        int            C
      resized_height  int            R * ph
      resized_width   int            C * pw
      total_patches   int            R * C
      padded          int            M - R * C
      patch_ids       list[[int,int]] 行优先 (row_id, col_id)，均从 1 起
    """
    raise NotImplementedError
