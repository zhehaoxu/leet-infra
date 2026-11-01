# Pix2Struct 可变宽高比 patch 网格账

## 题意

Pix2Struct 是把**整页截图**直接喂给视觉编码器的 OCR 模型：图像不再固定缩放到正方形，而是保持宽高比地缩放到「恰好铺满 `max_patches` 个 patch」的网格，再展平成序列、不足补 padding。本题转写 transformers v4.51.3 `Pix2StructImageProcessor.extract_flattened_patches` 的网格账：给定原图高宽、patch 预算与 patch 尺寸，输出行列数、缩放后尺寸、patch 总数、padding 数与逐 patch 的 (row_id, col_id)。

## 求解

口径题面钉死（与官方一致）：

1. `scale = sqrt(M * (ph / H) * (pw / W))`——在保持宽高比的前提下最大化缩放，使总 patch 数逼近预算 M；
2. `R = max(min(floor(scale * H / ph), M), 1)`、`C = max(min(floor(scale * W / pw), M), 1)`——先对 M clamp 上限，再对 1 clamp 下限；
3. `resized_height = max(R * ph, 1)`、`resized_width = max(C * pw, 1)`；
4. `total_patches = R * C`。记 `x = scale * H / ph`，则 `scale * W / pw = M / x`，由 `floor(x) * floor(M / x) ≤ x * (M / x) = M` 可证 **R*C ≤ M 恒成立**，故 `padded = M - R*C ≥ 0`；
5. `patch_ids` 按**行优先**展开，每项 `[row_id, col_id]`，且**偏移 +1**——0 被保留给 padding 位（官方注释原话）。

官方在得到网格后还会做 bilinear resize（antialias）与 `torch_extract_patches`，本题只考网格与索引账，不做像素重采样。

## 数据规模

- 1 ≤ H, W ≤ 10^4，1 ≤ M ≤ 10^4，1 ≤ ph, pw ≤ 64（全为正 int）。
- 任一入参非正/非 int/bool → `ValueError`。
