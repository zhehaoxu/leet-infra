# notes — ocr-pix2struct-patch-grid（order 688）

## 回源

- 出处：`huggingface/transformers` v4.51.3 `src/transformers/models/pix2struct/image_processing_pix2struct.py` 的 `Pix2StructImageProcessor.extract_flattened_patches`（GitHub API contents + base64，按 tag 抓取）。
- 核心四行：`scale = sqrt(M*(ph/H)*(pw/W))` → `R = max(min(floor(scale*H/ph), M), 1)` → `C = max(min(floor(scale*W/pw), M), 1)` → resize 后 `torch_extract_patches` 展平 + `row_ids/col_ids` 各 +1（注释原话 "Offset by 1 so the ids do not contain zeros, which represent padding"）+ pad 到 M 行。
- **R*C ≤ M 恒成立**（floor 乘积不等式：x = scale·H/ph 时另一维 = M/x，floor(x)·floor(M/x) ≤ M），2 万组随机实测 0 违例；故 `padded ≥ 0` 恒真，这是官方设计使然而非巧合。
- 双 clamp quirk：极端长宽比（如 1000000×1、M=64）下 R 被 `min(..., M)` 压到 64，C 的 floor 为 0 被 `max(..., 1)` 救回 1——两个 clamp 都可能同时触发（已钉锚点）。
- 与 `mm-qwen25vl-window-index` 划界：不同算法——pix2struct 是**单图变网格**（比例保持 + 预算约束），Qwen VL 是**窗口索引**；与 `mm-image-token-expand` 划界同理。
- 浮点 floor 临界：`scale*H/ph` 数学上恰为整数时浮点可能翻转 floor（224 案例实测 8.0 精确无翻转）；oracle 不走浮点——用恒等式 `floor(sqrt(A/B)) = isqrt(A//B)`（A = M·pw·H，B = W·ph，对 floor(scale·H/ph) 平方化消去 sqrt）纯整数复算，与浮点 solve 独立对拍。
- 像素重采样（bilinear antialias）与 `torch_extract_patches` 不入题：判题环境考浮点重采样无意义，账本只在网格与索引层。

## 判题面

- solve 入参：image_height/image_width/max_patches（必填正 int）+ patch_height/patch_width（默认 16/16）。返回 num_rows/num_cols/resized_height/resized_width/total_patches/padded/patch_ids。
- oracle：`test_cases._py` 用牛顿迭代 `_int_sqrt`（不用 `math.isqrt`，自实现）+ 整数 isqrt 恒等式独立复算 R/C；200 组随机对拍 + 500 组不变量扫描（R*C ≤ M、padded ≥ 0、双 clamp 下界）。
- 锚点：224×224/M=64 → R=C=8、total=64、pad=0；640×320/M=64 → R=11、C=5、55 patches、pad=9（patch_ids 首个 [1,1]、行末 [1,2]、次行 [2,1]、末个 [11,5]）；1000000×1 → R=64（max clamp）、C=1（min 救回）；1×1 → scale=128、R=C=8；7×13/M=32 → R=4、C=7、pad=4；非方形 patch 8×32。
- 非法输入：任一入参 ≤ 0 / 非 int / bool → ValueError。

## 变异记录（12 条可观察全抓 + 1 条等价）

scale-no-sqrt、ceil-instead-of-floor、rows-cols-swapped、clamp-max-off、clamp-min-off、ids-no-offset、ids-col-major、pad-sign-flipped、total-counts-pad、resized-height-uses-cols、resized-forgets-multiply、floor-arg-swapped——全抓。**等价变异**：scale-patch-swap（scale 公式里 (ph/H)(pw/W) 因子交换，乘积 M·ph·pw/(H·W) 不变，乘法交换律）。

## 红线

真图/PIL/torchvision 不出，尺寸全 int 入参；bilinear resize 数值与 patch 像素内容不出（无确定性对拍价值）；`mm-qwen25vl-window-index`、`mm-image-token-expand` 已占窗口/分辨率账勿重做。
