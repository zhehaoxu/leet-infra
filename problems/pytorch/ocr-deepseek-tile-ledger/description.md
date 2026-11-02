# DeepSeek-OCR 五档动态分辨率与 tile 账

## 题意

DeepSeek-OCR（arXiv 2510.18234）把文档压成少量视觉 token：五档分辨率配置 + Gundam 动态 tile。本题手写官方 vLLM 版 `image_process.py` 的 `dynamic_preprocess` tile 选择/切分账与 `tokenize_with_images` 的视觉 token 序列账——纯标量，不出 PIL 图。

## 求解

官方口径（deepseek-ai/DeepSeek-OCR `DeepSeek-OCR-vllm/process/image_process.py` + `config.py` 逐条钉死）：

- **五档**（config 注释原文）：Tiny base/image=512、Small 640、Base 1024、Large 1280 均 crop_mode=False；**Gundam base=1024、image=640、crop_mode=True**。config 默认 MIN_CROPS=2、MAX_CROPS=6（注释 max:9）；
- **小图门**：cropping 且 `width <= image_size 且 height <= image_size`（**and**——一维越界即切：500×700 切、640×640 不切）；cropping=False 直接 (1,1)；
- **ratio 集合 quirk**：`{(i,j): n∈[min,max], 1≤i,j≤n, min ≤ i*j ≤ max}`——**i/j 的上界是循环变量 n 而非 max_num**（对 min=2/max=6 与 i,j≤max 生成同一集合，等价变异实测 SURVIVED）；min=2 把 (1,1) 排除在外；sorted 按乘积升序（稳定）；
- **最近匹配**：`diff = |w/h − i/j|`，严格 `<` 更新（先到先得）；**diff 相等时 area 抢位**：`area > 0.5·image_size²·i·j` 即改选当前 ratio（原图面积恒定、阈值随 tile 数增大）——aspect 2.5 时 (2,1)/(3,1) 平分，area 4×10⁵ 不抢、2.5×10⁶ 抢；
- **tile 切分**：resize **直接拉伸**（非等比）到 `(image_size·nw, image_size·nh)`，再按行优先网格切 box：`x0=(k%nw)·image_size, y0=(k⌿nw)·image_size`；
- **token 序列账**：`q = ceil((image_size//16)/4)`（每 tile 每行 q 个 image token）、`qb` 同 base_size；全局视图 = `([qb 个 <image> + 1 sep])·qb` 再 +1 sep = **(qb+1)·qb+1**（Tiny 73 / Small 111 / Base·Gundam 全局 273 / Large 421）；`nw>1 或 nh>1` 时追加局部 `([q·nw 个 <image> + 1 sep])·(q·nh)` = **(q·nw+1)·q·nh**（行内跨 tile 连排、每行一个 sep）。README 的"vision tokens"（64/100/256/400）是 qb²/q² 纯编码器口径——与序列账差 sep，两口径并存。

## 数据规模

- 手算锚点：Gundam 2000×1500（4:3 → (3,2) 6 tiles、tokens 893、6 个行优先 box）、2000×2000（(2,2)、693）、3000×500（(6,1)、883）、小图门三态（640×640 / 500×700 → (2,3)、640×400 恰等）、平分双档（1000×400 不抢 (2,1) / 2500×1000 抢 (3,1)）、cropping=False 大图、min_crops=1 的 (1,1) 单 tile（官方 or 条件吞掉单 tile 局部账）、五档 native token 数；
- 随机对拍 200 组（oracle 为显式枚举 dict + 手写匹配的独立实现）。

## 评分

七键账逐项相等：crop_ratio / tiles / tile_boxes / global_tokens / local_tokens / num_image_tokens 精确相等（全整数账）。

## 红线

- 不出 PIL 图像、不得 import PIL（AST 节点级检查）；纯标量账；
- width/height/base_size/image_size 必须为正（否则 ValueError）。
