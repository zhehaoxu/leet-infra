# notes — ocr-deepseek-tile-ledger（order 719）

## 回源

- 权威源：deepseek-ai/DeepSeek-OCR `DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py` + `config.py`（main，2026-10 initial commit 后未改动）。HF 版模型代码 trust_remote_code 在 HF hub（同构）；torchvision/HF 版 dynamic_preprocess 同源（源自 InternLM/XComposer 系 anyres）。
- **config 钉死**：IMAGE_SIZE=640、BASE_SIZE=1024、CROP_MODE=True、MIN_CROPS=2、MAX_CROPS=6（注释 max:9，显存小建议 6）。五档 = 改配置：Tiny 512/512/False、Small 640/640/False、Base 1024/1024/False、Large 1280/1280/False、Gundam 1024/640/True。
- **ratio 集合构造**：`set((i,j) for n in range(min,max+1) for i in range(1,n+1) for j in range(1,n+1) if i*j<=max and i*j>=min)`——i/j 上界是 n。**等价性**：min=2/max=6 下与 i,j≤max 同集合（任意积∈[2,6] 的 (i,j) 有 i,j≤6，取 n=6 即生成）；等价变异 SURVIVED 实测确认。set → sorted by product：同积元素顺序由 set 迭代序定（同进程确定），closest 平分先到先得依赖它——**oracle 用 dict 显式枚举重建同序**（键序 = 插入序 = comprehension 序，与 set 迭代序对同积 tuple 一致性经随机对拍 200 组验证无分歧）。
- **平分 area 抢位**：`elif diff == best_diff: if area > 0.5*image_size²*i*j: best = ratio`——阈值随 i*j 增大、area 恒定；**抢位不看方向**（竖排 (1,n) 也能被横排抢）。aspect 2.5 的 (2,1)/(3,1) 平分链是干净分隔用例：area 4×10⁵（1000×400）不抢、2.5×10⁶（2500×1000）抢。
- **tile 切分**：`image.resize((tw,th))` 直接拉伸（非等比！长条图被压扁到 tile 网格）——与 Qwen-VL 系的等比 + 逐 tile resize 不同。box 行优先（k%nw 为列）。use_thumbnail 默认 False（Gundam 的全局视图是独立的 ImageOps.pad，不是 thumbnail）。
- **token 序列账**（tokenize_with_images）：全局 `([img]*qb + [sep])*qb + [sep]` = (qb+1)·qb+1；局部 `([img]*(q*nw) + [sep])*(q*nh)`——**行内跨 tile 连排**（一行 q·nw 个 img 共享一个行末 sep），无局部末尾独立 sep。README 的"64/100/256/400 vision tokens"是 qb² 纯编码器口径（SAM patch 16 → downsample 4 → (s/64)²）；序列账多 qb+1 或 q·nh 个 sep。num_image_tokens = 273 + local。
- **小图门 and quirk**：`image.size[0] <= 640 and image.size[1] <= 640`——500×700 切（一维越界）、640×640 恰等不切（<=）。cropping=False 完全跳过小图门。
- **min_crops=1 quirk**：(1,1) 入集 → 大方图可 diff 0 选中 → crop_ratio (1,1) → `if nw>1 or nh>1` 假 → dynamic_preprocess 产出的单 tile 被**丢弃**（images_crop_raw 不进序列）——局部账为 0。
- numq 的 ceil：`ceil((image_size//patch)/down)`——支持档位（512/640/1024/1280 均被 64 整除）下 ceil 与 floor 同值；`//down` 变异等价（SURVIVED 后换 downsample×2 实质变异 caught）。判题锚点不受影响。

## 测试设计

- oracle = dict 显式枚举 + 手写最近匹配 + 行/列双循环 box（与 solve 的 k 单循环不同写法）。
- 手算锚点全部整数：893/693/883/903/273/73/111/421。
- 平分双用例（同 aspect 异 area）分别钉 closest-strict-le（<=）与 closest-area-thresh（0.5→4.0）两条变异。

## 变异（13 全抓 + 1 等价）

- 全抓：小图门 and→or / <=→<、ratio 集 max/min 边界、closest 严格性、area 抢位阈值、box 行列交换、resize nw/nh 交换、全局 sep 摘除、局部 or→and、局部 sep 摘除、q 减半、total-1。
- 等价变异 1 条（SURVIVED 记此）：ratios-ij-n-bound——i/j≤n 放宽到 ≤max_num 生成同一集合（min/max 联合约束吸收）。
- **教训**：numq-floor 变异（ceil→floor+余数补偿）在参数域内真等价（is//ps 恒被 down 整除）——SURVIVED 不是坏事，等价也要记录口径。

## 坑

- README"vision tokens"与序列账两口径并存（差 sep），断言须指明口径。
- set 迭代序依赖：solve 与 oracle 的 ratio 序列必须同构（set comprehension vs dict 枚举），200 组随机对拍背书。
- 与 674（TMRoPE）/675（Qwen2-Audio token 账）划界：本题管 DeepSeek-OCR 五档分辨率/tile/token 序列账；`mm-qwen25vl-window-index`、`mm-image-token-expand` 已占 Qwen VL 分辨率账（todos 红线）。
