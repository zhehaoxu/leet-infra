# 解题说明 · 分辨率分桶与最近桶分配

## 出处

- `huggingface/diffusers` 的 `src/diffusers/trainers/training_utils.py`：`generate_aspect_ratio_buckets(resolution, divisibility=16, base_resolutions=None)` 与默认比例表 `_DEFAULT_BUCKET_ASPECT_RATIOS`；`find_nearest_bucket(h, w, bucket_options)` 用 `abs(h * bucket_w - w * bucket_h)` 打分、`<=` 比较。源码注释说明 `divisibility=16` 是为满足 Flux / Z-Image VAE 的 patch 尺寸约束。
- 本题把「生成桶 + 分配」做成纯函数；同文件的 `parse_buckets_string` 属于配置解析，不在本题范围内。

## 核心与易错点

**两套列表的朝向不同。** 默认表 `_DEFAULT_BUCKET_ASPECT_RATIOS` 的每一项被当作 `(rw, rh)`（宽、高），而用户传入的 `base_resolutions` 是 `(h, w)`，源码先做 `aspect_ratios = [(w, h) for (h, w) in base_resolutions]` 再进入同一循环。把两边都当 `(h, w)`，竖图基准（如 `(1080, 1920)`）会产出横桶——方向整体反 90°。

**先求高还是先求宽。** 源码先 `h = sqrt(target / aspect)` 再 `w = h * aspect`；也可以先 `w = sqrt(target * aspect)` 再 `h = w / aspect`。数学上等价，但浮点在 `.5` 舍入边界上可能差一个 ulp。本题的主解按源码顺序，独立 oracle 走另一条路径，两边在测试网格上必须给出同一份桶表。

**Python 的 `round` 是银行家舍入。** `round(2.5) == 2`、`round(3.5) == 4`；改用 `int(x + 0.5)` 或 `floor(x + 0.5)` 会在恰好 `.5` 处系统性偏移，桶表随之改变。

**`max(divisibility, ...)` 不是装饰。** 极端长宽比或低分辨率下，`round` 可能给出 0；这个兜底保证每条边至少一个对齐粒度。副作用是不同比例会塌缩到同一个桶，所以去重必须保序：低分辨率 `(16, 16)` 输入只剩一个桶。

**度量用交叉相乘，不是比例差。** `|h * bw - w * bh|` 是整数运算，等价于按 \(h/w\) 与 \(bh/bw\) 的距离比较（差一个正因子），但避免了除法；写成 `|h/w - bh/bw|` 在接近的比例上会出现浮点抖动，且与源码行为不一定一致。

**`<=` 让后一个桶胜出。** 这不是「随便选一个」：同一张图对不同比例可能恰好等距（如 `1088×976` 与 `1:1`、`576:464` 的度量都等于 `57344`），源码的 `<=` 使结果确定地落到后出现的桶。把比较改成 `<` 会让这道题的结果依赖实现细节，而题目要求对齐源码。

**输出的是下标，不是桶本身。** `assignments` 与 `image_sizes` 等长；空 `(0, 2)` 输入合法，返回空下标张量。桶张量保持生成顺序，不要排序。

**像素预算只是近似。** 对齐粒度越大、分辨率越低，`h*w` 与 `resolution²` 的偏差越大（`div=64`、`res=256` 时可超过 6%）；文档与测试只在 `divisibility=16` 时断言 5% 以内。

## 与其他题的分工

- `dt-*`（分词/数据整理）里的 padding/packing 处理的是序列长度；本题处理图像空间尺寸，单位不同。
- `mm-smart-resize`、`mm-clip-preprocess` 把单张图缩放到固定尺寸；本题保留长宽比做离散分桶，再按桶对齐形状，不做插值。
- `diff-sd3-timestep-density`、`diff-min-snr-weight` 同属训练侧数据管线，但处理的是时间步；本题处理空间分辨率。
- `vision-*` 检测题里的 resize/letterbox 是单图几何变换与坐标映射；本题只输出离散桶下标。
