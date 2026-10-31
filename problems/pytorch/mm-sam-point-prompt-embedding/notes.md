# 解析：SAM 稀疏提示嵌入与稠密位置网格

## 实现思路

先做纯 Python 的几何量，再做张量运算：`scale = image_size / max(H, W)`、`new_h = int(H * scale + 0.5)`、`new_w = int(W * scale + 0.5)` 都在 Python float 上算，与 dtype 无关，也绝不能改用 `round` 或 `//`。把两个逐轴因子组成一个 `(2,)` 张量 `(new_w / W, new_h / H)`，点乘一次即可；框用 `repeat(2)` 把同一个因子对复用给 `xyxy` 的四列，再 `reshape(-1, 2, 2)` 拆成角点。缩放和 `+0.5` 完成后才追加 padding 行，所以它天然是 `(0, 0)`，不参与任何偏移。

位置编码只有一个函数：输入已经在 `[0, 1]`，先 `2 * c - 1`，右乘 `(2, P)` 矩阵，乘 `2 * pi`，最后 `cat([sin, cos], dim=-1)`。sin 块必须在 cos 块之前，宽度才是 `2P`。稀疏提示走「除以 `image_size` 后调用」，稠密网格走「本来就在 `[0, 1]` 直接调用」，两条路径共用同一函数，避免归一化口径分叉。

标签写回不要按「padding 行的坐标特殊」来推，而要照抄上游的三步：先算出全部行的编码，再把 `labels == -1` 的整行清零，最后统一加 `not_a_point_embed`。用 `torch.where` 对 padding 行只取加数、其余行取「编码 + 加数」，就复现了这个顺序。框角的下标用 `arange(2B) % 2 + 2` 生成，`[2]`、`[3]` 自然交替。稠密网格的 `cumsum` 轴容易写反：`y_embed` 沿 `dim=0` 累加、`x_embed` 沿 `dim=1` 累加，各除以本轴长度，`stack(dim=-1)` 后 `permute(2, 0, 1)` 得到 `(2P, h, w)`；未请求时返回形状 `(0,)` 的空张量，并保持输入 dtype。

全程只做逐元素乘加、一次小矩阵乘和 `sin`/`cos`，不修改输入，也不要求输入连续。

## 边界与常见错误

- 取整口径：`int(x + 0.5)` 是「先加后截断」。`(1, 2048)` 配 `image_size = 1024` 时 `new_h = 1`，用 `round` 或向下取整会得到 0 或 1 的不同结果，短轴一旦低于 0.5 就暴露差异。
- 逐轴因子不可互换，也不能统一用 `scale`：短轴因子是 `new_h / H` 或 `new_w / W`，与 `scale` 一般差一个取整误差。把 x、y 系数写反是最常见的错误。
- padding 行只取决于「有没有框」，与点数无关：`N = 0, B = 0` 时仍有一行；`B >= 1` 时一行也不能多。输入标签只允许 0/1，`-1` 只作为内部 padding 标签出现。
- padding 行的坐标严格是 `(0, 0)`：若在 `+0.5` 之后再补 `0.5`，或先补 padding 再缩放，都会把它变成 `(0.5, 0.5)`。
- `-1` 行的编码必须「算完再清零」。少了这一步，padding 行会变成 `原始编码 + not_a_point_embed`，虽然仍与坐标同阶，却不再等于 `not_a_point_embed`。返回 `positional_raw` 正是为了让这一差异可观察。
- `sin` 块与 `cos` 块顺序颠倒、或用 `stack` 交错排列，都会破坏与稠密网格的一致性；`cat` 的 `dim=-1` 也别忘了。
- 归一化除以的是正方形 `image_size`，不是 `new_h`/`new_w`，也不是原图尺寸；框角与点必须共用同一个函数。
- 稠密网格的 `cumsum` 轴写反时，`(h, w)` 与 `(w, h)` 只在方阵下看不出来，因此要用 `(1, 4)`、`(4, 1)` 这类长方形去测。
- 全部点都是背景、只有一个框、`P = 1`、`H = W = 1`、`image_size` 远小于原图尺寸都是合法输入，不应特判。
- 浮点张量必须同 dtype、同 device，float32 不得被悄悄提升到 float64，非连续输入要照常工作。

## 独立验证

oracle 完全在 Python 列表与 `math` 里跑：逐轴因子、`int(x + 0.5)` 取整、逐点缩放、`2 * c - 1`、手写二维矩阵乘、逐元素 `math.sin`/`math.cos` 拼接，全部不经过 torch 的广播与 `cat`；padding 行按上游顺序「先算编码、再整行置零、再加 `not_a_point_embed`」显式重建，稠密网格按 `(i + 0.5) / h`、`(j + 0.5) / w` 逐格生成再手工转置成 `(2P, h, w)`。`_verify` 除了逐张量比对五个返回值，还独立复算形状与 dtype，并按标签把每行的加数还原成 `point_embeddings[0..3]` 或 `not_a_point_embed` 再检查一次，同时断言 padding 行的原始编码非零，确保「清零」真的发生过。

16 个 `check_*`（其中 `check_example_point_padding`、`check_example_box_corners` 是公开样例）覆盖：样例、padding 仅在无框时出现、`-1` 行编码被写回清零、全背景标签、无点提示、稠密网格的轴序与 sin/cos 次序、未请求网格返回空张量、预处理尺寸的四舍五入口径、逐轴因子不被写反、行序与框角下标、非默认 `image_size`、两种 dtype、非连续与不修改输入、跨形状与跨 `image_size` 的 20 组随机种子扫描，以及契约里每一条 `ValueError`。两条公开样例由测试中的 `_example`、`_box_example` 构造，文档数值取自参考实现的实际 dump。

## 复杂度

记 `M = N + [没有框] + 2B`，`C = 2P`。稀疏部分耗时 `O(M · P)`（每行一次 `(2, P)` 乘加），稠密网格耗时 `O(h · w · P)`，合计 `O((M + h · w) · P)`。峰值额外空间是返回量本身：`O(M · (C + 2) + h · w · C)`，其中 `positional_raw` 与 `sparse_embeddings` 各占 `M · C`。与上游一样，这里不涉及任何注意力或卷积，耗时与图像像素数无关，只与提示行数和网格大小线性相关。

## 来源与刻意简化

依据 [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)（Kirillov et al., arXiv:2304.02643）的 `segment_anything/prompt_encoder.py`（`PositionEmbeddingRandom.forward` / `_pe_encoding` / `forward_with_coords`、`PromptEmbedding._embed_points`、`_embed_boxes`、`PromptEncoder.forward`）和 `segment_anything/utils/transforms.py`（`ResizeLongestSide` 的 `get_preprocess_shape`、`apply_coords`）。保留的事实包括：像素中心 `+0.5`、`int(size * scale + 0.5)` 的取整、逐轴 `new_w / old_w` 与 `new_h / old_h` 因子、padding 点 `(0, 0)` 与标签 `-1`、`-1` 行「编码算完再整行清零」的顺序、框角 `point_embeddings[2]`/`[3]` 与左上—右下的追加次序、`sin` 块在 `cos` 块之前的 `2P` 拼接，以及稠密网格 `cumsum - 0.5` 的轴向。

刻意排除、不在本题范围内：图像缩放本身（PIL/torchvision 重采样）、`mask_postprocessing` 的双线性插值与裁剪、`mask_downscaling` 的 Conv2d 掩码输入通路、ViT/transformer/MLP 的任何权重，以及 mask 侧的稳定分数与单掩码选择。mask 侧的两处口径——`utils/amg.py::calculate_stability_score` 中间求和的 `int16` 溢出怪癖，以及上游 `union = 0` 时 `0/0` 返回 `nan`——属于姊妹题 `mm-sam-mask-stability-select`，本题不涉及；那两处的处理（我们使用精确整数求和，并把 union 为 0 时的稳定分数定义为 `0.0`）在该题的解析里作为刻意偏离记录。此外，上游用未播种的 `torch.randn` 初始化高斯矩阵与四个嵌入，本题把它们改成入参以保证可复现；`input_image_size` 固定为正方形 `image_size`，只实现 `_embed_points`/`_embed_boxes` 与随机位置编码这条纯几何通路，不是完整的 `PromptEncoder`。
