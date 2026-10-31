# 解析：SAM 稳定性分数与单掩码选择

## 实现思路

三段计数共用同一次逐元素比较：把 `masks` 分别与下界 `thr - offset`、上界 `thr + offset`、交付界 `thr` 比较，得到三个布尔张量，再 `flatten(2).sum(dim=-1)` 折叠成 `(B, C)` 的整数计数。求和顺序与上游一致——先按行求和、再按列求总，但这里全程用 int64，不引入 `int16` 的中间截断。`intersection <= union` 由集合包含关系自动成立，所以除法之前不需要额外夹紧。

除法要先处理空分母：用 `unions == 0` 造一个掩码，把除数替换成 1（`torch.where`，不改原张量），算完商再把空并集那些位置整体写成 `0.0`。这样既避开了 `0/0` 的 `nan`，也避免了「先加一个极小量」那种会污染比例的脏做法。整数计数要先转成 masks 的 dtype 再相除，否则 float32 输入会被悄悄提升到 float64。

偏置沿上游的写法构造一个 `(1, C)` 的常量行 `[1000, 0, 0, ...]`，乘 `(num_points - 2.5)` 再与 `stability` 相加，得到 `biased`；不要用切片赋值去改 `stability`，那会破坏要原样返回的那一份。`torch.argmax(biased, dim=1)` 直接给出并列时的最小下标。收集用「先建 `arange(B)` 再与 `best_idx` 一起索引」的花式索引，一次性拿到 `selected_mask`、`selected_score`、`pred_iou_selected`，三者的形状天然是 `(B, H, W)`、`(B,)`、`(B,)`。

`areas` 就是与 `thr` 比较的那份计数，与选择同阈值，因此 `areas[b, best_idx[b]]` 必然等于 `selected_mask[b]` 里真的像素数。整体没有任何排序、没有注意力、也没有第二次遍历像素。

## 边界与常见错误

- 三条界都是严格大于：等于 `thr + offset` 的像素不进交集，但只要它严格大于 `thr - offset` 就仍然进并集。把 `>` 写成 `>=` 会让整片等于 0 的背景被算进并集，分数被系统性地压低。
- `union == 0` 必须记 `0.0`：上游会得到 `nan`，`nan` 再进 `argmax` 的结果不可依赖，所以这里是刻意偏离，而不是巧合。
- 偏置只加在下标 0：加到所有通道等价于没加，选择会退化成纯 `stability` 的 argmax；加错通道（例如加到最后一条）会让样例数值全对不上。
- 忘记 `(num_points - 2.5)` 里的 `-2.5`，或写成 `(num_points - 2.5) * 1000 / C`，都会把「两分扣 500、三分加 500」这个翻转点移走。
- `C == 1` 时偏置仍然生效于 `biased`，尽管 argmax 没得可选；`biased` 是要返回的账目，不能因为没有比较就省掉。
- 并列取最小下标：先 `sort` 再取末位、或用 `topk`，在数值相同时可能给出更大的下标。本题的样例 1 里 `0.333333` 与 `0.666667` 不并列，但随机扫描里 `1/2` 与 `2/4` 这类同比值很常见。
- 用 `iou_preds` 参与选择是另一条通路：上游关掉稳定分数开关时才会走它，本题它只是回传账目。
- `selected_mask` 可以整片为假（选中通道的稳定分为 0），不要当成非法情况。
- 形状与 dtype：`masks` 必须 4 维且各维为正，`iou_preds` 必须是 `(B, C)` 且与 `masks` 同 dtype、同 device；含 NaN/Inf 要直接报 `ValueError`。`num_points` 必须是正整数，`bool` 不算整数；`offset` 必须为正实数。整型输出用 `int64`，布尔输出用 `bool`，浮点输出沿用输入浮点 dtype。
- 不修改输入、不要求输入连续；`flatten` 之前不要 `contiguous()` 复制一份大掩码。

## 独立验证

oracle 完全在 Python 列表上跑：`masks.tolist()` 取到嵌套列表后，用四层 `for` 逐像素做三次 `if value > bound` 计数，交并比是两个 Python 整数相除，空并集显式写 `0.0`，偏置用 `score + bias if channel == 0 else score`，最大下标用手写的从左到右严格大于扫描；布尔网格也是逐格 `append(True/False)` 重建的，最后才一次性包成张量。整条路径不经过 torch 的广播、求和与 `argmax`，与实现共享的只有公式本身。

样例里所有取值都是二分有理数（整数、四分之一、十六分之一），float32 与 float64 都能精确表示，因此 Python 的阈值比较与 torch 的逐位一致，不需要放宽容差去吸收浮点误差。

15 个 `check_*`（其中 `check_example_penalty_hides_best_channel`、`check_example_bias_prefers_first_token` 是公开样例）覆盖：两个样例、偏置在 `num_points` 从 1 到 12 时的翻转、并列取首个下标、严格大于的三条界、全前景与空并集、`areas` 跟随 `mask_threshold`、`mask_threshold`/`offset` 作为自由参数时的单调性、逐样本行的独立性与行序翻转、`iou_preds` 作为被动账目、最小形状与单候选、两种 dtype、非连续与不修改输入、跨形状与跨阈值的 24 组随机种子扫描，以及契约里每一条 `ValueError`。`_verify` 除逐张量比对七个返回值外，还独立复算形状与 dtype，断言分数落在 `[0, 1]`、`biased[:, 0]` 恰为 `stability[:, 0]` 加偏置而其余通道不变、`selected_score` 与 `pred_iou_selected` 都按 `best_idx` 读出、`areas[b, best_idx[b]]` 等于 `selected_mask[b]` 的计数，并要求比 `best_idx` 更靠前的通道严格更小、更靠后的通道不大——把「首个最大值」钉死。

## 复杂度

记 `N = B · C · H · W`。三次逐元素比较与折叠求和、逐 `(B, C)` 的除法、`(B, C)` 上的偏置与 argmax 各自线性，总耗时 `O(N + B · C)`，在 `H · W >= 1` 时即 `O(N)`。峰值额外空间是三个布尔计数张量与一个 `(B, C, H, W)` 的布尔保留图：`O(N)`，返回值另占 `O(B · H · W + B · C)`。相比 mask decoder 的注意力与上采样，这一段是纯线性扫描，没有任何排序。

## 来源与刻意简化

依据 [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)（Kirillov et al., arXiv:2304.02643）的 `segment_anything/utils/amg.py::calculate_stability_score` 与 `segment_anything/utils/onnx.py::SamOnnxModel.select_masks`。保留的事实包括：交集用 `thr + offset`、并集用 `thr - offset` 且两者都是严格大于，先按行求和再求总的两级计数，分数因此落在 `[0, 1]`，`Sam.mask_threshold = 0.0` 与 `stability_score_offset = 1.0` 的默认值，`(1, C)` 的 `[1000, 0, ...]` 权重行、`(num_points - 2.5)` 这个乘子、只作用于下标 0 的加分，`torch.argmax` 并列取最小下标，以及按 `arange(B)` 与 `best_idx` 收集掩码与分数、交付掩码仍是 `masks > mask_threshold` 的原始二值图。

刻意偏离与不在本题范围内：
- 上游 `SamOnnxModel.forward` 里有 `use_stability_score` 开关，关掉时选择用的是原始 IoU 预测。本题把开关删掉，**恒定使用稳定性分数**，`iou_preds` 退化为纯账目输出——这是我们声明的简化，不是上游行为。
- 上游在 `union == 0` 时会返回 `0/0 = nan`；本题显式定义为 `0.0`，也是声明的偏离。
- 上游 `calculate_stability_score` 的中间求和取 `dtype=torch.int16`，像素数超过 32767 会溢出后进入 `int32`；本题全程 int64 精确求和，不复现该怪癖。
- 上游 ONNX 的 `select_masks` 还接收 `prev_masks` 做重叠裁剪；本题入参里没有它，选择只看分数。
- `areas` 在上游是 `return_extra_metrics` 打开时对**上采样后**的掩码求和；本题始终返回，并且统计的是原始 `H × W` 掩码。
- 图像/像素缩放本身（PIL/torchvision 重采样）、`mask_postprocessing` 的双线性插值与裁剪、`mask_downscaling` 的 Conv2d 掩码输入通路、ViT 图像编码器与 mask 解码器的 transformer/MLP 权重、以及超网络（hyper-network）的网格多点掩码生成，全部不在范围内：候选掩码条数就是入参的 `C`，不存在按点网格批量产生 32 张掩码再逐点选择的过程。稳定性分数所需的 gaussian 位置编码矩阵与嵌入向量属于姊妹题 `mm-sam-point-prompt-embedding`，因为上游用未播种的 `torch.randn` 初始化，在那边改为入参。
