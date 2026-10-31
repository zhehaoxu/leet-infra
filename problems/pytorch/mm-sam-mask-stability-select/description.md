# 多模态：SAM 稳定分数与单掩码选择

## 任务

把 mask decoder 输出的原始掩码 logits 整理成 ONNX 导出路径最后一步真正消费的账目：每个掩码 token 的稳定分数（stability score）、按点提示数量加偏置后的选择分数、被选中的那一张二值掩码，以及面积与原始 IoU 预测的台账。

这是纯粹的逐元素比较与整数计数：没有插值、没有形态学后处理、没有掩码下采样，也不涉及任何可学习权重——`mask_threshold` 与 `offset` 在这里都由调用者给出，以便直接检验阈值口径。

参数、形状、数值范围和异常约定统一见 starter 的双语契约。以下只定义计算规则。

## 计算规则

1. 三条严格阈值：对每个 `(b, c)` 掩码，交集计数是 `masks > mask_threshold + offset` 的像素数，并集计数是 `masks > mask_threshold - offset` 的像素数，面积是 `masks > mask_threshold` 的像素数。三处都必须是严格大于，正好落在某条边界上的像素归入更宽的那一档。契约保证 `offset > 0`，所以并集集合总是包含交集集合。
2. 计数是精确整数：先对最后两维求和，每个 token 得到三个整数。计数过程与浮点 dtype 无关，也不允许任何中间截断带来的溢出。
3. 稳定分数 = 交集计数 / 并集计数；当并集计数为 0 时，本题把稳定分数定义为 `0.0`（上游此处是 `0/0 = nan`）。于是稳定分数总落在 `[0, 1]`，且 `0.0` 既可能来自空并集，也可能来自「交集为空但并集非空」。
4. 偏置只作用在 0 号 token：`biased[b, c] = stability[b, c] + (num_points - 2.5) * (1000.0 if c == 0 else 0.0)`。`num_points <= 2` 时 0 号 token 至少被压低 500，`num_points >= 3` 时至少被抬高 500，因此切换发生在 2 与 3 之间；`num_points` 是正整数，`2.5` 本身取不到，也不存在「偏置恰好为零」的情形。
5. 选择只看 `biased`：`best_idx[b]` 取 `biased[b]` 最大值的下标，同值时取最小下标（torch `argmax` 的行为，而不是稳定排序）。`iou_preds` 完全不参与选择。
6. 选中项：`selected_mask[b] = (masks[b, best_idx[b]] > mask_threshold)` 是布尔掩码；`selected_score[b] = stability[b, best_idx[b]]`，是未加偏置的稳定分数，不是 `biased`。
7. 台账：`areas[b, c]` 逐 token 记录 `masks > mask_threshold` 的像素数——用的是 `mask_threshold`，不是稳定分数的那两条边界；`pred_iou_selected[b] = iou_preds[b, best_idx[b]]` 只把原始预测按选中的 token 收拢一次。

返回七元组：`selected_mask` 布尔 `(B,H,W)`、`selected_score` `(B,)`、`stability` `(B,C)`、`biased` `(B,C)`、`best_idx` int64 `(B,)`、`areas` int64 `(B,C)`、`pred_iou_selected` `(B,)`。浮点输出沿用 `masks` 的 dtype，整型输出为 int64。

## 样例 1：两点提示让 0 号 token 出局

全部浮点张量为 float64，使用默认的 `mask_threshold = 0.0` 与 `offset = 1.0`。

```text
masks = [[[[2.0, 2.0], [2.0, 2.0]],
          [[1.0, 0.0], [-1.0, 2.0]],
          [[0.5, 2.0], [2.0, -2.0]]]]
iou_preds = [[0.9375, 0.5, 0.875]]
num_points = 2
mask_threshold = 0.0
offset = 1.0

stability = [[1.000000, 0.333333, 0.666667]]
biased = [[-499.000000, 0.333333, 0.666667]]
best_idx = [2]
selected_score = [0.666667]
selected_mask = [[[True, True], [True, False]]]
areas = [[4, 2, 3]]
pred_iou_selected = [0.875000]
```

三个 token 的交集/并集计数依次是 `4/4`、`1/3`、`2/3`。0 号 token 稳定分数最高，却被 `(2 - 2.5) * 1000 = -500` 的偏置压到 `-499`，于是选下标 2。下标 1 的掩码里，`1.0` 等于 `mask_threshold + offset`、`0.0` 等于 `mask_threshold`、`-1.0` 等于 `mask_threshold - offset`：三处边界都按严格大于排除，`-1.0` 那一格因此不计入并集。被选中的二值掩码用的就是 `mask_threshold`，与 `areas[0, 2] = 3` 完全一致。

## 样例 2：三点提示与空并集

全部浮点张量为 float64，`mask_threshold = 0.0`、`offset = 1.0`，`num_points = 3`。

```text
masks = [[[[1.5, -0.5], [-1.5, 0.5]],
          [[-1.0, -2.0], [-3.0, -1.5]],
          [[2.0, 2.0], [2.0, 2.0]]],
         [[[-2.0, -2.0], [-2.0, -2.0]],
          [[1.0, 1.0], [1.0, 1.0]],
          [[0.5, -0.5], [2.0, 3.0]]]]
iou_preds = [[0.25, 0.5, 0.75], [1.0, 0.0625, 0.5]]
num_points = 3
mask_threshold = 0.0
offset = 1.0

stability = [[0.333333, 0.000000, 1.000000], [0.000000, 0.000000, 0.500000]]
biased = [[500.333333, 0.000000, 1.000000], [500.000000, 0.000000, 0.500000]]
best_idx = [0, 0]
selected_score = [0.333333, 0.000000]
selected_mask = [[[True, False], [False, True]],
                 [[False, False], [False, False]]]
areas = [[2, 0, 4], [0, 4, 3]]
pred_iou_selected = [0.250000, 1.000000]
```

`+500` 的偏置让 0 号 token 在两个批次里都胜出，哪怕第二个批次的它的稳定分数是 `0.0`：那里四个像素全部 `<= mask_threshold - offset`，并集计数为 0，本题按约定返回 `0.0` 而不是 `nan`，选中的掩码也全为 `False`。第二个批次下标 1 的四个像素全都正好等于 `mask_threshold + offset`，交集为空而并集为 4，于是同样是 `0.0`——空交集与空并集两种 `0.0` 必须都能出现。逐批次的账目互不干扰：第一个批次下标 1 的面积是 0，第二个批次下标 1 的面积是 4。

以上输出均由参考实现实跑，并通过独立 oracle 校验；展示值保留六位小数。
