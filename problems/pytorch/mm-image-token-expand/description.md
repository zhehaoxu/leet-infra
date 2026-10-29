# 多模态：图像占位展开与混合 Batch 整理

每条文本序列含一个有效图像占位。先去除无效位置，再把该占位替换成该样本的全部视觉 embedding，最后将不同长度的结果右侧补齐成一个 batch。

本题限定为 LLaVA 原版占位展开子流程的**每行单图、右 padding、无截断**版本，并增加 `mm_token_type_ids`。输入 embedding 已提供，不需要模型权重或词表查找。这不等同于现代 HF 的另一条路径：由 processor 预先展开为 `N` 个图像标记，再由模型用 `masked_scatter` 填入视觉特征；本题是在这里把**一个**占位展开成多个位置。

## 输入与范围

所有张量均位于 CPU，且可以不连续。

| 输入 | 形状、类型与含义 |
| --- | --- |
| `text_embeds` | `float32` 或 `float64`，`[B,L,D]`，包含占位及 padding 位置的输入 embedding |
| `labels` | `int64`，`[B,L]`，逐位置标签 |
| `attention_mask` | `bool`，`[B,L]`，`True` 表示有效位置；允许左 padding、右 padding、内部空洞 |
| `image_mask` | `bool`，`[B,L]`，`True` 表示图像占位 |
| `image_features` | 长度为 `B` 的张量列表，第 `b` 项为 `[N_b,D]`，与 `text_embeds` 同 dtype、同 device |

范围：`1 <= B <= 4`，`1 <= L <= 16`，`1 <= D <= 8`，`1 <= N_b <= 12`。

每行有效位置必须**恰有一个**图像标记。标记数量不正确，或任意 `image_mask=True` 位于 `attention_mask=False` 的位置时，抛出 `ValueError`。其余输入保证合法。不得修改任何输入张量或图像特征列表。

## 展开规则

对每一行独立执行：

1. 按原始顺序保留 `attention_mask=True` 的位置，丢弃其余位置；不能假设有效位置连续。
2. 将唯一的图像占位替换为该行 `image_features[b]` 的全部 `N_b` 个向量，顺序不变。占位自身的 embedding 和 label 均丢弃。
3. 其他有效文本的 embedding 和 label 原样保留。所有新视觉位置的 label 为 `-100`。**不 shift labels，不截断**。
4. 有效位置的 `position_ids` 从 `0` 连续递增；视觉位置的 `mm_token_type_ids` 为 `1`，文本位置为 `0`。

文本 label 为 `-100` 只表示该文本位置不参与损失，不能据此判断图像位置。图像身份仅由占位展开产生。

## 返回结果

令 `A_b` 为第 `b` 行输入的有效位置数，展开长度为 `S_b = A_b - 1 + N_b`，batch 长度为 `S = max_b S_b`，因此 `1 <= S <= 27`。

返回字典，**恰有**以下五个 key。所有输出均在 CPU，每行右侧补齐到 `S`：

| key | 形状与 dtype | 有效位置 | 右侧 padding |
| --- | --- | --- | --- |
| `inputs_embeds` | `[B,S,D]`，与输入相同 dtype | 展开后的 embedding | `0` |
| `labels` | `[B,S]`，`int64` | 文本保留原 label，视觉为 `-100` | `-100` |
| `attention_mask` | `[B,S]`，`bool` | `True` | `False` |
| `position_ids` | `[B,S]`，`int64` | `0,1,...,S_b-1` | `0` |
| `mm_token_type_ids` | `[B,S]`，`int64` | 视觉为 `1`，文本为 `0` | `0` |

## 样例

`text_embeds` 与 `image_features` 均为 `float32`，`labels` 为 `int64`，两个输入 mask 为 `bool`：

```text
text_embeds = [
  [[1,10], [99,99], [2,20], [3,30]],
  [[88,88], [4,40], [77,77], [5,50]]
]
labels = [[11,91,-100,13], [81,14,92,15]]
attention_mask = [
  [True, True, True, True],
  [False, True, False, True]
]
image_mask = [
  [False, True, False, False],
  [False, False, False, True]
]
image_features = [
  [[101,102], [103,104]],
  [[201,202]]
]
```

输出：

```text
inputs_embeds = [
  [[1,10], [101,102], [103,104], [2,20], [3,30]],
  [[4,40], [201,202], [0,0], [0,0], [0,0]]
]
labels = [
  [11,-100,-100,-100,13],
  [14,-100,-100,-100,-100]
]
attention_mask = [
  [True, True, True, True, True],
  [True, True, False, False, False]
]
position_ids = [[0,1,2,3,4], [0,1,0,0,0]]
mm_token_type_ids = [[0,1,1,0,0], [0,1,0,0,0]]
```

两行展开长度分别为 `5` 和 `2`。第一行 `[2,20]` 的 label 仍为 `-100`，但它是文本，所以 type id 为 `0`。第二行去掉左 padding 和内部空洞，再在右侧补三个位置。样例对应 `check_example`，已实跑校验。
