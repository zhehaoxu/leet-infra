# 解析：图像占位展开与混合 Batch 整理

## 来源与限定

来源是 LLaVA 的 [`llava/model/llava_arch.py`](https://raw.githubusercontent.com/haotian-liu/LLaVA/main/llava/model/llava_arch.py)，`prepare_inputs_labels_for_multimodal` 第 212–324 行中的去 padding、占位展开与 batch 整理逻辑。该链接指向 `main`，行号对应核实的版本，后续可能移动。

本题不是整个原函数的等价复刻，而是以下受限改编：

- 每行必须恰有一个有效图像占位，视觉特征已提供，不编码图像。
- 文本 embedding 已提供，用显式 `image_mask` 标记占位，不依赖原始 token ID 或模型的 embedding 查表。
- 固定右 padding，移除原版可选的长度截断及其他 padding 方向。
- 始终返回五个输出张量，并额外提供 `mm_token_type_ids`；这不是声称原版返回该字段。
- label 只随片段搬移，视觉位置设为 `-100`，不进行 causal label shift。

这与现代 HF 的“processor 先展开成 `N` 个图像标记，模型再用 `masked_scatter` 替换对应 embedding”的路径不同。后者在 scatter 时已有足够位置；本题从单占位开始，必须改变序列长度，不能直接对原序列做等长替换。

## 参考方法

先检查每行图像标记数是否为一，并拒绝任何位于无效位置的标记。按行验证，不能只检查整个 batch 的标记总数。

对第 `b` 行：

1. 用 `attention_mask[b]` 提取有效 embedding、label 与图像标记，保持相对顺序。
2. 在压缩后的行中找到唯一占位位置 `i`，把序列划分为占位前、占位、占位后三段。
3. 拼接“前段文本、全部视觉向量、后段文本”。label 同样拼接，但视觉段长度为 `N_b`、值为 `-100`。
4. 构造相同长度的 type id，仅视觉片段置为 `1`。

求出最大展开长度 `S` 后，一次性创建五个输出张量并填好 padding 默认值。把每行结果放入前 `S_b` 个位置，将对应 mask 置真，position id 写为 `0..S_b-1`。

由于 `N_b>=1`，每行展开后非空。即使 `N_b=1`，仍必须丢弃占位原 label、写入视觉 label 与 type id。

## 正确性要点

去 padding 只删除无效位置，保留有效位置的相对顺序。唯一图像占位把有效序列唯一分成前后两个文本片段；拼接这两个片段与视觉序列，恰好执行要求的替换，长度为 `A_b-1+N_b`。

文本片段的 label 没有被重新计算或错位，视觉片段全部忽略损失。初始化后的右侧 padding 不会被有效片段覆盖；同一个长度 `S_b` 同时控制 embedding、label、mask、position 和 type 的写入边界，因此五个输出保持对齐。

文本的 `label=-100` 与图像身份相互独立；padding 和视觉位置虽然都使用 `-100`，仍可由 attention mask 和 type id 区分。

## 复杂度

令 `S = max_b(A_b-1+N_b)`。参考方法时间复杂度为 `O(BL+BSD)`，其中包含 mask 检查与 batch 输出初始化；空间复杂度为 `O(BL+BSD)`，包含校验时的临时 mask、逐行中间结果及最终输出。无需模型计算。

## 测试与易错点

共有 **14 个 `check_*`**，其中 `check_example.example = True`。独立 oracle 不复用参考解的按行切片拼接，而是逐原始有效位置发出“文本事件”或多个“视觉事件”，随后独立装配 Python 列表并转换成五个张量。

覆盖：图像在头、中、尾，`N_b=1`，不同视觉长度，左/右 padding 与内部空洞，文本 label 为 `-100`，padding 和占位哨兵不泄漏，所有输入的非连续带偏移视图，不修改张量、列表或底层存储，两种浮点 dtype，最小与最大范围，随机对拍，以及缺失、多余、无效位置和后续行的非法标记。

常见错误是按有效长度截取前缀而非按 mask 压缩、移动 label 一位、由 `label=-100` 推断图像、保留原始位置编号、把 pad 的 type id 设为 `1`，以及按输入 `L` 截断展开结果。

代码与独立测试先通过以下提交模式自测，再实跑样例并编写文档；结果为 **14/14 AC**：

```text
.venv-backend/bin/python scripts/add_problem.py --selftest mm-image-token-expand --mode submit
```
