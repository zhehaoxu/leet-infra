# 题解：隔离 prefix 的二维位置嵌入重采样

## 真实来源与范围区别

本题背景来自 **Hugging Face 的 DINOv2 位置编码插值路径**。已核对本地 `transformers==5.20.2`：

`.venv-eval/lib/python3.12/site-packages/transformers/models/dinov2/modeling_dinov2.py` 第 87–123 行，`Dinov2Embeddings.interpolate_pos_encoding`。

该实现取出一个 CLS 位置向量，用空间 token 数量的平方根恢复方形旧网格，再以显式 `size=(new_height,new_width)` 调用 bicubic、`align_corners=False`；未传 antialias，使用默认 False。插值前转 float32，结束后转回原 dtype。[HF 上游源码入口](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py)；行号与版本以上述本地文件为准。

**本题不是默认 DINOv2 或 timm 的逐位复刻：**

- 本题将 HF 的方形旧网格推广为显式矩形旧/新网格，并支持零个至多个 prefix，而不是固定一个 CLS。
- 本题保留 float64 输入的计算精度，不照搬 HF 的强制 float32 中间路径；相同尺寸返回 clone，而 HF 对特定相同尺寸分支可直接返回原位置张量。
- [官方 DINOv2](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py) 默认 `interpolate_offset=0.1` 的路径使用带 offset 的缩放因子，与本题无 offset 的显式 size 口径不同。
- [timm 的 `resample_abs_pos_embed`](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed.py) 默认 `antialias=True`，也不是本题固定的非抗锯齿路径。默认值不能混称为同一个精确算法。
- 这里只复用“隔离 prefix 后调整空间网格”的思路及指定的非 AA 数值口径，没有运行预训练模型或验证完整前向。Keys 核是一般三次卷积插值核；$a=-0.75$ 是本题采用的 PyTorch 非 AA bicubic 约定，不是 DINOv2 独创。

与已有 `antialias-upsample` 的区别：那题覆盖通用插值模式与抗锯齿，本题固定插值模式，重点是 **CLS/prefix 隔离、序列与矩形二维网格的互换、dtype 与前缀逐位保护**。

## 推导与实现

1. 在任何提前返回之前，检查序列长度是否等于 prefix 数量加旧网格面积。只要求这项不一致抛 ValueError；其他输入保证合法。
2. 高、宽分别相同时直接 clone。旧/新面积相同但形状不同时仍须插值。
3. 将前缀保留在原 dtype；剩余 token 按行主序恢复为 `[H0,W0,D]`。D 是独立特征轴，不是待缩放的空间轴。
4. 每轴用 half-pixel 坐标产生四个邻点，按题面 Keys 核算权重。距离使用原始邻点位置，读取索引才 clamp。
5. 显式累加两轴权重乘积对应的 16 项。边界重复索引仍各有一项，不能去重覆盖，也不要截断负权重或空间输出。
6. 按行主序展平空间部分、转回输入 dtype，与未经转换的 prefix 拼接。

参考实现使用 float64 计算坐标、权重与空间累加，float32 输入在空间结果写出时回投。这样实现了题面数学规则，但并不承诺与原生 float32 内部运算逐位一致；空间比较允许 float32 的 `atol=rtol=1e-5`、float64 的 `1e-10`。prefix 始终要求逐位一致，包括负零；不能用大容差比较整个序列掩盖前缀错误。

对每个目标格点仅需固定 16 项。参考实现的时间为 $O(H_0W_0D+16H_1W_1D+PD)$，空间为 $O(H_0W_0D+H_1W_1D+PD)$，包含工作网格与输出；尺寸相同只需克隆。

## 独立 oracle 与实跑

测试只在 oracle 中调用原生 `F.interpolate`，显式指定 `size`、`mode="bicubic"`、`align_corners=False`、`antialias=False`。空间插值先升为 float64，完成后转回输入类型，避免原生 float32 坐标舍入对大幅值输入造成误拒绝。它不调用参考实现的邻点或权重函数。prefix 使用整数视图按位比较，空间输出单独做浮点容差比较；同时校验输入不变、dtype、形状，以及恒等分支不共享输入存储。

本地 PyTorch **2.14.1** 上，`add_problem.py --selftest mm-pos-embed-resample --mode submit` 得到 **14/14 ACCEPTED**。覆盖巨大正负 prefix 哨兵与负零、HW 方向与行主序、零/多 prefix、单轴与单格、上下及混合采样、相同面积不同形状、常量场、两种 dtype、非连续视图、不变性、非法旧网格长度、尺寸上限和固定 `torch.Generator` 随机输入。`check_example.example=True`。

题面示例在参考实现与原生 oracle 中实跑一致：

```text
输入序列 = [[[99.0], [0.0], [8.0]]]
旧/新网格 = (1,2) -> (1,4)
输出序列 = [[[99.0], [-0.84375], [1.8125], [6.1875], [8.84375]]]
```

prefix `99.0` 没有进入空间加权；两端过冲体现了未截断坐标与负核权重。上述数值是运行记录，不是从完整 HF、官方 DINOv2 或 timm 模型输出中摘取的结果。
