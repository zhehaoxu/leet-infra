# 解析：Perceiver latent 注意力残差

## 实现思路

先检查 head 数的合法性，再在完整特征维上归一化；不要先拆 head 再做 LayerNorm。`torch.layer_norm` 的 normalized shape 只包含 `D`，且不传 affine 参数。

视觉与 latent 归一化后拼接，再做两个独立投影。查询只来自 latent。用 `reshape` 拆出 head 维，再交换 token/head 维，避免假设输入连续。视觉 mask 后追加全 True 的 latent mask，并广播到查询和 head。用负无穷屏蔽不可见分数，再调用稳定 softmax；因为至少有一个 latent，不需要全 mask 特判。

加权结果先交换回 token/head 次序，再合并特征。直接把 head-major 张量 reshape 为 token-major 会打乱顺序。输出投影使用 `wo.T`，残差必须加原始 latent，而不是归一化后的 latent。

参考实现只使用显式矩阵乘法、LayerNorm 和 softmax，不调用 SDPA 或 multi-head attention 封装。

## 边界与常见错误

- 遗漏 latent K/V 会改变概率分母，也会使空视觉输入失效。
- `True` 表示可见；反转 mask 语义会错误屏蔽有效视觉 token。
- 不能跨 token/batch 归一化，也不能使用无偏样本方差。`D = 1` 时归一化结果为零，不应 NaN。
- 全视觉 mask 不等于整个注意力行全 mask；latent 仍提供 key/value。
- 修改被遮挡视觉 token 的有限填充值不能影响任何输出。
- `wv` 或 `wo` 为零时，输出严格等于输入 latent，但仍须返回正常的注意力概率。
- 权重按 `x @ W.T` 解释；非对称权重才能可靠暴露漏转置错误。
- 不修改输入，不强转 float32，不要求连续存储。

## 独立验证

测试 oracle 逐 batch 先收集真正可见的视觉 key 和全部 latent key，用手写均值/总体方差归一化，再以独立的 `F.linear` 完成投影。逐 head 对收集后的 key/value 调用原生 PyTorch SDPA 验证输出，不复用参考实现的全局 mask 路径。

注意力权重另以逐查询点积、减最大值、指数归一化计算，并散回完整 key 轴。15 个 `check_*` 覆盖样例、latent 拼接、mask 方向与填充扰动、全视觉 mask、空视觉、单特征/单 latent、非默认 epsilon、不同 head 数、非对称权重、零投影、两种 dtype、非连续输入、不变性及非法 head 数；形状扫描还覆盖多个随机种子。两条公开样例可由测试中的 `_example`、`_empty_example` 构造，文档值来自实际执行的 dump。

## 复杂度

设 `L = N + K`。计入稠密投影，时间为 `O(B L D² + B K L D)`；主要中间张量空间为 `O(B L D + B H K L)`，不计输入权重。返回完整注意力矩阵本身就需要 `O(B H K L)` 空间。

## 来源与刻意简化

依据已回源的 [OpenFlamingo helpers.py](https://raw.githubusercontent.com/mlfoundations/open_flamingo/main/open_flamingo/src/helpers.py) 第 25–132 行的 `PerceiverAttention` / `PerceiverResampler`。保留了独立归一化视觉/latent、将 latent 拼入 K/V、以及注意力残差更新的结构。

本题增加视觉 mask 和可观察的注意力返回值；由调用者提供 latent 与四个独立权重，LayerNorm 固定无 affine。只执行一次注意力更新，不学习/复制内部 latent 参数，不执行前馈子层、完整多层 resampler、最终 LayerNorm 或时间/帧嵌入。它不是 Q-Former，也不代表完整 OpenFlamingo 模型。
