# 解析：最近媒体掩码与标量门

## 实现思路

先校验 head 数。将布尔媒体标记转成整数后沿文本维做累计和，再与从 1 开始的媒体编号比较。不能使用 `>=`，否则会允许全部历史媒体；也不能在比较前将累计和减一，否则标记本身看不到新媒体。把每幅媒体的可见性重复到它的全部 latent 上，再按媒体优先顺序展开。

只归一化文本特征，然后做查询投影。媒体保持原始特征，直接展开后做 key/value 投影。用 `reshape` 与维度交换完成 head 拆分，支持非连续张量。

## 安全处理全 mask 行

正常行将不可见分数置为负无穷。没有媒体的行在 softmax **之前**替换成有限的全零分数，计算后再把所有不可见位置的概率置为零。不能直接对全负无穷做 softmax，再乘零掩码，因为 NaN 乘零仍为 NaN。

在本题有限输入下，严格为零的概率乘 value 后仍严格为零。合并 head、进行无 bias 输出投影，再加上乘 `tanh(gate)` 的更新。没有媒体的 token 保持原始残差。返回的概率不乘 gate；零门也必须返回真实概率和可见性。

参考实现用显式矩阵乘法与稳定 softmax，不调用 SDPA 或 multi-head attention 封装。

## 门控语义

`gate` 是共享的实标量，`tanh` 将系数限制在 [-1,1]。零初始化使子层在前向上恰好是恒等映射，便于在预训练文本网络中引入新分支；并不表示注意力概率为零。正负相反的门产生相反的残差更新。这里不是为每个通道独立学习一个系数，也不是 sigmoid 的 [0,1] 门。

## 独立验证

oracle 逐 batch、逐 token 用 Python 控制流计数媒体标记，选择**唯一可见媒体的原始切片**。无标记时直接保留零概率和原始文本；有标记时逐 head 对该切片调用原生 PyTorch SDPA。它不构造与参考实现相同的全局 masked-softmax。文本归一化以手写均值/总体方差计算，投影使用 `F.linear`；概率另行通过逐 head 点积、减最大值和指数归一化生成，并放回完整媒体轴。

18 个 `check_*` 覆盖两个实跑样例、零门恒等、正负门与 ±20、标记即时可见、无标记、不同 batch 的首图前缀、历史媒体隔离、未引用媒体扰动、media 不归一化、逐 token LayerNorm、自定义 epsilon、非对称权重、多 head、最小维度、零投影、float32/64、非连续输入、不修改输入和非法 head 数。稳定性用例在两种 dtype 下构造约 ±1000 的 logits，同时包含全 mask 行，检查有限输出和精确零概率。形状扫描额外覆盖多个随机种子。

公开样例来自测试内的 `_example`、`_negative_example`，文档数值是独立 oracle 校验后的实际 dump。

## 复杂度

设 `L = M K`。当前向量化实现的时间为 `O(B (T+L) D² + B T L D)`，主要中间空间为 `O(B (T+L) D + B H T L)`，不计输入权重。虽然每个 token 最多只读取一幅媒体，返回的概率仍覆盖完整媒体轴，因此输出空间为 `O(B H T L)`。

## 来源与范围

依据已回源的 [OpenFlamingo helpers.py](https://raw.githubusercontent.com/mlfoundations/open_flamingo/main/open_flamingo/src/helpers.py) 第 136–279 行的 `MaskedCrossAttention` / `GatedCrossAttentionBlock`。保留媒体标记累计计数、最近媒体可见性、首图前零注意力及 tanh 门控残差的核心语义。

本题固定无缓存且只看最近媒体；LayerNorm 无 affine，权重由调用者提供，并额外返回概率及可见性供检查。只提取 attention 门控子层，不包含原 block 的前馈分支及其第二个门，也不包含完整视觉编码器、语言模型或多层融合。这里的 tanh 标量门不是逐通道门；本题也不是完整 Q-Former/OpenFlamingo 模型。
