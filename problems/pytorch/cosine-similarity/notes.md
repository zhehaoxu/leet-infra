# 余弦相似度详解

## 方向而非大小

点积 $a\cdot b$ 同时受**夹角**和**长度**影响；除以两个模长后就把长度归一掉，只留夹角信息：
$$ \cos = \frac{a\cdot b}{\lVert a\rVert\lVert b\rVert} = \hat a\cdot \hat b $$
因此 $a$ 与 $7a$ 相似度为 1（同向），正交为 0，反向为 $-1$。embedding 检索偏爱它，
正是因为「同一个概念说多说少」应被判为相似。

## 与 L2 距离、内积的关系

对**已归一化**的向量，$\lVert \hat a - \hat b\rVert^2 = 2 - 2\cos$，于是「最大化余弦」
等价于「最小化球面 L2」。未归一化时内积 (dot) 会把模长也算进去，常用于带权重的重要度打分；
选哪个取决于「长度是否有语义」。

## eps 的两种写法

- `clamp_min(eps)`（PyTorch / 本题）：模长小于 eps 才托底，正常值不受影响。
- `+eps`（常见于论文伪码）：所有分母都加 eps，会在小角度时轻微压低相似度。
两者对零向量都安全，但数值不同——务必与参考实现 `F.cosine_similarity` 保持同一种。

## 在对比学习里

InfoNCE/NT-Xent 把一正样本与若干负样本的相似度除以温度后做 softmax：正样本要最大化、
负样本最小化。所以「cosine → /τ → softmax」是 contrastive loss 的骨架，理解本题
是理解 SimCLR、MoCo、CLIP 损失的第一步。
