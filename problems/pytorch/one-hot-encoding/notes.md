# One-Hot 编码 详解

## 定义

把类别索引 $i\in\{0,\dots,K-1\}$ 映射成长度 $K$ 的向量，仅第 $i$ 位为 1、其余为 0：

$$ \operatorname{onehot}(i)_k=\mathbb 1[k=i] $$

形状 $(\dots)\to(\dots,K)$：在**末尾新增**类别维，索引张量每个位置各自展开成一个 0/1 向量。

## 为什么独热

- 类别是**无序名义量**，直接用整数 1/2/3 会引入虚假的大小与距离关系；独热把每个类别变成正交单位向量，配合 embedding 查表可学出任意表示。
- 也是交叉熵标签的稀疏形式（`F.cross_entropy` 收整数标签、内部即按独热的 one 位取对应 logit）。

## 实现与数值要点

- 参考解：`torch.zeros(*indices.shape, K, dtype=torch.long)` 再 `out.scatter_(-1, indices.unsqueeze(-1).long(), 1)`。`unsqueeze(-1)` 把索引对齐到新增的末维，`scatter_` 在每条 0 行的第 $i$ 列写 1。
- **类别数 $K$ 口径**：`num_classes=None` 时取 $\max+1$（含 0，故最小 $K\ge1$）；显式给定则用给定值——若 $K\le\max$ 索引会越界报错，若 $K>\max$ 则多出全 0 列。
- 空张量 `numel()==0` 时 $K=0$，返回空。
- 输出用整型（`long`），与 `F.one_hot` 一致；若要浮点独热需自行 `.float()`。

## 关联

embedding-lookup 是独热×权重矩阵的等价高效形式；gather-scatter 是本实现的底层原语；cross-entropy 的整数标签可视为「隐式独热」。
