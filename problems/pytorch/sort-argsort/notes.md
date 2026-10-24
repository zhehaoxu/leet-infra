# 排序与 argsort 详解

## values 与 indices 的分工

`sort` 返回排好序的**值**，`argsort` 只返回**位置置换**。真正灵活的是后者：拿到
`indices` 后可以用 `gather` 把**任意同形关联张量**按同一顺序重排——例如检测里按置信度
排 box 时，要把对应的坐标、类别一起搬；NLP 里按序列长度排序做打包（bucketing）。

$$ \text{values} = x[\text{indices}],\quad \text{即 } \text{values}_i = x_{\,\text{indices}_i} $$

## 选择排序实现

本解用选择法：每轮在未排序区取 `argmin`、写出值与原始索引、再把它置 `+inf` 移出候选，
$O(n^2)$ 但语义透明。`torch.sort` 内部是并行排序网络（bitonic/merge），深度 $O(\log^2 n)$。

## 稳定性与并列

「稳定排序」保证相等元素保持原相对次序。`torch.sort(stable=True)` 及 `argsort` 可控；
`argmin` 天然取**最小索引**，因此本实现对并列给出「靠前索引优先」的确定结果。判题用连续
随机数规避并列歧义，使与 `torch.sort` 精确对齐。

## 与其它算子的关系

- **top-k**（本合集另一题）＝只取排序前 k 个，无需排全。
- **argsort 的逆**（rank）：`rank[argsort[i]] = i`，可用于把「名次」映射回「原位置」。
- 双 argsort：`argsort(argsort(x))` 得到每个元素的**秩（rank）**，是非参数化去排名/
  分位数变换的常用技巧。
