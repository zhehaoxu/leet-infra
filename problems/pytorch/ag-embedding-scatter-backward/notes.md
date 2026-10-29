# 详解：查表反向就是一次稀疏归约

## 出处

Embedding 反向在两类面试里出现：一是「手写 `F.embedding` 的反向」这种基础手写题，二是显存/性能方向的追问——「为什么大词表的 embedding 梯度用 sparse 优化器能省一大截显存」。后者要求候选人真的理解：梯度天然只在被命中的 \(n\) 行上非零，写成稠密就是 \(O(V\cdot d)\) 的浪费。

## scatter-add 与「覆盖」的区别

前向 \(out_i=w_{idx_i}\)，所以 \(\partial L/\partial w_r=\sum_{i:\,idx_i=r} grad\_out_i\)。注意三点：其一，这是**求和**，任何 `index_put_(..., accumulate=False)` 都会让重复 id 只留下最后一次；其二，未命中行是常量参与前向（等价于没被选），梯度为 0；其三，`padding_idx` 更特殊——前向里它被强制为零向量且不更新，所以即使 `idx` 里有它，那一行也必须归零。用 `scatter_add_` 时，`src` 要广播成与输出同形，或直接 `index_add_(0, idx, grad)`，后者天然处理重复。

## scale_grad_by_freq 的正确顺序

PyTorch 的语义是「每个 token 的贡献除以该 id 的出现次数」：\(\partial L/\partial w_r=\sum_{i:\,idx_i=r} grad\_out_i/f_{idx_i}\)，其中 \(f_r\) 是 `idx` 中 \(r\) 的计数。先摊薄再累加，因此重复 token 的总贡献恰好等于一份 \(grad\_out\) 的平均，而不是把总和再除以 \(f_r\)（两种写法只在数值上差一个 \(f_r\) 因子，很容易在对拍里露馅）。

## 核心与易错点

- **稀疏分支**：`torch.sparse_coo_tensor(indices, values, size)` 后必须 `coalesce()`，判题会检查 `is_coalesced`、`layout` 与 `values().shape[0] <= n`（未合并时它会等于 \(n\)，但仍要保证不膨胀成 \(V\)）。
- **`padding_idx` 与 `scale_grad_by_freq` 同时开启**：先按频率摊薄，再把 padding 行整行清零。
- **越界检查**：`idx` 的负值不能靠 Python 的负索引语义蒙过去，必须显式抛 ValueError。
- **dtype**：累加在 float64 做，返回与 `grad_out` 同 dtype，否则 float32 用例的误差会比 autograd 大一个量级。

## 与邻近题的区别

`ag-pooling-backward` 也是「按索引把梯度送回原位」，但它的目标位置由 `max` 的比较决定；本题的目标位置由外部给定的 `idx` 决定，并额外带来行级合并（同一行多个 token）。`ag-conv2d-backward` 的 scatter 则是空间窗口级的。

## 对拍口径与复杂度

oracle 用 `weight = torch.empty(V,d).uniform_()` + `F.embedding(weight, idx, padding_idx=..., scale_grad_by_freq=..., sparse=...)` 再 `torch.autograd.grad`，float64 atol=1e-9。示例只用 \([2,4],[6,8],[10,12]\) 三行，重复行、命中行、空行三种情况一次覆盖。复杂度 \(O(nd)\) 累加 + 稀疏路径 \(O(n\log n)\) 合并；稠密写法需 \(O(Vd)\) 的清零开销。
