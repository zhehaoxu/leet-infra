# Embedding 查表详解

## 本质

Embedding 就是**按整数索引取行**：`weight` 是 $(V,d)$ 的词表矩阵，`indices` 里每个值是一个词 id，输出把对应行捞出来、保持 `indices` 的形状再加一维 $d$：

$$ \text{out}[\dots]=\text{weight}[\text{indices}[\dots]],\qquad \text{shape}=(\text{indices.shape},\,d) $$

Transformer 的 token embedding、位置编码表都是它。反向时梯度只回到被查到的那些行（稀疏），未用行梯度为 0。

## padding_idx 的真实语义（本关大坑）

- **`F.embedding`/`nn.Embedding` 的 `padding_idx` 前向并不把该行输出置零**！它只影响：① 初始化时把该行置零；② 反向时该行梯度强制为 0（不更新）。若 weight 里 padding 行本身非零，前向照样返回非零值。
- 因此本关约定的是「**凡 indices 等于 padding_idx 的位置，输出行强制置零**」——这是查询后置零语义，**不能**直接对拍 `F.embedding(...,padding_idx=)`，要手工参考：先普通取行，再 `torch.where(indices==padding_idx, 0, out)`。

## 实现坑

1. **展平再还原形状**：`weight[indices.reshape(-1)].reshape(*indices.shape, d)`，直接 `weight[indices]` 对多维索引结果形状是 `indices.shape+(d,)` 但易与高级索引混淆，展平最稳。
2. `pad_mask.unsqueeze(-1)` 把 $(...)$ 掩码扩到末维，才能与 $(...,d)$ 广播做 `where`。
3. 索引需为整型、在 $[0,V)$ 内（负索引 torch 支持）。

## 对拍建议

用行选择 + padding 置零的手工参考；若题目不考 padding 置零，则与 `F.embedding(indices, weight)` 一致。
