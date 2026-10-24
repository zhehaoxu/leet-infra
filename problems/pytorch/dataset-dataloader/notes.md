# Dataset 与 DataLoader 详解

## map-style 数据集协议

PyTorch 的 `torch.utils.data.Dataset` 只需实现两个方法即可被 DataLoader 消费：

- `__len__(self)` → 样本总数（本题固定 100）。
- `__getitem__(self, i)` → 第 $i$ 个样本（本题返回 `(x, y)`）。

另有 iter-style（实现 `__iter__`，如流式语料），本题是 map-style。

## 本题数据口径

`__getitem__(i)` 返回：

$$ x=\texttt{float32}\,[i,\,2i,\,3i,\,4i]\ \text{shape }(4,),\qquad y=\texttt{int64}\,[\,i\bmod 2\,] $$

dtype 必须严格：特征 `float32`、标签 `int64`（分类标签的标配，喂 `cross_entropy` 就要 int64）。`solve()` 返回数据集实例本身（不包 DataLoader）。

## 负索引必须拒绝（关键坑）

Python list 允许 `x[-1]` 取末元素，但**数据集协议约定非法索引抛 `IndexError`**（返回负索引语义会让 DataLoader 的 sampler 逻辑出错、并把越界静默变成合法样本）。参考解显式：

```python
if i < 0 or i >= len(self):
    raise IndexError(...)
```

DataLoader 的默认 `SequentialSampler`/`RandomSampler` 只发 $[0,\text{len})$ 的合法索引，所以正常训练不会触发；但判题会直接用越界/负索引探测，必须抛 `IndexError` 而非返回数据或抛 `ValueError`。

## 与 DataLoader 的配合

DataLoader 在此协议上做批量（`collate_fn` 把若干 `(x,y)` 堆成 `(B,4)`/`(B,)`）、打乱、多进程预取。`__len__` 决定步数，`__getitem__` 决定每个样本长相——两处任一模糊都会让下游形状/类型崩掉。

## 关联

optimizer-training-loop 是消费 DataLoader 的训练闭环；tensor-shape-indexing 解释 `__getitem__` 返回张量的形状操作；nn-module-block 提供被喂数据的模型。
