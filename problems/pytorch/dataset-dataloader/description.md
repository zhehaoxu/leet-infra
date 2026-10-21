# Dataset 与 DataLoader

## 题目描述

PyTorch 用 map-style `Dataset` 约定数据访问接口：实现 `__len__` 与 `__getitem__` 两个方法，`DataLoader` 就能按索引取样本、自动 collate 成批。接口契约写错（返回类型不对、索引越界不报错）， DataLoader 会在很远的地方以很难懂的栈炸掉。

## 规则

实现 `solve() -> torch.utils.data.Dataset`，返回一个 map-style 数据集实例：

1. `len(dataset) == 100`。
2. `dataset[i]`（`0 <= i < 100`）返回元组 `(x, y)`：
   - `x`：float32 张量 `[i, 2i, 3i, 4i]`，形状 `(4,)`
   - `y`：int64 标量张量 `i % 2`（奇偶标签）
3. `i` 为负数或 `>= 100` 时，按标准约定抛 `IndexError`（Python 负索引语义也不允许放行，`dataset[-1]` 同样要 `IndexError`）。

不要自己实现 batching——评测会用 `DataLoader(dataset, batch_size=8, shuffle=False)` 检查你的数据集能否被正确组批。

## 示例

```
ds = solve()
len(ds)      -> 100
ds[7]        -> (tensor([ 7., 14., 21., 28.]), tensor(1))
ds[0][0]     -> tensor([0., 0., 0., 0.])
```

## 约束

- `x` 必须 `dtype=torch.float32`，`y` 必须 `dtype=torch.int64` 且 `dim()==0`
- 样本值按公式生成即可，无需持有 100 条预存数据
