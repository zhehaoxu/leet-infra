# Concat / Stack / Split 详解

## 三种「合并/切分」的本质区别

- **cat（concat）**：沿**已存在**的维拼接，不增秩。$(2,3)+(4,3)\xrightarrow{\text{dim}=0}(6,3)$。除拼接维外其余维必须相同。
- **stack**：新建一维再堆叠，**升秩**。$(3,)+(3,)\xrightarrow{\text{dim}=0}(2,3)$。要求各输入形状完全一致。
- **split / chunk**：cat 的逆。`split(dim, size)` 按大小切、`chunk(dim, n)` 按份数切（尽量均分）。

选错 cat/stack 是最常见 bug：想「加一个 batch 维」用 stack，想「延长现有维」用 cat。

## 本关 concat 手写实现（理解内存布局）

```
out_shape[dim] = sum(t.shape[dim] for t in ts)
out = torch.empty(out_shape); off = 0
for t in ts:
    sl[dim] = slice(off, off + t.shape[dim]); out[sl] = t; off += t.shape[dim]
```

分配目标形状、按偏移把每块写进 `dim` 轴的对应切片——直观展示 cat 就是「沿 dim 依次排布」。负 `dim` 先 `+= ndim`。

## 坑

1. **非拼接维必须一致**：cat `dim=1` 时 dim=0/2… 要相同，否则报错。
2. 一维列表张量形状不齐无法 cat/stack——需先 reshape 对齐。
3. `cat` 不原地、会拷贝；反复在循环里 cat 累积是性能反模式（O(n²) 拷贝），应预分配或收集后一次 cat。
4. dtype/device 需统一，混用报错。

## 关联

多头注意力 `Concat(heads)@W_O`、通道拼接的 DenseNet/FPN、残差里的特征融合，都建立在 cat/stack 之上；把它和 `gather/scatter` 放一起，就掌握了「按维重组张量」的全套工具。
