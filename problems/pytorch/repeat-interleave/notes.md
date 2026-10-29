# 详解：repeat_interleave（逐元素重复与 index_select 等价）

## 出处

`torch.repeat_interleave` 与 numpy 的 `np.repeat` 同语义，而整段平铺的那一条属于 `np.tile` / `torch.repeat`。它在推理代码里出现得极频繁：GQA 把 $n_{kv}$ 个 KV 头扩成 $n_h$ 个（`kv.repeat_interleave(n_h // n_kv, dim=1)`）、把 per-layer 的标量阈值扩成逐头掩码、把 batch 索引扩成与 token 对齐的长向量、以及 `data-tokenization` 侧把一段文本按其 token 数重复展开。面试常考「`repeat` 和 `repeat_interleave` 的区别」，口头答对不难，手写出来就会暴露下标构造的功底。

## 下标向量是唯一的核心

两种语义的差别全在下标顺序上。设沿该轴有 $L$ 个元素、次数 $r_i$、总长 $T=\sum_i r_i$：

$$\text{interleave}:\; \underbrace{0,\dots,0}_{r_0}, \underbrace{1,\dots,1}_{r_1}, \dots \qquad\text{tile}:\; 0,1,\dots,L-1,\;0,1,\dots,L-1,\;\dots$$

拿到 interleave 下标后，一切归到 `index_select`：

```
idx = [i for i, k in enumerate(r) for _ in range(k)]   # 长度 T，允许 0 次
out = y.index_select(axis, torch.tensor(idx, dtype=torch.long))
```

`dim=None` 只是先 `y = x.reshape(-1)`、`axis = 0`，其余完全同一路径。`index_select` 天然是**拷贝**语义（新内存），正好满足「改输出不得影响输入」的断言；若改用 `expand` + `reshape` 来偷懒，`reshape` 在需要 contiguous 时会隐式 copy、看起来也对，但 `expand` 出的 0 长度维与变长次数根本无法表达，`repeats` 为张量时就直接写不出来。

不用 Python 循环的等价写法是「用前缀和定位」：令 $c_i = \sum_{j\le i} r_j$，则输出第 $t$ 个位置的下标是 $\min\{i : c_i > t\}$，即

$$\text{idx} = \texttt{searchsorted}(c,\ \texttt{arange}(T),\ \texttt{right}=\texttt{True})$$

这条式子也是变长展开类问题里避开 Python 循环的标准写法（省掉逐元素 append 的开销）；而 `repeats` 全 0 时 $T=0$、`arange(0)` 空、结果恰是保留轴数的空张量，无需特判。

## 核心与易错点

1. **`repeats` 张量的长度校验**：`dim` 给定时长度必须等于 `x.shape[dim]`；`dim=None` 时等于 `x.numel()`。少这一条就会把「长度 2 的 repeats 用在 4 行张量上」的非法用例放过去。
2. **必须是整数张量**：`torch.rand(4)` 这类浮点 repeats 要抛 ValueError，而不是被 `.long()` 悄悄截断。
3. **一维限制**：二维 repeats（`[[1,2],[3,4]]`）在 torch 里会报错，本题同样要求 ValueError。
4. **0 次不是错误**：输出长度可以为 0，此时形状是 `(0, ...)`——轴数与 dtype 都得保留，返回 `torch.tensor([])`（1-D）在 `dim=0` 的二维输入上是错的。
5. **int 与张量的统一**：int 次数等价于全填该值的长度 $L$ 向量，参考答案把两支合成 `_counts()` 一个入口，避免后续分支重复。

## 与邻近题的区别

- `gather-scatter` / `tensor-shape-indexing`：那是给任意下标做取/放，本题的下标由 `repeats` 唯一决定。
- `concat-stack-split`：`cat` 若干份切片也能实现同样结果，但没有统一的「下标向量」视角，`repeats` 为张量时会写成 150 行。
- `sequence-pack-pad`：把变长序列打包成定长块（涉及 padding 与截断），本题只重复不裁剪。

## 对拍口径与复杂度

- oracle 是 `torch.repeat_interleave`，用 `torch.equal` 做**精确**断言（纯索引、无算术，不需要浮点容差）；覆盖 None/正/负轴 × int/随机/全 0/int32 次数，另有 interleave≠tile 的反向断言、逐位置次数核对、拷贝语义与 7 组非法入参。
- 复杂度 $O(T\cdot \text{其余维})$，$T=\sum r_i$ 为输出沿该轴的长度；构造下标向量的 Python 循环是本题唯一非线性部分，判题规模下可忽略。
