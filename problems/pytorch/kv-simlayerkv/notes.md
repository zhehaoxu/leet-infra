# 解题说明 · SimLayerKV

## 直觉：层间差异不是"多少"，而是"要不要"

KV 缓存跨层压缩有两条路线。PyramidKV 式（题面 order 152）：每层给一个不同的**数量预算**（底部宽顶部窄的金字塔）。SimLayerKV（arXiv:2410.13846，`SimLayerKVPress`）观察到多数层的注意力高度集中——最近的 token 加上开头几个 sink 就吃掉了几乎全部概率质量，这类"懒惰层"里中段 token 是纯负担。于是压缩被二值化：**懒 = 只留头尾，不懒 = 全留**。没有排序、没有比例分配，每层只算一个标量和一个比较。

## 判定的准确形态（kvpress `simlayerkv_press.py`）

```python
attn_weights = compute_window_attention(..., n_last, ...)   # 最后 n_last 条 query 的注意力
attn_weights = attn_weights.mean((0, 1, 2))                 # batch × heads × 窗口 全部平均 → (L,)
score = attn_weights[:n_initial].sum() + attn_weights[-n_recent:].sum()
if score > lazy_threshold:   # 严格大于
    keys = cat(keys[:, :, :n_initial], keys[:, :, -n_recent + n_last:])
```

三个容易抄错的细节：
1. **保留区右段长 `n_recent - n_last`，不是 `n_recent`**。切片起点是 `-(n_recent - n_last)`：注意力求和覆盖最后 \(n\_recent\) 个 key（含窗口自身），但缓存保留只到窗口前 \(n\_last\) 位为止——窗口内最新 token 反正每步都在刷新。
2. **求和区间与保留区间不对称**：判定用 `[-n_recent:]`，保留用 `[-(n_recent-n_last):]`。
3. **`lazy_threshold == 1.0` 是硬开关**：源码第一步就短路返回（默认参数即"未启用"）。全 1 的 delta 注意力 + θ=1.0 也必须全保留——这是本题最阴的退化用例。
4. 序列过短（\(L\le n\_initial+n\_recent+n\_last\)）时不压缩，避免头尾区间重叠/覆盖整条序列。

## 与近邻题对比

- `kv-pyramid-layer-budget`（152）：层预算来自**闭式公式**，与注意力内容无关；本题每层的 0/1 决策由数据（注意力质量）驱动。
- `kv-duo-attention-heads`（168）：沿**头维**二分（检索头/流式头）；SimLayerKV 沿**层维**二分（懒惰/非懒惰），且每个懒惰层的保留形状恰好是 StreamingLLM 头尾模式（145）——相当于"逐层开关的 StreamingLLM"。
- 与任何 top-k 系（149/150/165/290）不同：本题输出不是下标集而是二值掩码 + 压缩比；懒惰层压缩比是**整数决定的常数**，与该层分数大小无关（分数只决定开关，不决定强度）。

## 陷阱

- 浮点求和顺序：判定式是若干非负数的和，纯 Python 与张量 `sum` 可能有 1ulp 差异；测试里的边界探针特意用二进制可精确表示的数（0.25/0.125），构造 \(s_l=\theta\) 严格不懒。
- `mask[lazy]` 花式索引返回拷贝，改完要写回。
- 全部层都不懒时 `ratios` 必须是 \(N\) 个 0（形状别塌）。
