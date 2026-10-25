# 解题说明 · FINCH

## 因果可见性带来的行尺度偏差

softmax 之后，位置 \(p\) 的 query 行只在 \([0,p]\) 上归一（KV 缓存语境里近似地以「可见 token 数」为分母尺度）。观察窗口里最老的 query 在全局位置 \(C\)，最新的在 \(C+W-1\)：同样是给某个上下文 key「投票」，老 query 的单次投票天然更大（分母小）。SnapKV 直接对窗口行取 `mean` 会把这种尺度差原样带进分数。`FinchPress.score` 的修正（kvpress `finch_press.py`，`normalize_scores=True`）：

```python
non_zero_counts = torch.arange(k_len - window_size, k_len)[None, None, :, None]
attn_weights = attn_weights * non_zero_counts   # 行 i（最老起算）乘 (C+i)
scores = attn_weights.mean(dim=-2)              # 再对窗口行平均
scores = scores.view(bsz, H_kv, groups, C).mean(dim=2)  # GQA 组内平均
scores = F.pad(scores, (0, window_size), value=scores.max().item() + 1)
```

乘 \(C+i\) 相当于把「归一化均值」还原成「未归一化的累计量级」，再平均；等价于给每条 query 按其视野宽度加权。注意乘数是 `arange(k_len - window, k_len)` = \(C+i\)（**不含 query 自身**），照抄上游、不要自行"修正"成 \(C+i+1\)。

## 与近邻题的对比

- `kv-observed-attention`（149）：**列**归一——score = 列和 ÷ (L−j)，把每个 key 收到的注意力按其"被看到的次数"平均。FINCH 是**行**加权——在跨 query 平均前先乘每行的可见宽度。两者是归一化方向上的镜像。
- `kv-snapkv-pooling`（147）：对窗口行取 **max** pooling；FINCH 取加权后的 mean。
- `kv-expected-attention`（150）：高斯期望估计未来 query 分布。FINCH 的 `F.pad(..., value=max+1)` 与 150 的 sink 保底同一手法：max 是全局的（跨所有 kv head），保证窗口位在任何 top-k 下都不掉。
- `kv-chunk-independent-topk`（165）：FINCH 论文/代码都有 chunk 模式，但分块 top-k 已由 165 覆盖，本题只考全局 top-k。

## 范围说明（回源结论）

FINCH **论文**（arXiv:2407.09112）还包含第二个机制：按 prompt 长度比例动态调整预算、以及 GQA 组内 key 的相似度合并。kvpress 的 `FinchPress` 实现的是「delimiter 动态窗口 + 上述归一化打分 + 可选重旋转」，其中 delimiter 检测需要 tokenizer、重旋转已有独立题（167），均未入本题。本題以 kvpress 源码为准，只考确定性打分/选择数学；不要按"聚类合并"理解本题。

## 陷阱

- 行序语义：行 0 是最老窗口 query（乘数最小 \(C\)），行 \(W-1\) 最新（乘数 \(C+W-1\)）。翻转输入行序是经典错误。
- pad 值来自**取 pad 前**的 `(H,C)` 全局 max +1，是标量、所有 kv head 共用。
- \(n\_kept\) 按 \(L=C+W\) 计（窗口位也占预算），不是只按 \(C\)。
- \(W=1\)：乘数是常数 \(C\)，行加权不改变序、只整体缩放——可作为 sanity 探针。
- 全零注意力：上下文分全 0、pad=1，top-k 落在窗口位的平局（小位置优先 → 从 \(C\) 开始连续取）。
