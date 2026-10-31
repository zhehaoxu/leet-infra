# NSA 三分支调度账：GQA 共享分数、top-n 块选择与门控聚合

DeepSeek 的 NSA（Native Sparse Attention，arXiv 2502.11089）把每个 query 位置的 KV 重映射进三条**独立**分支——压缩（cmp）、选择（slc）、滑窗（win）——再由可学习门控权重加权聚合。本题考察这条"调度账"：GQA 组内共享分数如何计算、top-n 块选择如何排名、各分支 KV 长度怎么数、门控如何聚合。注意力的 softmax/matmul 本体不在此题范围（以中间量直接给出）。

## 简化口径（全部钉死）

- 压缩块长 l、压缩步长 d、选择块长 l' 三者相等（l = d = l'）。此时论文式 (9) 的混合粒度分数映射退化为 p_slc = p_cmp 直接复用；候选选择块与压缩块一一对应。
- 已生成 L 个 token（编号 0..L-1）；压缩块 i 覆盖 token [i·l, (i+1)·l)，只有整块落在因果内（(i+1)·l ≤ L）的块才存在，故可用块数 n_blocks = L // l。
- 选择只发生在完整块上（式 12 整块 concat），未成块的部分由滑窗分支兜底。

## 计算规则（逐条对应论文）

1. **组内共享分数（式 10）**：GQA 组内各 query head 共享 KV，块选择必须组内一致——p_shared[i] = Σ_h p_cmp_heads[h][i]，是**求和**（不是平均、不是最大）。
2. **top-n 选择（式 11）**：对 p_shared 按分数降序排名（rank = 1 为最高），rank ≤ top_n 的块入选。并列分数按块索引升序依次排名（稳定）。实际入选块数 n_sel = min(top_n, n_blocks)。
3. **分支 KV 长度（式 6 的 size 项）**：cmp = n_blocks·l；slc = n_sel·l；win = min(window, L)。N_t = 三者之和（论文要求 N_t ≪ t 维持稀疏比）。
4. **门控聚合（式 5）**：output[j] = g_cmp·o_cmp[j] + g_slc·o_slc[j] + g_win·o_win[j]，三分支 KV 相互独立（论文的防 shortcut 设计），门控为 0 的分支输出完全不影响聚合。

## 输入

```python
solve(p_cmp_heads, L, l, top_n, window, gates, branch_outputs)
```

- `p_cmp_heads`：list[list[float]]，每个 query head 的压缩块分数（式 8 已 softmax；所有 head 等长，长度 = n_blocks）
- `L`：已生成 token 数；`l`：块长；`top_n`：选择块数；`window`：滑窗大小
- `gates`：[g_cmp, g_slc, g_win]（已过 sigmoid）
- `branch_outputs`：[[o_cmp], [o_slc], [o_win]]，三分支输出向量（等长）

输入不合法（空 head、长度不一、top_n < 1、window < 1、gates 长度 ≠ 3、输出向量长度不一或为空）抛 ValueError。

## 输出

```python
{
  "shared_scores": [...],       # 长度 n_blocks
  "selected_blocks": [...],     # 入选块索引，升序
  "cmp_kv_len": ..., "slc_kv_len": ..., "win_kv_len": ...,
  "n_remapped": ...,            # 三分支 KV 长度之和
  "output": [...]               # 门控聚合结果
}
```

## 例子

```
输入: p_cmp_heads = [[0.5, 0.1], [0.2, 0.3], [0.1, 0.2]]（组内 3 个 head）,
      L = 4, l = 2, top_n = 1, window = 2,
      gates = [1.0, 0.0, 0.0], branch_outputs = [[1], [1], [1]]
输出: shared_scores = [0.8, 0.6]（组内求和），
      selected_blocks = [0]，slc_kv_len = 2，
      cmp_kv_len = 4, win_kv_len = 2, n_remapped = 8，
      output = [1.0]
```
