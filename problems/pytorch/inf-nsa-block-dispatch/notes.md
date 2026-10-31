# notes — inf-nsa-block-dispatch（order 683）

## 回源

- 论文 arXiv 2502.11089（PDF 下载后 workbuddy-pdf 提取，逐字核对 §3.3）。**官方实现不存在**：deepseek-ai/NSA 仓 404（API 确认 Not Found），社区实现（tilde-research/nsa-impl、qflen/nsa-from-scratch、dhcode-cpp/NSA-pytorch、FLA 的并行 kernel）口径不一，故以论文公式 (5)-(12) 为唯一权威。
- 公式账（论文原样）：
  - (7) 压缩：K̃^cmp_t = {φ(k_{i·d+1:i·d+l}) : 0 ≤ i ≤ ⌊(t−l)/d⌋}，共 ⌊(t−l)/d⌋+1 个压缩 key。d < l 时块间重叠。φ 是带块内位置编码的 MLP。
  - (8) p^cmp_t = softmax(q_t^T K̃^cmp_t) ∈ R^{⌊(t−l)/d⌋+1}。
  - (9) 混合粒度：p^slc_t[j] = Σ_{m=0}^{l'/d−1} Σ_{n=0}^{l/d−1} p^cmp_t[(l'/d)·j − m − n]。**l=l'=d 时退化为恒等**（论文明确写"directly obtain … straightforwardly"）。PDF 提取中该公式的索引基（0/1）不可靠，出题避开混合粒度，只做 l=d=l' 主线。
  - (10) GQA 组内共享：p^slc,t' = Σ_{h=1}^{H} p^slc,(h)_t —— **求和**。动机：解码时 KV 加载量是组内所有 head 选择的并集，组内一致才能最小化加载。
  - (11) I_t = {i | rank(p^slc_t'[i]) ≤ n}，rank 降序、rank=1 最高。**并列口径论文未规定** → 出题钉死「并列按块索引升序依次排名（稳定）」。
  - (12) K̃^slc_t = Cat{k_{i·l'+1:(i+1)·l'} : i ∈ I_t} —— 整块 concat。
  - (6) N_t = Σ_c size[K̃^c_t]，论文要求 N_t ≪ t。
  - 滑窗：K̃^win_t = k_{t−w:t}（含当前 token）。
  - 门控：g^c_t 由 MLP+sigmoid 从输入特征导出；式 (5) 输出 = Σ_c g^c_t · o^c_t。三分支独立 KV 是防 shortcut learning 的设计（论文 3.3.3）。

## 出题抽象

- softmax/matmul 不考（手算不可行且别的题已覆盖注意力本体）：q/k/v 不出现，p_cmp 以"已 softmax 的压缩块分数"直接给出；三分支输出 o 直接给出。
- 因果口径：已生成 L 个 token（0..L-1），完整压缩块数 = L//l（(L−l)//d+1 在 d=l 时 = L//l ✓ 手算核对过 L=5,l=2 → 2 块 [0,2),[2,4)）。
- 选择候选 = 完整块（(i+1)·l ≤ L）。混合粒度块数不一致的场景（l ≠ l'）不出题（式 9 索引基存疑，见上）。
- 报错用 ValueError（与仓库惯例一致）。

## 账点（测试钉死）

1. **组内求和 vs 平均/最大**（check_gqa_group_sum；变异 gqa-mean / shared-first-only 抓）——用「单 head 格局与组和格局选出不同块」的构造，把"逐 head 独立选"的 Quest 式错误钉死。
2. **降序 rank + 并列稳定取小索引**（check_tie_stable_low_index；变异 rank-asc / tie-large 抓）。
3. **top_n > 块数时全选但 slc_len 仍按实际块数**（check_top_n_larger_than_blocks；变异 slc-len-topn 抓 slc_len = top_n·l 的账）。
4. **selected_blocks 输出升序**（rank 序 ≠ 升序的构造 check_selected_sorted_ascending；变异 sel-keep-rank-order 抓）。
5. **滑窗 min(window, L)**（check_window_cap / check_window_normal；变异 win-uncap / win-off 抓）。
6. **门控线性聚合 + 零门控分支沉默**（check_gated_aggregation / check_zero_gate_branch_silent；变异 gate-drop-win / gate-add 抓）。
7. **n_blocks = L//l 因果账**（check_l_equals_one_degenerate 的 l=1 退化；cmp-len-plus 变异抓 cmp_len）。
8. 输入校验（check_invalid_inputs，7 种非法形态）。

## 变异记录

11 个可观察变异全 CAUGHT：`gqa-mean`、`rank-asc`（选最低分）、`tie-large`（并列大索引）、`sel-keep-rank-order`（输出 rank 序）、`slc-len-topn`（slc_len 用 top_n 而非实际块数）、`win-uncap`、`win-off`（min(w,L+1)）、`gate-drop-win`、`gate-add`（g·a → a）、`cmp-len-plus`、`rank-strict`（少选一个）、`shared-first-only`。

**等价变异分析（sel-no-clip）**：`order[: min(top_n, n_blocks)]` → `order[: top_n]` 等价——Python 切片天然截断，选择集合不受影响；真实账点在 slc_len 的 min（已被 slc-len-topn 覆盖）。

## 手算锚点踩坑

- oracle 排序键写反（(shared[i], -i) 升序 = 选最低分）→ 对拍时 FAIL 方向恰好暴露："期望 [0,2] 得到 [1,3]"——solution 是对的，修 oracle。
- check_window_cap 初版残留 L=3,l=4 的非法调用（n_blocks=0 与 head 长度 1 冲突）→ 删。
- _compare 签名统一改关键字名（window/branch_outputs）后逐处更新。

## 与相邻题划界

- `sliding-window-attention`（operators）：纯因果滑窗注意力本体。
- `kvcache` 各题：KV 淘汰/回收；NSA 是**训练感知**的块选择（梯度可流过 gating，选择本身离散但不用 k-means/SimHash）。
- 本题：三分支**调度账**（GQA 共享、top-n、分支长度、门控），不含注意力 matmul/softmax 本体。
