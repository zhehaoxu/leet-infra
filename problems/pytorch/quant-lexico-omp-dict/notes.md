# 解题说明 · Lexico 的逐行 OMP 与 CSR 打包

## 一句话

把每条 KV 向量当信号，用贪心 OMP 在固定字典里挑 `n_atoms` 个原子：每轮先按相对能量判停，再取 $|\langle r,a_j\rangle|$ 最大的未选原子，然后对整个已选集合重解最小二乘；最后把每行的「原子 id + 权重」摊成 CSR 三元组。

## 上游到底是怎么跑的

`lexico/omp.py::omp_v0`（L10-92）是批量实现的经典 OMP，但没有显式的原子掩码，也没有真的去求逆：

- L28 一次性算 `projections = bmm(Xᵀ, yᵀ)ᵀ`，之后每轮在 L84 用 `projections -= temp_a_F * D_mybest[k]` 做 rank-1 downdate，L74 的 `rsqrt(1 - innerp(...))` 与新原子对老原子的 Gram 分量有关——这套 `F`/`a_F` 变量（L31-32、L87-88、L90 的 `F.permute(...) @ a_F`）就是 $(X_S^\top X_S)^{-1}X_S^\top y$ 的 Cholesky 式增量版本，数学上等价于「整组重解」，所以题面按直白的重解口径写。
- 选择步 L64 是 `sets[k] = projections.abs().argmax(2)`：绝对值、 argmax 取第一个最大值（即最小下标），并且**没有排除已选原子**——靠相关性被 downdate 成 0 来避免重复。
- 停止步 L43 的循环是 `range(n_nonzero_coefs + (tol is not None))`，L46 在**选原子之前**判 `normr2 <= tol`，L41 把阈值预成 `normr2_init * tol * tol`，L54 记 `result_lengths = k`。所以「先判停、lengths 是已选个数、误差报在停止点」这三条是上游原样。
- `omp()`（L94-112）负责打包：L101-104 在没有 tol 时把 `lengths` 兜底成满预算，L106 权重转 `float8_e4m3fn`，L107 原子 id 转 int16，L109-110 `indptr` 是 int32 的前缀和。
- `cache_utils.py::_compress`（L159-191）把 `tensor.permute(2, 0, 1, 3).reshape(1, -1, head_dim)`（L163）当成 $(1, N, D)$ 送进 `omp`，即「head × token」摊平成行；`CHUNK_SIZE = 20000`（L160）只是分块累加 `indptr`；K 与 V 各有一套按层训练的字典（L89-90、L136-137）。`max_sparsity` 默认 14（L14），`error_threshold` 默认 `None`（L15）——仓库的 `examples/generate.py` L20 用 8，`examples/lm_eval_harness.py` L67 用 14。

出处：`README.md` L7 的 badge 与 L76 的 bib 都写 arXiv 2412.08890，字典在 WikiText-103 上离线训练、约 4k 原子、跨任务跨模型通用（L15、L21-22）。

## 本题与上游的五处分歧（都已在题面钉死）

1. **`tol=0.0`**：上游 L37 写 `if tol:`，$0.0$ 是假值等于「关闭早停」；本题按 `tol is not None` 处理，$0.0$ 是合法的零阈值，零行会在第 0 轮停在 `lengths=0`。
2. **已选原子显式排除**：上游不屏蔽，极端并列（残差恰为 0）时会把同一个列再选一遍；本题钉成「只在未选原子里竞选」。
3. **dtype**：上游 `int16 / float8_e4m3fn / int32`，本题统一 int64 加输入 dtype，判题不做低比特截断。
4. **最小二乘的解法**：上游用 Gram 增量因子，本题用 `lstsq`；集合满秩时两者一致，秩亏时本题取**最小范数解**（`torch.linalg.lstsq` 与 `numpy.linalg.lstsq` 同一支），`recon` 仍是唯一投影。
5. **入参布局**：上游是 $(1,N,D)$ 批量加 chunk 拼接，本题把行直接给成 $(N,D)$，字典给成 $(D,M)$，`CHUNK_SIZE` 与 float8 落盘一律不进题。

## 判分点

六个键的 dtype 与形状都要对：`lengths/csr_indptr/csr_indices` int64，`csr_values/residual_norm/recon` 跟随输入 dtype；`csr_values` 与 `csr_indices` 的长度都等于 `lengths.sum()`。选原子顺序是「被选中的先后」而不是下标排序——CSR 段自己乘回原子列拼出 `recon` 就是这条顺序的检验。零行、`n_atoms == M`、整行放大 10 倍（顺序不变、权重同步放大）都有专门断言。

## 边界

`tol=None` 时每行必定选满 `n_atoms`；`tol` 很松（如 0.99）时可能在第 0 轮都不选（只有零行，或 $I=0$）。$D$ 方向没有整除要求，$M$ 只要 $\ge$ `n_atoms`。字典列不要求归一化，也不要求线性无关——`check_selection_is_abs_correlation_with_ties` 就给了三列重复的字典。

## 复杂度

每行每轮一次 $O(DM)$ 的相关、一次 $O(D\,k^2)$ 的最小二乘，总计 $O(N\cdot \text{n\_atoms}\cdot(DM + D\,\text{n\_atoms}^2))$；字典是入参，不涉及 K-SVD 的训练开销。

## 相关题目

`spqr-sparse-outlier` 是「离群值单独存」的一维稀疏，本题是**多原子线性组合**的稀疏字典编码；`quant-squeezellm-sensitive-kmeans` 用一个码字表示一段，本题用 $k$ 个原子加权求和。`retrieval-pq-codes` 的乘积量化分段与本题同为码本类，但那里没有逐行贪心搜索。
