# 详解：GaLore 低秩投影梯度 AdamW

## 出处
- Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian, *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection*, arXiv:2403.03507（ICML 2024 Oral；标题、作者与会议信息出题时已在 arXiv 检索页复核）。
- 官方实现 `jiaweizzhao/GaLore`：`galore_torch/galore_projector.py`（`GaLoreProjector.project/project_back/get_orthogonal_matrix`）与 `galore_torch/adamw.py`（`GaLoreAdamW`）。以下上游细节均出自该源码，可复核：基重算条件为 `self.ortho_matrix is None or iter % self.update_proj_gap == 0`；`GaLoreAdamW.step` 传入的 `iter` 是**自增前**的 `state["step"]`（即该参数已完成步数）；`project_back` 末尾再乘一个 `scale`；上游内嵌的 GaLoreAdamW 把 `eps` 加在**未偏差修正**的 \(\sqrt v\) 上并用 \(lr\sqrt{bc_2}/bc_1\) 作步长——这些与本题口径的差异见下。
- 与论文/上游的范围差异（均为本题设定，题面与 starter.py 是唯一 ground truth）：① 投影基 \(P\) 是显式输入（原论文每 gap 步用随机化 SVD 由梯度算出，本题去除全部随机性）；② 本题做**单侧投影**——梯度先 row-major 展平成 \(\mathbb{R}^d\) 再算 \(P^\top g\)，上游是按矩阵长短边选的左/右单侧投影或 `proj_type='full'` 双侧投影；③ 不乘 `scale` 因子；④ 偏差修正与 `eps` 摆放沿用 `torch.optim.AdamW` 口径（\(P=I\) 探针据此与原生 AdamW 对拍）。论文正文里 rank/scale 的具体默认超参（如常见的 rank=128、scale=25 说法）未在本次核验范围内展开——待回源。

## 完整推导与下标约定
设某参数 \(p\in\mathbb{R}^{m\times n}\)，\(d=mn\)，输入基 \(P\in\mathbb{R}^{d\times r}\)（约定列正交归一，\(P^\top P=I_r\)，但实现不得依赖）。下标 \(t\) 指**该参数实际被更新的第 \(t\) 次**（\(t=1,2,\dots\)，`grad is None` 不推进，逐参数独立，与 Adam 家族的 step 计数同口径；全局步号与它不等）。记 \(x_t=\operatorname{vec}(g_t)\in\mathbb{R}^d\)（row-major）。投影：\(\tilde g_t=P^\top x_t\in\mathbb{R}^r\)。投影空间矩：
\[ m_t=\beta_1 m_{t-1}+(1-\beta_1)\tilde g_t,\qquad v_t=\beta_2 v_{t-1}+(1-\beta_2)\,\tilde g_t\odot\tilde g_t, \]
偏差修正 \(bc_1=1-\beta_1^{t}\)、\(bc_2=1-\beta_2^{t}\)，自适应方向 \(\hat u_t=(m_t/bc_1)\big/\big(\sqrt{v_t}/\sqrt{bc_2}+\epsilon\big)\in\mathbb{R}^r\)。映回并施加：\(\Delta_t=P\hat u_t\in\mathbb{R}^d\) reshape 成 \((m,n)\)，先解耦衰减再更新 \(p_t=p_{t-1}(1-lr\cdot\text{wd})-lr\,\Delta_t\)。账本日程：记 \(U_t\) 为"第 \(t\) 步重算基"，规则 \(U_t\iff (t-1)\bmod \text{gap}=0\)。它与上游逐字等价：上游判据是 \(s\bmod\text{gap}=0\) 或基缺失，而 \(s=t-1\)，且 \(t=1\) 时基必缺失、\(0\bmod\text{gap}=0\) 恒真，故"首步建基 + 之后每 gap 步一次"就是同一条同余式。低秩假设的本质：经验上训练梯度谱衰减快（近低秩），取 \(P\) 为前 \(r\) 个主方向时 \(\Delta = P\hat u\) 承载了更新的绝大部分能量，于是 \(m,v\) 只需 \(r\) 份显存，而原论文里 SVD 只在每 gap 步做一次、摊薄了计算。

## 核心与易错点
1. **矩与偏差修正都住在 \(\mathbb{R}^r\)**：最常见的错法是"先全空间做 \(m,v\) 再投影"，两者数学不同、状态大小也退回 \(O(d)\)。本题的显式循环 oracle 与向量化的参考解恰好在这点结构不同，专杀此类实现。
2. **左右乘方向**：\(P^\top g\) 是 \([r,d]\cdot[d]=[r]\)，映回是 \(P\hat u\) 的 \([d,r]\cdot[r]=[d]\)；写成 \(P\tilde g\) 维数就对不上（除非 \(r=d\)，正好被 \(P=I\) 探针掩盖，所以另配了非方阵 \(P\) 的主用例）。
3. **flatten/reshape 同用 row-major**，映回时忘了 reshape 会形状广播出错或原地更新失败。
4. **不许正交化输入 \(P\)**：正交归一是入参约定而非实现步骤；再做一次 QR/SVD 不仅慢，还会破坏 \(P=\)I 与符号翻转等不变性探针的逐位一致性。
5. **账本不进数值路径**：\(P\) 全程固定，"重算"只是 would-be 日程；把 gap 的奇偶性写进更新公式是错的。账本断言用 `torch.equal` 整数列，`recompute_at(t)` 另作纯布尔日程单独核对。
6. **跳过口径**：`grad is None` 时不推进该参数的 \(t\)、不施加 wd、不记账；wd 先于矩估计且作用在原空间。
7. **r=0 免费退化**：\([d,0]\cdot[0]\) 型 matvec 返回零向量，更新恰为纯 wd——不必特判，但循环版实现要能处理空内积。

## 与邻近题的区别
- **opt-adafactor（184）**：同样主打省显存，但 Adafactor 压缩的是**二阶矩的表示**——把 \(v\) 因子分解为行和×列和（\(O(m+n)\) 状态），梯度本身原样进更新；GaLore 不动矩的因式分解，而是压缩**梯度所在的子空间**，矩在 \(r\) 维小空间里照常逐元素维护。一个压状态形状，一个压信号维度。
- **opt-muon（185）**：都在"更新矩阵的谱形状"上做文章，但 Muon 用 Newton-Schulz 把动量矩阵**正交化**（无需外部基，作用于全部奇异方向），GaLore 则把更新**限制在给定子空间**里，正交性发生在输入侧（\(P\) 已给）而非迭代侧。
- **opt-prodigy（195）**：D-Adaptation 解决的是"lr 超参免调"，与状态压缩正交；Prodigy 不改梯度的表示，也不引入任何投影。
- **opt-cautious（194）/ opt-copt（265）**：那一对逐元素/逐张量地调制更新**方向**（符号一致性门控），GaLore 调制的是**维度**（子空间限制），机制层完全正交，可与任何 Adam 变体叠加使用（上游正是 GaLore+AdamW）。
