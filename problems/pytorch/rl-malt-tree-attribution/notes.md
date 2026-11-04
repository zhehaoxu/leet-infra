# MALT 实现笔记（791）

回源：arXiv:2412.01928 v3（COLM 2025）§4.2 数据生成与信用分配、A.6.1 形式化。**官方无开源实现**（survey 表格 Code: -；arXiv 页无代码链接；第三方 Nimitkothari/MALT 是 Groq/LangChain 复现，不钉口径）。本题口径 = 论文公式唯一确定的算术账，沙箱 oracle 用独立 numpy 直译（np.mean(axis) 逐层均值）+ 手工锚点双通道对拍。

## 逐入口口径

### propagate_values

- 树编码 `rewards[j][k][l]` = r_{j,k,l} 的二元奖励。`V(v_{j,k}) = sum_l / n`、`V(g_j) = sum_k V(v_{j,k}) / n`。
- 论文原文锚点：'V(v_{i,j,k}) = E_l[V(r_{i,j,k,l})] ≈ (1/n)Σ_{l=1}^n V(r_{i,j,k,l})'、'V(g_{i,j}) = E_k[V(v_{i,j,k})]'；附录 A.6.1 等价形式 V(g)=1/n²ΣΣV(r)。
- **无折扣因子**：树深 3，公式无 γ。todos 旧候选描述的「深度折扣」经回源修正删除。
- 树形校验：len(rewards)==n 且各层 len==n，否则 ValueError（分支因子显式入参；论文实验 n=3 → 27 轨迹/题）。

### binarize

- `value > theta` 严格。论文原文：'Nodes with values greater than 0.5 are labeled as correct (✓), and those with values less than or equal to 0.5 are labeled as incorrect (×)'。
- n=3 时 V(v)/V(g) ∈ {0, 1/3, 2/3, 1}——0.5 不可达；n=2 时 {0, 0.5, 1}——0.5 边界由 TREE_B 钉死。

### build_preference_pairs

- 同父内正子集 × 负子集全组合，(正下标, 负下标)，父下标升序遍历。
- 论文原文锚点：D_R 'where r⁺ is a correct refinement (V(r⁺)=1) and r⁻ is an incorrect refinement (V(r⁻)=0) for the same verifier input v_{i,j,k}'；D_V 'by comparing outputs under the same generator output g_{i,j}'；D_G 'by comparing outputs under the same query q_i'。
- **不是跨分支笛卡尔积**：跨 (j,k) 父的正负不配。
- 二值化与标签同口径（严格 > θ）——0.5 值的 unit 落入负子集。

### sft_positives

- ✓ 下标保持输入顺序；逐父 (正数, 负数) 计数升序输出。
- 分层口径差异（论文）：R 层正样本 = V(r)=1（叶奖励本身）；V/G 层 = 二值 label ✓（V(g)>0.5 的 g 才进 SFT）。

### attribute_question

- 线性下标：r 叶 (j,k,l) → j·n²+k·n+l；v (j,k) → j·n+k；g → j。
- 组装顺序：propagate → 展平（v 值 + 父 id、r 值 + 父 id）→ binarize（v/g/r 三层）→ 三层 pairs → 三层 sft。
- G 层 parent_ids 恒 [0]*n（单题 = 单父）。

## 沙箱与测试

- /tmp/malt_check.py：numpy 直译对拍（A 树 n=3、B 树 n=2）+ 手工锚点（v00=1.0、v01=2/3、g0=7/9、g1=2/9）+ 偏好对计数（A 树 R 层 8 对 = (0,1) 2 + (0,2) 2 + (1,0) 2 + (1,2) 2——注意 (1,0) 是 1 正 2 负 = 2 对，初版手算漏计成 1）+ 0.5 边界 + ValueError。
- 手工锚点教训：配对计数按「每父 (正数×负数)」逐个算，别按「1 的个数」估——(j1,k0)=[1,0,0] 是 2 对不是 1 对。
- 测试：6 个 check 函数；TREE_A/TREE_B 内嵌，EXPECTED 为手工锚点（此题公式简单，无需 dump 大 JSON）。

## 变异清单（mut791 待做）

易漏点：均值分母 n vs len（同值不可变异——用树形校验钉）、两层均值轴序（axis=2 vs axis=1 对应 sum_l vs sum_k）、严格 > vs >=（B 树钉）、正负子集划分、父分组键、pair 顺序（正前负后/父升序）、线性编码系数 j*n*n、SFT 正样本口径（V 层 label vs 值）、ValueError 校验。

## 划界

- 782 `to-gmpo-geom-mean` / 787 `rl-ktae-token-advantage`：那些是 advantage/loss 侧 GRPO 族账；本题是数据构造侧——期望回传产出角色特定 SFT/DPO 数据集，产物直接喂 DPO 损失（损失侧已占不重做）。
- 784 `rl-vineppo-mc-value`：VinePPO 的 MC 值是每步 k 条续采均值作 value 基线；MALT 的期望回传是树形兄弟均值作数据标签，叶奖励来自终局判卷。
