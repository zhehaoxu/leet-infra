# 详解：llama.cpp 全管线阶段顺序装配

## 出处

- 算法出处：llama.cpp 的采样链设计（`llama_sampler_chain` 的装配顺序：logit-bias/penalties → dry → top-k → top-p → min-p → xtc → temp → dist；具体文件行号**待回源**，常见出处是 `common/sampling` 与 `src/llama-sampling.cpp`）。链式口径的核心主张：**截断判据读到的概率分布是"未过温度"的**——温度只重塑最终分布的锐度，不改变谁被截断。HuggingFace 的 `LogitsProcessorList` 习惯把 temperature processor 放在链首，两种生态的 top-p 因此不可互换。判题环境离线，**本题的阶段顺序与各阶段口径完全以 starter 注释为准**。
- 各单阶段算法此前已在本合集各自成题：`penalty-sampling`（重复/频率/presence 惩罚）、`dry-sampling`（贴尾重复惩罚）、`top-k-sampling`、`top-p-sampling`、`min-p-sampling`，以及本题链上的 `logit-bias-sampling`、`xtc-sampling`。它们的**判定细节**（取等方向、min_keep 兜底、逆序窗口匹配）在本题中沿用各自题面已接受的口径，本题新增的考点是**顺序**本身。
- 一句话记忆：**单算子是"做什么"，管线题是"按什么顺序做"**。同样的算子集合，温度换个位置，top-p 的存活集合就不同——顺序是行为的一部分，也是引擎兼容性 bug 的最大来源。

## 完整推导与下标约定

**快照语义**：`masks` 的 8 行按阶段下标 0–7 固定，第 \(i\) 行 = 第 \(i\) 个阶段执行完后的存活集合。这有一个隐蔽的推论：`temp_first=True` 时 temp **提前**执行，但它的快照仍然记在第 7 行——于是第 7 行的存活集合**不保证**是第 6 行的子集链的末端：HF 口径下第 7 行是"temp 刚跑完、截断未跑"的状态（评测里 temp=0.5 不删候选，所以第 7 行反而是全量 9 个），而第 3–6 行是过温度的截断链。这正是评测 `hf_vs_llama_temperature_order` 断言的形状。

**温度与截断的相互作用**（本题的数学核心）：截断判据（top-p 的累积、min-p 的地板、xtc 的阈值比较）作用在"当前存活 logits 的 softmax"上。logits 同除 \(1/T\)（\(T>1\)）会把 softmax 变平：头部的概率被压低、尾部抬高。于是：

- llama 口径（temp 在后）：截断看到**原始锐度**——高温不会让 top-p 多留候选；
- HF 口径（temp 在前）：截断看到**变平的分布**——同样的 \(p=0.6\)，过温度后头部两个 token 的累计概率更晚达到阈值……或更早塌缩到单 token（评测示例里 \(T=0.5\) 是锐化，top-p 直接塌到 1 个）。

所以 `temp_first` 是**行为开关**而非实现细节：评测用同一组参数跑两种口径，断言 top-p 行的存活集合不同、且 llama 口径下 temp 行 \(\subseteq\) xtc 行（temp 最后执行，只能删不增）。

**窗口与匹配的公共口径**：`penalty_last_n` 与 `dry_penalty_last_n` 都用"负数全量、0 关闭、正数取尾"的约定；dry 在**逆序**窗口上做最长公共前缀匹配，等价于"重复段必须贴着窗口末尾"——与 `dry-sampling` 已接受的口径逐字一致（含"只登记严格更长重复"与 `dry_allowed_length=0` 时不罚零长匹配）。

**min_keep 的三处兜底**：top-p 在"首次 \(\ge p\) 且已保留 \(\ge mk\)"处截断；min-p 在"首个低于地板且已保留 \(\ge mk\)"处停止；xtc 要求"砍完存活 \(\ge mk\)"。top-k 是唯一不受保护的截断（llama.cpp 口径），这正是示例里第 3 行敢从 9 砍到 8 的原因。

**最终逆变换**：`probs` 在存活集合上归一化、原顺序 CDF；命中判据是**前缀和首次 \(\ge u\)**（取等命中）。与兄弟题一致；兜底取降序存活里最后一个（浮点前缀和差一点点时不会抛错）。

## 核心与易错点

1. **顺序错误用快照抓**：只对最终分布对拍抓不到"bias 与 penalties 谁先"这类错误（两者都改 logit，交换后分布可能几乎不变）。评测对 8 行快照逐一 `torch.equal`，任何一行顺序错了立刻露馅。
2. **截断概率的"温度域"**：`temp_first=False` 时 top-p/min-p/xtc 的概率**不含**温度。把 softmax 提前过温度是 HF 习惯的自然泄漏，评测的专项断言（两种口径 top-p 存活集合不同）专打它。
3. **第 7 行快照的时机**（见推导）：HF 口径下第 7 行不是"链条末尾"，而是"temp 刚执行完"。把它实现成"最后执行、最后快照"，第 7 行就会变成截断后的 1–2 个存活，与断言矛盾。
4. **penalties 的乘除方向**：\(x\le0\) 乘 repeat、\(x>0\) 除 repeat（把正 logit 往下压、负的往上提，对称地"去尖"），随后统一减 \(c\cdot\text{freq}+\text{present}\)。写成统一乘或统一除、或忘了 presence/freq 里的计数下标，都会在行 1 的快照上翻车。
5. **dry 的贴尾约束**：重复段必须贴着窗口末尾（逆序匹配是它的实现），窗口中间的重复不算；每 token 只罚最长的那次重复，且是"严格更长"才更新——这些细则与 `dry-sampling` 完全一致，别在本题里发明新口径。
6. **xtc 的"留尾"**：砍的是达标头部前缀里**除最后一个之外**的所有 token——xtc 的语义是"把扎堆的头部削薄、留下边界上的那一个"。砍反了（留头去尾）会在随机对拍里立刻暴露。
7. **banned 只掩码不改 logit**：被 ban 的 token 的 logit 仍参与 min-p 的"最大 logit"计算吗？不参与——它不在存活集合里，所有"存活集合上归一化"的统计都天然排除它。把 ban 实现成 logit 置 −inf 是另一种口径，会让 min-p 地板、top-p 累计的数值不同。
8. **`min_keep` 不保护 top-k**；且所有兜底都是"砍不动就别砍"，不是"砍完再补回来"。

## 与邻近题的区别

- **vs `top-p/top-k/min-p/dry/penalty/logit-bias/xtc-sampling`（单算子题）**：它们各自隔离一个阶段、输入里没有"链"；本题的输入是**全套旋钮**，考点是阶段间的**顺序与相互作用**（温度位置、截断域、快照时机）。单算子题的对拍不涉及 masks 的多行快照。
- **vs `mirostat-v2-sampling` / `adaptive-p-sampling` / `dynatemp-sampling`**：那些是**自适应**采样（阈值/温度随生成反馈变化），本题所有旋钮都是常数——没有任何反馈回路，只有一次装配与一遍执行。
- **vs `inf-spec-decoding-rejection`（推测解码拒绝采样）**：同样消费"分布 + 随机数"，但推测解码验证的是**草稿 token 的接受/拒绝**（目标/草稿分布的比值），不涉及截断链；本题是**一次前向后的单步解码**管线。
- **vs `contrastive-sampling` / `moa-sampling`**：它们重排/合成多个候选分布（对比搜索、Mixture-of-Agents），不维护"存活集合"的截断语义；本题的中间态是 bool 掩码而非分布的凸组合。

## 对拍口径与复杂度

- 主口径：独立重算 `_oracle`——纯 Python 列表逐步模拟（不用 Z-algorithm，dry 用朴素逆序 LCP；不用向量化 softmax，每步在存活集合上重算），与参考解（torch 向量化）在 10 组随机 logits × {temp_first} × {xtc 开关} × {temp} × {top_p} × {min_p} 的全网格上对拍：masks `torch.equal`、probs `allclose(1e-5)`、chosen `torch.equal`。
- 专项断言：HF vs llama 的 top-p 存活集合差异（同参数对照）；"全旋钮关闭 = 原始 softmax"（管线退化的锚点）；llama 口径下 temp 行 ⊆ xtc 行。
- 复杂度：每阶段 \(O(V\log V)\)（排序）或 \(O(V)\)（扫描），整链 \(O(8\cdot V\log V)\)；快照 8×V 布尔——与真实引擎的 per-step 开销同阶，纯 CPU 可跑。
