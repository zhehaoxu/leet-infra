# 详解：SpecInfer 通用 token-tree 并行验证

## 出处
Miao et al., *SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification*（arXiv:2305.09781，实现随 FlexFlow 发布：[flexflow/FlexFlow](https://github.com/flexflow/FlexFlow) 的 SpecInfer/inference 分支，即 flexflow-ai/SpecInfer）。论文的两根支柱是：① **token tree**——draft 侧可以是小模型、多个 head、检索库或 n-gram，产出不是一条序列而是一棵树；② **tree attention**——把树上所有节点拼成一条序列送进 target，但注意力掩码改成「每个节点只看它的祖先链」，于是一次前向就同时算出树上每个位置的条件概率。验证沿已接受前缀**逐层**推进、被拒分支整条作废、最后取最长存活路径，就是论文 §Verification 描述的流程；被拒处从残差分布 $\mathrm{norm}(\max(P-Q,0))$ 重采一个 token，则是 Leviathan/Chen（arXiv:2211.17192 / 2302.01318）保留 target 分布的标准做法。

## 索引口径是本题最容易错的地方
`T` 的行按**父节点**索引、`qdist` 的行按**节点本身**索引，这不是笔误而是 tree attention 的语义：target 在节点 $v$ 上算出的 logits 描述的是「$v$ 之后那个位置」的分布，也就是 $v$ 的孩子们所在的那个位置。所以
$$ p_c=\texttt{T}[\text{parent}[c]][\texttt{token}[c]],\qquad q_c=\texttt{qdist}[c][\texttt{token}[c]]. $$
把 `T` 按节点自身索引（写成像 `T[c][token[c]]`）是本题第一号错法，会让第一层孩子的概率取到「本不该存在」的行、且整棵树的判定全部错位。反过来，路径末端补的那一个 token 也顺势落在 $\texttt{T}[v^\ast]$ 上：它是 $v^\ast$ 之后那个位置的分布，正是我们要的收尾 token（全接受时它扮演 bonus，被拒时它是残差的 P 侧）。

## 机制推导：为什么这样仍然严格等于 target 分布
单序列情形（155 题）的正确性来自逐位置拒绝采样 + 拒绝处残差重采：$x\sim\mathrm{norm}(\max(P-Q,0))$ 与被拒后重新采样的条件分布相同，故每个位置的边缘分布仍是 $P$。搬到树上时，一个位置挂着多个候选分支，判定必须**沿路径串起来**：节点 $v$ 有机会被接受的前提是它的整条祖先链都被接受，于是存活集合天然是一棵**连通的、含根的子树**。本题的实现按层推进（第 $d$ 层只从第 $d-1$ 层的存活者展开），因此：
1. **被拒分支的子树不再测试**——这正是 tree attention 的免费午餐：概率已经算出来了，但按树的语义它们不该影响输出；
2. **某层全军覆没即停止**——更深的位置已经没有合法前缀，继续读它们的概率没有意义；
3. 输出取**最深**的存活节点（并列取 token 字典序最小、再并列取编号最小），使「最长存活路径」是唯一确定的；
4. 收尾 token 只在路径末端取用**一次**：$v^\ast$ 有被拒孩子就用 $\mathrm{norm}(\max(\texttt T[v^\ast]-\texttt{qdist}[c],0))$（$c$ 取编号最小的被拒孩子），没有孩子就用 $\texttt T[v^\ast]$ 当 bonus。注意 $v^\ast$ 的孩子若存在必然全部被拒（否则 $v^\ast$ 不是最深），所以「有孩子」与「有被拒孩子」在这里等价。

顺带一个可证明的良定义性质：**拒绝必然给出正残差**。拒绝意味着 $p<q$（否则 $\min(1,p/q)=1$ 而 $u<1$ 恒接受），于是 $P\ne Q$ 且 $\sum_a\max(P_a-Q_a,0)=\mathrm{TV}(P,Q)>0$，归一化不会零除；这条把「要不要给零残差特判」的边界情况直接关掉了。

## 与合集内其它题的分界
- `inf-spec-decoding-rejection`(155)：一条线性 draft、首个拒绝处重采、后面全丢——是本题在「树退化成链且只有一条链」时的特例。
- `spec-medusa-tree-verify`(175)：树 + **typical acceptance**（只看 $P\ge\varepsilon$，不需要 $Q$、不做残差），最深存活路径的选法与本题神似但判据完全不同。
- `spec-sequoia-sample-verify`(176)：**同一个位置**上依次取 $k$ 个分支，每拒一次就把 $D$ 置零、把 $R$ 收缩为 $\mathrm{norm}(\max(R-D,0))$ 继续在同一位置试——残差被反复使用；本题的残差只在路径末端用一次，且树的分支分布在**不同深度**上。
- `spec-eagle2-dynamic-tree-prune`(284)：只决定「验证名额给谁」，不做任何接受判定；本题相反，树是给定的、只做接受判定。

## 易错点清单
1. 每层候选必须按节点编号升序收集，被拒孩子取**编号最小**者，否则残差的 $Q$ 侧不唯一。
2. 深度并列时的字典序比较要对 **token 序列**做，而不是对节点编号做（先长度、再 token、最后编号）。
3. 根不产出 token，也不参与判定（`u[0]`/`qdist[0]` 都不用），但 `T[0]` 是第一层孩子的分布行，必须给。
4. 采样用 CDF 反查「最小的 $j$ 使累积和 $>$ `u`」（严格大于），且 $u\in[0,1)$；用 `>=` 或与 155 的 `>=` 口径混用会在这种地方产生 off-by-one。
5. `M=1`（树上没有草稿节点）时存活路径为空，输出恰为 1 个 bonus token，不要返回空张量。

## 对拍口径
参考解按 frontier 逐层推进、共享已接受前缀（每个节点只判定一次）。判题 `_oracle` 走另一条路：对**每一条**根到节点的路径独立跑一遍线性拒绝采样（父节点的判定被重复计算，无前缀共享、无分层），由此得到「路径无拒绝」的节点集合；终点用 $(-\text{路径长度},\ \text{token 序列},\ \text{编号})$ 排序键选，残差与 CDF 用显式下标循环重算。二者等价的依据是「某层全拒 ⟹ 更深处不存在合法前缀」。此外用四类退化输入交叉验证：只有根（$M=1$）、第一层全拒、$\texttt{qdist}=\texttt{T}$ 的父行（ratio 恒为 1，全接受走 bonus）、星型一层树与同深字典序并列；并断言不变量 `len(out) == len(nodes)+1`。索引比较用 `torch.equal`。

## 复杂度
参考解 $O(M\cdot V)$ 级（每节点判定一次 + 末端一次残差与 CDF）；`_oracle` 因不做前缀共享是 $O(M^2)$ 量级，$M\le200$ 时仍远快于 30s 判题上限。
