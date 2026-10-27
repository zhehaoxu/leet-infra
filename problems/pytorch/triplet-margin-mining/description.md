# Triplet Margin 损失与在线难例挖掘（batch_hard / semihard）

## 题目描述

度量学习里一个 batch 能组出的三元组是 \(O(N^3)\) 量级，全用太贵、随机抽又太弱。FaceNet（Schroff et al., CVPR 2015, arXiv:1503.03832）的 **semihard** 与 batch-hard（Hermans et al., *In Defense of the Triplet Loss for Person Re-Identification*, arXiv:1703.07737）的 **batch_hard** 是两种"在线挖掘"：不给三元组，只给 (embeddings, labels)，让每个 anchor 自己在 batch 里挑正例和负例。本题实现这两套挖掘 + triplet margin 损失，含 PyTorch 特有的 `swap` 变体，返回单个 0 维标量。CPU、全程确定性、无随机数调用；并列距离一律取**最小下标**，因此判题可完全复现。

## 规则

- 完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。embeddings 是 (N,D) float（N=0 允许），labels 是 (N,) long；strategy ∈ {`"batch_hard"`, `"semihard"`}，其余抛 ValueError。
- **成对距离**：\(D_{ij}=\lVert x_i-x_j\rVert_p\)，按"逐元素差取 \(|\cdot|^p\) 求和再开 \(1/p\) 次方"计算（\(p=2\) 欧氏、\(p=1\) 曼哈顿；不用距离矩阵算子的近似路径）。
- **挖掘（对每个 anchor \(a\)）**：
  - 正例池 \(\{j:\text{labels}[j]=\text{labels}[a],\ j\ne a\}\)：取**距离最大**者为 pos（并列取最小 \(j\)）；池空 → **跳过该 anchor**。
  - 负例池 \(\{j:\text{labels}[j]\ne\text{labels}[a]\}\)（自身标签相同故自动排除自己）：
    * `batch_hard`：取**距离最小**者为 neg（并列取最小 \(j\)）；
    * `semihard`：优先在**开区间** \((d_{ap},\,d_{ap}+\text{margin})\) 内取距离最小者（并列取最小 \(j\)）；区间为空时**回退**为整个负例池中距离最小者（即退化成 batch_hard 的取法）。
  - 负例池为空 → 跳过该 anchor。
- **per-triplet 损失**（与 `F.triplet_margin_loss` 用固定三元组实测核对后的定稿口径）：
  \[\text{swap=False: }\ \ell_a=\operatorname{relu}(\text{margin}+d(a,\text{pos})-d(a,\text{neg})),\qquad
    \text{swap=True: }\ \ell_a=\operatorname{relu}(\text{margin}+d(a,\text{pos})-\min\{d(a,\text{neg}),\,d(\text{pos},\text{neg})\}).\]
  **注意 min 作用在负例一侧**，不是把 \(d_{ap}\) 换小。
- **返回**：所有**有效 anchor**（未被跳过的）损失的均值，0 维标量张量；没有有效 anchor 时返回标量 0。跳过的 anchor **不进分母**。

## 示例

embeddings \(=[[0,0],[3,0],[0,2],[1,2]]\)，labels \(=[0,0,1,1]\)，margin=1.0、p=2.0、swap=False。距离矩阵（部分）：\(d_{01}=3\)、\(d_{02}=2\)、\(d_{03}=\sqrt5=2.2361\)、\(d_{12}=\sqrt{13}=3.6056\)、\(d_{13}=\sqrt8=2.8284\)、\(d_{23}=1\)。

| anchor | pos（最远同类） | batch_hard 的 neg（最近异类） | \(\ell_a\) | semihard 的 neg（区间内最近，否则回退） | \(\ell_a\) |
|---|---|---|---|---|---|
| 0 | 1，\(d_{ap}=3\) | 2，\(d_{an}=2\) | \(\operatorname{relu}(1{+}3{-}2)=2\) | 区间 \((3,4)\) 空 → 回退 2，\(d_{an}=2\) | 2 |
| 1 | 0，\(d_{ap}=3\) | 3，\(d_{an}=\sqrt8=2.8284\) | \(1.1715729\) | 区间 \((3,4)\) 含 \(d_{12}=\sqrt{13}\) → 取 2 | \(0.3944487\) |
| 2 | 3，\(d_{ap}=1\) | 0，\(d_{an}=2\) | \(\operatorname{relu}(0)=0\) | 区间 \((1,2)\) 空 → 回退 2 | 0 |
| 3 | 2，\(d_{ap}=1\) | 0，\(d_{an}=\sqrt5\) | 0 | 区间 \((1,2)\) 空 → 回退 0，\(d_{an}=\sqrt5\) | 0 |

均值（实测回填 EXAMPLE_*）：

- `batch_hard` = \((2+1.1715729+0+0)/4=\) **0.7928932309**
- `semihard` = \((2+0.3944487+0+0)/4=\) **0.5986121893**

semihard 只在 anchor 1 上与 batch_hard 分道：它挑了更远的 \(\sqrt{13}\) 而不是最近的 \(\sqrt8\)，损失因此变小——这正是"半难"的设计意图（避开可能是误标的极近负例）。

## 约束

- strategy 只接受 `"batch_hard"`/`"semihard"`，其余 ValueError；margin、p 为普通实数（\(p=1.0\) 与 \(p=2.0\) 是主要口径，\(p\) 以 float 传入）
- 并列距离（含 semihard 区间内并列）**一律取最小下标**；这是判题可复现的前提
- anchor 的正例池或负例池为空则跳过：全部同标签、全部互异标签、N≤1、N=0 四种情形都返回标量 0
- 返回必须是 0 维标量张量（不是 Python float、不是 (1,) 张量）；float32 容差 \(10^{-5}\)
- 与 `F.triplet_margin_loss` 的一致性以"给定同一批三元组、同一 p/margin/swap"为判据；本题额外定义了挖掘与跳过/回退/并列三处 `F` 里没有的口径
