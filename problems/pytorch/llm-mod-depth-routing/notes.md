# 详解：Mixture-of-Depths 的逐层路由

## 出处

- Raposo et al., *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models*, DeepMind, 2024, arXiv **2404.02258**。核心式 (1)：

  \[
  x_i^{l+1}=
  \begin{cases}
  r_i^l\, f_i(\tilde X^l)+x_i^l, & r_i^l > P_\beta(R^l)\\
  x_i^l, & r_i^l < P_\beta(R^l)
  \end{cases}
  \]

  其中 router 是标量 \(r_i^l=w_\theta^\top x_i^l\)，阈值 \(P_\beta\) 是 router 权重的 \(\beta\) 分位点，\(\beta=1-C/S\)（\(C\) 容量、\(S\) 序列长），因此"高于分位点"就是"前 \(C\) 个"。\(\tilde X^l\) 只含被选中的 token。论文的辅助损失是对 router 输出做二分类交叉熵。
- 备注：题库待办里这条曾写作 2202.07736，那是错的编号；本文以 2404.02258 为准，实施时已核对标题与作者。
- 与 `moe` 合集的分工：MoE 沿**专家维度**做 softmax/top-k 组合，本题沿**深度（层）维度**做容量裁剪；每层只有一个标量 router 向量，没有专家权重矩阵的混合。库内没有第二题做深度路由。

## 核心与易错点

1. **容量阈值不是绝对值**。原文用 \(\beta\) 分位点，本题等价地写成"取 top-k，k=min(C,T)"；两者只在并列时有差别，而原文没规定并列语义。本题把它钉死为 \((分数降序, 下标升序)\)，这样同一输入永远给出同一 `route_mask`。
2. **router 分数乘在 block 输出上**，不是二值开关。漏乘 \(s_i\) 会让所有路由增益都变成系数 1，`check_single_token_block_scaling`（容量 1 时 \(x\leftarrow x+s\cdot xW\)）会立刻失败。
3. **block 只作用于子序列**。不是"对整条序列做 attention 再挑出被选中的行"——那样被丢弃 token 的信息会泄漏进保留 token 的注意力。`check_block_scope_selected_only` 通过扰动一个未被选中的 token 来验证它对其他 token 的输出毫无影响。
4. **子序列保序**。选中的下标必须按原始下标升序 gather，因果掩码才对应"原始序列里的先后关系"。按分数顺序 gather 会让高分 token 变成子序列首行、看不到本该在它前面的 token；`check_gather_preserves_order` 专门构造了"分数序 ≠ 下标序"的例子。
5. **router 用当前状态**。第 \(l+1\) 层的分数来自第 \(l\) 层写回之后的状态，而不是初始输入。`check_router_uses_current_state` 用两层的例子对比"当前状态"与"初始状态"会选出不同 token。
6. **未选中的 token 严格不变**。实现上要保证逐位相等，而不是"乘 0 后剩一点浮点残差"；`check_unselected_untouched` 用 `torch.equal` 断言。
7. **同一层内是快照语义**：block 一次性作用于整个子集，子集里靠后的 token 不能看到同层已写回的更新值。实现里先 `gather` 出副本再统一写回，天然满足。
8. **边界**：容量 0 时什么都不算但分数照常输出；容量 \(\ge T\) 时全选，等价于普通因果自注意力加分数缩放。`capacities` 不接受 `bool`/`float`，因为 `True` 会被当成 1 造成静默错误。
9. **block 是替身**：真实 MoD 的 block 是完整的 attention+MLP，这里用"共享 q/k/v 投影的因果自注意力 + 1/√d 缩放"替代，目的是让子序列内部的顺序敏感性可判、可复现，不涉及任何真实权重。

## 对拍口径

- `_oracle` 用 numpy 逐层重写：`sorted(..., key=lambda i: (-s[i], i))` 选容量、`sorted()` 保序 gather、逐行因果 softmax 手算，和向量化解法没有共享代码。
- 10 个 `check_*`：样例、容量与并列、单 token 缩放、未路由位不变、block 作用域、保序 gather、当前状态打分、返回契约（形状/dtype/分数即点积）、随机对拍（5 组配置含容量 0 与超容量）、22 种非法输入。
- 变异测试（升序选择、按分数序 gather、整序列做 block、漏乘 router 分数、用初始状态打分、无因果掩码、去掉 1/√d、容量 +1、子集行错位）全部被至少一个检查抓住。
- float32 与 float64 输入都要能跑；float32 输入的对拍口径是"先把输入升到 float64 再跑 oracle"，从而固定路由顺序（否则两种精度下的分数排序可能翻转）。

## 复杂度

- 每层：打分 \(O(Td)\)，子序列投影 \(O(cd^2)\)，因果注意力 \(O(c^2d)\)，写回 \(O(cd)\)，其中 \(c=\min(C_l,T)\)。
- 总时间 \(O\!\left(\sum_l (Td+c_l d^2+c_l^2 d)\right)\)，额外空间 \(O(Td+c\,d)\)；`route_mask` 与 `router_scores` 各占 \(O(LT)\)。
