# 解题笔记

## 思路

官方 mint-vu/Zero-Sum-SVD 的确定性核心，取两处：

1. **planning.py**（L90-207 主选择循环 + L383-435 rank/预算常量）：白化域谱、亏盈平衡秩、零和符号平衡选择器、effective_dense_delta 记账；
2. **correction_utils.py**（L97-275 + L289-342）：梯度白化 H = G@L⁻ᵀ（triangular solve）与 δ 估计的三种变体（first/second order、β-mixing）。

上游不入判题域：多阶段 planning（stage 间的投影修正）、projected gradient + re-truncation 修正步、remap 量化、非 zero_sum 的 selection_mode；模块一律从满秩 dense 起步。

与现有题划界：
- **773 SVD-LLM**：白化域做**逐模块固定 k 截断**（k 由 ratio 闭式给出），无全局分配无梯度；
- **771 LoSparse**：低秩+结构化分解，无白化、无敏感度、无跨模块预算；
- **772 LLM-Barber**：growth/prune 双阈值 + 粒度掩码，剪的是**权重元素**不是奇异分量。

## 关键实现点

- **梯度白化**：H = G@L⁻ᵀ 用 `solve(L, Gᵀ).t()`——不要去求逆再乘（数值稳定性 + 与官方 solve 语义对齐）。
- **敏感度投影**：`grad_σ = diag(Uᵀ H V)`，V 是白化域 A=W@L 的右奇异向量。丢 V（`UᵀH`）或换 W 自身的 V 都是可抓的变异。
- **偏好堆空落另一堆**：`for prefer in (prefer_pos, not prefer_pos)` 两轮循环——实现初版在偏好堆空时直接 break，导致符号失衡场景死锁/提前终止，与官方 `pop_prefer_then_any`（堆空换堆）语义不符，测试 `zero_sum_sign_balance` 的极端构造能稳定抓到。
- **effective_dense_delta 的三分段**：r>r* 记 0 不是 bug——低秩因子 (M+N)·r 在平衡点以上确实比稠密 mn 贵；r==r* 的 mn−r*(M+N) 是一次性结清。测试用多模块混合形状（12×8 / 8×16 / 10×10）让三段都被走到。
- **δ 的 β-mixing**：`−sign(g)·|σ|^β·|g|^(1−β)`，σ=g=0 时特判 0（0^0 防御）。β=0 与 β=1 分别退化为 |g| 与 |σ| 主导——测试里 β=0.7 抓指数互换（0.5 时互换恰等价，是无效变异）。
- **确定性**：候选遍历按模块名 sorted，`|δ|` 最小优先、名字 tie-break；oracle 用 numpy 独立复刻（`np.linalg.svd` + 排序），与 torch 的 svd 符号约定无关（断言只依赖 σ 非负与 δ 数值）。

## 测试设计（8 例）

- `zero_sum_oracle`：多组 ratio/keep_ratio/Beta/order 组合与 numpy oracle 全字段对拍（ranks/masks/truncated/running_sum/dense_removed/selections）；
- `delta_order_variants`：second_order 半平方项、β-mixing 三档（0/0.7/1）与 first_order 的 selections 差异可观测；
- `zero_sum_sign_balance`：构造一模块 δ 恒正、另一模块 δ 恒负的谱，断言 running_sum 被 |δ| 贪心压在 0 附近 + 偏好堆空换堆路径（两轮 prefer 循环都被走到）；
- `break_even_accounting`：三段记账逐值断言（构造让某模块恰剪到 r*、另一模块跨过 r*）；
- `keep_rank_floor`：keep_rank_ratio=1.0 时秩下限 = r*（不是满秩）且 min_rank 上限路径、max_svs=0 模块完全不参与选择；
- `no_whitening_path`：L=None 三模块对拍 + 白化可观测（同 weight/grad 有 L 与无 L 的 ranks 不同）；
- `purity`：输入张量不被原地修改；
- `args_validation`：空 dict/缺键/形状不匹配/非浮点/ratio≥1/keep_ratio>1/β 越界/order 非法/bool 混入，各报对应消息。

## 变异收口

41 条设计 → 39 条 CAUGHT + 2 条等价剔除：
- **m-cap-neg**（去掉 `max(0,·)`）：min_rank ≤ r* < min(M,N) 恒成立，被减数恒正，max(0,·) 冗余——上界语义已由 m-minrank-full 覆盖；
- **m-val-ratio-bool**（去掉 bool 拦截）：True 被 `< 1.0` 分支拦截抛错语义不变；False 与 int 0 数值全同且 ratio ≥ 0 本就合法。

覆盖：白化丢失/转置/solve 换逆序、SVD 升序、投影丢 V、δ 符号/半平方项/order 翻转/β 指数互换/β 符号/β 路由、偏好翻转/恒真偏好/去符号堆/tail 取头/consumed+2/truncated 去平方/truncated 索引、r* 公式/min_rank floor/min_rank 换满秩、记账三分段三变异/预算公式/预算冻结/running 冻结/dense 冻结、校验七变异 + 消息两变异。

## 工程学备注

- **选择器死循环防御**：budget 未满但所有模块分量耗尽时必须 break（两轮 prefer 都空即无候选），oracle 与 solution 语义必须一致——oracle 漏写该 break 时 `cands[0]` IndexError，两个用例假失败。
- 变异脚本替换后先 `compile()` 预检：丢失 if 冒号的替换串会产生语法错误，判题 0/0 会被误读成 SURVIVED（771 教训沿用）。
- σ 非负是 svd 的输出约定，`truncated_sigma_sq` 直接累加 σ²；δ 的符号只由 grad_σ 决定（一阶），这是符号堆设计成立的前提。
