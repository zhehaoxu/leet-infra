# notes（ag-toolformer-filter-ledger）

## 回源

- **Toolformer 官方未开源代码**。口径来源：① arXiv 2302.04761 **v5**（NeurIPS 2023 版）Sec 2.2 Filtering / Sec 4.1 weighting / App A 阈值——注意 v1 与 v5 的过滤小节口径相同（都是 L_i⁺/L_i⁻ 加权 CE），**社区流传的 P_plus/P_minus/P_corr 三条件（τ+δ 双阈值、3 个损坏样本）在 ar5iv 两版里都不存在**——todos 候选原记载（"perplexity 门、self-consistency 去重、加权 perplexity 增益"）系误记，已按 v5 钉死为 L 口径。ar5iv 的 ?v1 后缀疑似仍服务最新版，版本核对要靠多源。② lucidrains/toolformer-pytorch 的 filter_tokens_with_api_response / replace_all_but_first（星最多复现）——它实现 loss⁻=min(无响应, 无调用)、τ=1.0、去重保第一个。
- 论文默认值：τ_s=0.05、τ_f=1.0、k=5、m=5；calculator/MT 另设 τ_s=0、k=20、m=10、τ_f=0.5（App A 原文）。权重 w_t = w̃_t / Σ w̃_s，Σ=3.0；lucidrains 注释自认 "not sure what w_s is in the denominator"——本题钉全局常数 3.0 口径。
- 论文过滤小节无去重；"每输入至多一次 API 调用"是**推理期**限制（Sec 4.2）。文本级去重取 lucidrains 的 replace_all_but_first（训练数据管线）。

## 关官方 quirk（逐条核实）

1. **两处比较算子相反**：采样门 `p_i > τ_s` **严格**；过滤门 `L⁻ − L⁺ ≥ τ_f` **含等**（原文 "only keep API calls for which L_i⁻ − L_i⁺ ≥ τ_f holds"）。
2. **权重全局归一**：Σ_{s∈ℕ} max(0,1−0.2s) = 3.0 是常数——短后缀不重归一（2 token 全 -1 → L=0.6）；t≥5 权重 0 但 lp 列表仍须占位（极端 logp 不影响 L）。
3. **L⁻ 双对照 min**：L(ε)（无调用）与 L(e(c,ε))（只有调用没结果）取小——官方这样写使得"调用本身无害但响应有用"与"响应无用但调用文本有用"都可比。
4. **lucidrains 去重正则三 quirk**：前导空格 `' ['` 是 pattern 一部分（行首调用不匹配）；`\w+` 工具名（空名不匹配）；args `[^)]*` 遇 ')' 截断（含括号 args 整体不匹配）。删除是**替换为 ''**（不是删分隔符）。
5. has_calls 门：数据管线把无调用样本整对剔除（filter_and_keep_only_first_api_call 的 else 分支）。

## 出题工程

- **浮点恰等门是本题最大坑**：全 -1 后缀 score = 0.9999999999999999 < 1.0——想用"score 恰等 tau_f"锚点抓 `>=`→`>` 变异时被 `(w̃/3)·lp` 舍入坑了三轮。解法：含等证据改走精确路——**三表全同 → L⁺=L⁻ → score 恰 0.0，tau_f=0**（0.0>=0 True、0.0>0 False）；浮点恰等 quirk 本身反手入账（tau_f=1.0 判丢钉死）。教训：浮点恰等锚点必须找可精确表示的构造（如 0.2*5=1.0 exact 也会被先除后乘破坏——(0.2/3)*5 ≠ 1.0）。
- **lp 列表对齐二义**：第一版把权重步号写成 t = j − pos（绝对下标）与"后缀对齐"的测试构造矛盾——统一为**后缀对齐**（t = lp 列表下标，pos 只作身份与 per_pos 分组）。
- 校验须**前置到全体候选**：第一版只在评估循环里校验，pos=9 未过采样的候选漏校验（random/ValueError 锚点抓出）。
- scores 输出序 = evaluated 序（位置升序、组内输入序），不是候选输入序——第一版按输入序被 random 抓出。
- 变异 20 条实质全 CAUGHT + 3 条等价记 notes：eq-min-commute（min 交换律）、eq-order-tuple（top-k tie-break 用 passed 下标 t 与用位置号 i 等价——filter 保序使 t↔i 严格递增）、eq-kept-sorted（先排序后映射 vs 先映射后排序——t→i 严格递增故可交换）。
- regex-name-plus-star（\w+→\w*）第一轮 SURVIVED：空名锚点字符串写错（' [](2)' 是空方括号后跟 (2)，正确空名格式是 ' [(2)]'）——教训：构造反例前先逐字符打印字符串。
- 红线：真实 LM 前向/logits、tokenizer、API 执行器、训练循环不入题；推理期解码限制（at most one call per input）不入题只记 notes。

## 后续候选

- 采样侧账（τ_s 门前的 p_i 序列生成）依赖真实 LM 前向，不出；若出需 logits 显式入参化（softmax 归一 + <API> token 概率账）。
- ToolLLM DFS 决策树（2307.16789）为下一题，与本题共享"候选过滤"语境但机制独立。
