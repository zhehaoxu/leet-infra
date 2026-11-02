# notes（rs-ttrl-pseudo-reward-ledger）

## 回源

- 论文：TTRL（arXiv 2504.16084，NeurIPS 2025）。核心主张是「测试时用多数票造标签」：
  同一题采 `n` 条，投票得到伪标签，再用它当 ground truth 算 0/1 奖励做 RL。
  论文只给这套流程与曲线，**投票细节、比例分母、指标口径**都没写。
- 代码：`PRIME-RL/TTRL` main（官方实现，verl fork）。
  - `verl/verl/trainer/ppo/ttrl_utils.py`：20-32 `select_top_k_per_prompt`、
    38-47 `apply_original_gt`、50-83 `apply_ttrl_gt`、86-106 `_batch_majority_vote`、
    109-122 `_majority_vote`、128-155 `compute_ttrl_metrics`、
    158-190 `_batch_compute_ttrl_metrics`、192-214 `_prompt_compute_ttrl_metrics`。
  - `verl/verl/utils/reward_score/ttrl_math/__init__.py`：30-33 `extract_answer`、
    36-47 `grade`、49-65 `simplify_expression_string`（sympy）、67-103 `compute_score`。
  - `verl/verl/utils/reward_score/ttrl_math/math_utils.py`：891-915
    `last_boxed_only_string`、918-925 `remove_boxed`、928-932 `extract_boxed_answer`。
  - `verl/verl/utils/reward_score/ttrl_math/cluster.py`：1-44 `TTRLClusterCounter`
    （带等价函数的聚类计数器）——**注意 `_majority_vote` 没用它**，用的是裸 `Counter`，
    也就是「化简后逐字相同」才算同票。等价类聚合是仓库里没接上的另一条路。
- 本题把 LLM 生成、sympy 化简、mathd/sympy 判等三处不可判点全部外置成入参，只搬账。

## 官方实现里的坑

1. **`majority_ratio` 的分母是 `n`**（`majority_count / len(model_outputs)`），不是成功抽出
   的条数。抽坏一半的题，一条独苗当选也只有 0.25。
2. **一条都抽不出来时的兜底是字符串 `"None"`**，比例 `0.0`。它不是 `None`，会原样进入
   后面的 `label_accuracy` 比较——所以真值恰好写成 `"None"` 时会给 1.0 分。
3. **并列取先出现者**：`Counter.most_common(1)` 走 `sorted(..., key=count, reverse=True)`，
   Python 的 `sorted` 稳定且 `Counter` 保插入序，于是平票时最早出现的那条胜。
4. **只认最后一个 `\boxed`**：`last_boxed_only_string` 用 `rfind`，最后一个框写坏了整条
   采样就废，前面写对的框不救场。
5. **`\fbox` 分支是死代码**：`extract_answer` 先挡了「没有 `\boxed` 子串」的情况；就算直接
   调 `last_boxed_only_string`，`remove_boxed` 也只认 `\boxed{` 前缀，`\fbox{7}` 必返回
   `None`。所以本题只说「`\fbox` 不算」。
6. **配平从 `\boxed` 的位置开始**，所以「`\boxed` 后面先出现 `}`」会让计数变负、永远回不到
   0；而「`\boxed` 后面跟空格、花括号在更远处」会配平成功但前缀检查失败。两种都是 `None`，
   原因不同。
7. **空 box 是合法票**：`\boxed{}` 抽出 `""`，`simplify` 失败也返回 `""`，于是空串能当选、
   能被真值判中。
8. **`select_top_k_per_prompt` 不检查 `k <= n`**：`range(start, start + k)` 直接跨块，最后
   一块越界抛 `IndexError`。而且选完之后的流水线把 `k` 当新的 `n`，所有分母跟着变。
9. **`reward_accuracy` 是逐样本相等比例**，不是「均值相等」：它数的是
   `majority_reward[i] == gt_reward[i]` 的样本数除以 `n`。
10. **`pass@{n}` 用 `sum(gt_reward) >= 1`**，不是 `any`；对 0/1 奖励两者等价，但写成了和式。
11. **跨题平均的键序取第一题**（`for k in ttrl_metrics[0]`），`majority_ratio` 在这之后才
    追加，所以恒排最后。
12. **`_batch_compute_ttrl_metrics` 里那条「每题 n 条标签全同」的 assert 是构造性冗余**：
    `apply_ttrl_gt` 给同一题的 n 条样本写的是同一个 label，不可能不同。本题不复刻这条死断言。

## 本题的规范化

- **化简表代替 sympy**：官方 `simplify_expression_string` 先 `parse_expr(transformations="all",
  evaluate=False)`，失败退 `latex2sympy`，再失败返回原串。本题把结果做成 `canon` 表，未命中
  即原样（等价于两级 fallback 都失败）。每个原始值只查一次表，所以链式表不会连锁。
- **判等用字符串相等代替 mathd/sympy**：官方 `grade` 是 `grade_answer_mathd or
  grade_answer_sympy`（后者还会 `_normalize` + `split_tuple` + `are_equal_under_sympy`）。
  判题环境没有 sympy，本题退化成「化简后逐字相同」，这是唯一一处**语义**降级，其余口径不变。
- **真值抽不出来时兜底成 False**：官方 `grade` 拿到 `None` 真值会在 mathd 里抛 `TypeError`；
  本题把它规范化成「不匹配」，不再抛异常。
- **空批抛 `ValueError`**：官方空批会在 `_batch_compute_ttrl_metrics` 的 `ttrl_metrics[0]`
  处 `IndexError`；本题提前给显式 `ValueError`。
- **删掉不可达的整除复检**：降采样后长度恒为 `题数 * keep`，`n` 又等于 `keep`，第二次
  「长度能被 n 整除」检查永远成立，已删除（等价变异可证）。

## 变异

45 条变异：**39 条实质变异全部 CAUGHT**，6 条判为等价，全部用「3000 组随机试卷（含
`keep`/空 box/坏真值/链式化简表）+ 30000 条随机串」逐一对拍账本，0 差异：

- `m-depth-le`（配平判据 `depth == 0` → `depth <= 0`）：计数变负只可能发生在「`}` 先于
  `{`」，那时截出的候选串以 `\boxed}` 开头、前缀检查照样返回 `None`，两种写法同结论。
- `m-end-check`（`end < 0` → `end == 0`）：没找到闭合时 `end` 是 `-1`，`text[idx:0]` 是空串，
  前缀检查同样返回 `None`；找到时 `end >= 1` 恒不为 0，两条分支都没有差别。
- `m-truth-item-none-only`（去掉 `truth is None` 那半个条件）：`None` 真值时比较的是
  `canon(答案) == None`，字符串永不等于 `None`，结果仍是 0。
- `m-mvr-div-len`（`sum(rewards) / n` → `/ len(rewards)`）：`rewards` 的长度构造上恒等于
  `n`，两个分母是同一个数。
- `m-pass-ge-half`（`>= 1` → `>= 0.5`）：真值奖励只有 0/1，和式取整，两个阈值等价。
- `m-metrics-first-last`（平均键序取 `prompts[-1]` 而不是 `prompts[0]`）：每题的指标键集
  完全相同，键序也一样，对外不可观测。

过程中首轮 SURVIVED 又救回来的：

- `m-div-guard-off`：把「降采样前先查整除」整个短路后，原本由**第二处**复检兜住；而那处
  复检是我多写的死代码。补了「不整除 + 给 keep」的用例（官方此时必须先 assert，否则尾巴
  会被静悄悄吞掉）后，再删掉死代码，这条才真正被抓。
- `m-scan-after-prefix`（扫描起点 `idx` → `idx + 6`）：`\boxed` 正好 6 个字符，`+6` 落在
  开括号上，与原写法等价；改成 `+7`（跳过开括号）才是有意义的变异。

## 查重

- 与 `rs-self-consistency-vote`（order 654）划界：那题输入已是**答案段**，考的是 8 条词法
  归一化规则 + 加权计票；本题输入是**原始输出**，考的是 `\boxed` 抽取（rfind + 花括号配平 +
  前缀检查）、伪标签、双路奖励、指标口径与降采样，归一化反而被外置成入参。
- 与 `rs-prm-best-of-n` / `rs-budget-forcing` / `rs-overthinking-metrics` 无交集：那些是
  步级聚合、预算状态机、长度统计，都不涉及伪标签与奖励账。
- 全库无 TTRL / 伪奖励 / `\boxed` 抽取相关题目；`agentic-systems` 里的投票题（ToT、CoA）
  都是树/链调度，没有「用自采样造标签再反算奖励」这条轴。
