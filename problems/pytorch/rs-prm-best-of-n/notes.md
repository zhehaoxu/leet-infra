# 解题说明 · PRM best-of-N 聚合账

## 四种口径的机理

2305.20050 附录里对比了多种把步级分数压成标量的方法，结论是 **product（连乘）最贴合
best-of-N 的需求**：一条 10 步的解答只要任何一步的真值概率低（比如模型悄悄算错一步），
连乘就会把这个缺陷指数级暴露；而 `max` 会被单步高光迷惑，`last` 又完全无视前面的过程。
题面用例 `[[0.99]*9 + [0.01], [0.9]*2]` 就是这个机理的最小复现：product 下短链 0.81 胜过
长链 0.0092，max 下长链 0.99 胜出。

`answer_scores` 乘项对应「PRM 分数 × 终答验证器（ORM）分数」的组合用法：步级过程好 + 终答
对，两者缺一不可，所以是乘法而非加法。

## 实现要点

- **校验顺序**：rule 白名单最先（未知的 rule 在聚合前就该报错）；`top_m` 要同时排除
  `top_m < 1` 和 `isinstance(top_m, bool)`（`True` 是合法 int 但语义荒谬）；空 `step_scores`
  与空候选分两条校验，错误信息不同。
- **排序键 `(-agg[i], i)`**：Python 的 `sorted` 稳定，但显式把 index 放进键里能让「并列取小」
  的意图自文档化；`tie` 用 `agg.count(max_agg) > 1` 严格计数——浮点相等在此口径下是合法的
  （构造用例全是精确小数）。
- **`min_len` 账本项**：为 best-of-N 的「最短候选」统计口径服务（如最少步数优先的次级排序
  场景），单独一次 `min(len(s) ...)` 即可。

## Oracle 与变异测试

- `test_cases.py` 的 `_oracle` 是独立实现的归并账，随机用例 40 组覆盖四种规则 × 有无
  answer_scores × top_m 全区间。
- anchor 用例的手算锚点：product 口径 `[0.36, 0.63, 0.001]` → `best=1`、`top=[1,0,2]`。
  注意 max 口径下 `best=2`（0.9, 0.9, 1.0 的最大值在候选 2），last 口径下 `best=1`
  （0.8, 0.9, 0.1）——四个口径的 best 各不相同才是好锚点。
- 变异测试：last→first、min→max、product→sum、去 answer 乘项、tie 恒真、排序改按 index、
  min_len 改 max 共 8 条全抓；`no-rule-check`（删除 solve 内 rule 白名单）**存活**，原因是
  `_agg_one` 末尾的 `raise ValueError(rule)` 兜底抛出同类型异常，`check_invalid_inputs` 只
  断言 ValueError 类型——该白名单校验是冗余防御（提前于 top_m 等校验、错误信息更明确），
  行为上等价，记录为等价变异。
