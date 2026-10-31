# notes — rs-overthinking-metrics（order 663）

## 回源

- 机制出处：overthinking / token-efficiency 评估的通用口径（2503.16419「Do NOT Think That Much」、s1 的 token-vs-accuracy 曲线均为此类；无唯一权威实现），按「确定性账本」自定口径并在题面钉死——判题只认题面语义。
- 与 654 `rs-self-consistency-vote`（答案归一化投票）、655 `rs-prm-best-of-n`（PRM 聚合）、662 `rs-budget-forcing`（预算状态机）划界：本题是**评估侧指标账**，无生成语义。

## 判题面

- solve 入参：correct（bool 列表）、tokens（非负 int 列表）、bin_edges（严格递增 int，≥2 条）。返回 n/accuracy/total_tokens/bins/len_correct/len_wrong/efficiency_ratio/best_bin。
- oracle：`test_cases._py` 独立纯 Python（按样本重算分箱），随机 200 组对拍全字段。
- 锚点：① 4 样本 2 箱手算（accuracy 0.75、counts [1,3]、efficiency 20/(135/3)、best_bin=1）；② 边界值 t=10 落箱0、t=30 落末箱（左闭右开 + 末箱右含）；③ 空箱 None 与 best_bin 跳过；④ 全对（len_wrong=None、efficiency=None、best_bin=0）；⑤ 全错（best_bin 仍取首个非空箱）；⑥ 准确率并列取最小箱号；⑦ K=1 单箱。
- 非法输入：长度不齐、空记录、bool/负数/浮点混入、edges <2 条/不严格递增/含非整 → ValueError。

## 易错点（出题时自踩）

- **K = len(edges) − 1**：edges=[10,30] 只有 1 个箱（[10,∞)）——两次把「2 条边当 2 箱」写错锚点，靠 oracle 对拍暴露。edges=[10,30,100] 才是 2 箱。
- 低于 edges[0] 的样本没有专属箱，题面钉死落入末箱。
- `efficiency_ratio` 的 None 语义含 len_correct == 0（避免除零）。

## 变异记录（9 条全抓）

bin-left-open（边界归属翻转）、eff-inverted、len-wrong-all（答错均值算全体）、best-bin-last（并列取大）、bin-acc-denom-n（箱内准确率分母换成 N）、bin-mean-correct-only（箱内均值只算答对）、total-correct-only、acc-wrong-denom、last-bin-excl（末箱归属错位）。

## 红线

模型生成、题库统计显著性检验、画图不出；correct/tokens 全显式入参。
