# notes — ocr-rec-norm-edit-dis（order 687）

## 回源

- 出处：`PaddlePaddle/PaddleOCR` `ppocr/metrics/rec_metric.py` 的 `RecMetric.__call__`（rapidfuzz 版本；更早版本用 python-Levenshtein，语义等价）。默认 `ignore_space=True`、`is_filter=False`、`eps=1e-5`。
- `rapidfuzz.distance.Levenshtein.normalized_distance` 语义（回源 `Levenshtein_py.py` 钉死）：`dist / maximum`，`maximum = max(len(s1), len(s2))`（weights (1,1,1) 下）；**`maximum == 0` 时返回 0**（`dist / maximum if maximum else 0`），即空-空对 ned=0、空-非空 ned=1.0。
- 归一化只保留 `string.digits + string.ascii_letters` 再 lower（`_normalize_text`），中文字符会被剔除；官方执行顺序：先 `replace(" ", "")` 再 normalize（filter 已删空格，顺序无行为差，照抄官方）。
- 经典 Levenshtein 非 Damerau：`"ab"→"ba"` dist=2。rapidfuzz 的相邻换位是 `DamerauLevenshtein` 类，本题不走。
- 与 `eval-*`（bleu/rouge/shingle）划界：本题是**字符级逐行**编辑距离 + per-sample 均值；HF `evaluate` 的 CER 是 corpus 级 `Σdist/Σlen`，口径不同勿混。

## 判题面

- solve 入参：pairs（[[pred, target], ...]，可空）、ignore_space、is_filter。返回 rows/acc/norm_edit_dis。
- oracle：`test_cases._py` 用递归 + `lru_cache` 独立实现 Levenshtein，语义复刻（`dist/maximum if maximum else 0`）；200 组随机对拍（插入/删除/替换扰动 + 两开关随机组合）。
- 锚点：kitten/sitting dist=3、ned=3/7；`abcd/ab` ned=0.5（分母 max 口径，len(target) 口径会错成 1.0）；空-空 ned=0 且 exact=True；空-非空 ned=1.0；`ab→ba` dist=2（Damerau 陷阱）；`A-1 x!` vs `a1X` filter 后相等；空批 acc=0.0、norm_edit_dis=1.0（官方公式直接计算的反直觉行为）；全对批 acc = n/(n+eps) 严格小于 1。
- 非法输入：pairs 非 list/元素非二元组/含非 str、开关非 bool → ValueError。

## 变异记录（13 条全抓）

ned-denominator-target（分母换 len(target)）、ned-empty-one（空批 ned 置 1）、ignore-space-off-pred/target（去空格开关失效，pred/target 各一）、is-filter-no-lower（漏 lower）、is-filter-no-filter（漏 filter）、lev-no-substitute（DP 漏替换分支）、lev-diag-free（对角零代价）、acc-no-eps（分母去 eps）、ned-no-1minus（漏 1-）、rows-reversed（行序逆转）、exact-before-filter（exact 用原始串）、ned-round2（ned 舍入两位）。

## 红线

不出 HF CER/WER corpus 级口径（与本题 per-sample 均值区分）；不做 diff 对齐输出（editops 不入题）；真模型/评测框架 IO 不出，字符串全入参。
