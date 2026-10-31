# notes — dt-gopher-repetition-filters（order 680）

## 回源

- 权威源：Gopher 论文（Rae et al., arXiv 2112.11446）**附录 A.1.1** Repetition Removal 小节与 **Table A1** 阈值表（PDF 逐字提取核对）。原文关键句："For lines and paragraphs separately, we calculate over the document both the fraction that are duplicates, and the fraction of characters contained within those duplicates; for each n ∈ {2,…,4}, we calculate the fraction of characters contained within the most frequently-occurring n-gram; and for each n ∈ {5,…,10}, we calculate the fraction of characters contained within all duplicate n-grams, taking care not to count characters that occur in overlapping n-grams more than once."
- Table A1 十三条：dup line 0.30 / dup para 0.30 / dup line char 0.20 / dup para char 0.20 / top2 0.20 / top3 0.18 / top4 0.16 / dup5 0.15 / dup6 0.14 / dup7 0.13 / dup8 0.12 / dup9 0.11 / dup10 0.10。拒绝词 "surpasses" → 严格大于。
- DeepMind 未开源该管道；论文未给行/段/n-gram 的精确定义，以下口径为题面钉死（notes 记录偏离）：行 = splitlines 原样（空行参与重复）；段 = 空行分隔的极大非空行块、段字符 = 行字符和；词 = 空白切分并记录字符 span；n-gram 为 word 级、精确匹配（论文的「归一化空白/忽略标点」只写在 MinHash 13-gram 段，不适用于 repetition 段）；top n-gram 并列取首现最早；覆盖 fraction 分母统一 len(text)。
- 与 `dt-minhash-lsh-dedup`（order 551）划界：那是文档间 MinHash 13-gram Jaccard ≥ 0.8 近似去重（论文 Document Deduplication 段）；本题是文档内 Repetition Removal 的精确切片账，两段机制在论文中相邻但完全不同。

## 判题面

- solve(text, thresholds=None) → {accepted, reject_reason, metrics(13 名)}；thresholds 覆盖任意子集，键校验 + 值域 [0,1]（bool 拒）。
- oracle：test_cases._py 独立重写——行/段重复用双循环计数、n-gram 覆盖用逐字符 boolean mask（与 solution 的区间合并不同路径），全字段对拍。
- 锚点：13 度量全量对拍；行实例 vs distinct（A×3 文档）；0.30 行 fraction 与 0.20 行 char 恰等不拒（thresh-ge 边界）；空行参与行重复 quirk（5 空行 + x×2 = 7/15）；段账可成为 reason 的构造（6 段 27 行、行账全部压线）；段 char 被行 char 支配（段重复 ⇒ 行重复且分母同源，段 char 永不先触发）；top 并列取首现（count=1 全并列 + count=2 场景）；重叠 span 合并（周期文档 dup5 = 1.0 vs 朴素累加 3.79）；Table A1 顺序账（多处超限取行序第一）；阈值覆盖入参改判；空文档/纯空白/词数不足。
- ValueError：text 类型、thresholds 非 dict/未知键/bool/字符串/越界。

## 易错点（出题自踩）

- 手算锚点三连错：① 段重复必导致行重复（段由行组成），且两 char fraction 分母同源，故 dup line char ≥ dup para char——「段字符账先触发」的用例构造不出来；② 空行本身是行且空行占比高，多段文档行 fraction 几乎总先于段 fraction 触发（段 reason 需 27 行 6 段的精细构造）；③ 周期文档 dup5 覆盖 = 1.0（末词 end 截到 len(text)=29），手算 30/29 错。
- 词重复 ≠ 行重复：行级是整行精确匹配，构造行账用例必须整行重复。
- 行字符含行内空白；覆盖 fraction 分母 len(text) 含换行与段间空白，覆盖只计词字符——两口径不同需分清。

## 变异记录（10 条全抓）

thresh-ge（≥ 边界）、dup-line-distinct（distinct 化）、span-no-merge（覆盖不合并）、top-tie-last（并列取末现）、top-span-single（top 只算首现 span）、para-always-split（段账清零）、line-char-denom-text（分母换 len(text)）、metric-order-reversed（Table A1 倒序）、dup-ngram-min-count（count≥3）、skip-blank-lines（剔除空行）。无等价变异。

## 红线

Gopher 质量启发式（词数/符号比/列表行）归 `dt-gopher-rules`（order 656）；MinHash 近似去重归 `dt-minhash-lsh-dedup`；C4 行清洗归 `dt-c4-filter`。测试集过滤（13-gram Jaccard vs test set）不出（随机移除哪个之一不确定）。无 torch 依赖，纯 Python 账本题。
