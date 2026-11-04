# QUITO 查询注意力过滤 — 实现笔记

## 回源与口径

- 官方仓 `Wenshansilvia/attention_compressor`（`quito/utils.py` + `quito/compressor.py`），论文 arXiv:2408.00274（QUITO, CCIR 2024）。QUITO-X（信息瓶颈视角，EMNLP 2025 Findings）不在本题范围。
- **打分离线入参化**：官方 `attention_score` 依赖 HF 前向（`apply_chat_template` + trigger token、`prefix=14` 硬编码偏移、最后一层注意力按头平均后取**末行**、`softmax(axis=0)`）。判题只有 torch+numpy 无 transformers，注意力分数直接给定，钉的是打分之后的规则链。
- **官方缺陷口径（关键！）**：`text_filter` 调用 `get_sorted_ids(score)`，但 utils.py **没有定义该函数**（照抄会 NameError）。语义按论文「保留高注意力内容」钉为**按分降序的 stable 索引**；同文件的 `get_rank_list`（升序 rank 映射）是它的镜像参照。勿再找「官方定义」——不存在。

## 逐入口细节

### reconstruct
- `Ġ`（GPT-2 byte-BPE 空格前缀）或 `▁`（SentencePiece 下划线前缀）**开头** → 开新词；否则拼入当前词，注意力取 **max**，cnt += 1。
- `Ċ` → `\n` 的替换发生在**前缀判断之前**：`"ĊĠnew"` 替换后是 `"\nĠnew"`，不以 Ġ 开头 → 整段一词（换行不切词）。
- **官方怪癖**：首个无前缀 token（如 `["hel","lo","Ġworld"]` 的 "hel"）走 else 分支，cnt 从初始 1 自增 → 首词 cnt=3 而非直觉的 2。照钉官方。
- 末词在循环后补 flush。

### gaussian_smooth
- numpy 复现 `scipy.ndimage.gaussian_filter1d`（默认 `mode='reflect'`、`truncate=4.0`）：
  - `radius = int(truncate*sigma + 0.5)`（σ=0.7 → radius 3；漏掉 +0.5 会得 2，可观测）；
  - 核 `exp(-x²/(2σ²))` 归一化；padding 用 numpy 的 **`symmetric`**（scipy 的 `reflect` == numpy 的 `symmetric`，命名陷阱）；
  - σ=0 时恒等返回。
- **验证**：与 scipy 在 σ∈{0.5,0.7,1.0,1.5,2.0,3.0} × n∈{5,13,24,41} 全组合逐位一致（max diff 5.6e-17）。scipy 用系统 python3 一次性互证（两个 venv 都没有 scipy；`--target /tmp/scipy_chk` 装到隔离目录，不污染任何 venv）。

### text_filter（词级）
- rank = 按分**降序** stable 索引；`threshold = sum(cnt)*ratio`（是 **sum** 不是 len——fixture 里 cnt 全 1 时两者重合，必须用 cnt 含 >1 的用例区分）。
- **严格 `<`**：累积恰好等于阈值的词被丢弃并立即 break（宁可少留）。与 sentence_filter 的 `>` 口径**相反**——这是两个模式各自钉死的怪癖。
- break 后不继续扫（数学上无差：acc 单调递增，后续条件恒 false——变异 m-tf-nobreak 因此是真等价，已剔除记录，勿再钉）。
- 输出保持**原序**（不是分数序）：fixture `[0.5, 0.9, 0.2]` 保留 {m, n} 时输出 `[m, n]` 而非 `[n, m]`。

### sentence_filter（句级）
- 官方用 nltk `sent_tokenize` 切句 + 脆弱的子串匹配循环；本题把句子边界**入参化**为 `sizes`（每句词数），语义钉为：句分 = 句内词分 **max**（非 mean、非 sum）。
- `budget = len(words) * ratio`（浮点）；按分降序贪心，`acc + size > budget` 才断句（**等额可过**），断句时记录 breakpoint 句与其前的 `budget_left = budget - acc`。
- 选中句集合 `sort()` 后按原序拼接输出。
- **全放得下时**官方代码会走进 `except` 分支（breakpoint 未定义）打印后返回 None——本题钉为 `breakpoint_seg = budget_left = None`。
- 并列句分 stable：同分按原句序。

### dynamic_span（动态句级）
- `compress_sentence_token = sentence_filter + _select_tuples`：对放不下的 breakpoint 句，从句内**最高分词**（max 首见，即并列取最左）出发，左右指针交替扩展（两边都有则一次插两个），直到凑够 `budget_left` 个词或句长耗尽（break）；最后按原 index 排序输出。
- m 超句长时取整句。

## 验证链

- 沙箱 `/tmp/quito_sandbox.py`：四段互证（reconstruct 官方怪癖三条、text_filter 严格< + len/sum 阈值 + 原序、sentence max + 等额可过 + None 断点、dynamic 中心扩展四形态）+ scipy 全组合高斯互证。
- selftest 双模式 5/5 ACCEPTED；mut801 **23/23 全 CAUGHT**，1 条真等价剔除（m-tf-nobreak，理由如上）。

## 与 ctx 组其他题的分界

- LLMLingua-2（=ctx-llmlingua2-word-compress）钉的是**分类器打分**的词级压缩；LongLLMLingua 钉困惑度预算分配；GemFilter 钉早层注意力 top-k 裁输入。
- QUITO 增量：**子词→词的 max 聚合与 cnt 账、高斯平滑、cnt 严格<预算、句级 max 贪心、断点句中心扩展**——打分后的规则链整体，尤其是 text/sentence 两级相反的边界口径。
