# 参考做法：词首归并、force 置一与「+1 截断」阈值

## 实现

1. 先做类型与取值校验（等长、概率在 `[0,1]`、`rate ∈ (0,1)`、枚举值白名单、`token_count` 可调用且返回 `>=1` 的 int）。
2. **词首归并**（对应上游 `llmlingua/prompt_compressor.py::__merge_token_to_word`，L2255-2292）：
   - `special_tokens` 命中的 token 直接 `continue`；
   - `is_begin_of_new_word`（`llmlingua/utils.py` L81-101）：BERT 系看「去 `##` 后命中 force/map → 词首，否则非 `##` 开头即词首」；xlm-roberta 看「token 是标点或在 force/map → 词首，否则 `▁` 开头即词首」；
   - 词首 token 取 `get_pure_token`（BERT 去 `##`、xlm 去 `▁`），若纯词形命中 `force_tokens` 或 `token_map.values()`，概率置 `1.0`；随后 `replace_added_token` 把 `token_map` 的替换 token 还原成原词，`force_reserve_digit` 的数字判定作用在**还原后的词**上；
   - 续接 token 把纯词形拼进当前词，概率 append 进当前词的子词列表，`force_reserve_digit` 的数字判定作用在**原 token**上；
   - 第一个非特殊 token 不是词首时上游会 `IndexError`，本题改为显式 `ValueError`。
3. **词概率合并**（`__token_prob_to_word_prob`，L2294-2302）：`mean` 对子词列表取均值、`first` 取首子词，其余模式上游抛 `NotImplementedError`、本题收窄为 `ValueError`。
4. **drop_consecutive**（`__compress` 内 L2385-2397）：词级阈值 `percentile(word_probs, int(100*rate))`；扫到 force 词时若 `is_token_between` 已置位则只复位，否则与前一个 force 词相同就把当前词概率置 `0`；非 force 词用 `|=` 累积「是否出现过高于词间阈值的词」。
5. **展开与阈值**（L2399-2409）：每个词概率按目标分词器 token 数展开（上游是 `len(self.oai_tokenizer.encode(word))`，本题作显式入参 `token_count`），阈值 `percentile(expanded, int(100*rate + 1))`——**`+1` 与 int 截断**是本题最容易丢的两个字。
6. **保留判定**（L2414-2429）：`p > threshold`，或 `threshold == 1.0` 时 `p == threshold` 也保留（防全 1.0 输入一个不剩）；`drop_consecutive` 下，与已保留末词相同的 force 词标签记 `0` 但不进 `keep_words`。

## 正确性检验

9 个 `check_*` 覆盖：题面示例（`▁Hello/▁world` 交替，阈值 0.612）、10 组多旋钮组合对拍（mean/first × 两分词器 × force/map/digit/特殊 token/token_count/drop_consecutive）、force 与 map 命中置 1.0、BERT `##` 归并的 mean/first 差异、连续 force 词置 0、特殊 token 跳过、token 展开改变阈值与标签、全 1.0 时阈值等值保留、21 条非法输入。

oracle（`test_cases.py::_oracle`）按同一契约独立重写（先归并再逐条判定），与参考解共享契约不共享代码。

变异测试（6/6 全抓）：mean 改 first（`bert_continuation_merge`/`matches_oracle` 抓）、force 不置 1.0、阈值去掉 `+1`（6 个用例抓，示例与对拍首当其冲）、`>` 改 `>=`、drop_consecutive 不置 0、digit 不强制。

## 范围与来源

上游 microsoft/LLMLingua（2403.12968）：

- `prompt_compressor.py::__merge_token_to_word`：special 跳过、词首判定、force/map 纯词形命中置 1.0、`replace_added_token` 还原、数字判定分支（词首看还原后 token、续接看原 token）。
- `prompt_compressor.py::__token_prob_to_word_prob`：mean/first 二选一。
- `prompt_compressor.py::__compress`：词间阈值 `int(100*rate)`、展开 `oai_tokenizer.encode`、最终阈值 `int(100*rate + 1)`、保留条件 `p > thr or (thr==1.0 and p==thr)`、连续 force 词 label 0。
- `utils.py::is_begin_of_new_word / get_pure_token / replace_added_token`：两分词器的词首与纯词形规则（L81-120）。

本题的口径差异（题面自定契约）：

- **不加载任何模型/分词器**：BERT 级编码器的 keep 概率直接作 `token_probs` 入参；tiktoken 的 `oai_tokenizer.encode` 改为显式 `token_count` 可调用。
- 输出 `words` 保留 `▁`/`##` 原样（上游在 xlm 侧末尾会把 `words` 逐个 `lstrip("▁")`，但那发生在 `keep_str` 与标签判定之后，不影响压缩决策；本题为了让 force 匹配口径自洽，保留前缀）。
- 上游 `reduce_rate <= 0` 有「全保留」的旁路分支；本题把 `rate` 限定在开区间 `(0,1)`，0 与 1 都抛 `ValueError`。
- 上游按 `max_seq_len` 分 chunk 后逐 chunk 压缩再拼接；本题输入就是一个 chunk 的 token 串，不做分块。
