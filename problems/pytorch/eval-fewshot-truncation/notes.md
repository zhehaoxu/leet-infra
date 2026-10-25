# 详解：few-shot 提示的预算截断

## 出处
- Wang, Prasad, Weber, Watcharawanotai, Clark, Bowman, Gurevych, *How does in-context learning work? A framework for understanding the successes and failures of LLMs*（arXiv:2201.11903）。该文把 few-shot 演示的排布拆成 demonstration selection / ordering / formatting 三个自由度，并明确给出"在长度预算下按顺序**截断**演示（truncation）"这一处理：预算不足时后面的演示整体丢弃，而不是重排或挑选。
- 工程实现：EleutherAI `lm-evaluation-harness` 的模板拼装（`TemplateLM` 先算 `fewshot` 再与 doc/目标文本拼接，超长就报 `context length exceeded`）、OpenCompass 的 `max_num_tokens` 截断、以及各家 eval script 里的 `--max-shots` + tokenizer 计数。
- 本题与 `retrieval-rag-chunk`（把文档切成有重叠的定长块）不同：那是**单文档内部**的长度切分，本题是**多条完整演示之间**的取舍，且顺序固定、不许挑。

## 形式化
令 \(\tau(s)\) 为玩具分词器的 token 数，\(w_i = \tau(\text{demo}_i)\)，\(b = \tau(\text{query}) + \tau(\text{tail})\)。设
\[ m = \max\Big\{\,m' \in \{0,\dots,M\} : b + \sum_{i < m'} w_i \le budget\,\Big\} \]
则 `kept = [0, 1, ..., m-1]`、\(\text{final\_len} = b + \sum_{i<m} w_i\)。注意两点结构性事实：
1. **解一定是前缀**：因为 \(w_i \ge 0\)，"是否收下第 \(i\) 条"只取决于前 \(i\) 条的累计成本，所以贪心的输出必然形如 \([0..m-1]\)。这也意味着可以二分 \(m\)（前缀和单调不减），但逐条循环更贴合真实框架的写法。
2. **保留量是被"预留"出来的**：\(b\) 先扣，等价于给 query+tail 留了不可动用的额度。真实评测里这一步常被写成"预算减去模板长度"，忘了减就表现为**尾巴被裁**，模型看不到 `"Answer:"`，输出格式与正则抽取一起崩——这类静默失败比报错更难查。

## 玩具分词器的规则与陷阱
`tokenize` 的规则：空白切词 → 命中 `VOCAB` 的词值 1 个 token → 未命中的词**逐字符**展开（每字符一个 `<unk>`，id 0）。
- 大小写敏感：`"here"` 在词表里值 1，`"HERE"` 不在词表就值 4。示例里 `"the dog is very nice"` 之所以是 8 而不是 5，就是因为 `very` 被拆成 4 个字符 token。
- 因为切分只看空白，所以 \(\tau(a + b)\)（\(+\) 表示用单个空格连接）恰好等于 \(\tau(a) + \tau(b)\)，"逐条累加"与"整串一次分词"**恰好相等**，`final_len` 两种算法一致——判题里专门有这条断言。
- **换成真实 BPE 就不再成立**：`byte-level BPE` 会在拼接边界产生合并/分裂，\(\tau(a)+\tau(b) \ne \tau(a\,\Vert\,b)\) 常有 1~几个 token 的出入。生产实现要么整串一次编码，要么按 chat template 的 `add_special_tokens` 精确算，不能靠"各部分长度相加"估预算——这是本玩具设定最主要的"不可外推"处。

## 核心与易错点
1. **硬截断，不是背包**：见到第 \(i\) 条放不下就 `break`。写成"跳过这条继续看下一条"（或按长度排序后装箱）会让 `kept` 出现非前缀下标，是最典型的错误；判题用「长演示后面跟着 1-token 演示」的构造专门抓它。
2. **边界用 \(\le\)**：恰好用满预算应保留该条。写成 `<` 会少一条演示，few-shot 数量随机波动。
3. **空集与零成本**：`demos = []` 时返回空张量；全是空白串的演示成本 0，必须能被保留（`+0 <= budget` 成立）。别用"成本为 0 就跳过"的想当然写法。
4. **超支如实返回**：\(b > budget\) 时返回的 `final_len` 大于 `budget`——这不是错误，而是"模板本身放不下"的真实信号；把它 clip 成 `budget` 会骗过下游的长度检查。
5. **返回类型**：下标是整数语义，用 int64 张量（空时也要是 `torch.long` 的空张量，别退化成 float）；长度是普通整数，不是张量。
6. **顺序即语义**：演示顺序影响 in-context learning 的表现（锚定效应、多数标签偏置），所以"固定顺序 + 前缀截断"是指标口径的一部分，不能为了多塞几条而重排。

## 对拍口径
参考解：预留 \(b\) → 逐条 `len(tokenize(demo))` 累加 → 首次超预算即 `break`。判题 `_oracle` 另起一路：自己手写**逐字符扫描**的空白切分（不用 `str.split`）计算 token 数，主循环用 `while` 加显式越界判断，返回普通 list。断言包含：样例值、预算极宽（全保留）、预算为 2（一条不放）、预算恰好卡边界（\(\le\) 语义）、硬截断不回头捡、tail 超支仍保留、空演示集、OOV 字符爆炸，以及把 `kept` 真拼回 prompt 后重新分词核对 `final_len`；另有 25 组随机演示/预算全量对拍。整数与张量比较用 `torch.equal` / `==`，不涉及浮点。

## 复杂度
每条演示一次线性扫描，总计 \(O\!\left(\sum_i |\text{demo}_i| + |query| + |tail|\right)\) 时间、\(O(m)\) 输出空间；与前缀和 + 二分的写法同阶。

## 实践提醒
真实框架里预算还要减去**生成长度**（`max_new_tokens`）与模板 special tokens；few-shot 数量本身是超参，同一模型同一数据集，截断到 5-shot 与 8-shot 可以差好几个点，因此报数必须写清预算与截断策略。
