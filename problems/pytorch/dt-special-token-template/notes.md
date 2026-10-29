# 题解：按位置来源拼接模板

## 参考实现

先验证 `style`，再根据 `ids_b is not None` 决定是否为句对。不要写 `if ids_b`，否则空 B 会丢失自己的分隔符。

把输出拆成少量 segment，每段携带 `(IDs, type_id, special_flag, sequence_id)`：原 A 为 `(0,0,0)`，原 B 在 BERT 下为 `(1,0,1)`、RoBERTa 下为 `(0,0,1)`；插入段的 special flag 为 1、sequence ID 为 `None`。按模板顺序逐段扩展四个结果列表。

关闭插入只是省去插入段，不改变原始段的 type 和 sequence 归属。输入 ID 即使与 CLS/SEP 相等，仍是原始段中的普通位置；不能用集合成员判断特殊身份。`extend` 到新列表，不在输入上追加或原地相加。

## 正确性

每个原 A/B 位置按原顺序且恰好一次进入对应段，每个模板插入位置也恰好一次出现。因此 `input_ids` 满足模板。

段的四种属性在扩展时使用相同长度，同一输出下标始终对应同一个来源事件。普通位置与插入事件互不混淆，数值碰撞不会改变标记。B 的存在性与长度分开判断，因此空 B 仍能触发句对插入。以上也覆盖空 A、两侧均空以及完全关闭插入的情况。

令 $n_A,n_B$ 为两侧长度，缺失 B 的长度视为零，$s$ 为插入数：关闭时为 0；单句为 2；BERT 句对为 3；RoBERTa 句对为 4。时间和输出空间均为 $O(n_A+n_B+s)$，除返回值外只需常数数量的段描述。

## 独立测试

参考实现是 segment 拼接；oracle 是逐位置发射的有限状态机，状态依次覆盖开始、A、A 结尾、桥接、B、B 结尾。它不调用参考实现，不共享模板拼接助手。测试比较完整字典、键集合、list/元素类型，并检查输入不变。

`--selftest dt-special-token-template --mode submit` 已通过 **12/12**：

- 两种非空句对，以及默认插入参数；
- 空 A、None B、空 B、双空；
- 关闭插入、ID 碰撞和 CLS/SEP 相同；
- 非对称长度、最大两侧 30 个 ID、边界 ID；
- 小规模穷举与固定种子 458 的 150 组随机输入；
- 非法风格在空输入/关闭插入前仍报错；
- 重复调用及 A/B 共用同一列表时不修改输入。

## 原生冻结依据与实跑

代码与独立测试先用 `.venv-backend/bin/python` 通过 submit 自测，然后才用 `.venv-eval/bin/python` 执行文档样例与原生核验。交付的 starter、solution、tests 都不依赖 tokenizers/transformers，也不依赖冻结环境。

本地版本为 transformers **5.20.2**、tokenizers **0.23.3**。已回源：

- `transformers/models/bert/tokenization_bert.py:128–135`：`TemplateProcessing` 的 single/pair 模板与类型配置。
- `transformers/models/roberta/tokenization_roberta.py:170–175`：`RobertaProcessing` 的构造。
- `tokenizers/__init__.pyi:279–402,931–967`：Encoding 的归属字段与 `encode` 的可选 pair。

原生测试使用 `WordLevel` 词表 `{'a':11,'c':12,'b':21,'[UNK]':99}` 和 `WhitespaceSplit`，从而把文本分词因素隔离掉。BERT 使用源码中的 `TemplateProcessing`，CLS/SEP 为 101/102；RoBERTa 使用 `RobertaProcessing(('</s>',2),('<s>',0), trim_offsets=False, add_prefix_space=False)`。

对 A 为 `''/'a'/'a c'`、B 为 `None/''/'b'`、两种 style、插入开关两种状态，**36 组**原生结果的四字段都与参考解一致。字面量 fixtures 包括普通句对和 `backend.encode('a','')`：

| 字段 | BERT 空 B | RoBERTa 空 B |
| --- | --- | --- |
| input_ids | `[101,11,102,102]` | `[0,11,2,2,2]` |
| token_type_ids | `[0,0,0,1]` | `[0,0,0,0,0]` |
| special_tokens_mask | `[1,0,1,1]` | `[1,0,1,1,1]` |
| sequence_ids | `[None,0,None,None]` | `[None,0,None,None,None]` |

description 的四个样例均实际运行后记录，包含关闭插入时的 ID 碰撞。

## API 边界，不夸大一致性

本地 `transformers/tokenization_utils_tokenizers.py:1047` 的单条 wrapper 按 `text_pair` 真值决定是否传 pair。使用小词表中 CLS=2、SEP=3、a=5 实跑：`BertTokenizer` 的单条 `tokenizer('a','')` 得到 `[2,5,3]`，但 `tokenizer.backend_tokenizer.encode('a','')` 得到 `[2,5,3,3]`。

本题明确选择后者：显式空 B 仍是句对。并非所有 HF API 都一致。题目只模拟模板字段，不覆盖分词、注册 added token、规范化、截断、padding 或 batch wrapper 行为。
