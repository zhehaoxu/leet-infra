# 题解：UTF-8 边界二分与位置归属

## 参考算法

令原文码点数为 $L$，构造累积 UTF-8 边界数组 $P$：

$$
P_0=0,\qquad P_{i+1}=P_i+|\operatorname{UTF8}(text_i)|.
$$

每个 Unicode 标量占 1..4 字节，因此 $P$ 严格递增。对普通字节区间 $[a,b)$，先验证 $0\le a<b\le P_L$，再求：

$$
s=\operatorname{bisect\_right}(P,a)-1,\qquad
 e=\operatorname{bisect\_left}(P,b).
$$

结果为码点半开区间 $[s,e)$。这不需要 decode 残缺字节：start 位于某字符内部时落在该字符起点；end 在字符内部时上取整，在边界时保持原边界。

特殊项 `(-1,-1)` 必须先单独处理，生成 `[0,0]` 和 `None`。其他负数、空区间或越界都不能冒充特殊项。

普通 token 保证属于某一个显式 word，因此在 `word_spans` 的 start 上二分，取不大于 $s$ 的最后一个 start，即可得到 word 下标。不要按文本空格重新分词。按 token 顺序扫描时，记录每个 word 的首个 token 下标，并不断把末端更新为当前下标加一；中间 special 不会终止这个范围。

每个字符查询按 token 原顺序扫描，返回第一个非 special 且覆盖查询的 offset。负数、越界、没有 token 覆盖的字及 gaps 自然得到 `None`。

## 正确性

二分给出 $P_s\le a<P_{s+1}$ 和 $P_{e-1}<b\le P_e$。于是 $[P_s,P_e)$ 包含全部 $[a,b)$；删除任一端的覆盖字符都会漏掉至少一个 token 字节，因此 $[s,e)$ 是最小覆盖。

区间包含于某 word 的前置条件以及 word 的非重叠性保证归属唯一。对每个 word，首次命中的下标最小，末次命中的下标最大，因此记录得到的半开范围恰为全部该 word token 的最小包围范围。字符查询按下标从小到大搜索，所以重复、重叠、end 非单调均不会破坏 first-hit 规则。

## 复杂度

设 token、word、查询数分别为 $T,W,Q$。时间为

$$
O\bigl(L+W+T(1+\log(L+1)+\log(W+1))+Q(T+1)\bigr).
$$

包括结果的空间为 $O(L+T+W+Q)$。源码仅使用标准库 `bisect`，无 tokenizer 或张量库依赖。

## 独立 oracle 与测试

oracle 不使用二分或累积边界转换。它逐字符展开 `byte_owner` 与 `byte_word` 数组，逐 token 枚举其每个字节的 owner，用字符下标集合的最小值和最大值构造覆盖；查询直接查 owner 集合，word 范围则另行枚举所有匹配 token 下标。它不调用参考解。

先执行 `--selftest dt-byte-char-offsets --mode submit`，结果 **15/15**。覆盖中文与非 BMP 字符、组合重音/预组合重音、ZWJ/变体选择符、所有 1..4 字节字符内的非空子区间、跨字符覆盖、边界、重复与重叠、first hit、特殊项与字符 0、gaps/空文本、无 token 的 word、夹 special 的 word 范围、含空格的显式单元、100 码点/200 token/30 word/100 查询上限、固定种子 460 的 120 组随机输入、输入不变和非法字节范围。

## 原生冻结样例

独立测试通过后，使用 `.venv-eval/bin/python` 的 tokenizers **0.23.3** 执行完整原生流程，而不是用参考解生成期望值。最小复现配置如下，仅用于本地核验，不是运行本题的依赖：

```python
from tokenizers import Tokenizer, models, pre_tokenizers
alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
backend = Tokenizer(models.BPE(
    vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
backend.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False, use_regex=True)
encoding = backend.encode('é😊X')
```

词表恰有 256 个 byte-level 符号，无 merges，无 normalizer、无 offset-trimming postprocessor。预分词单元的原文码点区间为 `[(0,1),(1,2),(2,3)]`。原生输出冻结为：

```text
offsets        = [(0,1),(0,1),(1,2),(1,2),(1,2),(1,2),(2,3)]
word_ids       = [0,0,1,1,1,1,2]
char_to_token  = [0,2,6]             # 查询 0,1,2
word_to_tokens = [(0,2),(2,6),(6,7)]
```

把原生 tuple 范围转为 list 后，四字段与题目的逐字节区间输入完全一致；该结果以字面量冻结在 `check_native_bytelevel_fixture`，tests 不导入第三方库。description 两个样例随后均实跑记录；第二例的任意 special 插入是题目模型的独立示例，不冒称由上述无 postprocessor 的原生流程产生。

## 回源与限定：完整流程不等于独立 converter

已阅读本地 `tokenizers/__init__.pyi:279–402,931–967`，并在内存中读取同版 [tokenizers 0.23.3 源码包](https://files.pythonhosted.org/packages/e0/7c/2cabb2174e772636683008f2c5621949b645da7d303c596589e84516a184/tokenizers-0.23.3.tar.gz)，没有安装或解压到项目：

- `tokenizers/src/tokenizer/pre_tokenizer.rs:165–194`：先通过 normalized alignment 的 `convert_offsets` 回到原文范围，再进行 byte-to-char 转换；word 来自预分词 split 或显式 word 下标。
- 同文件 `267–300`：内部 `BytesToCharOffsetConverter` 的两端 map 查找。在 `é` 的字节内部，直接查 `(0,1)` 可得到 `(0,0)`，**并非本题要求的向上取整最小覆盖**。
- `tokenizers/src/tokenizer/encoding.rs:228–292`：word token 包围范围与字符查询的第一个半开区间命中。

因此本题是**完整 ByteLevel、无规范化对齐的受限模型**，不是对独立 Rust `BytesToCharOffsetConverter` 任意 span 行为的逐项复刻。即使没有 normalizer，ByteLevel 的字节符号表示仍携带到原文的 alignment；不能跳过这一层后宣称内部 converter 自己会扩张所有碎字节区间。

本题额外显式给出原文 byte spans、word spans 和 special 哨兵，允许满足契约的重复/重叠区间。冻结原生例证明指定配置下的对应关系，不证明任意 tokenizer、任意规范化或任意 wrapper 等价。这里不做 UTF-16、grapheme 映射，不做 NFC/NFD 转换、前缀空格或 offset trim。
