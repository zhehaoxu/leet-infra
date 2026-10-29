# 分词：UTF-8 字节与字符偏移对齐

## 任务

把 token 在原文 UTF-8 字节流中的区间转换为最小字符覆盖区间，并建立 token、显式预分词单元和字符查询之间的映射。这里的“字符”是 **Python Unicode 码点**，不是 UTF-16 单元，也不是 grapheme（字素簇）。不规范化文本、不添加前缀、不 trim 空白，不修改输入。

## 输入与合法性

- `text`：仅含合法 Unicode 标量的字符串，码点长度为 0..100。
- `word_spans`：0..30 个 `(start,end)` 组成的列表，每项是原文中的非空码点半开区间，按 start 排序且不重叠。允许空白 gaps。这些区间就是预分词单元，不要按空格重新切词；一个单元也可以包含空白。
- `byte_offsets`：长度为 0..200 的 `(a,b)` tuple 列表。
  - `(-1,-1)` 专指插入的特殊 token，可以出现在任何位置。
  - 其余项必须满足 $0\le a<b\le |\operatorname{UTF8}(text)|$，否则抛出 `ValueError`。空区间、逆序、越界、只有一个端点为 -1 都非法。
  - 普通区间允许端点落在字符内部，允许重复和重叠；忽略特殊 token 后，start 非递减，end 不要求递增。
  - 每个普通区间保证完全位于某个 `word_span` 对应的 UTF-8 字节范围内。
- `char_queries`：最多 100 个 Python 整数组成的列表，允许负数和越界索引。

除了普通字节区间的上述范围要求，其余输入保证合法；不要求检测排序或单元归属等前置条件。所有区间均为半开区间。

## 返回值

返回恰有以下四键的字典。各字段均为 **Python list**；字段内部的区间也必须用二元素 list，而不是 tuple。

1. **`offsets`**：每个普通 token 的最小码点覆盖区间。start 向下取到所属码点，end 向上取到码点边界；恰好在边界上的端点保持不变。不要对残缺 UTF-8 字节做 decode。特殊 token 输出 `[0,0]`。
2. **`word_ids`**：普通 token 所属 `word_spans` 的下标，即使只包含某字符的部分字节也属于该单元；特殊 token 为 `None`。
3. **`char_to_token`**：每个查询对应第一个覆盖该码点的 token 下标，按输入 token 顺序决定“第一个”。没有覆盖，包括负数、越界或 gap，输出 `None`。特殊 token **绝不覆盖字符 0**。
4. **`word_to_tokens`**：每个 word 对应包含其全部 token 的最小 token 下标半开范围。若相关下标集合为 $I_w$，结果为 $[\min I_w,\max I_w+1)$；没有 token 则为 `None`。该范围允许包含插入的特殊 token。

## 实跑样例

### 1. 一字符可对应多个 byte-level token

```text
text = 'é😊X'
byte_offsets = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7)]
word_spans = [(0,1), (1,2), (2,3)]
char_queries = [0,1,2]

输出：
{
  'offsets': [[0,1], [0,1], [1,2], [1,2], [1,2], [1,2], [2,3]],
  'word_ids': [0,0,1,1,1,1,2],
  'char_to_token': [0,2,6],
  'word_to_tokens': [[0,2], [2,6], [6,7]]
}
```

`é` 占 2 个 UTF-8 字节，U+1F60A 占 4 个，`X` 占 1 个。重复 offset 是合法结果，字符查询返回第一个命中。该输出还与本地 tokenizers 0.23.3 的完整 ByteLevel、256 字符词表、无 merges 流程实测一致。

### 2. 特殊 token、空白 gap 与 word 范围

```text
text = 'é X'
byte_offsets = [(-1,-1), (0,1), (-1,-1), (1,2), (3,4), (-1,-1)]
word_spans = [(0,1), (2,3)]
char_queries = [0,1,2,-1,3]

输出：
{
  'offsets': [[0,0], [0,1], [0,0], [0,1], [2,3], [0,0]],
  'word_ids': [None,0,None,0,1,None],
  'char_to_token': [1,None,4,None,None],
  'word_to_tokens': [[1,4], [4,5]]
}
```

word 0 的 token 下标是 1 和 3，因此范围为 `[1,4]`，中间包含特殊 token 2。字符 1 是空白 gap；字符 0 首次由 token 1 覆盖，不会误命中特殊 token 0。

## 范围限定

这是无规范化、无前缀、无 trim 的完整 ByteLevel 对齐思想的受限模型，不是单独 Rust `BytesToCharOffsetConverter` 的任意 span 接口。组合重音、ZWJ 等仍按独立码点计数；不做 UTF-16 或字素簇映射，也不要求实现 tokenizer 或 BPE。
