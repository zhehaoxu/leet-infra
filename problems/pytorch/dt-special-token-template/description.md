# 分词：单句与句对特殊 Token 模板

## 任务

给定已经分词的 ID 列表，为单句或句对应用 BERT / RoBERTa 特殊 token 模板，并同时生成类型、特殊位置标记和序列归属。只处理列表，不调用 tokenizer，不截断、不 padding，也不修改输入。

## 输入与模板

- `ids_a`：长度为 0..30 的 Python 列表。
- `ids_b`：`None` 或长度为 0..30 的 Python 列表。**只有 `None` 表示没有 B；空列表仍表示句对。**
- 列表 ID、`cls_id`、`sep_id` 均为 $0\le x\le 10^6$ 的整数，允许任意数值碰撞，包括 `cls_id == sep_id`。
- `style` 只能为 `"bert"` 或 `"roberta"`；否则抛出 `ValueError`，即使输入为空或关闭插入也必须报错。
- `add_special_tokens` 为 bool，默认 `True`。其余输入保证合法。

开启插入时，`[CLS]` 使用 `cls_id`，`[SEP]` 使用 `sep_id`：

| 风格 | 单句：B 为 None | 句对：B 非 None，包括空列表 |
| --- | --- | --- |
| BERT | `[CLS] A [SEP]` | `[CLS] A [SEP] B [SEP]` |
| RoBERTa | `[CLS] A [SEP]` | `[CLS] A [SEP] [SEP] B [SEP]` |

关闭插入时，只按顺序输出 A 后接 B；B 为 `None` 时只输出 A。

## 返回值

返回恰有以下四键的字典，各值均为等长的 **Python list**：

| 键 | 规则 |
| --- | --- |
| `input_ids` | 模板得到的 ID 序列。 |
| `token_type_ids` | BERT：CLS、A、A 后的 SEP 为整数 0；B、B 后的 SEP 为整数 1。RoBERTa：全部为整数 0。关闭插入时，BERT 的 B 仍为 1。 |
| `special_tokens_mask` | **仅本次插入的位置**为整数 1，所有原始输入位置为整数 0；不得通过 ID 数值判断。关闭插入时全 0。 |
| `sequence_ids` | 原 A 的位置为整数 0，原 B 的位置为整数 1，插入位置为 `None`；不要把 type ID 当成序列归属。 |

空 B 的末尾 SEP 仍存在；在 BERT 中其 type 为 1，但其 sequence ID 为 `None`。

## 实跑样例

### 1. BERT 句对

```text
ids_a = [11, 12]
ids_b = [21]
style = "bert"
cls_id = 101
sep_id = 102
add_special_tokens = True

输出：
{
  'input_ids': [101, 11, 12, 102, 21, 102],
  'token_type_ids': [0, 0, 0, 0, 1, 1],
  'special_tokens_mask': [1, 0, 0, 1, 0, 1],
  'sequence_ids': [None, 0, 0, None, 1, None]
}
```

### 2. RoBERTa 句对

```text
ids_a = [11, 12]
ids_b = [21]
style = "roberta"
cls_id = 0
sep_id = 2
add_special_tokens = True

输出：
{
  'input_ids': [0, 11, 12, 2, 2, 21, 2],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0],
  'special_tokens_mask': [1, 0, 0, 1, 1, 0, 1],
  'sequence_ids': [None, 0, 0, None, None, 1, None]
}
```

### 3. 空 B 不是单句

```text
ids_a = [11]
ids_b = []
style = "roberta"
cls_id = 0
sep_id = 2
add_special_tokens = True

输出：
{
  'input_ids': [0, 11, 2, 2, 2],
  'token_type_ids': [0, 0, 0, 0, 0],
  'special_tokens_mask': [1, 0, 1, 1, 1],
  'sequence_ids': [None, 0, None, None, None]
}
```

### 4. 关闭插入且 ID 碰撞

```text
ids_a = [101, 102]
ids_b = [102]
style = "bert"
cls_id = 101
sep_id = 102
add_special_tokens = False

输出：
{
  'input_ids': [101, 102, 102],
  'token_type_ids': [0, 0, 1],
  'special_tokens_mask': [0, 0, 0],
  'sequence_ids': [0, 0, 1]
}
```

## 语义边界

题目选择原生 backend 保留显式空 B 的语义。某些高层单条调用 wrapper 会把空 B 当作不存在；不能据此把本题的空列表改成单句，也不声称所有 Hugging Face API 对空句对行为一致。
