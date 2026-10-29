# Special-Token Templates for Single and Paired Sequences

## Task

Apply a BERT or RoBERTa special-token template to already-tokenized ID lists, producing aligned type IDs, special-position masks and sequence ownership. Operate on lists only: no tokenizer calls, truncation, padding or input mutation.

## Inputs and templates

- `ids_a`: a Python list of length 0..30.
- `ids_b`: `None` or a Python list of length 0..30. **Only `None` means no B; an empty list still means a pair.**
- All list IDs, `cls_id` and `sep_id` are integers satisfying $0\le x\le 10^6$. Any collisions are allowed, including `cls_id == sep_id`.
- `style` must be `"bert"` or `"roberta"`; otherwise raise `ValueError`, even for empty input or disabled insertion.
- `add_special_tokens` is bool, default `True`. All other inputs are guaranteed valid.

When insertion is enabled, `[CLS]` uses `cls_id` and `[SEP]` uses `sep_id`:

| Style | Single: B is None | Pair: B is not None, including an empty list |
| --- | --- | --- |
| BERT | `[CLS] A [SEP]` | `[CLS] A [SEP] B [SEP]` |
| RoBERTa | `[CLS] A [SEP]` | `[CLS] A [SEP] [SEP] B [SEP]` |

Without insertion, output A followed by B, or just A when B is `None`.

## Return value

Return a dictionary with exactly these four keys, each containing an equally long **Python list**:

| Key | Rule |
| --- | --- |
| `input_ids` | IDs produced by the template. |
| `token_type_ids` | BERT: integer 0 for CLS, A and A's final SEP; integer 1 for B and B's final SEP. RoBERTa: integer 0 everywhere. Without insertion, BERT still assigns type 1 to B. |
| `special_tokens_mask` | Integer 1 **only at positions inserted by this operation**, integer 0 at every original position. Never infer this from numeric IDs. All zeros without insertion. |
| `sequence_ids` | Integer 0 for original A positions, integer 1 for original B positions, `None` for inserted positions. Type IDs and sequence ownership are different fields. |

An empty B still receives its final SEP. For BERT that SEP has type 1 but sequence ID `None`.

## Executed examples

### 1. BERT pair

```text
ids_a = [11, 12]
ids_b = [21]
style = "bert"
cls_id = 101
sep_id = 102
add_special_tokens = True

Output:
{
  'input_ids': [101, 11, 12, 102, 21, 102],
  'token_type_ids': [0, 0, 0, 0, 1, 1],
  'special_tokens_mask': [1, 0, 0, 1, 0, 1],
  'sequence_ids': [None, 0, 0, None, 1, None]
}
```

### 2. RoBERTa pair

```text
ids_a = [11, 12]
ids_b = [21]
style = "roberta"
cls_id = 0
sep_id = 2
add_special_tokens = True

Output:
{
  'input_ids': [0, 11, 12, 2, 2, 21, 2],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0],
  'special_tokens_mask': [1, 0, 0, 1, 1, 0, 1],
  'sequence_ids': [None, 0, 0, None, None, 1, None]
}
```

### 3. An empty B is not a single sequence

```text
ids_a = [11]
ids_b = []
style = "roberta"
cls_id = 0
sep_id = 2
add_special_tokens = True

Output:
{
  'input_ids': [0, 11, 2, 2, 2],
  'token_type_ids': [0, 0, 0, 0, 0],
  'special_tokens_mask': [1, 0, 1, 1, 1],
  'sequence_ids': [None, 0, None, None, None]
}
```

### 4. Disabled insertion with ID collisions

```text
ids_a = [101, 102]
ids_b = [102]
style = "bert"
cls_id = 101
sep_id = 102
add_special_tokens = False

Output:
{
  'input_ids': [101, 102, 102],
  'token_type_ids': [0, 0, 1],
  'special_tokens_mask': [0, 0, 0],
  'sequence_ids': [0, 0, 1]
}
```

## Semantic boundary

This exercise chooses the native backend convention that preserves an explicitly empty B. Some high-level single-item wrappers treat an empty B as absent. That does not change this exercise's empty-list rule, and we do not claim that every Hugging Face API handles empty pairs identically.
