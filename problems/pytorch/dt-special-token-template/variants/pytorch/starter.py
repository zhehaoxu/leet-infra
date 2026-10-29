def solve(ids_a, ids_b, style, cls_id, sep_id, add_special_tokens=True):
    """单句/句对特殊 token 模板 / Special-token templates for sequences.

    中文契约：
    - ids_a 为长度 0..30 的 Python 列表；ids_b 为 None 或长度 0..30 的列表。
      所有输入 ID、cls_id、sep_id 都是 0..10^6 的整数，允许彼此相同。
      style 仅为 'bert' 或 'roberta'，否则 ValueError，即使不插入或输入为空。
      add_special_tokens 是 bool，默认 True；其余输入保证合法。不得修改输入。
    - 只有 ids_b is None 才是 single；空列表 B 仍是 pair。
      插入开启时，BERT single: [CLS] A [SEP]；pair: [CLS] A [SEP] B [SEP]。
      RoBERTa single: [CLS] A [SEP]；pair: [CLS] A [SEP] [SEP] B [SEP]。
      [CLS]/[SEP] 的 ID 分别为 cls_id/sep_id，不截断、不 padding。
    - 返回恰四键 dict，值均为等长 Python list：
      input_ids：模板输出 ID；
      token_type_ids：BERT 的 CLS、A、A 后 SEP 为 0，B、B 后 SEP 为 1；
          RoBERTa 全部为 0；
      special_tokens_mask：仅本次插入的位置为整数 1，原输入位置为整数 0，
          即使输入 ID 与 cls_id/sep_id 相同也为 0；
      sequence_ids：原 A 位置为 0，原 B 位置为 1，插入位置为 None。
    - 关闭插入时只输出 A+B（B 为 None 时仅 A）；BERT 的 B type 仍为 1，
      RoBERTa type 全 0；mask 全 0；sequence_ids 仍按来源为 0/1。

    English contract:
    - ids_a is a Python list of length 0..30; ids_b is None or a list of length
      0..30. All input IDs, cls_id and sep_id are integers in 0..10^6; collisions
      are allowed. style must be 'bert' or 'roberta', otherwise raise ValueError,
      even for empty inputs or disabled insertion. add_special_tokens is bool,
      default True. All other inputs are valid. Do not mutate inputs.
    - Only ids_b is None means single; an empty B list still means pair.
      With insertion: BERT single = [CLS] A [SEP], pair = [CLS] A [SEP] B [SEP];
      RoBERTa single = [CLS] A [SEP], pair = [CLS] A [SEP] [SEP] B [SEP].
      [CLS]/[SEP] use cls_id/sep_id. No truncation or padding.
    - Return a dict with exactly four equal-length Python list values:
      input_ids: output IDs;
      token_type_ids: BERT uses 0 for CLS, A and A's final SEP, and 1 for B and
          B's final SEP; RoBERTa uses 0 everywhere;
      special_tokens_mask: integer 1 only for inserted positions, integer 0 for
          every original position, even if its ID equals cls_id or sep_id;
      sequence_ids: 0 for original A, 1 for original B, None for inserted slots.
    - Without insertion, output only A+B (just A when B is None). BERT still
      assigns type 1 to B; RoBERTa types are all 0. All masks are 0 and sequence
      IDs remain 0/1 according to origin.
    """
    raise NotImplementedError
