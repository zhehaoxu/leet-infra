def solve(messages, add_generation_prompt=False, continue_final_message=False):
    """渲染本题固定 ChatML，并返回 UTF-8 字节级标注。仅需 Python 标准库。

    中文契约：
    - messages 是长度 0..20 的列表；每项必须是恰含 role/content 的 dict，
      role 仅可为 system/user/assistant；记录结构或角色不合法抛 ValueError。
      content 保证是长度 0..1000 的 Unicode 标量字符串（无代理码点）。
      两个 flag 为 bool；其余未要求报错的输入保证合法。
    - 每条消息严格拼成 '<|im_start|>'+role+'\n'+content+'<|im_end|>\n'。
      不插入隐式 system，不 trim、不转义、不规范化、不解释 Jinja 或正文标记。
    - continue_final_message=True 时仅最后一条省略 '<|im_end|>\n'，
      与最后角色无关，保留正文所有尾部空白；空 messages 时抛 ValueError。
    - add_generation_prompt=True 时，在全部已闭合消息后追加
      '<|im_start|>assistant\n'；空 messages 也允许。两 flag 同开抛 ValueError。
    - 返回 dict 恰含 text/input_bytes/role_ids/assistant_mask/content_spans：
      text 为完整字符串；input_bytes 为 list(text.encode('utf-8'))，不是模型 token ID。
      role_ids 是等字节长度的 Python int 列表，正文 system=0/user=1/assistant=2，
      一切语法、头尾标记及新增生成头为 -1；assistant_mask 是等长 bool 列表，
      仅 assistant 正文字节为 True，结束标记不监督。
      content_spans 为每条原消息的 (start, end) tuple 列表，UTF-8 字节半开区间，
      空正文 start=end；新增生成头不添加 span。不得修改 messages 或内部 dict。

    English contract:
    - messages is a list of 0..20 records. Each must be a dict with exactly
      role/content; role must be system, user, or assistant. Invalid record
      structure or role raises ValueError. content is guaranteed to be a
      string of 0..1000 Unicode scalars, with no surrogate code points.
      Both flags are bool; all other inputs not specified as errors are valid.
    - Render each record exactly as '<|im_start|>'+role+'\n'+content+'<|im_end|>\n'.
      No implicit system message, trimming, escaping, normalization, Jinja
      evaluation, or interpretation of marker-like text inside content.
    - Continuation removes only the last record's '<|im_end|>\n', for any role,
      preserving every trailing content character. Empty continuation raises
      ValueError. Generation appends '<|im_start|>assistant\n' after all closed
      records, even for an empty list. Both flags true raises ValueError.
    - Return exactly five dict keys: text (str), input_bytes (list of Python
      ints equal to list(text.encode('utf-8')), NOT model token IDs), role_ids
      (one Python int per byte: content system=0/user=1/assistant=2; syntax,
      headers, terminators and the new generation header=-1), assistant_mask
      (one bool per byte, true only for assistant content, never terminators),
      and content_spans (one (start, end) tuple per original record, a half-open
      UTF-8 byte interval; empty content has start=end). A generation header
      adds no span. Do not mutate the list or its dictionaries. No tokenizer,
      BPE, or external library is involved.
    """
    raise NotImplementedError
