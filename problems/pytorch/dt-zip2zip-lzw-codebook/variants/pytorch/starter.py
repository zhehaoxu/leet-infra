def solve(ids, initial_vocab_size, max_codebook_size, max_subtokens,
          pad_token_id, disabled_ids=None, truncation=False, max_length=None,
          padding="do_not_pad"):
    """zip2zip LZW 动态码本编码（2610.13605，CPU 参考口径）。

    ids：基词 id 序列，全部 < initial_vocab_size 且非 bool；
    LZW 编码：buffer 逐 token 扩展，扩出未知短语时
      ① 码本未满（next_id < initial_vocab_size + max_codebook_size）先把新短语
         以 next_id 入码本并 next_id += 1；
      ② 弹出末 token、发射 buffer（单 token 发基词 id，多 token 发码本超词 id）；
      ③ buffer 重置为 [当前 id]；短语达到 max_subtokens 时直接发射并清空；
    disabled_ids 里的 id 不参与合并：先发射 buffer 残余，再单独发射该 id；
    truncation=True 时发射上限 max_length 且输入提前终止（码本仍照常生长）；
    padding="max_length" 时左填充 pad_token_id 到 max_length（压缩后更长则报错）。
    返回 dict：compressed_ids、attention_mask（id != pad）、
      codebook（{超词 id: 短语}，按超词 id 升序）。
    非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
