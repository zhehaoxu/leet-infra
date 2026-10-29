def solve(pattern, vocab, prefix):
    """正则约束解码：给出词表里每个候选 token 的合法性掩码。

    pattern : 文法字符串。支持
                字面字符
                "[...]"   字符类，类内每个字符都是字面字符（不支持 "^" 与 "-"）
                连接      相邻单元直接拼接
                "A|B"     选择
                "X*" "X+" "X?"   重复 0+ / 1+ / 0-1 次
                "(X)"     分组
              不会出现空分支（如 "a|"）、空组 "()"、转义字符、"."、"^"、"$"、"{m,n}"。
    vocab   : 候选 token 列表，每个 token 是非空字符串；其字符可以不在 pattern 的字母表里。
    prefix  : 已经生成的文本，可以为空。

    记 L(pattern) 为被 pattern 完整匹配的全部字符串。

    返回三元组 (mask, accept, complete)：
      mask    : list[bool]，长度 len(vocab)；mask[i] 为真当且仅当存在（可为空的）
                字符串 w 使 prefix + vocab[i] + w 属于 L(pattern)
      accept  : bool，prefix 本身是否属于 L(pattern)，即"现在就可以结束"
      complete: list[bool]，长度 len(vocab)；complete[i] 为真当且仅当
                prefix + vocab[i] 属于 L(pattern)；恒有 complete[i] 蕴含 mask[i]

    注意可空模式：pattern="a*" 时 accept 对空前缀为真（空串也是完整匹配），
    但这不影响 mask —— mask 只回答"还能不能继续"。
    """
    # TODO
    raise NotImplementedError
