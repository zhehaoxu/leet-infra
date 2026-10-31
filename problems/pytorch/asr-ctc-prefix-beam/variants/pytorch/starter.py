def solve(probs, beam_width=3, blank_id=0, k=None):
    """CTC prefix beam search 账（Graves 1504.08129 口径，线性概率域、无 LM）。

    输入
    ----
    probs      : [T, V] 概率矩阵（线性域，每帧对全部 V 类展开）
    beam_width : 每帧剪枝后保留的前缀数（≥1）
    blank_id   : CTC blank 类 id
    k          : 最终返回的假设数；None 时取 beam_width

    记账（题面钉死）：
      每个前缀 s 维护 (p_b, p_nb)：以 blank 结尾 / 以非 blank 结尾的路径概率和。
      逐帧扩展：
        - blank 类：p_b(s) += (p_b(s) + p_nb(s)) * P(blank)
        - 非 blank 类 c：
          * s 末尾恰为 c（重复）：加新字符的路径只能来自 blank 结尾
            （p_nb(s+c) += p_b(s) * P(c)）；重复延续不改变前缀
            （p_nb(s) += p_nb(s) * P(c)）
          * 否则：p_nb(s+c) += (p_b(s) + p_nb(s)) * P(c)
      每帧后按总概率 p_b + p_nb 降序剪枝到 beam_width；
      **并列按前缀字典序升序**（稳定可复现）。

    返回
    ----
    dict：
      hypotheses  list[{"prefix": list[int], "prob": float}]
                  按总概率降序（并列按前缀字典序升序）取前 k 个
    """
    raise NotImplementedError
