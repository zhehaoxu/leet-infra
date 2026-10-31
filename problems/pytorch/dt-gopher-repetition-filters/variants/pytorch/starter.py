import re
from collections import Counter


def solve(text, thresholds=None):
    """逐条转写 Gopher（Rae et al., 2112.11446 附录 A.1.1）MassiveWeb 的
    Repetition Removal：对文档计算 13 个重复度量（Table A1），
    任一度量严格大于阈值（"surpasses"）即整文档拒绝，检查顺序 = Table A1 行序。

    度量与计算口径（题面钉死，均精确匹配、区分大小写与标点）：
      行（line）   = text.splitlines() 的每一行，原样（不 strip）；
      段落（para） = splitlines 后按「空行」（strip 后为空的行）分隔的
                     极大连续非空行块；段字符串 = "\\n".join(块内行)，
                     段字符数 = sum(len(line))（不含分隔空行与段内换行）；
      词（word）   = text 中由空白分隔的最大非空白连续段（记录字符区间 [s, e)）；
      word n-gram  = 连续 n 个词，出现实例的覆盖区间 = [首词 s, 末词 e)；
      重复（duplicate）= 出现次数 >= 2（对 top n-gram 则是出现次数最多者）。

      duplicate_line_fraction        = 重复行实例数 / 总行数
      duplicate_line_character_fraction   = 重复行实例的字符和 / 全部行字符和
      duplicate_paragraph_fraction   = 重复段实例数 / 总段数
      duplicate_paragraph_character_fraction = 重复段实例字符和 / 全部段字符和
      top_n_gram_character_fraction (n=2,3,4)：出现次数最多的 n-gram，
            并列取文档中首次出现位置最早者；度量 = 其全部出现实例
            的覆盖区间并集长度 / len(text)；
      duplicate_n_gram_character_fraction (n=5..10)：全部出现次数 >= 2 的
            n-gram 的全部出现实例覆盖区间并集长度 / len(text)
            （重叠区间合并，字符不重复计——论文 "taking care not to count
            characters that occur in overlapping n-grams more than once"）。

      行/段分数分母为 0、len(text) == 0 或词数 < n 时该度量记 0.0。
      所有覆盖 fraction 的分母 = len(text)（含空白与换行）；覆盖只计词字符。

    参数
    ----
    text       : str
    thresholds : None 或 dict；None 用 Table A1 全部默认；给出时按键覆盖
                 （键必须是 13 个合法度量名，值必须是 0..1 的非 bool 数字）。

    返回
    ----
    dict：
      accepted      bool
      reject_reason None | 首个超阈值度量名（Table A1 行序）
      metrics       dict（13 个度量名 -> float）
    """
    raise NotImplementedError
