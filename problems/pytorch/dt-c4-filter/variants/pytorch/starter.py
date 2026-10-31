import hashlib
import re


def solve(text, badwords=None, seen_hashes=None,
          min_words_per_line=5, min_num_sentences=3, max_word_length=1000):
    """逐行转写 C4（HF datasets legacy c4_utils.py）clean_page 的行级清洗账，
    附上行级 md5 去重（上游 remove_duplicate_text 的单页内 + 跨页显式状态版）。

    参数
    ----
    text       : str           页面文本（按 splitlines 切行）
    badwords   : None 或 list[str]；None 表示不做 badword 过滤
    seen_hashes: None 或 list/set[str]，此前页面已保留行的 md5（跨页去重状态）
    min_words_per_line / min_num_sentences / max_word_length : 上游缺省 5 / 3 / 1000

    每行按下列顺序处理（行先 line.strip()）：
      1. 任一词 len > max_word_length → 丢行，计数 lines_with_too_long_word
      2. citation_regex = r"\\[\\d*\\]|\\[edit\\]|\\[citation needed\\]" 替换为空
      3. 不以 (".", "?", "!", '"') 任一结尾，或以 "..." 结尾 → 丢行 lines_no_endmark
      4. len(line.split()) < min_words_per_line → 丢行 lines_too_short
      5. "lorem ipsum" 在 line.lower() → 整页拒绝 filtered_page_loremipsum
      6. "javascript" 在 line.lower() → 丢行 lines_javascript
      7. "{" 在 line → 整页拒绝 filtered_page_squigglybracket
      8. _POLICY_SUBSTRINGS 任一在 line.lower() → 丢行 lines_policy
      9. badwords 给定时：正则 [^a-z](bw1|bw2|...)[^a-z] 在 line.lower() 上命中
         → 整页拒绝 filtered_page_badword（词首/词尾无锚点保护，上游原样）
     10. 去重：h = md5(line.strip().lower())；h 在 seen（初始 seen_hashes ∪ 本页已保留）
         → 丢行 lines_dedup；否则保留并入 seen
     11. 保留：valid_lines.append(line)；num_sentences += 1（punkt 的确定性代理：
         每个有效行记 1 句，题面钉死）

    整页拒绝（5/7/9）立即终止：valid_lines 为空，计数保留已发生的部分；
    处理完后 num_sentences < min_num_sentences → 拒绝 filtered_page_toofewsentences。

    返回
    ----
    dict：
      accepted      bool
      reject_reason None | 'filtered_page_loremipsum' | 'filtered_page_squigglybracket'
                    | 'filtered_page_badword' | 'filtered_page_toofewsentences'
      valid_lines   list[str]（被拒页为 []）
      clean_text    str（"\\n".join(valid_lines).strip()；被拒页为 ""）
      num_sentences int
      drop_counters dict（含上述计数键，未发生的键值为 0）
      new_hashes    list[str]  本页新保留行的 md5（按序）
    """
    raise NotImplementedError
