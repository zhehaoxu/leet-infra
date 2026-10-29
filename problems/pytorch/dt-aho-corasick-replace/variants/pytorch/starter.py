def solve(text, patterns, replacements):
    """中文契约：
    text 是长 0..500 的 Unicode 标量字符 str。patterns 是含 0..30 项的
    str 列表，每项长 1..30，允许重复；replacements 是等长的 str 列表，
    其中字符串可为空。任一 pattern 为空时抛 ValueError，其余输入保证合法。
    所有长度与偏移按 Python Unicode code point 计，不按字节或字素簇计。

    返回 Python tuple (rewritten, all_matches, selected)：rewritten 为 str，
    后两项为 list[tuple[int, int, int]]，每项为 (start, end, pattern_index)，
    使用原始 text 的零基半开区间 [start, end) 和 patterns 的零基下标。
    all_matches 含原文全部匹配（包括重叠、后缀和重复 pattern 的各自记录），
    按 (start, end, pattern_index) 升序排列。
    selected 从 all_matches 贪心选择互不重叠的匹配：初始消费位置为 0；
    每次在 start >= 消费位置的匹配中选 start 最小者，同 start 选最长者，
    再平局选 pattern_index 最小者；消费位置更新为该匹配的 end。
    selected 按选择顺序排列。将选中的原文片段一次性替换为对应 replacement，
    未覆盖原文原样保留，替换文本不再扫描。不做大小写转换或 Unicode 归一化。
    返回值只能使用上述 Python 内置类型，不修改任何输入。
    请用 trie、BFS failure links 和继承输出实现 Aho–Corasick 多模式检测。

    English contract:
    text is a str of 0..500 Unicode scalar characters. patterns is a list of
    0..30 nonempty strings of length 1..30; duplicates are allowed. replacements
    is an equally long list of strings, each possibly empty. Raise ValueError
    if any pattern is empty; all other inputs are guaranteed valid. Lengths and
    offsets count Python code points, not bytes or grapheme clusters.

    Return a Python tuple (rewritten, all_matches, selected): a str and two
    lists of Python (int, int, int) tuples (start, end, pattern_index). Offsets
    are zero-based half-open intervals in the original text; indices refer to
    patterns. all_matches includes every overlapping and suffix match, with
    separate entries for duplicate patterns, sorted by (start, end, index).
    For selected, start with consumed position 0. Among matches whose start is
    at least that position, choose the smallest start, then greatest length,
    then smallest pattern_index. Advance the consumed position to that end
    and repeat. Return selected in selection order. Replace selected spans
    once with their indexed replacements, preserve uncovered original text,
    and never scan replacement text. No case folding or Unicode normalization.
    Use only the stated Python built-in return types and do not mutate inputs.
    Implement detection with an Aho–Corasick trie, BFS failure links, and
    inherited outputs.
    """
    raise NotImplementedError
