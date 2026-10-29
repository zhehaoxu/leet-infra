def solve(
    text: str, added_tokens: list[dict], split_special_tokens: bool = False
) -> list[tuple[str, int, int, int]]:
    """中文契约：
    text 为长度 0..300 的 ASCII 字符串。added_tokens 是长度 0..30 的注册
    列表，每项恰含 content / single_word / special；content 为长度 1..30
    的非空、无空白 ASCII 字符串，两个 flag 均为 bool。content 为空或重复
    必须抛 ValueError（即使原文为空或重复项是被跳过的 special）；其他
    输入保证合法。不修改 text、列表及字典。

    返回 (原文 piece, start, end, added_id) 的列表，偏移为原文 Python
    字符下标、左闭右开；added_id 是原注册列表下标，普通文本为 -1。
    split_special_tokens=True 时，先移除 special=True 的候选，但不能
    重排其余 ID。默认 False。特殊 token 没有额外优先权。
    在剩余候选中先产生非重叠原始匹配：左起最早，同一起点取最长；
    不是全局最长，也不是先检查边界再选最长。选中区间后扫描从其末尾
    继续。只有随后才检查 single_word：紧邻左右字符若存在，均不能是
    ASCII A-Z/a-z/0-9/_。失败的长匹配归普通原文，不能回退短 pattern，
    不能重新扫描其内部。匹配内容本身不必以字母数字开头或结尾。

    每个成功 added token 单独输出；连续未匹配原文（含失败候选）全部
    合并为普通片段；不要输出空片段；空原文返回 []，但仍须校验注册表。
    大小写敏感，identity normalization，lstrip=rstrip=False，所有 pattern
    同一阶段，无数字 priority，不做 BPE，只用 Python 标准库。
    这是受限 ASCII 题意；special 预先移除规则不同于 tokenizers 0.23.3
    的候选生成后跳过规则，不是完整 AddedVocabulary 实现。

    English contract:
    text is ASCII, length 0..300. added_tokens has 0..30 registration dictionaries,
    each with exactly content, single_word, and special. content is nonempty
    whitespace-free ASCII, length 1..30; both flags are bool. Empty or duplicate
    contents raise ValueError, even for empty text or skipped special tokens.
    All other inputs are valid. Do not mutate the input list or dictionaries.

    Return (original_piece, start, end, added_id) tuples with half-open source
    Python character offsets. added_id is the ORIGINAL registration index;
    ordinary text uses -1. With split_special_tokens=True, remove special=True
    patterns BEFORE candidate selection, preserving remaining IDs; default is
    False. Special tokens have no inherent priority.
    Select nonoverlapping raw occurrences leftmost first, longest at equal
    start, advancing to the end of each selected interval. This is not globally
    longest matching. Only THEN apply single_word: immediate neighbors, if any,
    must not be ASCII A-Z/a-z/0-9/_. A rejected longest candidate becomes ordinary
    text: never fall back to a shorter pattern or rescan inside its interval.
    The content itself need not start or end with a word character.

    Emit each accepted added token separately. Merge all contiguous unmatched
    source text, including rejected candidates, into ordinary pieces. Emit no
    empty pieces. Empty text returns [], but registration validation still runs.
    Matching is case-sensitive; normalization is identity; lstrip=rstrip=False;
    all patterns share one stage; there is no numeric priority and no BPE.
    Standard library only. This ASCII subset is not full AddedVocabulary:
    pre-removing special patterns deliberately differs from tokenizers 0.23.3,
    which skips them after raw candidate selection.
    """
    raise NotImplementedError
