def solve(text: str) -> list[tuple[str, int, int]]:
    r"""中文契约：
    输入 text 为 0..300 个合法 Unicode 标量，限定 Unicode 15.0 / Python 3.12
    已定义字符；不处理后来版本新增字符。不规范化、不补前缀空格、不做 BPE。
    返回有序列表 (byte_encoded_piece, start, end)，偏移是原文 Python 码点
    下标，左闭右开；所有非空片段连续覆盖原文。空串返回 []。

    必须按以下 GPT-2 正则的顺序、大小写敏感地匹配：
    's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    L/N 分别为 unicodedata.category 以 L/N 开头，包括 Nd/Nl/No；组合标记
    不属于 L。可选前导空格只允许 U+0020。缩写只识别小写，且仅在当前
    扫描位置优先；标点串可吞掉后面的撇号，不在串内重新启动缩写匹配。
    空白精确为 U+0009–000D、0020、0085、00A0、1680、2000–200A、
    2028–2029、202F、205F、3000。不可用 isspace()，001C–001F 不是空白。
    走空白分支时，尾部整段保留；内部长度 >1 时先匹配除最后一个以外
    的空白；内部单个空白走最后一支。

    每段先 UTF-8 编码：字节 33..126、161..172、174..255 映射到同值
    Unicode 字符；其余字节按数值升序映射到 U+0100 起的连续字符。
    仅用 Python 标准库、unicodedata.category 和手写有序 scanner；禁止
    导入 tokenizers / regex，不要把编码后长度或字节长度当作原文偏移。

    English contract:
    text contains 0..300 valid Unicode scalars defined in Unicode 15.0 (Python
    3.12); newly assigned characters are out of scope. No normalization, prefix
    space, or BPE merges. Return ordered (byte_encoded_piece, start, end) tuples
    with half-open ORIGINAL Python code-point offsets, not byte offsets.
    Nonempty pieces partition the input; the empty string returns [].

    Follow the exact case-sensitive ordered GPT-2 pattern printed above.
    L and N mean Unicode categories starting with L and N (Nd/Nl/No included).
    Combining marks are neither letters nor numbers. Only ASCII U+0020 is an
    optional leading space. Lowercase contractions have priority only at the
    current position; a punctuation run may consume a later apostrophe.
    Whitespace is exactly U+0009–000D, 0020, 0085, 00A0, 1680, 2000–200A,
    2028–2029, 202F, 205F, 3000, NOT str.isspace(); U+001C–001F are not whitespace.
    At a whitespace branch, take the entire trailing run; internally take all
    but the last whitespace if the run has length >1, otherwise take one.

    UTF-8 encode each matched piece, then map bytes 33..126, 161..172, 174..255
    to the same-valued character. Map remaining bytes, in ascending order, to
    consecutive characters starting at U+0100. Use only the standard library,
    unicodedata.category, and a handwritten ordered scanner. No tokenizers or
    third-party regex imports. Encoded-piece length is not source length.
    """
    raise NotImplementedError
