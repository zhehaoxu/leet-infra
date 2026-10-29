def solve(text, byte_offsets, word_spans, char_queries):
    """UTF-8 字节与码点对齐 / UTF-8 byte-to-code-point alignment.

    中文契约：
    - text 是只含合法 Unicode 标量的 str，Python 码点长度 0..100；不做
      normalize、前缀添加或空白 trim。字符索引是 Python 码点，不是 UTF-16
      单元或 grapheme（字素簇）；所有区间均为半开区间。不得修改任何输入。
    - word_spans 是 0..30 个 (start,end) 组成的列表：码点区间非空、位于
      text 内、按 start 排序且互不重叠。允许空白 gaps；这是显式预分词单元，
      不能按空格重新分词，单元也可含空白。
    - byte_offsets 是长度 0..200 的 tuple(a,b) 列表。(-1,-1) 是插入 special。
      其余项必须满足 0 <= a < b <= len(text.encode('utf-8'))，否则 ValueError
      （包括空区间、逆序、越界和只出现一个 -1）。允许端点在字符内部，允许
      重复和重叠；忽略 special 后 start 非递减。每个普通 byte 区间保证完全
      位于一个 word_span 对应的 UTF-8 区间内。其他输入保证合法。
    - char_queries 是最多 100 个 Python 整数组成的列表，可负或越界。
    - 返回恰四键 dict，所有容器均为 Python list，区间也用 [start,end]：
      offsets：每个普通 token 的最小码点覆盖区间；起点向下取所在码点，
          终点向上取边界。不能 decode 残缺字节。special 输出 [0,0]。
      word_ids：普通 token 所属 word_spans 下标，部分字节也归此 word；
          special 为 None。
      char_to_token：与查询等长，每项为第一个覆盖该码点的 token 下标；
          未覆盖（含负数或越界）为 None。special 绝不覆盖字符 0。
      word_to_tokens：与 word_spans 等长，含该 word 全部 token 的最小
          token 下标半开范围 [min_index,max_index+1]，没有 token 为 None；
          该范围允许包含插入的 special。

    English contract:
    - text is a str of valid Unicode scalar values, 0..100 Python code points.
      No normalization, prefix insertion or whitespace trimming. Character indices
      count Python code points, not UTF-16 units or grapheme clusters. All ranges
      are half-open. Do not mutate any input.
    - word_spans is a list of 0..30 nonempty (start,end) code-point spans inside
      text, sorted by start and nonoverlapping. Whitespace gaps are allowed.
      These are explicit pre-tokenization units; do not split again on spaces.
      A unit itself may contain whitespace.
    - byte_offsets is a list of 0..200 tuples (a,b). (-1,-1) marks an inserted
      special. Every other item must satisfy 0 <= a < b <= len(text.encode('utf-8'));
      otherwise raise ValueError, including empty/reversed/out-of-bounds intervals
      and a lone -1. Endpoints may lie inside characters; duplicates and overlaps
      are allowed. Ignoring specials, starts are nondecreasing. Each ordinary byte
      span is guaranteed to lie entirely in one word_span's UTF-8 range. All other
      inputs are valid.
    - char_queries is a list of at most 100 Python integers, possibly negative or
      out of bounds.
    - Return a dict with exactly four keys. All containers, including [start,end]
      ranges, are Python lists:
      offsets: minimal code-point cover per ordinary token; round the start down
          to its owning code point and the end up to a boundary. Never decode
          partial bytes. A special has [0,0].
      word_ids: index of the supplied word span owning each ordinary token,
          including partial-character bytes; None for specials.
      char_to_token: one entry per query, the first token index covering that
          code point, or None if uncovered/negative/out of bounds. Specials never
          cover character 0.
      word_to_tokens: one entry per word, the minimal half-open token-index range
          [min_index,max_index+1] containing all its tokens, or None if it has none.
          This range may include inserted specials.
    """
    raise NotImplementedError
