"""Minimal code-point covers of UTF-8 byte intervals; no tokenizer dependency."""

from bisect import bisect_left, bisect_right


def solve(text, byte_offsets, word_spans, char_queries):
    boundaries = [0]
    for char in text:
        boundaries.append(boundaries[-1] + len(char.encode("utf-8")))
    word_starts = [start for start, _ in word_spans]
    offsets, word_ids = [], []
    word_to_tokens = [None] * len(word_spans)

    for token_index, (a, b) in enumerate(byte_offsets):
        if (a, b) == (-1, -1):
            offsets.append([0, 0])
            word_ids.append(None)
            continue
        if not 0 <= a < b <= boundaries[-1]:
            raise ValueError("ordinary byte offsets must satisfy 0 <= a < b <= UTF-8 length")
        start = bisect_right(boundaries, a) - 1
        end = bisect_left(boundaries, b)
        offsets.append([start, end])
        # Containment in exactly one supplied word span is guaranteed.
        word = bisect_right(word_starts, start) - 1
        word_ids.append(word)
        if word_to_tokens[word] is None:
            word_to_tokens[word] = [token_index, token_index + 1]
        else:
            word_to_tokens[word][1] = token_index + 1

    char_to_token = []
    for query in char_queries:
        char_to_token.append(next((i for i, (start, end) in enumerate(offsets)
                                   if word_ids[i] is not None and start <= query < end), None))
    return dict(offsets=offsets, word_ids=word_ids,
                char_to_token=char_to_token, word_to_tokens=word_to_tokens)
