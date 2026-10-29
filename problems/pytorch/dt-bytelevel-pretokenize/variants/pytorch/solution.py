"""Standard-library GPT-2 ordered scanner and byte encoder."""

import unicodedata


_WHITESPACE = frozenset(
    "\t\n\v\f\r \u0085\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _byte_table():
    table = {}
    next_codepoint = 256
    for byte in range(256):
        if 33 <= byte <= 126 or 161 <= byte <= 172 or 174 <= byte <= 255:
            table[byte] = chr(byte)
        else:
            table[byte] = chr(next_codepoint)
            next_codepoint += 1
    return table


_BYTE_TABLE = _byte_table()


def _kind(char):
    if char in _WHITESPACE:
        return "space"
    category = unicodedata.category(char)[0]
    return category if category in ("L", "N") else "other"


def solve(text: str) -> list[tuple[str, int, int]]:
    result = []
    start = 0
    size = len(text)
    while start < size:
        stop = None
        for contraction in _CONTRACTIONS:
            if text.startswith(contraction, start):
                stop = start + len(contraction)
                break
        if stop is None:
            # Only ASCII space can be absorbed by the L/N/other branches.
            body = start + (text[start] == " ")
            kind = _kind(text[body]) if body < size else "space"
            if kind != "space":
                stop = body + 1
                while stop < size and _kind(text[stop]) == kind:
                    stop += 1
            else:
                stop = start + 1
                while stop < size and text[stop] in _WHITESPACE:
                    stop += 1
                # Greedy whitespace with (?!\S) backtracks once internally.
                if stop < size and stop - start > 1:
                    stop -= 1
        encoded = "".join(_BYTE_TABLE[byte] for byte in text[start:stop].encode("utf-8"))
        result.append((encoded, start, stop))
        start = stop
    return result
