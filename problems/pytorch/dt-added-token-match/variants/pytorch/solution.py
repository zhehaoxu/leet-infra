"""Trie matching with post-selection word-boundary filtering."""


def _word(char):
    return "a" <= char <= "z" or "A" <= char <= "Z" or "0" <= char <= "9" or char == "_"


def solve(
    text: str, added_tokens: list[dict], split_special_tokens: bool = False
) -> list[tuple[str, int, int, int]]:
    seen = set()
    trie = {}
    for added_id, token in enumerate(added_tokens):
        content = token["content"]
        if not content or content in seen:
            raise ValueError("Added-token contents must be nonempty and unique")
        seen.add(content)
        if split_special_tokens and token["special"]:
            continue
        node = trie
        for char in content:
            node = node.setdefault(char, {})
        node[None] = added_id

    result = []
    cursor = 0
    ordinary_start = 0
    size = len(text)
    while cursor < size:
        node = trie
        end = cursor
        best_id = None
        best_end = cursor
        while end < size and text[end] in node:
            node = node[text[end]]
            end += 1
            if None in node:
                best_id = node[None]
                best_end = end
        if best_id is None:
            cursor += 1
            continue
        token = added_tokens[best_id]
        allowed = not token["single_word"] or (
            (cursor == 0 or not _word(text[cursor - 1]))
            and (best_end == size or not _word(text[best_end]))
        )
        if allowed:
            if ordinary_start < cursor:
                result.append((text[ordinary_start:cursor], ordinary_start, cursor, -1))
            result.append((text[cursor:best_end], cursor, best_end, best_id))
            ordinary_start = best_end
        # A rejected raw candidate still consumes its entire interval.
        cursor = best_end
    if ordinary_start < size:
        result.append((text[ordinary_start:], ordinary_start, size, -1))
    return result
