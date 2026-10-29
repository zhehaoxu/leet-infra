def _parse(pattern):
    """递归下降：alt -> seq ('|' seq)*；seq -> rep+；rep -> atom ('*'|'+'|'?')*。"""
    pos = 0

    def peek():
        return pattern[pos] if pos < len(pattern) else None

    def parse_alt():
        nonlocal pos
        node = parse_seq()
        while peek() == "|":
            pos += 1
            node = ("alt", node, parse_seq())
        return node

    def parse_seq():
        nonlocal pos
        parts = []
        while peek() is not None and peek() not in "|)":
            parts.append(parse_rep())
        if not parts:
            raise ValueError("empty sequence")
        node = parts[0]
        for nxt in parts[1:]:
            node = ("cat", node, nxt)
        return node

    def parse_rep():
        nonlocal pos
        node = parse_atom()
        while peek() in ("*", "+", "?"):
            op = pattern[pos]
            pos += 1
            node = ({"*": "star", "+": "plus", "?": "opt"}[op], node)
        return node

    def parse_atom():
        nonlocal pos
        ch = pattern[pos]
        if ch == "(":
            pos += 1
            node = parse_alt()
            if pattern[pos] != ")":
                raise ValueError("unbalanced group")
            pos += 1
            return node
        if ch == "[":
            end = pattern.index("]", pos)
            chars = frozenset(pattern[pos + 1 : end])
            pos = end + 1
            return ("class", chars)
        if ch in "*+?|)":
            raise ValueError(f"unexpected {ch}")
        pos += 1
        return ("lit", frozenset([ch]))

    node = parse_alt()
    if pos != len(pattern):
        raise ValueError("trailing input")
    return node


def _positions(root):
    """Glushkov 位置自动机：每个字面位置一个状态，返回 nullable/first/last/符号表/follow。"""
    sym = {}
    follow = {}
    count = 0

    def walk(node):
        nonlocal count
        kind = node[0]
        if kind in ("lit", "class"):
            count += 1
            sym[count] = node[1]
            follow[count] = set()
            return False, {count}, {count}
        if kind == "cat":
            left_null, left_first, left_last = walk(node[1])
            right_null, right_first, right_last = walk(node[2])
            for p in left_last:
                follow[p] |= right_first
            return (
                left_null and right_null,
                left_first | (right_first if left_null else set()),
                right_last | (left_last if right_null else set()),
            )
        if kind == "alt":
            left_null, left_first, left_last = walk(node[1])
            right_null, right_first, right_last = walk(node[2])
            return left_null or right_null, left_first | right_first, left_last | right_last
        inner_null, inner_first, inner_last = walk(node[1])
        if kind in ("star", "plus"):
            for p in inner_last:
                follow[p] |= inner_first
        if kind == "star":
            return True, inner_first, inner_last
        if kind == "plus":
            return inner_null, inner_first, inner_last
        return True, inner_first, inner_last

    nullable, first, last = walk(root)
    return nullable, first, last, sym, follow


def solve(pattern, vocab, prefix):
    nullable, first, last, sym, follow = _positions(_parse(pattern))

    def step(state, ch):
        """state=None 表示还没读过字符；否则是"刚读完的字符落在哪些位置"的集合。"""
        if state is None:
            candidates = first
        else:
            candidates = set()
            for p in state:
                candidates |= follow[p]
        return frozenset(q for q in candidates if ch in sym[q])

    state = None
    for ch in prefix:
        state = step(state, ch)
        if not state:
            break

    accept = nullable if state is None else bool(state & last)

    mask = []
    complete = []
    for tok in vocab:
        cur = state
        for ch in tok:
            cur = step(cur, ch)
            if not cur:
                break
        if cur is None:
            mask.append(nullable)
            complete.append(nullable)
        else:
            mask.append(bool(cur))
            complete.append(bool(cur & last))
    return mask, accept, complete
