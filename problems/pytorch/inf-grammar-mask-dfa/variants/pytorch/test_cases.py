"""独立 oracle：Thompson NFA（显式 ε 转移）+ 子集构造，与参考实现的结构不共享。"""

import itertools
import random
import re


def _parse(pattern):
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


def _alphabet(root):
    out = set()

    def walk(node):
        if node[0] in ("lit", "class"):
            out.update(node[1])
            return
        for child in node[1:]:
            walk(child)

    walk(root)
    return "".join(sorted(out))


def _thompson(root):
    trans = {}
    eps = {}
    counter = [0]

    def new():
        counter[0] += 1
        trans[counter[0]] = {}
        eps[counter[0]] = set()
        return counter[0]

    def build(node):
        kind = node[0]
        if kind in ("lit", "class"):
            src, dst = new(), new()
            for ch in node[1]:
                trans[src].setdefault(ch, set()).add(dst)
            return src, dst
        if kind == "cat":
            a1, b1 = build(node[1])
            a2, b2 = build(node[2])
            eps[b1].add(a2)
            return a1, b2
        if kind == "alt":
            src, dst = new(), new()
            a1, b1 = build(node[1])
            a2, b2 = build(node[2])
            eps[src] |= {a1, a2}
            eps[b1].add(dst)
            eps[b2].add(dst)
            return src, dst
        src, dst = new(), new()
        a1, b1 = build(node[1])
        eps[src].add(a1)
        eps[b1].add(dst)
        if kind in ("star", "plus"):
            eps[b1].add(a1)
        if kind in ("star", "opt"):
            eps[src].add(dst)
        return src, dst

    start, accept = build(root)
    return trans, eps, start, accept


def _closure(states, eps):
    out = set(states)
    stack = list(states)
    while stack:
        s = stack.pop()
        for t in eps[s]:
            if t not in out:
                out.add(t)
                stack.append(t)
    return frozenset(out)


def _oracle(pattern, vocab, prefix):
    root = _parse(pattern)
    trans, eps, start, accept_state = _thompson(root)

    def step(subset, ch):
        nxt = set()
        for s in subset:
            nxt |= trans[s].get(ch, set())
        return _closure(nxt, eps) if nxt else frozenset()

    state = _closure({start}, eps)
    for ch in prefix:
        state = step(state, ch)
        if not state:
            break

    mask = []
    complete = []
    for tok in vocab:
        cur = state
        for ch in tok:
            cur = step(cur, ch)
            if not cur:
                break
        mask.append(bool(cur))
        complete.append(accept_state in cur)
    return mask, accept_state in state, complete


def _witness(pattern, text, max_suffix=4):
    """用 re.fullmatch 暴力找后缀，作为与两种自动机都无关的第三方见证。"""
    for depth in range(max_suffix + 1):
        for combo in itertools.product(_alphabet(_parse(pattern)), repeat=depth):
            if re.fullmatch(pattern, text + "".join(combo)):
                return True
    return False


def _rand_pattern(rng, depth=0):
    r = rng.random()
    if depth > 2 or r < 0.35:
        if rng.random() < 0.3:
            chars = "".join(rng.sample("abc", rng.randrange(1, 3)))
            return f"[{chars}]"
        return rng.choice("abc")
    if r < 0.5:
        return "(" + _rand_pattern(rng, depth + 1) + ")(" + _rand_pattern(rng, depth + 1) + ")"
    if r < 0.65:
        return "(" + _rand_pattern(rng, depth + 1) + "|" + _rand_pattern(rng, depth + 1) + ")"
    op = rng.choice("*+?")
    return f"({_rand_pattern(rng, depth + 1)}){op}"


def check_example(solve):
    mask, accept, complete = solve("a(b|c)*d", ["b", "c", "d", "e", "bb"], "a")
    assert mask == [True, True, True, False, True], f"mask 应为 [T,T,T,F,T]，got {mask}"
    assert accept is False, f"prefix 'a' 不是完整匹配，got {accept}"
    assert complete == [False, False, True, False, False], f"complete 错：{complete}"


check_example.example = True


def check_completed_prefix(solve):
    """prefix 已经完整匹配：accept 为真，且再也没有 token 可以接。"""
    mask, accept, complete = solve("abc", ["a", "abc", "d"], "abc")
    assert accept is True, f"got {accept}"
    assert mask == [False, False, False], f"got {mask}"
    assert complete == [False, False, False], f"got {complete}"


def check_nullable_pattern(solve):
    """a* 接受空串：空前缀下 accept 为真，但 mask 仍按"能不能继续"判断。"""
    mask, accept, complete = solve("a*", ["a", "b", "aa"], "")
    assert (mask, accept, complete) == ([True, False, True], True, [True, False, True]), (
        mask,
        accept,
        complete,
    )
    mask, accept, complete = solve("a*", ["a", "b", "aa"], "aa")
    assert (mask, accept, complete) == ([True, False, True], True, [True, False, True]), (
        mask,
        accept,
        complete,
    )


def check_optional_class(solve):
    """可选位已被吃掉后再补一位就非法；多位 token 要逐字符走完。"""
    mask, accept, complete = solve("ab[cd]?e", ["c", "d", "e", "cde", "ce", "b"], "ab")
    assert mask == [True, True, True, False, True, False], f"got {mask}"
    assert accept is False, f"got {accept}"
    assert complete == [False, False, True, False, True, False], f"got {complete}"


def check_multichar_token(solve):
    """token 跨多个状态：'cc' 在 [ab]+c 下第一个字符就死。"""
    mask, accept, complete = solve("[ab]+c", ["c", "ab", "abc", "aab", "cc"], "")
    assert mask == [False, True, True, True, False], f"got {mask}"
    assert accept is False, f"got {accept}"
    assert complete == [False, False, True, False, False], f"got {complete}"


def check_star_loop(solve):
    """(ab)* 循环后还能再开一轮，但 'c' 之后没有回头路。"""
    mask, accept, complete = solve("(ab)*c", ["a", "b", "ab", "abc", "cab"], "ab")
    assert mask == [True, False, True, True, False], f"got {mask}"
    assert accept is False, f"got {accept}"
    assert complete == [False, False, False, True, False], f"got {complete}"


def check_dead_prefix(solve):
    """prefix 本身就走死了：全部 token 与 accept 都为假。"""
    for prefix in ("ac", "b", "abx"):
        mask, accept, complete = solve("abc", ["a", "b", "abc", "c"], prefix)
        assert mask == [False] * 4, f"{prefix} got {mask}"
        assert complete == [False] * 4, f"{prefix} got {complete}"
        assert accept is False, f"{prefix} got {accept}"


def check_outside_alphabet(solve):
    """文法外字符不可能被消费，第一步即死。"""
    mask, accept, complete = solve("ab", ["a", "ax", "ab", "x"], "")
    assert mask == [True, False, True, False], f"got {mask}"
    assert accept is False, f"got {accept}"
    assert complete == [False, False, True, False], f"got {complete}"


def check_witnesses(solve):
    """用 re.fullmatch 做第三方见证：合法 token 必须存在可补完的后缀。"""
    cases = [
        ("a(b|c)*d", ["b", "c", "d", "e", "bb", "bd"], "a"),
        ("a?b", ["b", "a", "ab", "bb"], ""),
        ("[ab]+c", ["c", "ab", "abc", "aab", "cc", "b"], ""),
        ("(ab)*c", ["a", "b", "ab", "abc", "cab"], "ab"),
        ("ab[cd]?e", ["c", "d", "e", "cde", "ce", "b"], "ab"),
        ("a*", ["a", "b", "aa"], "aa"),
    ]
    for pattern, vocab, prefix in cases:
        mask, accept, complete = solve(pattern, vocab, prefix)
        assert accept == bool(re.fullmatch(pattern, prefix)), (pattern, prefix, accept)
        for tok, legal, done in zip(vocab, mask, complete):
            text = prefix + tok
            if done:
                assert re.fullmatch(pattern, text), f"complete 但 re 不匹配：{pattern} {text}"
            assert not (done and not legal), f"complete 必须蕴含 mask：{pattern} {text}"
            if legal:
                assert _witness(pattern, text), f"合法 token 找不到见证：{pattern} {text}"
            else:
                assert not _witness(pattern, text), f"非法 token 却有见证：{pattern} {text}"


def check_random_against_oracle(solve):
    """随机文法/词表/前缀与 Thompson 子集构造 oracle 三元组全等。"""
    rng = random.Random(680)
    for _ in range(60):
        pattern = _rand_pattern(rng)
        vocab = [
            "".join(rng.choice("abcd") for _ in range(rng.randint(1, 3))) for _ in range(6)
        ]
        prefix = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 3)))
        got = solve(pattern, vocab, prefix)
        want = _oracle(pattern, vocab, prefix)
        assert got[0] == want[0], f"mask 不一致：{pattern} {vocab} {prefix} {got[0]} vs {want[0]}"
        assert got[1] == want[1], f"accept 不一致：{pattern} {prefix} {got[1]} vs {want[1]}"
        assert got[2] == want[2], f"complete 不一致：{pattern} {vocab} {prefix} {got[2]} vs {want[2]}"


def check_decoding_trajectory(solve):
    """固定一条解码轨迹：每一步选的 token 都必须合法，complete 与 re 的判定一致。"""
    vocab = ["a", "b", "c", "d", "e", "ab", "bc", "de", "cd"]
    plans = [
        ("(ab|c)*d", ["ab", "c", "d"]),
        ("a[bc]+d", ["a", "b", "c", "d"]),
        ("(a|bc)(de)+d", ["bc", "de", "d"]),
    ]
    for pattern, tokens in plans:
        prefix = ""
        for tok in tokens:
            idx = vocab.index(tok)
            mask, accept, complete = solve(pattern, vocab, prefix)
            assert accept is False, f"{pattern} {prefix!r} 不该已经接受"
            assert bool(mask[idx]) is True, f"{pattern} {prefix!r} + {tok!r} 应合法"
            assert bool(complete[idx]) == bool(re.fullmatch(pattern, prefix + tok)), (
                f"{pattern} {prefix!r} + {tok!r} complete 判定错：{complete[idx]}"
            )
            prefix += tok
        mask, accept, complete = solve(pattern, vocab, prefix)
        assert accept is True, f"{pattern} {prefix!r} 应已接受"
        assert re.fullmatch(pattern, prefix)


def check_prefix_invariants(solve):
    """沿随机轨迹逐步核对：未接受时掩码必非空，complete 的 token 拼上必被 re 接受。"""
    rng = random.Random(99)
    vocab = ["a", "b", "c", "d", "e", "ab", "bc", "de", "cd", "x"]
    for pattern in ["(ab|c)*d", "a[bc]+d", "(a|bc)(de)*", "a?b"]:
        prefix = ""
        for _ in range(25):
            mask, accept, complete = solve(pattern, vocab, prefix)
            assert (mask, accept, complete) == _oracle(pattern, vocab, prefix), (
                pattern,
                prefix,
            )
            assert accept == bool(re.fullmatch(pattern, prefix)), (pattern, prefix)
            if accept:
                break
            legal = [i for i in range(len(vocab)) if mask[i]]
            assert legal, f"{pattern} {prefix!r} 未接受却无 token 可选"
            for i in legal:
                if complete[i]:
                    assert re.fullmatch(pattern, prefix + vocab[i])
            prefix += vocab[rng.choice(legal)]
