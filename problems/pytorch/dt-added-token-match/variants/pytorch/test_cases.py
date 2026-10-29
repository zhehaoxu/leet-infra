"""Independent occurrence oracle with native and explicitly prefiltered fixtures; see notes."""

import copy
import random
import string


_WORD = frozenset(string.ascii_letters + string.digits + '_')


def _tokens(specs):
    return [dict(content=content, single_word=single_word, special=special)
            for content, single_word, special in specs]


def _oracle(text, tokens, split=False):
    contents = [token['content'] for token in tokens]
    if '' in contents or len(set(contents)) != len(contents):
        raise ValueError('Invalid registration')
    # Enumerate ALL occurrences without a trie, including overlaps.
    occurrences = []
    for index, token in enumerate(tokens):
        if split and token['special']:
            continue
        content = token['content']
        for start in range(len(text) - len(content) + 1):
            if text[start:start + len(content)] == content:
                occurrences.append((start, start + len(content), index))
    occurrences.sort(key=lambda row: (row[0], row[0] - row[1]))
    raw = []
    selected_end = 0
    for start, end, index in occurrences:
        if start >= selected_end:
            raw.append((start, end, index))
            selected_end = end
    # Word boundaries are inspected only AFTER the raw interval list is fixed.
    accepted = {}
    for start, end, index in raw:
        neighbors = text[max(0, start - 1):start] + text[end:end + 1]
        if not tokens[index]['single_word'] or not any(c in _WORD for c in neighbors):
            accepted[start] = (end, index)
    # Independently assemble maximal ordinary runs from accepted starts.
    output = []
    pos = 0
    while pos < len(text):
        if pos in accepted:
            end, index = accepted[pos]
        else:
            end = pos + 1
            while end < len(text) and end not in accepted:
                end += 1
            index = -1
        output.append((text[pos:end], pos, end, index))
        pos = end
    return output


def _assert_result(text, tokens, split, actual, expected):
    assert isinstance(actual, list), 'Return a list'
    cursor = 0
    last_id = None
    for row in actual:
        assert isinstance(row, tuple) and len(row) == 4, row
        piece, start, end, index = row
        assert isinstance(piece, str) and piece, row
        assert all(type(value) is int for value in (start, end, index)), row
        assert start == cursor and start < end <= len(text), row
        assert piece == text[start:end], row
        assert -1 <= index < len(tokens), row
        if index == -1:
            assert last_id != -1, 'Adjacent ordinary runs must be merged'
        else:
            assert piece == tokens[index]['content'], row
            assert not (split and tokens[index]['special']), row
        cursor = end
        last_id = index
    assert cursor == len(text)
    assert ''.join(row[0] for row in actual) == text
    assert actual == expected, (text, tokens, split, actual, expected)


def _case(solve, text, specs, split=False, expected=None):
    tokens = _tokens(specs)
    snapshot = copy.deepcopy(tokens)
    reference = _oracle(text, tokens, split)
    if expected is not None:
        assert reference == expected, ('Oracle versus frozen fixture', reference, expected)
    actual = solve(text, tokens, split)
    assert tokens == snapshot, 'Do not mutate the registration list or dictionaries'
    _assert_result(text, tokens, split, actual, reference)


_NATIVE_FIXTURES = [
    ('abcde', [('ab', False, False), ('bcde', False, False)], False, [('ab', 0, 2, 0), ('cde', 2, 5, -1)]),
    ('ab abc abcd', [('ab', False, False), ('abc', False, False), ('abcd', False, False)], False, [('ab', 0, 2, 0), (' ', 2, 3, -1), ('abc', 3, 6, 1), (' ', 6, 7, -1), ('abcd', 7, 11, 2)]),
    ('zabc!', [('ab', False, False), ('abc', True, False), ('b', False, False)], False, [('zabc!', 0, 5, -1)]),
    ('zabcd!', [('abc', True, False), ('cd', False, False)], False, [('zabcd!', 0, 6, -1)]),
    ('cat _cat cat2 2cat cat_ cat!', [('cat', True, False)], False, [('cat', 0, 3, 0), (' _cat cat2 2cat cat_ ', 3, 24, -1), ('cat', 24, 27, 0), ('!', 27, 28, -1)]),
    ('a<tag> <tag>!', [('<tag>', True, False)], False, [('a<tag> ', 0, 7, -1), ('<tag>', 7, 12, 0), ('!', 12, 13, -1)]),
    ('xabc!', [('ab', False, True), ('abc', False, False)], False, [('x', 0, 1, -1), ('abc', 1, 4, 1), ('!', 4, 5, -1)]),
    ('a<s>b<s>', [('<s>', False, True)], False, [('a', 0, 1, -1), ('<s>', 1, 4, 0), ('b', 4, 5, -1), ('<s>', 5, 8, 0)]),
    ('a<s>b<s>', [('<s>', False, True)], True, [('a<s>b<s>', 0, 8, -1)]),
    ('', [('abc', False, True)], False, []),
]
_PREFILTERED_FIXTURES = [
    # Native 0.23.3 output: [('abc!', 0, 4, -1)] (no fallback to 'ab').
    ('abc!', [('ab', False, False), ('abc', True, True)], True, [('ab', 0, 2, 0), ('c!', 2, 4, -1)]),
    # Native 0.23.3 output: [('abcd!', 0, 5, -1)] (no internal 'bcd' rescan).
    ('abcd!', [('abc', False, True), ('bcd', False, False)], True, [('a', 0, 1, -1), ('bcd', 1, 4, 1), ('!', 4, 5, -1)]),
]


def check_leftmost_not_globally_longest(solve):
    _case(solve, 'abcde', [('ab', False, False), ('bcde', False, False)],
          expected=[('ab', 0, 2, 0), ('cde', 2, 5, -1)])
    _case(solve, 'xabcdef!', [('a', False, False), ('bcdef', False, False), ('abcdef', False, False)])


def check_rejected_longest_blocks_shorter(solve):
    specs = [('ab', False, False), ('abc', True, False), ('b', False, False)]
    _case(solve, 'zabc!', specs, expected=[('zabc!', 0, 5, -1)])
    _case(solve, 'abcx', specs, expected=[('abcx', 0, 4, -1)])
    _case(solve, 'abc ab', specs)


def check_special_prefilter_changes_candidates(solve):
    for text, specs, split, expected in _PREFILTERED_FIXTURES:
        _case(solve, text, specs, split, expected)
        _case(solve, text, specs, False)
    _case(solve, 'abcd', [('ab', False, True), ('abc', False, True), ('abcd', False, False)], True,
          [('abcd', 0, 4, 2)])


def check_empty_text_and_no_patterns(solve):
    for specs in [[], [('a', False, False)], [('a', True, True)]]:
        for split in [False, True]:
            _case(solve, '', specs, split, [])
    _case(solve, 'abc \t\x00!?', [], expected=[('abc \t\x00!?', 0, 8, -1)])
    _case(solve, 'abc', [('a', False, True)], True, [('abc', 0, 3, -1)])


def check_longest_at_same_start_and_registration_ids(solve):
    for specs in [
        [('a', False, False), ('ab', False, False), ('abc', False, False)],
        [('abc', False, False), ('ab', False, False), ('a', False, False)],
        [('ab', False, False), ('a', False, False), ('abc', False, False)],
    ]:
        _case(solve, 'abc ab a abcabc', specs)


def check_single_word_ascii_neighbors(solve):
    specs = [('cat', True, False)]
    for left in ['', 'a', 'Z', '0', '9', '_', ' ', '.', '\t', '\n', '\x00', '\x7f']:
        for right in ['', 'a', 'Z', '0', '9', '_', ' ', '.', '\t', '\n', '\x00', '\x7f']:
            _case(solve, left + 'cat' + right, specs)


def check_punctuation_contents_and_control_neighbors(solve):
    _case(solve, 'a<tag> <tag>! _<tag> <tag>2', [('<tag>', True, False)])
    _case(solve, '!a! !!', [('!', True, False), ('!!', True, False)])
    _case(solve, '\x00\x01!\x7f \x00', [('\x00', False, False), ('\x01!', True, False)])
    _case(solve, ''.join(map(chr, range(128))), [('A', True, False), ('!', False, False)])


def check_no_rescan_inside_rejected_interval(solve):
    _case(solve, 'zabcd!', [('abc', True, False), ('cd', False, False)],
          expected=[('zabcd!', 0, 6, -1)])
    _case(solve, 'zabc!bc', [('abc', True, False), ('bc', False, False)],
          expected=[('zabc!', 0, 5, -1), ('bc', 5, 7, 1)])
    _case(solve, 'x<tag>y !', [('<tag>', True, False), ('tag', False, False), ('!', False, False)])


def check_special_has_no_extra_priority(solve):
    _case(solve, 'xabc!', [('ab', False, True), ('abc', False, False)],
          expected=[('x', 0, 1, -1), ('abc', 1, 4, 1), ('!', 4, 5, -1)])
    _case(solve, 'abcde', [('ab', False, False), ('bcde', False, True)])
    _case(solve, 'xabc!', [('ab', False, False), ('abc', True, True)])


def check_case_sensitive_and_adjacent_matches(solve):
    specs = [('cat', False, False), ('CAT', False, True), ('a', False, False)]
    _case(solve, 'catCATCatcatcat', specs)
    _case(solve, 'catCATCatcatcat', specs, True)
    _case(solve, 'aaa', [('a', False, False)], expected=[('a', 0, 1, 0), ('a', 1, 2, 0), ('a', 2, 3, 0)])


def check_merge_ordinary_and_preserve_whitespace(solve):
    _case(solve, '  <s>\t\nword  <s> ', [('<s>', False, True), ('word', False, False)])
    _case(solve, 'xx abcY ?? <s> zz', [('abc', True, False), ('<s>', False, False)])
    _case(solve, '  nothing\t\n', [('missing', False, True)])


def check_invalid_registration_even_if_skipped(solve):
    invalid = [
        [('', False, False)],
        [('', False, True)],
        [('abc', False, False), ('abc', False, False)],
        [('abc', True, True), ('abc', False, False)],
        [('abc', False, True), ('abc', True, True)],
    ]
    for text in ['', 'abc!']:
        for specs in invalid:
            for split in [False, True]:
                tokens = _tokens(specs)
                snapshot = copy.deepcopy(tokens)
                try:
                    solve(text, tokens, split)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Empty/duplicate content must raise ValueError')
                assert tokens == snapshot


def check_input_immutability_default_and_repeated_calls(solve):
    tokens = _tokens([('abc', False, True), ('ab', True, False), ('!', False, False)])
    snapshot = copy.deepcopy(tokens)
    original_items = list(tokens)
    for split in [True, False, True, False]:
        text = 'abc ab!'
        _assert_result(text, tokens, split, solve(text, tokens, split), _oracle(text, snapshot, split))
        assert tokens == snapshot
        assert all(a is b for a, b in zip(tokens, original_items))
    assert solve('abc ab!', tokens) == _oracle('abc ab!', snapshot, False)
    assert tokens == snapshot


def check_maximum_input_and_pattern_limits(solve):
    specs = [('a' * length, False, False) for length in range(1, 31)]
    _case(solve, 'a' * 300, specs,
          expected=[('a' * 30, i, i + 30, 29) for i in range(0, 300, 30)])
    specs[-1] = ('a' * 30, True, False)
    _case(solve, 'a' * 300, specs, expected=[('a' * 300, 0, 300, -1)])
    specs[-1] = ('a' * 30, False, True)
    _case(solve, 'a' * 300, specs, True)


def check_seeded_random_against_independent_oracle(solve):
    rng = random.Random(4592026)
    alphabet = 'abAB019_-<>!'
    for _ in range(180):
        contents = set()
        target = rng.randrange(31)
        while len(contents) < target:
            contents.add(''.join(rng.choice(alphabet) for _ in range(rng.randrange(1, 9))))
        ordered = sorted(contents)
        rng.shuffle(ordered)
        specs = [(content, bool(rng.getrandbits(1)), bool(rng.getrandbits(1))) for content in ordered]
        chunks = ordered + list(alphabet + ' \t\n\x00\x7f')
        text = ''.join(rng.choice(chunks) for _ in range(rng.randrange(61)))[:300]
        for split in [False, True]:
            _case(solve, text, specs, split)
    # High-overlap random prefix families, including boundary rejections.
    for _ in range(60):
        base = ''.join(rng.choice('ab') for _ in range(8))
        specs = [(base[:length], bool(rng.getrandbits(1)), bool(rng.getrandbits(1))) for length in range(1, 9)]
        text = ''.join(rng.choice([base, 'x' + base, ' ', '_', base[1:]]) for _ in range(20))
        for split in [False, True]:
            _case(solve, text, specs, split)


def check_frozen_actual_tokenizers_probes(solve):
    for text, specs, split, expected in _NATIVE_FIXTURES + _PREFILTERED_FIXTURES:
        _case(solve, text, specs, split, expected)


check_leftmost_not_globally_longest.example = True
check_rejected_longest_blocks_shorter.example = True
check_special_prefilter_changes_candidates.example = True
