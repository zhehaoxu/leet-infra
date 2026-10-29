import copy
import random


# Independent from cumulative-boundary/bisect conversion: expand every byte's
# owning character and word, then enumerate all byte positions in each token.
def _oracle(text, byte_offsets, word_spans, char_queries):
    byte_owner, byte_word = [], []
    for char_index, char in enumerate(text):
        word = next((w for w, (s, e) in enumerate(word_spans) if s <= char_index < e), None)
        for _ in char.encode("utf-8"):
            byte_owner.append(char_index)
            byte_word.append(word)
    offsets, word_ids, covers = [], [], []
    for a, b in byte_offsets:
        if (a, b) == (-1, -1):
            offsets.append([0, 0])
            word_ids.append(None)
            covers.append(set())
            continue
        if not 0 <= a < b <= len(byte_owner):
            raise ValueError("invalid bytes")
        owners = {byte_owner[pos] for pos in range(a, b)}
        words = {byte_word[pos] for pos in range(a, b)}
        assert len(words) == 1 and None not in words, "invalid test fixture"
        offsets.append([min(owners), max(owners) + 1])
        word_ids.append(words.pop())
        covers.append(owners)
    char_to_token = [next((i for i, owners in enumerate(covers) if q in owners), None)
                     for q in char_queries]
    word_to_tokens = []
    for word in range(len(word_spans)):
        indices = [i for i, w in enumerate(word_ids) if w == word]
        word_to_tokens.append([min(indices), max(indices) + 1] if indices else None)
    return dict(offsets=offsets, word_ids=word_ids,
                char_to_token=char_to_token, word_to_tokens=word_to_tokens)


def _verify(solve, args, frozen=None):
    snapshot = copy.deepcopy(args)
    expected = _oracle(*args)
    if frozen is not None:
        assert expected == frozen, (expected, frozen)
    actual = solve(*args)
    assert type(actual) is dict and actual == expected, (args, actual, expected)
    assert all(type(v) is list for v in actual.values())
    for span in actual["offsets"] + actual["word_to_tokens"]:
        assert span is None or (type(span) is list and len(span) == 2
                                and all(type(x) is int for x in span))
    for item in actual["word_ids"] + actual["char_to_token"]:
        assert item is None or type(item) is int
    assert args == snapshot, "input mutation"
    return actual


def check_native_bytelevel_fixture(solve):
    # tokenizers 0.23.3: full ByteLevel pipeline, alphabet=256, no BPE merges,
    # add_prefix_space=False, no normalizer and no offset-trimming processor.
    _verify(solve, ("é😊X", [(i, i + 1) for i in range(7)],
                    [(0, 1), (1, 2), (2, 3)], [0, 1, 2]), {
        "offsets": [[0, 1], [0, 1], [1, 2], [1, 2], [1, 2], [1, 2], [2, 3]],
        "word_ids": [0, 0, 1, 1, 1, 1, 2],
        "char_to_token": [0, 2, 6],
        "word_to_tokens": [[0, 2], [2, 6], [6, 7]],
    })


def check_chinese_and_emoji(solve):
    _verify(solve, ("中文😊", [(0, 3), (3, 6), (6, 10)],
                    [(0, 2), (2, 3)], [-1, 0, 1, 2, 3]))
    _verify(solve, ("𠀀中🚀", [(0, 1), (1, 4), (4, 7), (7, 11)],
                    [(0, 3)], [0, 1, 2, 3]))


def check_combining_and_accent_no_normalization(solve):
    _verify(solve, ("e\u0301é", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
                    [(0, 2), (2, 3)], [0, 1, 2, 3]))
    # ZWJ and variation selector are separate code points, not grapheme units.
    text = "👩\u200d💻\ufe0f"
    _verify(solve, (text, [(i, i + 1) for i in range(len(text.encode('utf-8')))],
                    [(0, len(text))], [0, 1, 2, 3, 4]))


def check_one_byte_partials(solve):
    for char in ("a", "é", "中", "😊", "\U0010ffff"):
        width = len(char.encode("utf-8"))
        for a in range(width):
            for b in range(a + 1, width + 1):
                _verify(solve, (char, [(a, b)], [(0, 1)], [-1, 0, 1]))


def check_boundaries_and_cross_character_covers(solve):
    _verify(solve, ("aé中😊z", [(0, 1), (1, 3), (2, 4), (3, 6),
                                (5, 7), (6, 10), (10, 11)],
                    [(0, 5)], list(range(-1, 7))))
    _verify(solve, ("é中", [(1, 4)], [(0, 2)], [0, 1, 2]))


def check_duplicates_overlaps_and_first_hit(solve):
    _verify(solve, ("éab", [(0, 4), (0, 1), (0, 4), (1, 2), (2, 3), (3, 4)],
                    [(0, 3)], [2, 0, 1, 2, -1, 3]), {
        "offsets": [[0, 3], [0, 1], [0, 3], [0, 1], [1, 2], [2, 3]],
        "word_ids": [0, 0, 0, 0, 0, 0],
        "char_to_token": [0, 0, 0, 0, None, None],
        "word_to_tokens": [[0, 6]],
    })


def check_special_never_covers_character_zero(solve):
    _verify(solve, ("a", [(-1, -1), (0, 1), (-1, -1)], [(0, 1)], [0, 1, -1]), {
        "offsets": [[0, 0], [0, 1], [0, 0]],
        "word_ids": [None, 0, None],
        "char_to_token": [1, None, None],
        "word_to_tokens": [[1, 2]],
    })
    _verify(solve, ("a", [(-1, -1)], [(0, 1)], [0]))


def check_gaps_no_hit_and_empty(solve):
    _verify(solve, ("é  X\t中", [(0, 1), (4, 5)], [(0, 1), (3, 4), (5, 6)],
                    [-100, -1, 0, 1, 2, 3, 4, 5, 6, 100]))
    for tokens in ([], [(-1, -1), (-1, -1)]):
        _verify(solve, ("", tokens, [], [-1, 0, 1]))
        _verify(solve, (" \t", tokens, [], []))
    _verify(solve, ("abc", [], [(0, 3)], [0, 1, 2]))


def check_words_without_tokens(solve):
    _verify(solve, ("a b c d", [(2, 3), (-1, -1), (6, 7)],
                    [(0, 1), (2, 3), (4, 5), (6, 7)], list(range(7))), {
        "offsets": [[2, 3], [0, 0], [6, 7]],
        "word_ids": [1, None, 3],
        "char_to_token": [None, None, 0, None, None, None, 2],
        "word_to_tokens": [None, [0, 1], None, [2, 3]],
    })


def check_word_range_includes_special(solve):
    _verify(solve, ("é X", [(-1, -1), (0, 1), (-1, -1), (1, 2), (3, 4), (-1, -1)],
                    [(0, 1), (2, 3)], [0, 1, 2, -1, 3]), {
        "offsets": [[0, 0], [0, 1], [0, 0], [0, 1], [2, 3], [0, 0]],
        "word_ids": [None, 0, None, 0, 1, None],
        "char_to_token": [1, None, 4, None, None],
        "word_to_tokens": [[1, 4], [4, 5]],
    })


def check_explicit_pretokenized_units(solve):
    # The first unit contains whitespace; adjacent units need no whitespace.
    _verify(solve, ("a b,c", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
                    [(0, 3), (3, 4), (4, 5)], list(range(5))))
    _verify(solve, ("\x00é\n", [(0, 1), (1, 3), (3, 4)], [(0, 3)], [0, 1, 2]))


def check_maximum_limits(solve):
    text = "é😊中a" * 25
    total = len(text.encode("utf-8"))
    spans = [(i * 3, (i + 1) * 3) for i in range(29)] + [(87, 100)]
    tokens = [(i * total // 200, i * total // 200 + 1) for i in range(200)]
    _verify(solve, (text, tokens, spans, [-1] + list(range(98)) + [100]))


def check_seeded_random(solve):
    rng = random.Random(460)
    alphabet = ["a", " ", "é", "中", "😊", "\u0301", "\x00", "\U0010ffff"]
    for _ in range(120):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(101)))
        spans, cursor = [], 0
        while cursor < len(text) and len(spans) < 30:
            cursor += rng.randrange(3)
            if cursor >= len(text):
                break
            end = min(len(text), cursor + rng.randrange(1, 7))
            spans.append((cursor, end))
            cursor = end
        byte_words = [(len(text[:s].encode("utf-8")), len(text[:e].encode("utf-8")))
                      for s, e in spans]
        ordinary = []
        count = rng.randrange(201)
        for _ in range(count):
            if byte_words and rng.randrange(5) != 0:
                s, e = rng.choice(byte_words)
                a = rng.randrange(s, e)
                ordinary.append((a, rng.randrange(a + 1, e + 1)))
        ordinary.sort(key=lambda span: span[0])
        tokens = ordinary[:]
        for _ in range(count - len(ordinary)):
            tokens.insert(rng.randrange(len(tokens) + 1), (-1, -1))
        queries = [rng.randrange(-5, len(text) + 6) for _ in range(rng.randrange(101))]
        _verify(solve, (text, tokens, spans, queries))


def check_immutability_and_repeat_calls(solve):
    args = ("éX", [(0, 1), (-1, -1), (1, 2), (2, 3)], [(0, 1), (1, 2)], [0, 1, 0])
    first = _verify(solve, args)
    second = _verify(solve, args)
    assert first == second
    _verify(solve, ("x", [], [], []))
    assert _verify(solve, args) == first


def check_invalid_byte_offsets(solve):
    for invalid in ((-1, 0), (0, -1), (-2, -2), (-2, 1), (0, 0), (1, 1),
                    (2, 1), (0, 3), (2, 3), (3, 4), (2, 2)):
        for tokens in ([invalid], [(-1, -1), (0, 1), invalid]):
            args = ("é", tokens, [(0, 1)], [0])
            snapshot = copy.deepcopy(args)
            try:
                solve(*args)
            except ValueError:
                assert args == snapshot, "input mutation on error"
            else:
                raise AssertionError(("expected ValueError", invalid))
    for invalid in ((0, 0), (0, 1), (-1, 0)):
        try:
            solve("", [invalid], [], [])
        except ValueError:
            pass
        else:
            raise AssertionError("empty text does not allow ordinary byte spans")


check_native_bytelevel_fixture.example = True
check_word_range_includes_special.example = True
