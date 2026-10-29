from copy import deepcopy
import random


def _oracle(text, patterns, replacements):
    if "" in patterns:
        raise ValueError("empty pattern")
    matches = []
    for start in range(len(text)):
        for index, pattern in enumerate(patterns):
            if text.startswith(pattern, start):
                matches.append((start, start + len(pattern), index))
    matches.sort()

    selected = []
    pieces = []
    position = 0
    while position < len(text):
        best_index = None
        best_length = 0
        for index, pattern in enumerate(patterns):
            if len(pattern) > best_length and text.startswith(pattern, position):
                best_index = index
                best_length = len(pattern)
        if best_index is None:
            pieces.append(text[position])
            position += 1
        else:
            selected.append((position, position + best_length, best_index))
            pieces.append(replacements[best_index])
            position += best_length
    return "".join(pieces), matches, selected


def _assert_case(solve, text, patterns, replacements, expected=None):
    before = deepcopy((text, patterns, replacements))
    reference = _oracle(text, patterns, replacements)
    if expected is not None:
        assert reference == expected, (reference, expected)
    result = solve(text, patterns, replacements)
    assert (text, patterns, replacements) == before, "inputs were mutated"
    assert type(result) is tuple and len(result) == 3, "return a three-item Python tuple"
    assert type(result[0]) is str
    for matches in result[1:]:
        assert type(matches) is list
        for match in matches:
            assert type(match) is tuple and len(match) == 3
            assert all(type(value) is int for value in match)
    assert result == reference, (result, reference)
    return result


def check_failure_inheritance_example(solve):
    _assert_case(
        solve, "ushers", ["he", "she", "hers", "he"], ["H", "S", "R", "D"],
        ("uSrs", [(1, 4, 1), (2, 4, 0), (2, 4, 3), (2, 6, 2)], [(1, 4, 1)]),
    )


def check_overlapping_suffixes(solve):
    _assert_case(solve, "bananana", ["a", "na", "ana", "nana", "banana"], ["A", "N", "X", "Y", "B"])
    _assert_case(solve, "aaaaa", ["a", "aa", "aaa", "aaaa"], ["1", "2", "3", "4"])


def check_deep_failure_fallback(solve):
    _assert_case(
        solve, "abcabcaby!aaababcabx", ["abcabx", "bcaby", "cab", "aab", "aaab", "b", "aby"],
        ["X", "Y", "C", "D", "E", "B", "Z"],
    )
    _assert_case(solve, "abcdxbcdcd", ["abcdq", "bcd", "cd", "d"], ["Q", "B", "C", "D"])


def check_duplicate_pattern_tie(solve):
    _assert_case(
        solve, "zz", ["z", "zz", "zz", "z"], ["short", "first", "second", "last"],
        ("first", [(0, 1, 0), (0, 1, 3), (0, 2, 1), (0, 2, 2), (1, 2, 0), (1, 2, 3)], [(0, 2, 1)]),
    )


def check_leftmost_before_global_longest(solve):
    _assert_case(
        solve, "abcdef", ["ab", "bcdef", "cde", "f"], ["A", "B", "C", "F"],
        ("ACF", [(0, 2, 0), (1, 6, 1), (2, 5, 2), (5, 6, 3)], [(0, 2, 0), (2, 5, 2), (5, 6, 3)]),
    )


def check_same_start_longest(solve):
    _assert_case(solve, "abcdabc", ["a", "ab", "abc", "abcd", "bc"], ["1", "2", "3", "4", "5"])
    _assert_case(solve, "ababab", ["ab", "abab", "babab"], ["S", "L", "later"])


def check_unicode_example(solve):
    _assert_case(
        solve, "e\u0301é\U0001f642é", ["é", "e\u0301", "\U0001f642é"], ["E", "C", ""],
        ("CE", [(0, 2, 1), (2, 3, 0), (3, 5, 2), (4, 5, 0)], [(0, 2, 1), (2, 3, 0), (3, 5, 2)]),
    )
    _assert_case(solve, "猫e\u0301Éé\U0001f469\u200d\U0001f4bb猫", ["猫", "\u0301", "é", "\U0001f4bb", "\u200d"], ["C", "M", "E", "P", "J"])
    _assert_case(solve, "a\x00a", ["\x00", "a\x00"], ["zero", "pair"])


def check_empty_inputs(solve):
    _assert_case(solve, "", [], [], ("", [], []))
    _assert_case(solve, "", ["a", "a"], ["", "x"], ("", [], []))
    _assert_case(solve, "猫 untouched", [], [], ("猫 untouched", [], []))
    _assert_case(solve, "abc", ["xyz", "abcd"], ["X", "Y"], ("abc", [], []))


def check_empty_replacements(solve):
    _assert_case(solve, "ababa", ["aba", "ba"], ["", ""], ("", [(0, 3, 0), (1, 3, 1), (2, 5, 0), (3, 5, 1)], [(0, 3, 0), (3, 5, 1)]))
    _assert_case(solve, "a--b--a", ["a", "b"], ["", ""], ("----", [(0, 1, 0), (3, 4, 1), (6, 7, 0)], [(0, 1, 0), (3, 4, 1), (6, 7, 0)]))


def check_no_cascading_replacement(solve):
    _assert_case(solve, "ab", ["a", "b", "bb"], ["bb", "a", "X"], ("bba", [(0, 1, 0), (1, 2, 1)], [(0, 1, 0), (1, 2, 1)]))
    _assert_case(solve, "axb", ["x", "ab"], ["", "cascade"], ("ab", [(1, 2, 0)], [(1, 2, 0)]))
    _assert_case(solve, "a", ["a"], ["a"], ("a", [(0, 1, 0)], [(0, 1, 0)]))


def check_uncovered_text_and_adjacent_matches(solve):
    _assert_case(solve, "!abxcd?", ["ab", "cd", "x"], ["A", "C", "X"])
    _assert_case(solve, " abab ", ["ab", "b"], ["Q", "R"])
    _assert_case(solve, "a\nb\tC", ["\n", "b\t", "c"], ["|", "B", "lower"])


def check_invalid_empty_pattern(solve):
    for text, patterns in [("", [""]), ("abc", ["a", "", "b"]), ("x", ["x", ""]), ("", ["", ""])]:
        replacements = ["R"] * len(patterns)
        before = deepcopy((text, patterns, replacements))
        try:
            solve(text, patterns, replacements)
        except ValueError:
            pass
        else:
            raise AssertionError("an empty pattern must raise ValueError")
        assert (text, patterns, replacements) == before


def check_input_immutability(solve):
    patterns = ["ba", "a", "ba", "aba"]
    replacements = ["aba", "", "ba", "a"]
    for text in ["ababa", "", "baaba", "ababa"]:
        _assert_case(solve, text, patterns, replacements)
    shared = ["ab", "a", "ab"]
    _assert_case(solve, "aba", shared, shared)


def check_maximum_bounds_dense_matches(solve):
    patterns = ["a" * length for length in range(1, 31)]
    _assert_case(solve, "a" * 500, patterns, [str(length) for length in range(1, 31)])
    _assert_case(solve, "猫" * 500, ["猫" * 30] * 30, [""] * 30)


def check_seeded_random(solve):
    rng = random.Random(8201)
    alphabet = ["a", "b", "c", "é", "\u0301", "猫", "\U0001f642"]
    for _ in range(180):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(71)))
        patterns = []
        for _ in range(rng.randrange(16)):
            if patterns and rng.random() < 0.2:
                patterns.append(rng.choice(patterns))
            elif text and rng.random() < 0.65:
                start = rng.randrange(len(text))
                patterns.append(text[start:start + rng.randint(1, 9)])
            else:
                patterns.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 9))))
        replacements = ["".join(rng.choice(alphabet) for _ in range(rng.randrange(6))) for _ in patterns]
        _assert_case(solve, text, patterns, replacements)


check_failure_inheritance_example.example = True
check_unicode_example.example = True
