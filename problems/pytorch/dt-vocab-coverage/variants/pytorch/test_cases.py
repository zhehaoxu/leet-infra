from copy import deepcopy
import math
import random


_KEYS = {
    "vocab_counts", "vocab_doc_freq", "oov_tokens", "oov_counts",
    "oov_doc_freq", "document_stats", "token_coverage", "type_coverage",
}


def _oracle(documents, vocab, ignored_tokens):
    if len(set(vocab)) != len(vocab):
        raise ValueError("duplicate vocabulary token")
    retained = [[token for token in document if token not in ignored_tokens] for document in documents]
    flat = [token for document in retained for token in document]
    observed = set(flat)
    unknown = observed.difference(vocab)
    ranked = sorted((-flat.count(token), token) for token in unknown)
    oov_tokens = [token for _, token in ranked]
    known_occurrences = sum(token in vocab for token in flat)
    return {
        "vocab_counts": [sum(item == token for item in flat) for token in vocab],
        "vocab_doc_freq": [sum(token in document for document in retained) for token in vocab],
        "oov_tokens": oov_tokens,
        "oov_counts": [-negative_count for negative_count, _ in ranked],
        "oov_doc_freq": [sum(token in document for document in retained) for token in oov_tokens],
        "document_stats": [(len(document), sum(token not in vocab for token in document)) for document in retained],
        "token_coverage": known_occurrences / len(flat) if flat else 0.0,
        "type_coverage": len(observed.intersection(vocab)) / len(observed) if observed else 0.0,
    }


def _assert_case(solve, documents, vocab, ignored_tokens, expected=None):
    before = deepcopy((documents, vocab, ignored_tokens))
    reference = _oracle(documents, vocab, ignored_tokens)
    if expected is not None:
        assert reference == expected, (reference, expected)
    result = solve(documents, vocab, ignored_tokens)
    assert (documents, vocab, ignored_tokens) == before, "inputs were mutated"
    assert type(result) is dict and set(result) == _KEYS, "return exactly the eight specified keys"
    for key in ("vocab_counts", "vocab_doc_freq", "oov_counts", "oov_doc_freq"):
        assert type(result[key]) is list
        assert all(type(value) is int for value in result[key])
    assert type(result["oov_tokens"]) is list
    assert all(type(token) is str for token in result["oov_tokens"])
    assert type(result["document_stats"]) is list
    for item in result["document_stats"]:
        assert type(item) is tuple and len(item) == 2
        assert all(type(value) is int for value in item)
    for key in _KEYS:
        if key in ("token_coverage", "type_coverage"):
            assert type(result[key]) is float
            assert math.isclose(result[key], reference[key], rel_tol=1e-12, abs_tol=1e-12), (key, result, reference)
        else:
            assert result[key] == reference[key], (key, result, reference)
    return result


def check_basic_example(solve):
    _assert_case(solve, [["a", "x", "a"], [], ["x", "b"]], ["b", "a", "z"], ["b"], {
        "vocab_counts": [0, 2, 0],
        "vocab_doc_freq": [0, 1, 0],
        "oov_tokens": ["x"],
        "oov_counts": [2],
        "oov_doc_freq": [2],
        "document_stats": [(3, 1), (0, 0), (1, 1)],
        "token_coverage": 0.5,
        "type_coverage": 0.5,
    })


def check_occurrences_versus_document_frequency(solve):
    result = _assert_case(solve, [["a"] * 7 + ["x"] * 4, ["a", "x", "x"], [], ["y"] * 3], ["a"], [])
    assert result["vocab_counts"] == [8] and result["vocab_doc_freq"] == [2]
    assert result["oov_counts"] == [6, 3] and result["oov_doc_freq"] == [2, 1]
    assert result["document_stats"] == [(11, 4), (3, 2), (0, 0), (3, 3)]


def check_coverage_denominators(solve):
    result = _assert_case(solve, [["a"] * 8 + ["b", "x", "y"]], ["unused", "b", "a", "never"], [])
    assert math.isclose(result["token_coverage"], 9 / 11)
    assert result["type_coverage"] == 0.5
    result = _assert_case(solve, [["a", "a", "x", "x"], ["a", "y", "z"]], ["a"], ["x", "x"])
    assert result["token_coverage"] == 0.6
    assert math.isclose(result["type_coverage"], 1 / 3)


def check_ignored_inside_and_outside_vocab(solve):
    result = _assert_case(solve, [["known", "stop", "noise", "other"], ["stop", "noise", "known"]], ["stop", "known", "absent"], ["stop", "noise", "stop", "unused"])
    assert result["vocab_counts"] == [0, 2, 0]
    assert result["oov_tokens"] == ["other"]
    assert result["document_stats"] == [(2, 1), (1, 0)]


def check_empty_corpus_and_empty_documents(solve):
    for documents in ([], [[]], [[], [], []]):
        for vocab in ([], ["unused"]):
            result = _assert_case(solve, documents, vocab, ["ignored", "ignored"])
            assert result["document_stats"] == [(0, 0)] * len(documents)
            assert result["token_coverage"] == 0.0 and result["type_coverage"] == 0.0
    _assert_case(solve, [[], ["a"], [], ["x"], []], ["a"], [])


def check_all_ignored(solve):
    result = _assert_case(solve, [["a", "x", "x"], [], ["a"]], ["a", "unused"], ["x", "a", "x"])
    assert result["vocab_counts"] == [0, 0] and result["vocab_doc_freq"] == [0, 0]
    assert result["oov_tokens"] == [] and result["document_stats"] == [(0, 0), (0, 0), (0, 0)]
    assert result["token_coverage"] == 0.0 and result["type_coverage"] == 0.0


def check_unused_vocab_and_all_known(solve):
    result = _assert_case(solve, [["a", "b", "a"], [], ["b"]], ["never", "b", "a", "unused"], [])
    assert result["vocab_counts"] == [0, 2, 2, 0]
    assert result["vocab_doc_freq"] == [0, 2, 1, 0]
    assert result["token_coverage"] == 1.0 and result["type_coverage"] == 1.0
    assert result["oov_tokens"] == []


def check_empty_vocab_and_all_oov(solve):
    documents = [["x", "y", "x"], ["z", "y", "x"]]
    for vocab in ([], ["unused"]):
        result = _assert_case(solve, documents, vocab, [])
        assert result["oov_tokens"] == ["x", "y", "z"]
        assert result["oov_counts"] == [3, 2, 1]
        assert result["token_coverage"] == 0.0 and result["type_coverage"] == 0.0


def check_oov_count_and_lexical_ties(solve):
    result = _assert_case(solve, [["z", "b", "A", "a", "10", "2", "é", "猫", "\U0001f642"], ["z", "z"]], [], [])
    assert result["oov_tokens"] == ["z", "10", "2", "A", "a", "b", "é", "猫", "\U0001f642"]
    result = _assert_case(solve, [["z", "z", "a"], ["a"]], [], [])
    assert result["oov_tokens"] == ["a", "z"] and result["oov_doc_freq"] == [2, 1]


def check_unicode_example(solve):
    _assert_case(solve, [["é", "e\u0301", "É", "猫", "\U0001f642", "\U0001f642"], ["é", "É", "\U0001f642"]], ["é", "unused"], ["É"], {
        "vocab_counts": [2, 0],
        "vocab_doc_freq": [2, 0],
        "oov_tokens": ["\U0001f642", "e\u0301", "猫"],
        "oov_counts": [3, 1, 1],
        "oov_doc_freq": [2, 1, 1],
        "document_stats": [(5, 4), (2, 1)],
        "token_coverage": 2 / 7,
        "type_coverage": 0.25,
    })


def check_exact_matching_not_substrings_or_lowercase(solve):
    result = _assert_case(solve, [["a", "A", "aa", "a!", "é", "e\u0301", "\x00"]], ["A", "é", "aa"], ["a", "e\u0301"])
    assert result["vocab_counts"] == [1, 1, 1]
    assert result["oov_tokens"] == ["\x00", "a!"]
    assert result["document_stats"] == [(5, 2)]


def check_invalid_duplicate_vocab(solve):
    cases = [([], ["x", "x"], []), ([[]], ["a", "b", "a"], ["a"]), ([["é"]], ["é", "é"], ["é", "é"])]
    for documents, vocab, ignored_tokens in cases:
        before = deepcopy((documents, vocab, ignored_tokens))
        try:
            solve(documents, vocab, ignored_tokens)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate vocab entries must raise ValueError")
        assert (documents, vocab, ignored_tokens) == before


def check_input_immutability(solve):
    shared = ["b", "a", "x", "a"]
    documents = [shared, [], shared]
    vocab = ["b", "a", "z"]
    ignored_tokens = ["x", "x"]
    first = _assert_case(solve, documents, vocab, ignored_tokens)
    _assert_case(solve, [], vocab, ignored_tokens)
    assert _assert_case(solve, documents, vocab, ignored_tokens) == first
    assert documents[0] is shared and documents[2] is shared
    _assert_case(solve, [vocab], vocab, vocab)


def check_maximum_bounds(solve):
    vocab = ["v" + str(index).zfill(29) for index in range(100)]
    pool = vocab + ["u" + str(index).zfill(29) for index in range(40)]
    documents = [[pool[(doc * 17 + position * 3) % len(pool)] for position in range(100)] for doc in range(30)]
    _assert_case(solve, documents, vocab, [vocab[0], vocab[-1], pool[-1], vocab[0]])
    _assert_case(solve, [["猫" * 30] * 100 for _ in range(30)], ["猫" * 30], [])


def check_seeded_random(solve):
    rng = random.Random(8202)
    pool = ["a", "A", "aa", "x", "é", "e\u0301", "É", "猫", "\U0001f642", "2", "10", " ", "\x00"]
    for _ in range(180):
        documents = [[rng.choice(pool) for _ in range(rng.randrange(36))] for _ in range(rng.randrange(11))]
        vocab = rng.sample(pool + ["unused", "never"], rng.randrange(len(pool) + 3))
        ignored_tokens = [rng.choice(pool + ["absent"]) for _ in range(rng.randrange(12))]
        _assert_case(solve, documents, vocab, ignored_tokens)


check_basic_example.example = True
check_unicode_example.example = True
