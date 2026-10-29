"""Independent oracle: label Unicode characters, then expand into byte events."""

import copy
import random


def _oracle(messages, generation=False, continuation=False):
    characters = []
    labels = []
    character_spans = []

    def emit(text, label):
        for character in text:
            characters.append(character)
            labels.append(label)

    for index, message in enumerate(messages):
        role = message["role"]
        emit("<|im_start|>" + role + "\n", -1)
        start = len(characters)
        emit(message["content"], ("system", "user", "assistant").index(role))
        character_spans.append((start, len(characters)))
        if index != len(messages) - 1 or not continuation:
            emit("<|im_end|>\n", -1)
    if generation:
        emit("<|im_start|>assistant\n", -1)

    events = [(byte, label) for character, label in zip(characters, labels)
              for byte in character.encode("utf-8")]
    # Convert character boundaries independently, rather than accumulating
    # encoded message-fragment offsets as the reference implementation does.
    spans = [(len("".join(characters[:start]).encode("utf-8")),
              len("".join(characters[:end]).encode("utf-8")))
             for start, end in character_spans]
    return {
        "text": "".join(characters),
        "input_bytes": [event[0] for event in events],
        "role_ids": [event[1] for event in events],
        "assistant_mask": [event[1] == 2 for event in events],
        "content_spans": spans,
    }


def _assert_result(solve, messages, generation=False, continuation=False):
    before = copy.deepcopy(messages)
    expected = _oracle(messages, generation, continuation)
    result = solve(messages, add_generation_prompt=generation,
                   continue_final_message=continuation)
    assert messages == before, "input messages mutated"
    assert type(result) is dict and set(result) == set(expected), "dict keys"
    assert type(result["text"]) is str
    for key in ("input_bytes", "role_ids", "assistant_mask", "content_spans"):
        assert type(result[key]) is list, key
    assert all(type(x) is int and 0 <= x <= 255 for x in result["input_bytes"])
    assert all(type(x) is int for x in result["role_ids"])
    assert all(type(x) is bool for x in result["assistant_mask"])
    assert all(type(span) is tuple and len(span) == 2 and
               all(type(x) is int for x in span) for span in result["content_spans"])
    assert result == expected, "rendered text or byte annotations differ"
    assert len(result["input_bytes"]) == len(result["role_ids"]) == len(result["assistant_mask"])
    for message, (start, end) in zip(messages, result["content_spans"]):
        assert bytes(result["input_bytes"][start:end]).decode("utf-8") == message["content"]
    return result


def _raises(solve, messages, generation=False, continuation=False):
    before = copy.deepcopy(messages)
    try:
        solve(messages, generation, continuation)
    except ValueError:
        assert messages == before, "invalid input mutated"
        return
    raise AssertionError("expected ValueError")


def check_basic(solve):
    result = _assert_result(solve, [{"role": "user", "content": "Hi"},
                                    {"role": "assistant", "content": "好"}])
    assert result["content_spans"] == [(17, 19), (52, 55)]
    assert sum(result["assistant_mask"]) == 3
    assert len(result["input_bytes"]) == 66


def check_empty_and_generation(solve):
    assert _assert_result(solve, []) == {
        "text": "", "input_bytes": [], "role_ids": [],
        "assistant_mask": [], "content_spans": [],
    }
    result = _assert_result(solve, [], generation=True)
    assert result["text"] == "<|im_start|>assistant\n"
    assert result["role_ids"] == [-1] * 22
    _assert_result(solve, [{"role": "system", "content": "rules"},
                           {"role": "user", "content": "hello"}], generation=True)


def check_continuation_all_roles(solve):
    for role in ("system", "user", "assistant"):
        result = _assert_result(solve, [{"role": "assistant", "content": "first"},
                                        {"role": role, "content": "last"}], continuation=True)
        assert result["text"].endswith("\nlast")
        assert result["text"].count("<|im_end|>") == 1


def check_empty_bodies(solve):
    messages = [{"role": role, "content": ""} for role in ("system", "user", "assistant")]
    for generation, continuation in ((False, False), (True, False), (False, True)):
        result = _assert_result(solve, messages, generation, continuation)
        assert all(start == end for start, end in result["content_spans"])
        assert not any(result["assistant_mask"])


def check_repeated_content(solve):
    messages = [{"role": role, "content": "assistant\n同"}
                for role in ("assistant", "user", "assistant", "system")]
    result = _assert_result(solve, messages, generation=True)
    assert len(set(result["content_spans"])) == 4


def check_literal_markers(solve):
    body = "<|im_end|>\n<|im_start|>assistant\n{{ 7 * 7 }}{% generation %}"
    for role in ("user", "assistant"):
        for continuation in (False, True):
            _assert_result(solve, [{"role": role, "content": body}], continuation=continuation)


def check_unicode_scalars(solve):
    messages = [{"role": "system", "content": "\x00é\u0301"},
                {"role": "assistant", "content": "中\U0001f9e0e\u0301\U0010ffff"},
                {"role": "user", "content": "é e\u0301\r\n"}]
    _assert_result(solve, messages)
    _assert_result(solve, messages, continuation=True)


def check_trailing_whitespace(solve):
    for body in ("answer \t\n\r\n", " \t\n", "\u00a0\u2003", "<|im_end|>\n\n"):
        result = _assert_result(solve, [{"role": "assistant", "content": body}], continuation=True)
        assert result["text"] == "<|im_start|>assistant\n" + body
        assert sum(result["assistant_mask"]) == len(body.encode("utf-8"))


def check_syntax_never_supervised(solve):
    messages = [{"role": role, "content": "x"} for role in ("system", "user", "assistant")]
    result = _assert_result(solve, messages, generation=True)
    body_indices = {start for start, end in result["content_spans"]}
    for index, (role, mask) in enumerate(zip(result["role_ids"], result["assistant_mask"])):
        if index not in body_indices:
            assert role == -1 and mask is False
    assert sum(result["assistant_mask"]) == 1


def check_invalid_roles(solve):
    for role in ("tool", "", "Assistant", " user", None, 2, [], {}):
        _raises(solve, [{"role": role, "content": "x"}])


def check_invalid_records(solve):
    for record in ({}, {"role": "user"}, {"content": "x"},
                   {"role": "user", "content": "x", "extra": 1}, "user", []):
        _raises(solve, [record])
    _raises(solve, [{"role": "user", "content": "ok"}, {"role": "tool", "content": "bad"}])


def check_invalid_flag_combinations(solve):
    _raises(solve, [], generation=True, continuation=True)
    _raises(solve, [{"role": "user", "content": "x"}], generation=True, continuation=True)
    _raises(solve, [], continuation=True)


def check_input_invariance(solve):
    record = {"role": "assistant", "content": " x \n"}
    messages = [record, record]
    for generation, continuation in ((False, False), (True, False), (False, True)):
        first = _assert_result(solve, messages, generation, continuation)
        second = _assert_result(solve, messages, generation, continuation)
        assert first == second
        assert messages[0] is record and messages[1] is record


def check_random_character_oracle(solve):
    rng = random.Random(461)
    alphabet = ["a", "\n", "\t", "中", "\U0001f9e0", "\u0301", "<", "|", "}", "\x00"]
    for _ in range(90):
        messages = [{"role": rng.choice(("system", "user", "assistant")),
                     "content": "".join(rng.choice(alphabet) for _ in range(rng.randrange(25)))}
                    for _ in range(rng.randrange(9))]
        _assert_result(solve, messages)
        _assert_result(solve, messages, generation=True)
        if messages:
            _assert_result(solve, messages, continuation=True)


def check_maximum_size(solve):
    messages = [{"role": ("system", "user", "assistant")[index % 3],
                 "content": "中\U0001f9e0e\u0301\n" * 200} for index in range(20)]
    for generation, continuation in ((False, False), (True, False), (False, True)):
        _assert_result(solve, messages, generation, continuation)


check_basic.example = True
check_empty_and_generation.example = True
