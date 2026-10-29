# Frozen outputs from transformers 5.20.2 smart_resize avoid a cloned formula oracle.
_FIXTURES = [
    ((560, 1120, 14, 2, 3136, 156800), (280, 560)),
    ((28, 5600, 14, 2, 3136, 3136), (28, 784)),
    ((35, 63, 7, 2, 196, 100000), (28, 56)),
    ((14, 14, 14, 2, 3136, 1003520), (56, 56)),
    ((500, 1000, 14, 2, 3136, 1003520), (504, 1008)),
    ((1001, 337, 14, 2, 3136, 100352), (532, 168)),
    ((337, 1001, 14, 2, 3136, 100352), (168, 532)),
    ((1, 200, 1, 1, 4, 8), (1, 40)),
    ((200, 1, 1, 1, 4, 8), (40, 1)),
    ((442, 884, 5, 2, 1300, 7200), (50, 110)),
    ((3, 5, 1, 2, 4, 64), (4, 4)),
    ((5, 7, 1, 2, 4, 64), (4, 8)),
    ((7, 9, 1, 2, 4, 64), (8, 8)),
    ((9, 11, 1, 2, 4, 200), (8, 12)),
    ((63, 91, 7, 2, 196, 100000), (56, 84)),
    ((100, 101, 3, 3, 162, 810), (27, 27)),
    ((100, 101, 3, 3, 162, 20000), (99, 99)),
    ((1, 1, 7, 4, 784, 784), (28, 28)),
    ((101, 99, 2, 3, 36, 36), (6, 6)),
    ((127, 251, 4, 2, 128, 4096), (40, 88)),
    ((250, 50, 5, 2, 400, 900), (60, 10)),
    ((29, 29, 14, 2, 3136, 3136), (56, 56)),
    ((27, 27, 14, 2, 784, 784), (28, 28)),
    ((20, 20, 14, 2, 784, 784), (28, 28)),
    ((100, 100, 5, 2, 10000, 10000), (100, 100)),
    ((3, 100, 5, 2, 100, 100), (10, 60)),
]


def _assert_case(solve, args, expected_hw):
    result = solve(*args)
    h, w = expected_hw
    _, _, patch, merge, lower, upper = args
    assert isinstance(result, dict)
    assert set(result) == {"resized_hw", "grid_thw", "num_patches", "num_visual_tokens", "in_budget"}
    assert type(result["resized_hw"]) is tuple and result["resized_hw"] == (h, w)
    assert type(result["grid_thw"]) is tuple and result["grid_thw"] == (1, h // patch, w // patch)
    assert all(type(v) is int for v in result["resized_hw"] + result["grid_thw"])
    assert h > 0 and w > 0 and h % (patch * merge) == w % (patch * merge) == 0
    assert type(result["num_patches"]) is int
    assert result["num_patches"] * patch**2 == h * w
    assert type(result["num_visual_tokens"]) is int
    assert result["num_visual_tokens"] * merge**2 == result["num_patches"]
    assert type(result["in_budget"]) is bool
    assert result["in_budget"] == (lower <= h * w <= upper)
    return result


def check_example(solve):
    result = _assert_case(solve, *_FIXTURES[0])
    assert result == {
        "resized_hw": (280, 560), "grid_thw": (1, 20, 40),
        "num_patches": 800, "num_visual_tokens": 200, "in_budget": True,
    }


check_example.example = True


def check_upstream_fixtures(solve):
    for args, expected in _FIXTURES:
        _assert_case(solve, args, expected)


def check_transpose_symmetry(solve):
    for (h, w, p, m, lower, upper), (oh, ow) in _FIXTURES:
        _assert_case(solve, (w, h, p, m, lower, upper), (ow, oh))


def check_ties_to_even_without_scaling(solve):
    for factor in (2, 4, 10, 28):
        for h_multiple in range(1, 8):
            for w_multiple in range(1, 8):
                h = factor * h_multiple + factor // 2
                w = factor * w_multiple + factor // 2
                expected = ((h_multiple + h_multiple % 2) * factor,
                            (w_multiple + w_multiple % 2) * factor)
                _assert_case(solve, (h, w, factor, 1, factor**2, (factor * 20)**2), expected)


def check_already_aligned_exact_budget(solve):
    for patch, merge in ((1, 1), (2, 3), (7, 2), (14, 2)):
        for a, b in ((1, 1), (2, 7), (7, 2), (11, 19)):
            h, w = patch * merge * a, patch * merge * b
            _assert_case(solve, (h, w, patch, merge, h*w, h*w), (h, w))


def check_same_factor_different_patch_merge(solve):
    for patch, merge in ((1, 28), (2, 14), (4, 7), (7, 4), (14, 2), (28, 1)):
        result = _assert_case(solve, (560, 1120, patch, merge, 3136, 156800), (280, 560))
        assert result["num_visual_tokens"] == 200


def check_budget_is_not_repaired_after_rounding(solve):
    result = _assert_case(solve, (28, 5600, 14, 2, 3136, 3136), (28, 784))
    assert result["in_budget"] is False
    result = _assert_case(solve, (250, 50, 5, 2, 800, 900), (60, 10))
    assert result["in_budget"] is False


def check_python_float_boundary(solve):
    _assert_case(solve, (442, 884, 5, 2, 1300, 7200), (50, 110))


def check_invalid_arguments(solve):
    valid = [56, 112, 14, 2, 3136, 1003520]
    for index in range(6):
        for invalid in (0, -1, True, 1.5, "2", None):
            args = valid.copy()
            args[index] = invalid
            try:
                solve(*args)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected ValueError: {args}")
    for args in ((1, 201, 1, 1, 1, 1000), (201, 1, 1, 1, 1, 1000),
                 (56, 112, 14, 2, 100, 1000), (56, 112, 14, 2, 4000, 3000)):
        try:
            solve(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: {args}")
