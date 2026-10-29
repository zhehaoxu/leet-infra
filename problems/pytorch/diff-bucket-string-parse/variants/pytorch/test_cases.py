import warnings

BAD = (
    "",
    "   ",
    None,
    123,
    "512x768",
    "512,",
    ",768",
    "512,768,1024",
    "-512,768",
    "+512,768",
    "5e2,768",
    "512.0,768",
    "512,768;;256,256",
    "512,768;",
    "abc",
    "512 768",
    "0,512",
    "512,0",
    "0,0",
    "12a,16",
)


def _alt_parse(buckets_str):
    """独立实现：不用正则，改成 split(',') + str.isdigit 判定，用于对拍接受/拒绝。"""
    out = []
    for segment in buckets_str.strip().split(";"):
        parts = segment.split(",")
        if len(parts) != 2:
            raise ValueError(segment)
        height, width = parts[0].strip(), parts[1].strip()
        if not height.isdigit() or not width.isdigit():
            raise ValueError(segment)
        height, width = int(height), int(width)
        if height <= 0 or width <= 0:
            raise ValueError(segment)
        out.append((height, width))
    return out


def _record(solve, text):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = solve(text)
    return value, [str(item.message) for item in caught]


def check_example(solve):
    assert solve("1024,1024; 768,1344; 1344,768") == [(1024, 1024), (768, 1344), (1344, 768)]
    # 逗号两侧与段首尾的空白都允许
    assert solve("  512 , 768 ;  256 ,256 ") == [(512, 768), (256, 256)]
    # 保持顺序、重复项不合并
    assert solve("512,512;16,16;512,512") == [(512, 512), (16, 16), (512, 512)]


check_example.example = True


def check_return_types(solve):
    got = solve("1024,1024;768,1344")
    assert isinstance(got, list) and len(got) == 2
    for pair in got:
        assert isinstance(pair, tuple) and len(pair) == 2
        assert isinstance(pair[0], int) and isinstance(pair[1], int)
        assert not isinstance(pair[0], bool) and not isinstance(pair[1], bool)
    assert solve("8,8") == [(8, 8)], "单段也要能解析"


def check_warning_on_misalignment(solve):
    value, messages = _record(solve, "500,1008")
    assert value == [(500, 1008)], "不能整除 8 只是警告，桶要保留"
    assert len(messages) == 1, messages
    assert "not divisible by 8" in messages[0]

    _, none_warn = _record(solve, "512,768")
    assert none_warn == [], "两边都能被 8 整除时不应发警告"

    _, both_warn = _record(solve, "3,5")
    assert len(both_warn) == 1, "警告是按桶发的：两条边都不合规也只有一条"

    _, mixed = _record(solve, "1001,512;64,64")
    assert len(mixed) == 1, "只有第一段的 height 不能整除，第二段干净"

    _, two_bad = _record(solve, "3,5;7,9")
    assert len(two_bad) == 2, "两个不合规的桶应各发一条"


def check_rejects_malformed(solve):
    for text in BAD:
        try:
            solve(text)
            raise AssertionError(f"应当拒绝：{text!r}")
        except ValueError:
            pass


def check_error_messages(solve):
    try:
        solve("0,0")
        raise AssertionError("0 尺寸应当报错")
    except ValueError as error:
        message = str(error)
        assert "Invalid integer in bucket pair '0,0'" in message, message
        assert "positive integers" in message, message

    try:
        solve("512x768")
        raise AssertionError("非法分隔符应当报错")
    except ValueError as error:
        assert "Invalid bucket format" in str(error), str(error)


def check_against_alt(solve):
    cases = [
        "1024,1024",
        "1024,1024; 768,1344",
        " 8 , 8 ; 16 , 16 ",
        "512,512;512,512",
        "1000,24",
        "24,1000",
        "8192,8192; 4096,8192; 8192,4096",
    ]
    for text in cases:
        assert solve(text) == _alt_parse(text), text

    for text in BAD:
        if not isinstance(text, str) or not text:
            continue
        try:
            _alt_parse(text)
            alt_ok = True
        except ValueError:
            alt_ok = False
        assert not alt_ok, f"两种实现都应拒绝 {text!r}"


def check_single_pass_semantics(solve):
    # 只有 ";" 切分，"|" 与换行都不是分隔符
    try:
        solve("512,768|256,256")
        raise AssertionError("'|' 不是分隔符，应报错")
    except ValueError:
        pass
    assert solve("512,768\n") == [(512, 768)], "末尾换行会被 strip 掉"
