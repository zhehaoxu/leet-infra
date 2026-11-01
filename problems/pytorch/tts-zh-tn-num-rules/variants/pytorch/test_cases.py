from collections.abc import Callable

DIGITS = {str(i): tran for i, tran in enumerate('零一二三四五六七八九')}
UNITS = {1: '十', 2: '百', 3: '千', 4: '万', 8: '亿'}


def _oracle_get_value(value_string, use_zero=True):
    stripped = value_string.lstrip('0')
    if len(stripped) == 0:
        return []
    if len(stripped) == 1:
        if use_zero and len(stripped) < len(value_string):
            return [DIGITS['0'], DIGITS[stripped]]
        return [DIGITS[stripped]]
    largest_unit = next(p for p in reversed(sorted(UNITS.keys())) if p < len(stripped))
    return (_oracle_get_value(value_string[:-largest_unit])
            + [UNITS[largest_unit]]
            + _oracle_get_value(value_string[-largest_unit:]))


def _oracle_cardinal(v):
    if not v:
        return ''
    v = v.lstrip('0')
    if not v:
        return DIGITS['0']
    syms = _oracle_get_value(v)
    if len(syms) >= 2 and syms[0] == DIGITS['1'] and syms[1] == UNITS[1]:
        syms = syms[1:]
    return ''.join(syms)


def _oracle_digit(v, alt_one=False):
    r = ''.join(DIGITS[d] for d in v)
    return r.replace('一', '幺') if alt_one else r


def _oracle_num(v):
    a, _, b = v.partition('.')
    res = _oracle_cardinal(a)
    b = b.rstrip('0')
    if b:
        res = res if res else '零'
        res += '点' + _oracle_digit(b)
    return res


def _oracle_time(v):
    h, rest = v.split(':', 1)
    parts = rest.split(':')
    minute = parts[0]
    second = parts[1] if len(parts) == 2 else None
    res = f"{_oracle_num(h)}点"
    if minute.lstrip('0'):
        res += "半" if int(minute) == 30 else f"{_oracle_time_sub(minute)}分"
    if second and second.lstrip('0'):
        res += f"{_oracle_time_sub(second)}秒"
    return res


def _oracle_time_sub(s):
    r = _oracle_num(s.lstrip('0'))
    if s.startswith('0'):
        r = DIGITS['0'] + r
    return r


def _int2zh_v2(n):
    # 独立数值法（万进制标准读数），与官方字符串递归在无前导零整数上必须一致
    if n == 0:
        return '零'
    d = '零一二三四五六七八九'
    units = ['', '十', '百', '千']

    def four(x, head):
        out = []
        need_zero = False
        for i in (3, 2, 1, 0):
            digit = x // 10 ** i % 10
            if digit == 0:
                need_zero = need_zero or bool(out)
                continue
            if need_zero:
                out.append('零')
                need_zero = False
            out.append(d[digit] + units[i])
        r = ''.join(out)
        if head and r.startswith('一十'):
            r = r[1:]
        return r

    yi = n // 10 ** 8
    wan = n // 10 ** 4 % 10 ** 4
    ge = n % 10 ** 4
    parts = []
    if yi:
        parts.append(four(yi, True) + '亿')
    if wan:
        if yi and wan < 1000:
            parts.append('零')
        parts.append(four(wan, not yi) + '万')
    elif yi:
        parts.append('零')
    if ge:
        if (yi or wan) and ge < 1000:
            parts.append('零')
        parts.append(four(ge, False))
    return ''.join(parts)


def _oracle_solve(value, mode, alt_one):
    if mode == "cardinal":
        return _oracle_cardinal(value)
    if mode == "digit":
        return _oracle_digit(value, alt_one)
    if mode == "num":
        return _oracle_num(value)
    if mode == "time":
        return _oracle_time(value)
    if mode == "phone_mobile":
        return '，'.join(_oracle_digit(p, alt_one=True) for p in value.strip('+').split())
    if mode == "phone_fixed":
        return '，'.join(_oracle_digit(p, alt_one=True) for p in value.split('-'))
    raise AssertionError


def _compare(solve: Callable, value, mode, alt_one=False):
    led = solve(value, mode, alt_one)
    exp = _oracle_solve(value, mode, alt_one)
    assert led == exp, f"{value!r}/{mode}: led={led!r} exp={exp!r}"
    return led


def check_cardinal_sectioning(solve):
    assert _compare(solve, "1001", "cardinal") == "一千零一"
    assert _compare(solve, "100000001", "cardinal") == "一亿零一"
    assert _compare(solve, "100010001", "cardinal") == "一亿零一万零一"
    assert _compare(solve, "12345678", "cardinal") == "一千二百三十四万五千六百七十八"
    assert _compare(solve, "10000", "cardinal") == "一万"
    assert _compare(solve, "100000", "cardinal") == "十万"


def check_leading_zero_and_all_zero(solve):
    assert _compare(solve, "0", "cardinal") == "零"
    assert _compare(solve, "000", "cardinal") == "零"
    assert _compare(solve, "07", "cardinal") == "七"     # 前导零丢，不补
    assert _compare(solve, "0078", "cardinal") == "七十八"
    assert _compare(solve, "07", "digit", True) == "零七"  # digit 逐位不丢零；幺只替换一


def check_inner_ten_no_abbrev(solve):
    # 一十缩写只对整体符号串首位生效，段内不缩
    assert _compare(solve, "10", "cardinal") == "十"
    assert _compare(solve, "110", "cardinal") == "一百一十"
    assert _compare(solve, "100010", "cardinal") == "十万零一十"
    assert _compare(solve, "1000000010", "cardinal") == "十亿零一十"


def check_decimal_rules(solve):
    assert _compare(solve, "3.20", "num") == "三点二"       # 尾零剥离
    assert _compare(solve, "10.100", "num") == "十点一"
    assert _compare(solve, ".22", "num") == "零点二二"      # 空整数部分补零
    assert _compare(solve, "0.05", "num") == "零点零五"     # 小数零逐位
    assert _compare(solve, "5", "num") == "五"


def check_time_rules(solve):
    assert _compare(solve, "8:30", "time") == "八点半"
    assert _compare(solve, "8:05", "time") == "八点零五分"
    assert _compare(solve, "8:00", "time") == "八点"
    assert _compare(solve, "0:00", "time") == "零点"
    assert _compare(solve, "20:30", "time") == "二十点半"
    assert _compare(solve, "9:15:30", "time") == "九点十五分三十秒"
    assert _compare(solve, "23:59:59", "time") == "二十三点五十九分五十九秒"
    assert _compare(solve, "0:30", "time") == "零点半"


def check_phone_rules(solve):
    assert _compare(solve, "138 1234 5678", "phone_mobile") == "幺三八，幺二三四，五六七八"
    led = solve("+86 138 1234 5678", "phone_mobile", False)
    # '+' strip 后按空格分段：86 / 138 / 1234 / 5678
    assert led == "八六，幺三八，幺二三四，五六七八", led
    assert _compare(solve, "010-62765678", "phone_fixed") == "零幺零，六二七六五六七八"
    assert _compare(solve, "410", "digit", True) == "四幺零"  # alt_one 只幺化一


def check_digit_alt_one(solve):
    assert _compare(solve, "13812345678", "digit", True) == "幺三八幺二三四五六七八"
    assert _compare(solve, "13812345678", "digit", False) == "一三八一二三四五六七八"
    assert _compare(solve, "0", "digit", False) == "零"


def check_invalid_inputs(solve):
    # num 模式两个点 → 断言
    try:
        solve("1.2.3", "num", False)
        raise SystemError("expected assertion")
    except AssertionError:
        pass
    # 未知 mode → 断言
    try:
        solve("1", "banana", False)
        raise SystemError("expected assertion")
    except AssertionError:
        pass


def check_random_cross_check(solve):
    import random
    rng = random.Random(696969)
    # 整数：官方字符串递归 vs 独立数值法（万进制标准读数）
    for _ in range(300):
        n = rng.randint(0, 10 ** 9)
        assert _oracle_cardinal(str(n)) == _int2zh_v2(n), n
        assert solve(str(n), "cardinal", False) == _int2zh_v2(n), n
    # 小数 / 时刻 / 号码 随机
    for _ in range(200):
        a = str(rng.randint(0, 10 ** 6))
        b = str(rng.randint(0, 999999)).rstrip('0') or '1'
        solve(f"{a}.{b}", "num", False) == _oracle_solve(f"{a}.{b}", "num", False)
        h = rng.randint(0, 23)
        m = rng.randint(0, 59)
        s = rng.randint(0, 59)
        tv = f"{h}:{m:02d}:{s:02d}"
        assert solve(tv, "time", False) == _oracle_solve(tv, "time", False), tv
        ph = " ".join(str(rng.randint(0, 9999)) for _ in range(rng.randint(1, 4)))
        assert solve(ph, "phone_mobile", False) == _oracle_solve(ph, "phone_mobile", False)
