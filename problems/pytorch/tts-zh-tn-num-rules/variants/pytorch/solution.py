DIGITS = {str(i): tran for i, tran in enumerate('零一二三四五六七八九')}
UNITS = {1: '十', 2: '百', 3: '千', 4: '万', 8: '亿'}


def _get_value(value_string, use_zero=True):
    stripped = value_string.lstrip('0')
    if len(stripped) == 0:
        return []
    if len(stripped) == 1:
        if use_zero and len(stripped) < len(value_string):
            return [DIGITS['0'], DIGITS[stripped]]
        return [DIGITS[stripped]]
    largest_unit = next(power for power in reversed(sorted(UNITS.keys()))
                        if power < len(stripped))
    first_part = value_string[:-largest_unit]
    second_part = value_string[-largest_unit:]
    return _get_value(first_part) + [UNITS[largest_unit]] + _get_value(second_part)


def _verbalize_cardinal(value_string):
    if not value_string:
        return ''
    value_string = value_string.lstrip('0')
    if len(value_string) == 0:
        return DIGITS['0']
    result_symbols = _get_value(value_string)
    if len(result_symbols) >= 2 and result_symbols[0] == DIGITS['1'] \
            and result_symbols[1] == UNITS[1]:
        result_symbols = result_symbols[1:]
    return ''.join(result_symbols)


def _verbalize_digit(value_string, alt_one=False):
    result = ''.join(DIGITS[d] for d in value_string)
    if alt_one:
        result = result.replace('一', '幺')
    return result


def _num2str(value_string):
    parts = value_string.split('.')
    assert len(parts) <= 2
    integer = parts[0]
    decimal = parts[1] if len(parts) == 2 else ''
    result = _verbalize_cardinal(integer)
    decimal = decimal.rstrip('0')
    if decimal:
        result = result if result else '零'
        result += '点' + _verbalize_digit(decimal)
    return result


def _time_num2str(num_string):
    result = _num2str(num_string.lstrip('0'))
    if num_string.startswith('0'):
        result = DIGITS['0'] + result
    return result


def _solve_time(value):
    h, rest = value.split(':', 1)
    parts = rest.split(':')
    minute = parts[0]
    second = parts[1] if len(parts) == 2 else None
    result = f"{_num2str(h)}点"
    if minute.lstrip('0'):
        if int(minute) == 30:
            result += "半"
        else:
            result += f"{_time_num2str(minute)}分"
    if second and second.lstrip('0'):
        result += f"{_time_num2str(second)}秒"
    return result


def solve(value, mode, alt_one):
    if mode == "cardinal":
        return _verbalize_cardinal(value)
    if mode == "digit":
        return _verbalize_digit(value, alt_one)
    if mode == "num":
        return _num2str(value)
    if mode == "time":
        return _solve_time(value)
    if mode == "phone_mobile":
        parts = value.strip('+').split()
        return '，'.join(_verbalize_digit(p, alt_one=True) for p in parts)
    if mode == "phone_fixed":
        parts = value.split('-')
        return '，'.join(_verbalize_digit(p, alt_one=True) for p in parts)
    raise AssertionError(f"unknown mode {mode}")
