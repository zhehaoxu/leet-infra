# Chinese TN number verbalization rule engine

## Problem

The TTS text frontend's TN (text normalization) turns number strings into Chinese
readings. Reproduce the purely rule-based parts of PaddleSpeech `zh_normalization`
(`num.py` / `chronology.py` / `phonecode.py`):

1. **cardinal myriad sectioning**: `verbalize_cardinal(value)`
   - `lstrip('0')` first (**leading zeros dropped**: "07" -> "七"); all zeros -> "零".
   - `_get_value` recursion: the largest power in `UNITS = {1:十, 2:百, 3:千, 4:万, 8:亿}`
     with `power < len(stripped)`; split `value[:-k] / value[-k:]` **on the original
     string** (not stripped — this is the key convention), then recurse.
   - **Mid-string zero padding**: at leaves where stripped has length 1 but is shorter
     than its input, prepend "零" ("1001" -> "一千零一", "100010001" -> "一亿零一万零一").
   - **"一十" abbreviation applies only to the whole symbol string's head**: "10"->"十",
     "100000"->"十万"; no abbreviation inside segments ("110" -> "一百一十").
2. **digit reading**: `verbalize_digit(value, alt_one)`; `alt_one=True` maps "一"->"幺"
   (only digit 1; "07" -> "零七", "410" -> "四幺零").
3. **num2str**: integer part via cardinal; decimal part `rstrip('0')` first
   ("3.20" -> "三点二"), digits read one by one (zeros included); empty integer reading
   becomes "零" (".22" -> "零点二二"); more than one "." is an illegal input (assert).
4. **time**: `H:MM` or `H:MM:SS`. Minute == 30 -> "半"; otherwise num2str after lstrip
   with a "零" prepended when the original started with 0 ("05" -> "零五"); all-zero
   fields omitted ("8:00" -> "八点", "0:00" -> "零点"); seconds follow the same rule.
   Hour goes through num2str ("08" -> "八", "20" -> "二十").
5. **phone_mobile**: strip a leading '+' then split on **spaces**, per-digit alt_one
   reading, segments joined with full-width "，". **phone_fixed**: split on "-".

## Interface

```python
solve(value: str, mode: str, alt_one: bool) -> str
# mode in {"cardinal", "digit", "num", "time", "phone_mobile", "phone_fixed"}
```

## Example

```python
solve("1001", "cardinal", False)       # -> "一千零一"
solve("100000001", "cardinal", False)  # -> "一亿零一"
solve("110", "cardinal", False)        # -> "一百一十" (no in-segment abbreviation)
solve("07", "cardinal", False)         # -> "七" (leading zeros dropped)
solve("3.20", "num", False)            # -> "三点二"
solve(".22", "num", False)             # -> "零点二二"
solve("8:05", "time", False)           # -> "八点零五分"
solve("8:30", "time", False)           # -> "八点半"
solve("8:00", "time", False)           # -> "八点"
solve("138 1234 5678", "phone_mobile", False)  # -> "幺三八，幺二三四，五六七八"
solve("010-62765678", "phone_fixed", False)    # -> "零幺零，六二七六五六七八"
```

## Scale and judging

- value length <= 12 (cardinal/num); time format pre-validated; phone segments only 0-9.
- Random cross-checks: 300 random integers where the official string recursion and an
  independent numeric method (myriad standard reading) must agree; 200 more for
  decimals/times/phones.
- Mutations 14/14 caught (dropping lstrip, flipping the power condition, dropping zero
  padding, loosening the abbreviation, keeping trailing zeros, dropping the 半 special
  case, wrong alt_one string, wrong split character, full-width -> ASCII comma, etc.).
