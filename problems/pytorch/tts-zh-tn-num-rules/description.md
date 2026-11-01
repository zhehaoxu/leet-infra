# 中文 TN 数字读法规引擎

## 题意

TTS 文本前端的 TN（Text Normalization）把数字串转成中文读法。本题复现 PaddleSpeech
`zh_normalization`（`num.py` / `chronology.py` / `phonecode.py`）的纯规则部分：

1. **cardinal 万/亿分节**：`verbalize_cardinal(value)`
   - 先 `lstrip('0')`（**前导零全丢**："07"→"七"）；全零 → "零"。
   - `_get_value` 递归：`UNITS = {1:十, 2:百, 3:千, 4:万, 8:亿}` 中满足
     `power < len(stripped)` 的最大幂，按 `value[:-k] / value[-k:]` **基于原串**
     切分递归（切分原串而非 stripped 是口径关键）。
   - **中间零补零**：叶子 stripped 长度 1 且小于其输入串 → 先补 "零"（"1001"→
     "一千零一"、"100010001"→"一亿零一万零一"）。
   - **一十缩写只对整体符号串首位**："10"→"十"、"100000"→"十万"；段内不缩
     （"110"→"一百一十"）。
2. **digit 逐位**：`verbalize_digit(value, alt_one)`；`alt_one=True` 把 "一"→"幺"
   （只影响数字 1，"07"→"零七"、"410"→"四幺零"）。
3. **num2str**：整数走 cardinal；小数部分先 `rstrip('0')`（"3.20"→"三点二"），
   逐位读（零也读）；整数读数为空补 "零"（".22"→"零点二二"）；两个以上点非法断言。
4. **time**：`H:MM` 或 `H:MM:SS`。分钟 == 30 → "半"；否则去前导零 num2str、原串
   以 0 开头先补 "零"（"05"→"零五"）；全零省略（"8:00"→"八点"、"0:00"→"零点"）；
   秒同规则。hour 走 num2str（"08"→"八"、"20"→"二十"）。
5. **phone_mobile**：strip 前导 '+' 后按**空格**分段，各段逐位幺化，全角 "，" 连接。
   **phone_fixed**：按 "-" 分段，其余同。

## 接口

```python
solve(value: str, mode: str, alt_one: bool) -> str
# mode ∈ {"cardinal", "digit", "num", "time", "phone_mobile", "phone_fixed"}
```

## 示例

```python
solve("1001", "cardinal", False)       # -> "一千零一"
solve("100000001", "cardinal", False)  # -> "一亿零一"
solve("110", "cardinal", False)        # -> "一百一十"（段内不缩）
solve("07", "cardinal", False)         # -> "七"（前导零丢）
solve("3.20", "num", False)            # -> "三点二"
solve(".22", "num", False)             # -> "零点二二"
solve("8:05", "time", False)           # -> "八点零五分"
solve("8:30", "time", False)           # -> "八点半"
solve("8:00", "time", False)           # -> "八点"
solve("138 1234 5678", "phone_mobile", False)  # -> "幺三八，幺二三四，五六七八"
solve("010-62765678", "phone_fixed", False)    # -> "零幺零，六二七六五六七八"
```

## 数据规模与判题

- value 长度 ≤ 12（cardinal/num）；time 已校验格式；phone 各段只含 0-9。
- 随机对拍：300 组整数上官方字符串递归与独立数值法（万进制标准读数）互证一致；
  小数/时刻/号码 200 组。
- 变异 14 条全抓（去 lstrip、幂条件翻转、去补零、缩写放宽、尾零不剥、半特判、
  幺化错串、分段符错、全角→ASCII 等）。
