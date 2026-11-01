# 中文 TN（Text Normalization）数字读法规引擎

PaddleSpeech TTS 前端 `zh_normalization`（`num.py` / `chronology.py` / `phonecode.py`）
把数字字符串转成可发音的中文读法。本题复现其中**纯规则部分**（不含正则调度，
输入已是剥好的数字/时刻/号码串）：

## 接口

```
solve(value: str, mode: str, alt_one: bool) -> str
```

- `mode="cardinal"`：基数百读法 `verbalize_cardinal(value)`
- `mode="digit"`：逐位读 `verbalize_digit(value, alt_one)`（alt_one=True 时 "一"→"幺"）
- `mode="num"`：`num2str(value)`——整数部分走 cardinal，小数部分逐位（尾零剥离）
- `mode="time"`：`value` 为 `H:MM` 或 `H:MM:SS`（H∈0-23 可 1-2 位，M/S 恒 2 位）
- `mode="phone_mobile"`：按**空格**分段（先 strip 掉开头的 '+'），各段逐位 alt_one 语义，"，" 连接
- `mode="phone_fixed"`：按 **"-"** 分段，其余同上

## 官方口径（对齐源码，逐条）

1. **cardinal 分节**：先 `lstrip('0')`（前导零全丢，"07"→"七"）；全零 → "零"。
   `_get_value` 递归：取 `stripped` 长度内最大单位幂（10=十、100=百、1000=千、
   10000=万、1e8=亿，条件 `power < len(stripped)`，**基于原串切分**
   `value[:-k] / value[-k:]`），左右两半递归。
2. **中间零补零**：递归中 stripped 长度 1 且小于原串长度（如 "001" 的 "1"）→
   先补 "零" 再读位；stripped 为空 → 空。于是 "1001"→"一千零一"、"100010001"→
   "一亿零一万零一"；但**串首前导零不补**（lstrip 先行）。
3. **一十缩写**：仅当整体符号串以 "一十" 开头时去掉（"10"→"十"、"10 万"→"十万"）；
   **段内不缩**（"110"→"一百一十"）。
4. **num2str 小数**：按 "." 拆（两个以上点为非法输入，断言）；小数部分先
   `rstrip('0')`（"3.20"→"三点二"、"10.100"→"十点一"）；整数部分读数为空串时补 "零"
   （".22"→"零点二二"）；小数部分逐位读（含零，"0.05"→"零点零五"）。
5. **时刻**：`H点` + 分钟（minute=="30" → "半"；分钟去前导零后非空 →
   `_time_num2str`：**先补回一个 "零" 再 num2str**，"05"→"零五"）+ 秒同规则；
   分钟/秒全零则省略（"8:00"→"八点"、"0:00"→"零点"）。hour 走 num2str
   （"08"→"八"、"20"→"二十"）。
6. **号码**：分段逐位读，alt_one 强制开启（幺化），段间 "，"（全角逗号）连接；
   mobile 段按空格切（strip 前导 '+'），fixed 段按 "-" 切。

## 示例

```python
solve("1001", "cardinal", False)            # -> "一千零一"
solve("100000001", "cardinal", False)       # -> "一亿零一"
solve("110", "cardinal", False)             # -> "一百一十"（段内不缩）
solve("07", "cardinal", False)              # -> "七"（前导零丢，不补）
solve("100010001", "cardinal", False)       # -> "一亿零一万零一"
solve("3.20", "num", False)                 # -> "三点二"
solve(".22", "num", False)                  # -> "零点二二"
solve("8:05", "time", False)                # -> "八点零五分"
solve("8:30", "time", False)                # -> "八点半"
solve("8:00", "time", False)                # -> "八点"
solve("9:15:30", "time", False)             # -> "九点十五分三十秒"
solve("0:00", "time", False)                # -> "零点"
solve("20:30", "time", False)               # -> "二十点半"
solve("138 1234 5678", "phone_mobile", False)  # -> "幺三八，幺二三四，五六七八"
solve("010-62765678", "phone_fixed", False)    # -> "零幺零，幺零六二七六五六七八"
```

## 约束

- cardinal/digit/num 的 value 只含 `0-9` 与至多一个 "."（num）；time 已校验格式；
  phone 各段只含 `0-9`。
- 读法中间不出现连续两个 "零"（官方递归天然保证：空段返回空）。

写你的实现到 `solve`。
