# 工具循环熔断器：指纹去重账

## 题意

LLM agent 运行时常见故障模式是工具调用死循环（同参数反复调用）。工程护栏是对每步调用做**指纹去重计数**，同一指纹出现次数超过阈值即熔断。本题给定逐步调用列表 `calls`（每项 `{"tool", "args"}`）与阈值 `threshold`，逐账输出：每步是否放行、指纹、调用前该指纹已出现次数、以及首次熔断步号。

## 求解

指纹口径题面钉死：

1. `args` 规范化：递归把所有 `float` round 到 6 位小数（`round(v, 6)`；`bool` 优先于 `int` 判定），再 `json.dumps(..., sort_keys=True, separators=(',', ':'))`——键序无关、1e-7 级浮点抖动视为相同；
2. `fingerprint = tool + "|" + sha256(canonical).hexdigest()`（64 位十六进制）。

判定账目：设本次调用前该指纹已出现 `c` 次（记入 `counts_at_call`），`c < threshold` 则放行并把计数 +1，否则熔断（`allowed=False`，计数不增）。`threshold=0` 时第一步即熔断。`first_block_index` 取首个熔断步号（0 起），从未熔断为 `None`。返回 `{"allowed", "fingerprints", "counts_at_call", "first_block_index"}`。

## 数据规模

- 1 ≤ len(calls) ≤ 10^4，args 可含 float/int/str/bool/None 及嵌套 list/dict。
- calls 为空/非列表、条目缺键或多键、tool 非非空 str、args 非 dict、threshold 为 bool/负数/非 int → `ValueError`。
