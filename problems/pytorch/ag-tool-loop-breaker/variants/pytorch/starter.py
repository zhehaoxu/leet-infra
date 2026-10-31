import hashlib
import json


def solve(calls, threshold):
    """工具循环熔断器：对逐步工具调用做指纹去重计数，超过阈值即熔断。

    每步调用 {'tool': str, 'args': dict}。指纹口径（题面钉死）：

      1. args 规范化：递归把所有 float round 到 6 位小数（round(v, 6)），
         然后 json.dumps(..., sort_keys=True, separators=(',', ':'))——
         键序无关、1e-7 级浮点抖动视为相同；
      2. fingerprint = tool + '|' + sha256(canonical).hexdigest()

    判定：同一 fingerprint 在**本次调用之前**已出现 c 次（counts_at_call），
    c < threshold 则允许并计数 +1，否则本步熔断（allowed=False，计数不增）。
    threshold=0 时第一步即熔断。

    参数
    ----
    calls     : 非空 list[dict]，每项恰有键 {'tool', 'args'}；
                tool 为非空 str；args 为 dict（可含 float/int/str/bool/None/
                嵌套 list/dict）
    threshold : 正 int（且非 bool），同一指纹允许出现的最大次数

    返回
    ----
    dict：
      allowed          list[bool]      每步是否放行
      fingerprints     list[str]       每步指纹（tool + '|' + 64 位十六进制）
      counts_at_call   list[int]       每步调用前该指纹已出现次数
      first_block_index  int | None    首次熔断步号（0 起），从未熔断为 None
    """
    raise NotImplementedError
