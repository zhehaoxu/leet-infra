import json
import re


def solve(instruction_ids, kwargs_list, response):
    """转写 google-research/instruction-following-eval 的三类结构判别器
    （官方 check_following 原样口径）：json_format / postscript / two_responses。

    参数
    ----
    instruction_ids : list[str]，与 kwargs_list 等长（>=1），id 必须属于 3 类白名单
    kwargs_list     : list[dict]，每类参数（json_format/two_responses 传 {}）
    response        : str，被评响应

    3 类指令（observed 与 pass 口径，与官方逐字对齐）：
      detectable_format:json_format   kwargs {}
        预处理：value.strip() 后依次 removeprefix("```json")、"```Json"、
        "```JSON"、"```"（每个只剥一次），再 removesuffix("```")（只剥一次），
        再 strip()；json.loads 成功（捕获 ValueError）则 pass。
        observed = bool（能否解析）。注意 JSON 标量 "42"/"null"/"true" 合法，
        Python 大写 "True" 非法；"NaN"/"Infinity" 是 Python json 扩展，合法。
      detectable_content:postscript   kwargs {'postscript_marker': str}
        marker 先 strip()；value.lower() 后按 marker 精确分支：
          marker == "P.P.S" -> r"\\s*p\\.\\s?p\\.\\s?s.*$"
          marker == "P.S."  -> r"\\s*p\\.\\s?s\\..*$"
          其他             -> r"\\s*<marker.lower()>.*$"（marker 拼接进正则、
                              不 re.escape，marker 里的 . 等元字符按正则解释）
        用 re.findall(pattern, value, flags=re.MULTILINE) 匹配，
        observed = 匹配条数（任意行命中即一行一个），pass = observed > 0。
      combination:two_responses       kwargs {}
        value.split("******")；空段：位于 index 0 或末段则跳过，出现在中间
        直接 fail；pass = 非空段恰 2 段且 strip 后互不相等。
        observed = 非空段总数（含中间空段场景的全部非空段）。

    返回
    ----
    dict：
      results  list[dict]：{'instruction_id', 'pass' bool, 'observed'} 逐条
      all_pass bool
    """
    if not isinstance(response, str):
        raise ValueError("response must be a string")
    if not isinstance(instruction_ids, (list, tuple)) or len(instruction_ids) == 0:
        raise ValueError("instruction_ids must be a non-empty list")
    if len(instruction_ids) != len(kwargs_list):
        raise ValueError("instruction_ids/kwargs_list length mismatch")
    valid_ids = {
        "detectable_format:json_format",
        "detectable_content:postscript",
        "combination:two_responses",
    }
    for iid in instruction_ids:
        if iid not in valid_ids:
            raise ValueError(f"unknown instruction id: {iid}")
    if not all(isinstance(k, dict) for k in kwargs_list):
        raise ValueError("kwargs_list entries must be dicts")

    results = []
    for iid, kw in zip(instruction_ids, kwargs_list):
        if iid == "detectable_format:json_format":
            value = (
                response.strip()
                .removeprefix("```json")
                .removeprefix("```Json")
                .removeprefix("```JSON")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            try:
                json.loads(value)
            except ValueError:
                ok = False
            else:
                ok = True
            results.append({"instruction_id": iid, "observed": ok, "pass": ok})
        elif iid == "detectable_content:postscript":
            marker_raw = kw.get("postscript_marker")
            if not isinstance(marker_raw, str) or not marker_raw.strip():
                raise ValueError("postscript requires a non-empty postscript_marker")
            marker = marker_raw.strip()
            value = response.lower()
            if marker == "P.P.S":
                pattern = r"\s*p\.\s?p\.\s?s.*$"
            elif marker == "P.S.":
                pattern = r"\s*p\.\s?s\..*$"
            else:
                pattern = r"\s*" + marker.lower() + r".*$"
            hits = re.findall(pattern, value, flags=re.MULTILINE)
            results.append({"instruction_id": iid, "observed": len(hits),
                            "pass": len(hits) > 0})
        else:  # combination:two_responses
            parts = response.split("******")
            valid = []
            ok = True
            for index, seg in enumerate(parts):
                if not seg.strip():
                    if index != 0 and index != len(parts) - 1:
                        ok = False
                else:
                    valid.append(seg)
            n_valid = len(valid)
            ok = ok and n_valid == 2 and valid[0].strip() != valid[1].strip()
            results.append({"instruction_id": iid, "observed": n_valid, "pass": ok})

    return {
        "results": results,
        "all_pass": all(r["pass"] for r in results),
    }
