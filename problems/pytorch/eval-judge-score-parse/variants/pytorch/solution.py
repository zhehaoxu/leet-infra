"""LMSYS FastChat LLM-as-judge：裁判输出里的数值分数串解析与单点评分表。"""

import ast
import math
import re

# 与 fastchat/llm_judge/common.py:34-37 逐字同源（成对 / 单点两套，各有主模式与备份模式）
TWO_SCORE = re.compile(r"\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
TWO_SCORE_BACKUP = re.compile(r"\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
ONE_SCORE = re.compile(r"\[\[(\d+\.?\d*)\]\]")
ONE_SCORE_BACKUP = re.compile(r"\[(\d+\.?\d*)\]")

TIE_DELTA = 0.1


def _number(tok):
    """官方 `ast.literal_eval(match.groups()[0])` 口径：无小数点得 int，有得 float。

    `007` / `08` 这类前导零不是合法 Python 字面量，官方此处抛 SyntaxError（未捕获）；
    本题统一归一化成 ValueError。
    """
    try:
        return ast.literal_eval(tok)
    except (ValueError, SyntaxError):
        raise ValueError(f"非法分数字面量: {tok!r}")


def _parse_pair(text):
    """成对打分 `[[a, b]]`：先主模式全串搜，未命中再备份模式。返回 (pattern, scores|None)。"""
    m = TWO_SCORE.search(text)
    if m is not None:
        a, b = m.groups()
        return "two", [_number(a), _number(b)]
    m = TWO_SCORE_BACKUP.search(text)
    if m is not None:
        a, b = m.groups()
        return "two_backup", [_number(a), _number(b)]
    return "none", None


def _parse_one(text):
    """单点打分 `[[r]]`：同样的主 / 备份两级。返回 (pattern, score|None)。"""
    m = ONE_SCORE.search(text)
    if m is not None:
        return "one", _number(m.groups()[0])
    m = ONE_SCORE_BACKUP.search(text)
    if m is not None:
        return "one_backup", _number(m.groups()[0])
    return "none", None


def solve(entries, mode="pair", tie_delta=TIE_DELTA):
    if not isinstance(mode, str) or mode not in ("pair", "single"):
        raise ValueError("mode 必须是 'pair' 或 'single'")
    if isinstance(tie_delta, bool) or not isinstance(tie_delta, (int, float)):
        raise ValueError("tie_delta 必须是实数")
    td = float(tie_delta)
    if not math.isfinite(td) or td < 0:
        raise ValueError("tie_delta 必须是有限非负数")
    if not isinstance(entries, (list, tuple)):
        raise ValueError("entries 必须是列表")

    if mode == "pair":
        rows = []
        for text in entries:
            if not isinstance(text, str):
                raise ValueError("成对模式每条必须是字符串")
            pat, sc = _parse_pair(text)
            if sc is None:
                verdict = "error"
            elif abs(sc[0] - sc[1]) <= td:
                verdict = "tie"
            else:
                verdict = "A" if sc[0] > sc[1] else "B"
            rows.append({"pattern": pat, "scores": sc, "verdict": verdict})
        return {"mode": mode, "rows": rows, "table": {}, "rank": []}

    rows = []
    for e in entries:
        if not isinstance(e, dict):
            raise ValueError("单点模式每条必须是 dict")
        for key in ("model", "turn", "text"):
            if key not in e:
                raise ValueError(f"单点模式缺少字段 {key}")
        if not isinstance(e["model"], str) or not e["model"]:
            raise ValueError("model 必须是非空字符串")
        if not isinstance(e["turn"], int) or isinstance(e["turn"], bool):
            raise ValueError("turn 必须是整数")
        if e["turn"] not in (1, 2):
            raise ValueError("turn 只能是 1 或 2")
        if not isinstance(e["text"], str):
            raise ValueError("text 必须是字符串")
        pat, sc = _parse_one(e["text"])
        rows.append(
            {
                "model": e["model"],
                "turn": e["turn"],
                "pattern": pat,
                "score": -1 if sc is None else sc,
            }
        )

    buckets = {}
    for r in rows:
        if r["score"] == -1:
            continue  # show_result.py:20 剔除解析失败（-1）的行
        b = buckets.setdefault(r["model"], {1: [], 2: []})
        b[r["turn"]].append(r["score"])

    table = {}
    for name, b in buckets.items():
        every = b[1] + b[2]
        table[name] = {
            "turn_1": sum(b[1]) / len(b[1]) if b[1] else None,
            "turn_2": sum(b[2]) / len(b[2]) if b[2] else None,
            "avg": sum(every) / len(every),
        }
    rank = sorted(table, key=lambda n: (-table[n]["avg"], n))
    return {"mode": mode, "rows": rows, "table": table, "rank": rank}
