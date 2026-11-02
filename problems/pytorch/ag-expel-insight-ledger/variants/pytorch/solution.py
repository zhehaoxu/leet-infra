"""ExpeL 洞见账：规则操作的解析、计数增减与满额淘汰。

对照 LeapLabTHU/ExpeL 的 `agent/expel.py`：
- `parse_rules`(L665-681)：`((?:REMOVE|EDIT|ADD|AGREE)(?: \\d+|)): (?:[a-zA-Z\\s\\d]+: |)(.*)`
  抽两列，再按「非空 / 不含 ADD·AGREE·EDIT / 以句号结尾」三道门过滤。
- `update_rules`(L696-743)：先做一遍合法性修复（ADD 去重、EDIT 有歧义转 AGREE、编号越界丢弃、
  REMOVE/AGREE 找不到原文丢弃），再按 `REMOVE → AGREE → EDIT → ADD` 的固定顺序落账：
  REMOVE 减 1（列表满额时减 3）、AGREE 加 1、EDIT 换文本并加 1、ADD 追加且初始计数为 2。
- 收尾：丢掉计数 <= 0 的规则，再按计数**降序稳定排序**。
- 渲染（L399）：`"<i>. <文本>"` 逐行拼接，序号从 1 起。
LLM 输出、embedding、retriever 全部换成显式入参。
"""
import re

_OP_RE = re.compile(r"((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)")
_BANNED = ("ADD", "AGREE", "EDIT")
_ORDER = ("REMOVE", "AGREE", "EDIT", "ADD")


def _as_text(value, name):
    if not isinstance(value, str):
        raise ValueError("%s 必须是字符串" % name)
    return value


def _as_int(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s 必须是整数" % name)
    if minimum is not None and value < minimum:
        raise ValueError("%s 必须 >= %d" % (name, minimum))
    return value


def _op_type(operation):
    return operation.split(" ")[0]


def _op_number(operation):
    return int(operation.split(" ")[1]) if " " in operation else None


def parse_ops(text):
    out = []
    for operation, body in _OP_RE.findall(text):
        body = body.strip()
        if body == "" or not body.endswith("."):
            continue
        if any(word in body for word in _BANNED):
            continue
        if "ADD" in operation:
            out.append(["ADD", body])
        else:
            out.append([operation.strip(), body])
    return out


def _match_index(rules, text):
    for i, item in enumerate(rules):
        if item[0] in text:
            return i
    return None


def _update(rules, operations, list_full):
    fixed = [[op, txt] for op, txt in operations]
    dropped = set()
    for i in range(len(operations)):
        operation, text = operations[i]
        kind = _op_type(operation)
        number = _op_number(operation)
        if kind == "ADD":
            if _match_index(rules, text) is not None:
                dropped.add(i)
            continue
        if kind == "EDIT":
            index = _match_index(rules, text)
            if index is not None:
                fixed[i] = ["AGREE %d" % (index + 1), rules[index][0]]
            elif number is None or number > len(rules):
                dropped.add(i)
            continue
        if _match_index(rules, text) is None:
            dropped.add(i)
    applied = [tuple(fixed[i]) for i in range(len(operations)) if i not in dropped]

    for wanted in _ORDER:
        for operation, text in applied:
            if _op_type(operation) != wanted:
                continue
            if wanted == "REMOVE":
                rules[_match_index(rules, text)][1] -= 3 if list_full else 1
            elif wanted == "AGREE":
                rules[_match_index(rules, text)][1] += 1
            elif wanted == "EDIT":
                index = _op_number(operation) - 1
                rules[index] = [text, rules[index][1] + 1]
            else:
                rules.append([text, 2])

    rules = [item for item in rules if item[1] > 0]
    rules.sort(key=lambda item: item[1], reverse=True)
    return rules, applied


def solve(rounds, initial_rules=None, max_num_rules=10):
    if not isinstance(rounds, (list, tuple)):
        raise ValueError("rounds 必须是序列")
    max_rules = _as_int(max_num_rules, "max_num_rules", 0)
    rules = []
    for i, pair in enumerate(initial_rules or ()):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("initial_rules[%d] 必须是 (文本, 计数)" % i)
        text = _as_text(pair[0], "initial_rules[%d] 文本" % i)
        rules.append([text, _as_int(pair[1], "initial_rules[%d] 计数" % i)])

    log = []
    for i, raw in enumerate(rounds):
        if not isinstance(raw, str):
            raise ValueError("rounds[%d] 必须是字符串" % i)
        parsed = parse_ops(raw)
        list_full = max_rules + 5 <= len(rules)
        rules, applied = _update(rules, parsed, list_full)
        log.append({"round": i, "parsed": [list(op) for op in parsed],
                    "applied": [list(op) for op in applied],
                    "list_full": list_full, "rules": [list(r) for r in rules]})

    prompt = "\n".join("%d. %s" % (i, item[0]) for i, item in enumerate(rules, 1))
    return {"rules": rules, "prompt": prompt, "log": log, "count": len(rules)}
