"""ACE 技能书：delta 批次应用、计数器语义与提示词渲染账。

对照 kayba-ai/agentic-context-engine 的 `ace/core/skillbook.py`（899 行）：
- `apply_update` 顺序执行 `_apply_operation`，四种操作 ADD / UPDATE / TAG / REMOVE；
- `add_skill` 做 section 归一化、关键词去重、section 既非 context 也非 harness 时把
  section 名当关键词并把 section 折叠成 `context`、context 缺 insight 时用 issue 兜底；
- `update_skill` 用 `_UNSET` 哨兵区分「不改」与「改成空」；
- `tag_skill` 的 delta 只按符号取 ±1，0 记 neutral；
- `remove_skill` 默认软删（`active=False`），`purge` 才真删；
- `mark_used` 只给 active 的条目加计数；
- `as_prompt` 按固定顺序渲染两节、跳过空节、只有 insight 非空才输出该行。

随机与时钟全部剔除：`updated_at` / embedding 不进账本。
"""
import re as _re

SECTIONS = ("context", "harness")
LEGACY_SECTION = "context"
_WS = _re.compile(r"\s+")
_OPS = ("ADD", "UPDATE", "TAG", "REMOVE", "MARK_USED")


def _text(value):
    if value is None:
        return None
    out = str(value).strip()
    return out or None


def _required(value, name):
    out = _text(value)
    if out is None:
        raise ValueError("%s 必须是非空文本" % name)
    return out


def _keyword(value):
    out = _text(value)
    if out is None:
        return None
    out = _WS.sub("_", out.lower())
    return out or None


def _keywords(values):
    out, seen = [], set()
    if values is None:
        return out
    for value in values:
        kw = _keyword(value)
        if kw is None or kw in seen:
            continue
        out.append(kw)
        seen.add(kw)
    return out


def _as_skill(raw, name):
    if not isinstance(raw, dict):
        raise ValueError("%s 必须是字典" % name)
    for key in ("id", "section", "keywords", "issue"):
        if key not in raw:
            raise ValueError("%s 缺少字段 %r" % (name, key))
    if not isinstance(raw["id"], str) or not raw["id"]:
        raise ValueError("%s.id 必须是非空字符串" % name)
    section = raw["section"]
    if section not in SECTIONS:
        raise ValueError("%s.section 必须是 context / harness" % name)
    if not isinstance(raw["keywords"], (list, tuple)):
        raise ValueError("%s.keywords 必须是序列" % name)
    rec = {"id": raw["id"], "section": section,
           "keywords": [str(k) for k in raw["keywords"]],
           "issue": _required(raw["issue"], "%s.issue" % name),
           "insight": _text(raw.get("insight")),
           "active": bool(raw.get("active", True))}
    for key in ("used", "helpful", "harmful", "neutral"):
        value = raw.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("%s.%s 必须是非负整数" % (name, key))
        rec[key] = value
    return rec


def _lookup(skills, skill_id):
    return skills.get(skill_id)


def _add(skills, sections, op):
    raw_section = _required(op.get("section"), "ADD.section").lower()
    keywords = _keywords(op.get("keywords"))
    if raw_section in SECTIONS:
        section = raw_section
    else:                                   # 遗留的自由 section 折叠成 context
        section = LEGACY_SECTION
        keywords = _keywords([raw_section] + keywords)
    issue = _required(op.get("issue"), "ADD.issue")
    insight = _text(op.get("insight"))
    if not keywords:
        keywords = _keywords([section])
    if section == "context" and insight is None:
        if raw_section not in SECTIONS:
            insight = issue                 # 遗留 section 用 issue 兜底
        else:
            raise ValueError("context 条目必须带非空 insight")
    skill_id = op.get("skill_id")
    if skill_id is None:
        skill_id = "%s-%d" % (section, len(skills) + 1)
    if not isinstance(skill_id, str) or not skill_id:
        raise ValueError("ADD.skill_id 必须是非空字符串")
    skills[skill_id] = {"id": skill_id, "section": section, "keywords": keywords,
                        "issue": issue, "insight": insight, "active": True,
                        "used": 0, "helpful": 0, "harmful": 0, "neutral": 0}
    sections.setdefault(section, []).append(skill_id)   # 同 id 重复 ADD 会重复登记
    return skill_id


def _update(skills, op):
    skill_id = op.get("skill_id")
    if skill_id is None:
        return None, None
    skill = _lookup(skills, skill_id)
    if skill is None:
        return skill_id, None
    issue = op.get("issue")
    if issue is not None:
        skill["issue"] = _required(issue, "UPDATE.issue")
    raw_keywords = op.get("keywords")
    if raw_keywords:                        # 空列表等价于「不改」
        normalized = _keywords(raw_keywords)
        if not normalized:
            raise ValueError("keywords 不能全为空")
        skill["keywords"] = normalized
    insight = op.get("insight")
    if insight is not None:
        skill["insight"] = _text(insight)
    if skill["section"] == "context" and not skill["insight"]:
        raise ValueError("context 条目必须带非空 insight")
    return skill_id, skill


def _tag(skills, op):
    skill_id = op.get("skill_id")
    if skill_id is None:
        return None, None
    skill = _lookup(skills, skill_id)
    if skill is None:
        return skill_id, None
    delta = op.get("delta", 0)
    if isinstance(delta, bool) or not isinstance(delta, int):
        raise ValueError("TAG.delta 必须是整数")
    if delta > 0:
        skill["helpful"] += 1
    elif delta < 0:
        skill["harmful"] += 1
    else:
        skill["neutral"] += 1
    return skill_id, skill


def _remove(skills, sections, op):
    skill_id = op.get("skill_id")
    if skill_id is None:
        return None, "noop"
    skill = _lookup(skills, skill_id)
    if skill is None:
        return skill_id, "noop"
    if op.get("purge"):
        del skills[skill_id]
        bucket = sections.get(skill["section"])
        if bucket:
            sections[skill["section"]] = [s for s in bucket if s != skill_id]
            if not sections[skill["section"]]:
                del sections[skill["section"]]
        return skill_id, "purged"
    skill["active"] = False
    return skill_id, "removed"


def _mark_used(skills, op):
    ids = op.get("skill_ids")
    if not isinstance(ids, (list, tuple)):
        raise ValueError("MARK_USED.skill_ids 必须是序列")
    hit = []
    for skill_id in ids:
        skill = _lookup(skills, skill_id)
        if skill is not None and skill["active"]:
            skill["used"] += 1
            hit.append(skill_id)
    return hit


def _prompt(skills, sections):
    parts = []
    for section in SECTIONS:
        ids = sections.get(section, [])
        rows = [skills[sid] for sid in ids if skills[sid]["active"]]
        if not rows:
            continue
        parts.append("## %s" % section)
        for skill in rows:
            parts.append("- [%s]" % skill["id"])
            parts.append("  Keywords: %s" % ", ".join(skill["keywords"]))
            parts.append("  Issue: %s" % skill["issue"])
            if skill["insight"]:
                parts.append("  Insight: %s" % skill["insight"])
            parts.append("")
    return "\n".join(parts).rstrip()


def solve(skills, operations):
    if not isinstance(skills, (list, tuple)):
        raise ValueError("skills 必须是序列")
    if not isinstance(operations, (list, tuple)):
        raise ValueError("operations 必须是序列")
    book, sections = {}, {}
    for i, raw in enumerate(skills):
        rec = _as_skill(raw, "skills[%d]" % i)
        if rec["id"] in book:
            raise ValueError("skills 内 id 重复")
        book[rec["id"]] = rec
        sections.setdefault(rec["section"], []).append(rec["id"])

    log = []
    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            raise ValueError("operations[%d] 必须是字典" % i)
        kind = op.get("type")
        if kind not in _OPS:
            raise ValueError("未知的操作类型: %r" % (kind,))
        if kind == "ADD":
            skill_id = _add(book, sections, op)
            log.append({"op": i, "type": kind, "result": "added",
                        "skill_id": skill_id})
        elif kind == "UPDATE":
            skill_id, skill = _update(book, op)
            log.append({"op": i, "type": kind,
                        "result": "noop" if skill is None else "updated",
                        "skill_id": skill_id})
        elif kind == "TAG":
            skill_id, skill = _tag(book, op)
            log.append({"op": i, "type": kind,
                        "result": "noop" if skill is None else "tagged",
                        "skill_id": skill_id})
        elif kind == "REMOVE":
            skill_id, result = _remove(book, sections, op)
            log.append({"op": i, "type": kind, "result": result,
                        "skill_id": skill_id})
        else:
            hit = _mark_used(book, op)
            log.append({"op": i, "type": kind, "result": "used", "skill_id": hit})

    active = [sid for sid in book if book[sid]["active"]]
    by_section = {section: 0 for section in SECTIONS}
    for sid in active:
        sec = book[sid]["section"]
        by_section[sec] = by_section.get(sec, 0) + 1
    stats = {"sections": len([s for s, n in by_section.items() if n]),
             "skills": len(book), "active_skills": len(active),
             "by_section": by_section}
    return {"order": list(book), "skills": book,
            "sections": {k: list(v) for k, v in sections.items()},
            "active": active, "prompt": _prompt(book, sections),
            "stats": stats, "log": log}
