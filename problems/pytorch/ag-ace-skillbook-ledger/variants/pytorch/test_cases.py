import re

TOL = 1e-12


def _eq(got, ref, path=""):
    if isinstance(ref, dict):
        assert isinstance(got, dict), path
        assert set(got) == set(ref), (path, sorted(got), sorted(ref))
        for k in ref:
            _eq(got[k], ref[k], path + "/" + str(k))
        return
    if isinstance(ref, (list, tuple)):
        assert isinstance(got, (list, tuple)), (path, type(got))
        assert len(got) == len(ref), (path, len(got), len(ref))
        for i, (g, r) in enumerate(zip(got, ref)):
            _eq(g, r, path + "[%d]" % i)
        return
    assert got == ref, (path, got, ref)


SECTIONS = ("context", "harness")


def _t(v):
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _kw1(v):
    s = _t(v)
    if s is None:
        return None
    return re.sub(r"\s+", "_", s.lower()) or None


def _kwlist(vals):
    if vals is None:
        return []
    out = []
    for v in vals:
        k = _kw1(v)
        if k and k not in out:
            out.append(k)
    return out


def _py(skills, operations):
    """独立参考实现：条目放在有序列表里，id -> 下标。"""
    rows = []
    for raw in skills:
        rows.append({"id": raw["id"], "section": raw["section"],
                     "keywords": list(raw["keywords"]), "issue": raw["issue"],
                     "insight": _t(raw.get("insight")),
                     "active": bool(raw.get("active", True)),
                     "used": raw.get("used", 0), "helpful": raw.get("helpful", 0),
                     "harmful": raw.get("harmful", 0),
                     "neutral": raw.get("neutral", 0)})
    pos = {r["id"]: i for i, r in enumerate(rows)}
    sec = {}
    for r in rows:
        sec.setdefault(r["section"], []).append(r["id"])

    log = []
    for i, op in enumerate(operations):
        kind = op["type"]
        if kind == "ADD":
            raw_sec = _t(op.get("section")).lower()
            kws = _kwlist(op.get("keywords"))
            if raw_sec in SECTIONS:
                section = raw_sec
            else:
                section = "context"
                kws = _kwlist([raw_sec] + kws)
            issue = _t(op.get("issue"))
            insight = _t(op.get("insight"))
            if not kws:
                kws = _kwlist([section])
            if section == "context" and insight is None:
                if raw_sec not in SECTIONS:
                    insight = issue
                else:
                    raise ValueError("context 缺 insight")
            sid = op.get("skill_id") or "%s-%d" % (section, len(rows) + 1)
            rec = {"id": sid, "section": section, "keywords": kws,
                   "issue": issue, "insight": insight, "active": True,
                   "used": 0, "helpful": 0, "harmful": 0, "neutral": 0}
            if sid in pos:
                rows[pos[sid]] = rec
            else:
                pos[sid] = len(rows)
                rows.append(rec)
            sec.setdefault(section, []).append(sid)
            log.append({"op": i, "type": kind, "result": "added", "skill_id": sid})
        elif kind == "UPDATE":
            sid = op.get("skill_id")
            if sid is None or sid not in pos:
                log.append({"op": i, "type": kind, "result": "noop", "skill_id": sid})
                continue
            r = rows[pos[sid]]
            if op.get("issue") is not None:
                r["issue"] = _t(op["issue"])
            if op.get("keywords"):
                nk = _kwlist(op["keywords"])
                if not nk:
                    raise ValueError("keywords 全空")
                r["keywords"] = nk
            if op.get("insight") is not None:
                r["insight"] = _t(op["insight"])
            if r["section"] == "context" and not r["insight"]:
                raise ValueError("context 缺 insight")
            log.append({"op": i, "type": kind, "result": "updated", "skill_id": sid})
        elif kind == "TAG":
            sid = op.get("skill_id")
            if sid is None or sid not in pos:
                log.append({"op": i, "type": kind, "result": "noop", "skill_id": sid})
                continue
            r = rows[pos[sid]]
            d = op.get("delta", 0)
            if d > 0:
                r["helpful"] += 1
            elif d < 0:
                r["harmful"] += 1
            else:
                r["neutral"] += 1
            log.append({"op": i, "type": kind, "result": "tagged", "skill_id": sid})
        elif kind == "REMOVE":
            sid = op.get("skill_id")
            if sid is None or sid not in pos:
                log.append({"op": i, "type": kind, "result": "noop", "skill_id": sid})
                continue
            r = rows[pos[sid]]
            if op.get("purge"):
                sec[r["section"]] = [x for x in sec[r["section"]] if x != sid]
                if not sec[r["section"]]:
                    del sec[r["section"]]
                del pos[sid]
                rows = [x for x in rows if x["id"] != sid]
                pos.clear()
                pos.update({x["id"]: j for j, x in enumerate(rows)})
                log.append({"op": i, "type": kind, "result": "purged",
                            "skill_id": sid})
            else:
                r["active"] = False
                log.append({"op": i, "type": kind, "result": "removed",
                            "skill_id": sid})
        else:
            hit = []
            for sid in op["skill_ids"]:
                if sid in pos and rows[pos[sid]]["active"]:
                    rows[pos[sid]]["used"] += 1
                    hit.append(sid)
            log.append({"op": i, "type": kind, "result": "used", "skill_id": hit})

    book = {r["id"]: r for r in rows}
    parts = []
    for section in SECTIONS:
        live = [book[x] for x in sec.get(section, []) if book[x]["active"]]
        if not live:
            continue
        parts.append("## " + section)
        for r in live:
            parts.append("- [%s]" % r["id"])
            parts.append("  Keywords: " + ", ".join(r["keywords"]))
            parts.append("  Issue: " + r["issue"])
            if r["insight"]:
                parts.append("  Insight: " + r["insight"])
            parts.append("")
    by_section = {s: 0 for s in SECTIONS}
    for r in rows:
        if r["active"]:
            by_section[r["section"]] = by_section.get(r["section"], 0) + 1
    active = [r["id"] for r in rows if r["active"]]
    return {"order": [r["id"] for r in rows], "skills": book, "sections": sec,
            "active": active, "prompt": "\n".join(parts).rstrip(),
            "stats": {"sections": len([1 for n in by_section.values() if n]),
                      "skills": len(rows), "active_skills": len(active),
                      "by_section": by_section},
            "log": log}


def _compare(solve, skills, operations):
    ref = _py([dict(s) for s in skills], [dict(o) for o in operations])
    got = solve([dict(s) for s in skills], [dict(o) for o in operations])
    _eq(got, ref)
    return got


def _s(sid, section="harness", keywords=None, issue=None, insight=None,
       active=True, **kw):
    rec = {"id": sid, "section": section,
           "keywords": list(keywords) if keywords else [section],
           "issue": issue or ("issue of " + sid), "insight": insight,
           "active": active}
    rec.update(kw)
    return rec


def check_keyword_normalization(solve):
    """关键词归一化：strip → lower → `\\s+` 变 `_`，空串丢弃，按首次出现去重。"""
    led = _compare(solve, [], [
        {"type": "ADD", "section": "harness", "issue": "  use grep  ",
         "keywords": ["File  IO", "file_io", "  ", None, "RE"], "skill_id": "k1"},
    ])
    rec = led["skills"]["k1"]
    assert rec["keywords"] == ["file_io", "re"]
    assert rec["issue"] == "use grep"
    assert rec["insight"] is None


def check_legacy_section_folds(solve):
    """既非 context 也非 harness 的 section：折叠成 context、section 名进关键词、insight 用 issue 兜底。"""
    led = _compare(solve, [], [
        {"type": "ADD", "section": "  Tooling ", "issue": "prefer ripgrep",
         "keywords": ["search"], "skill_id": "l1"},
    ])
    rec = led["skills"]["l1"]
    assert rec["section"] == "context"
    assert rec["keywords"] == ["tooling", "search"]
    assert rec["insight"] == "prefer ripgrep"
    assert led["sections"] == {"context": ["l1"]}

    # 真正的 context 节缺 insight 要报错
    try:
        solve([], [{"type": "ADD", "section": "context", "issue": "x"}])
    except ValueError:
        pass
    else:
        raise AssertionError("context 缺 insight 应当报错")

    # harness 节没有这个要求
    led = _compare(solve, [], [
        {"type": "ADD", "section": "HARNESS", "issue": "x", "skill_id": "h1"}])
    assert led["skills"]["h1"]["section"] == "harness"
    assert led["skills"]["h1"]["insight"] is None


def check_update_unset_semantics(solve):
    """UPDATE 的空 keywords 表示「不改」，非空但全空白表示「非法」。"""
    base = [_s("a", insight="keep me")]
    led = _compare(solve, base, [
        {"type": "UPDATE", "skill_id": "a", "issue": "new issue", "keywords": []},
    ])
    rec = led["skills"]["a"]
    assert rec["issue"] == "new issue"
    assert rec["keywords"] == ["harness"]
    assert rec["insight"] == "keep me"

    try:
        solve(base, [{"type": "UPDATE", "skill_id": "a", "keywords": ["  "]},
                     ])
    except ValueError:
        pass
    else:
        raise AssertionError("全空白 keywords 应当报错")


def check_update_missing_skill(solve):
    """UPDATE/TAG/REMOVE 缺 skill_id 或指向不存在的条目都记 noop，不报错。"""
    base = [_s("a")]
    led = _compare(solve, base, [
        {"type": "UPDATE", "issue": "x"},
        {"type": "UPDATE", "skill_id": "ghost", "issue": "x"},
        {"type": "TAG", "skill_id": "ghost", "delta": 1},
        {"type": "REMOVE", "skill_id": "ghost"},
        {"type": "TAG", "skill_id": "a", "delta": 1},
    ])
    assert [x["result"] for x in led["log"]] == [
        "noop", "noop", "noop", "noop", "tagged"]
    assert led["skills"]["a"]["helpful"] == 1


def check_tag_delta_sign_only(solve):
    """TAG 的 delta 只按符号取 ±1，0 记 neutral——不是累加原值。"""
    base = [_s("a")]
    led = _compare(solve, base, [
        {"type": "TAG", "skill_id": "a", "delta": 7},
        {"type": "TAG", "skill_id": "a", "delta": -3},
        {"type": "TAG", "skill_id": "a", "delta": 0},
        {"type": "TAG", "skill_id": "a", "delta": -1},
    ])
    rec = led["skills"]["a"]
    assert (rec["helpful"], rec["harmful"], rec["neutral"]) == (1, 2, 1)


def check_remove_soft_vs_purge(solve):
    """REMOVE 默认软删（还留在 skills 里但 inactive），purge 才真删并清空 section 桶。"""
    base = [_s("a"), _s("b", section="context", insight="i")]
    led = _compare(solve, base, [{"type": "REMOVE", "skill_id": "a"},
                                 {"type": "REMOVE", "skill_id": "b", "purge": True}])
    assert led["order"] == ["a"]
    assert led["skills"]["a"]["active"] is False
    assert led["active"] == []
    assert led["sections"] == {"harness": ["a"]}
    assert led["stats"]["skills"] == 1
    assert led["stats"]["active_skills"] == 0
    assert led["stats"]["by_section"] == {"context": 0, "harness": 0}
    assert led["prompt"] == ""


def check_mark_used_active_only(solve):
    """mark_used 只给 active 的条目加计数，且不改 helpful/harmful。"""
    base = [_s("a"), _s("b", active=False), _s("c")]
    led = _compare(solve, base, [
        {"type": "MARK_USED", "skill_ids": ["a", "b", "c", "ghost"]},
        {"type": "MARK_USED", "skill_ids": ["a"]},
    ])
    assert led["skills"]["a"]["used"] == 2
    assert led["skills"]["b"]["used"] == 0
    assert led["skills"]["c"]["used"] == 1
    assert led["log"][0]["skill_id"] == ["a", "c"]


def check_prompt_rendering(solve):
    """提示词：固定 context → harness 顺序、跳过空节、只有 insight 非空才输出该行。"""
    base = [_s("h1", section="harness", keywords=["grep"], issue="use grep",
               insight=None),
            _s("c1", section="context", keywords=["io", "hint"],
               issue="cache first", insight="read twice")]
    led = _compare(solve, base, [])
    assert led["prompt"] == (
        "## context\n"
        "- [c1]\n"
        "  Keywords: io, hint\n"
        "  Issue: cache first\n"
        "  Insight: read twice\n"
        "\n"
        "## harness\n"
        "- [h1]\n"
        "  Keywords: grep\n"
        "  Issue: use grep")
    # 空书 -> 空串
    assert _compare(solve, [], [])["prompt"] == ""


def check_prompt_skips_removed(solve):
    """软删的条目不再出现在提示词里，但仍在 stats['skills'] 里计数。"""
    base = [_s("a", section="context", insight="i"), _s("b", section="context",
                                                        insight="j")]
    led = _compare(solve, base, [{"type": "REMOVE", "skill_id": "a"}])
    assert "- [a]" not in led["prompt"]
    assert "- [b]" in led["prompt"]
    assert led["stats"]["skills"] == 2
    assert led["stats"]["by_section"]["context"] == 1


def check_duplicate_add_repeats_section(solve):
    """同一 skill_id 重复 ADD：记录被覆盖，但 section 桶再登记一次（提示词里会出现两遍）。"""
    led = _compare(solve, [], [
        {"type": "ADD", "section": "harness", "issue": "v1", "skill_id": "d1"},
        {"type": "ADD", "section": "harness", "issue": "v2", "skill_id": "d1"},
    ])
    assert led["order"] == ["d1"]
    assert led["skills"]["d1"]["issue"] == "v2"
    assert led["sections"]["harness"] == ["d1", "d1"]
    assert led["prompt"].count("- [d1]") == 2
    assert led["stats"]["skills"] == 1
    assert led["stats"]["active_skills"] == 1


def check_update_insight_blank_clears(solve):
    """harness 条目的 insight 用空串 UPDATE 会真的被清空（不是「不改」）。"""
    base = [_s("h", section="harness", insight="old")]
    led = _compare(solve, base, [{"type": "UPDATE", "skill_id": "h", "insight": ""}])
    assert led["skills"]["h"]["insight"] is None
    led = _compare(solve, base, [{"type": "UPDATE", "skill_id": "h", "insight": "  "}])
    assert led["skills"]["h"]["insight"] is None


def check_context_update_clears_insight(solve):
    """把 context 条目的 insight 改成空串会报错（不是静默变空）。"""
    base = [_s("c", section="context", insight="i")]
    try:
        solve(base, [{"type": "UPDATE", "skill_id": "c", "insight": "   "}])
    except ValueError:
        pass
    else:
        raise AssertionError("把 context 的 insight 清空应当报错")


def check_validation(solve):
    """结构与标量校验。"""
    for bad in ([{"id": "x", "section": "nope", "keywords": [], "issue": "i"}],
                [{"id": "x", "section": "harness", "keywords": [], "issue": ""}],
                [{"id": "x", "section": "harness", "keywords": [], "issue": "i"},
                 {"id": "x", "section": "harness", "keywords": [], "issue": "j"}],
                [{"id": "x", "section": "harness", "keywords": [], "issue": "i",
                  "helpful": -1}]):
        try:
            solve(bad, [])
        except ValueError:
            pass
        else:
            raise AssertionError("%r 应当报错" % (bad,))
    for ops in ([{"type": "NOPE"}], [{"type": "NOPE", "skill_ids": ["x"]}],
                [{"type": "TAG", "delta": True, "skill_id": "x"}],
                [{"type": "MARK_USED", "skill_ids": "x"}]):
        try:
            solve([_s("x")], ops)
        except ValueError:
            pass
        else:
            raise AssertionError("%r 应当报错" % (ops,))


def check_against_oracle(solve):
    """混合批次整体对拍：ADD → TAG → UPDATE → REMOVE → MARK_USED。"""
    base = [_s("seed", section="harness", keywords=["grep"], issue="use grep")]
    ops = [
        {"type": "ADD", "section": "Context", "issue": "cache twice",
         "insight": "read once, memo", "keywords": ["CACHE  IO"], "skill_id": "c1"},
        {"type": "ADD", "section": "scratch", "issue": "note it"},
        {"type": "TAG", "skill_id": "c1", "delta": 2},
        {"type": "TAG", "skill_id": "c1", "delta": -2},
        {"type": "UPDATE", "skill_id": "c1", "keywords": ["cache", "memo"]},
        {"type": "MARK_USED", "skill_ids": ["c1", "seed"]},
        {"type": "REMOVE", "skill_id": "seed"},
        {"type": "ADD", "section": "harness", "issue": "lint first",
         "skill_id": "h2", "insight": "run ruff"},
    ]
    led = _compare(solve, base, ops)
    assert led["skills"]["c1"]["keywords"] == ["cache", "memo"]
    assert led["skills"]["c1"]["insight"] == "read once, memo"
    assert (led["skills"]["c1"]["helpful"], led["skills"]["c1"]["harmful"]) == (1, 1)
    assert led["stats"]["by_section"] == {"context": 2, "harness": 1}
    assert led["prompt"].startswith("## context")


def run_examples(solve):
    print(_compare(solve, [], [
        {"type": "ADD", "section": "tooling", "issue": "prefer ripgrep",
         "skill_id": "l1"}])["skills"]["l1"])
    print(_compare(solve, [_s("a")],
                   [{"type": "TAG", "skill_id": "a", "delta": -5}])["skills"]["a"])


check_legacy_section_folds.example = True
check_tag_delta_sign_only.example = True
