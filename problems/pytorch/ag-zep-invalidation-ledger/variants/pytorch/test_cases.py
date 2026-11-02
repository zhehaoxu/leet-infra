"""Zep 双时态边失效账的检查点。

每个检查点都拿一份**另一套写法**的 oracle（先落表、再逐段判）与 `solve`
比对，再补若干硬断言钉住容易写反的边界：严格大于 / 含等号、降序补位、
None 排最后、原地排序改变遍历序、重复下标不重复计入。
"""

import re
from datetime import datetime, timezone

_UTC = timezone.utc
_NOW = "2024-06-01T00:00:00Z"


def _dt(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_UTC)
    return dt.astimezone(_UTC)


def _iso(dt):
    return None if dt is None else dt.isoformat().replace("+00:00", "Z")


def _edge(uid, fact, valid=None, invalid=None, expired=None,
          source="alice", target="berlin"):
    return {"uuid": uid, "fact": fact, "source": source, "target": target,
            "valid_at": valid, "invalid_at": invalid, "expired_at": expired}


# ---------------------------------------------------------------- oracle --
def _oracle(extracted, related_edges=(), existing_edges=(), decisions=None,
            now=None, timestamps=None):
    if now is None:
        raise ValueError("now")
    now_dt = _dt(now)
    related = list(related_edges)
    existing = list(existing_edges)

    rows = []

    def add(edge):
        rows.append({
            "uuid": edge["uuid"],
            "fact": edge["fact"],
            "source": edge.get("source"),
            "target": edge.get("target"),
            "valid": _dt(edge.get("valid_at")),
            "invalid": _dt(edge.get("invalid_at")),
            "expired": _dt(edge.get("expired_at")),
        })
        return rows[-1]

    fresh = add(extracted)
    for edge in related:
        add(edge)
    for edge in existing:
        add(edge)
    table = {row["uuid"]: row for row in rows}

    log = []

    def assign(row, field, value):
        row[field] = value
        log.append({"uuid": row["uuid"], "field": field + "_at",
                    "value": _iso(value)})

    def maybe_stamp(row):
        if row["valid"] is not None or row["invalid"] is not None:
            return
        if not timestamps:
            return
        for field in ("valid", "invalid"):
            raw = timestamps.get(field + "_at")
            if raw:
                assign(row, field, _dt(raw))

    def view(row):
        return {"uuid": row["uuid"], "fact": row["fact"],
                "valid_at": _iso(row["valid"]),
                "invalid_at": _iso(row["invalid"]),
                "expired_at": _iso(row["expired"])}

    def report(path, head, dup_ids, bad_dups, bad_contras, picked, walk,
               killed):
        return {
            "path": path,
            "resolved": view(head),
            "resolved_is_new": head["uuid"] == fresh["uuid"],
            "duplicate_ids": list(dup_ids),
            "duplicate_uuids": [related[i]["uuid"] for i in dup_ids],
            "invalid_duplicate_ids": list(bad_dups),
            "invalid_contradiction_ids": list(bad_contras),
            "candidate_uuids": [r["uuid"] for r in picked],
            "sorted_candidate_uuids": [r["uuid"] for r in walk],
            "invalidated": [view(r) for r in killed],
            "edges": {row["uuid"]: view(row) for row in rows},
            "writes": log,
        }

    # 一、没有可比的边：只抽时间戳，整段失效跳过。
    if not related and not existing:
        maybe_stamp(fresh)
        return report("empty", fresh, [], [], [], [], [], [])

    # 二、快速路径：归一后逐字相同且两端点相同。
    key = re.sub(r"\s+", " ", extracted["fact"].lower()).strip()
    for edge in related:
        ok_end = (edge.get("source") == extracted.get("source")
                  and edge.get("target") == extracted.get("target"))
        if ok_end:
            text = re.sub(r"\s+", " ", edge["fact"].lower()).strip()
            if text == key:
                return report("fast", table[edge["uuid"]], [], [], [], [], [],
                              [])

    if decisions is None:
        raise ValueError("decisions")

    n = len(related)
    last = n + len(existing) - 1
    dup_raw = list(decisions.get("duplicate_facts") or [])
    con_raw = list(decisions.get("contradicted_facts") or [])

    bad_dups = [i for i in dup_raw if i < 0 or i >= n]
    dup_ids = [i for i in dup_raw if 0 <= i < n]
    bad_contras = [i for i in con_raw if i < 0 or i > last]

    head = fresh
    if dup_ids:
        head = table[related[dup_ids[0]]["uuid"]]

    picked = []
    for i in con_raw:
        if 0 <= i < n:
            picked.append(table[related[i]["uuid"]])
        elif n <= i <= last:
            picked.append(table[existing[i - n]["uuid"]])

    if head["uuid"] == fresh["uuid"]:
        maybe_stamp(head)

    # 三、新边自述已是历史 -> 只盖事务时间戳。
    if head["invalid"] is not None and head["expired"] is None:
        assign(head, "expired", now_dt)

    # 四、更晚的候选顶掉新边；这一步会重排候选。
    walk = list(picked)
    if head["expired"] is None:
        walk = sorted(walk, key=lambda r: (r["valid"] is None, r["valid"]))
        for cand in walk:
            later = (cand["valid"] is not None
                     and head["valid"] is not None
                     and cand["valid"] > head["valid"])
            if later:
                assign(head, "invalid", cand["valid"])
                assign(head, "expired", now_dt)
                break

    # 五、逐条判候选该不该失效。
    killed = []
    for cand in walk:
        dead_early = (cand["invalid"] is not None
                      and head["valid"] is not None
                      and not cand["invalid"] > head["valid"])
        head_early = (cand["valid"] is not None
                      and head["invalid"] is not None
                      and not head["invalid"] > cand["valid"])
        if dead_early or head_early:
            continue
        older = (cand["valid"] is not None
                 and head["valid"] is not None
                 and cand["valid"] < head["valid"])
        if older:
            assign(cand, "invalid", head["valid"])
            if cand["expired"] is None:
                assign(cand, "expired", now_dt)
            killed.append(cand)

    return report("resolve", head, dup_ids, bad_dups, bad_contras, picked,
                  walk, killed)


def _same(solve, *args, **kwargs):
    got = solve(*args, **kwargs)
    want = _oracle(*args, **kwargs)
    assert got == want, "账本不一致\n got=%r\n exp=%r" % (got, want)
    return got


# ------------------------------------------------------------ fixtures ----
def _simple(**over):
    """一条新事实顶掉一条旧事实的最小场景。"""
    base = dict(
        extracted=_edge("new", "Alice lives in Berlin",
                        valid="2024-03-01T00:00:00Z"),
        related_edges=[_edge("old", "Alice lives in Paris",
                             valid="2024-01-01T00:00:00Z")],
        existing_edges=[],
        decisions={"duplicate_facts": [], "contradicted_facts": [0]},
        now=_NOW,
    )
    base.update(over)
    return base


# ------------------------------------------------------------- checks ----
def check_empty_pool(solve):
    """related 与 existing 都空：只抽时间戳，不做任何失效。"""
    r = _same(solve,
              _edge("new", "Alice lives in Berlin"),
              [], [],
              now=_NOW,
              timestamps={"valid_at": "2024-02-02T00:00:00Z"})
    assert r["path"] == "empty"
    assert r["resolved_is_new"] is True
    assert r["resolved"]["valid_at"] == "2024-02-02T00:00:00Z"
    assert r["resolved"]["invalid_at"] is None
    assert r["candidate_uuids"] == []
    assert [w["field"] for w in r["writes"]] == ["valid_at"]
    assert set(r["edges"]) == {"new"}

    # 两个时间戳都给：写入顺序固定 valid -> invalid。
    r = _same(solve,
              _edge("new", "Alice lived in Rome"),
              [], [],
              now=_NOW,
              timestamps={"valid_at": "2020-01-01T00:00:00Z",
                          "invalid_at": "2021-01-01T00:00:00Z"})
    assert [w["field"] for w in r["writes"]] == ["valid_at", "invalid_at"]
    # 没给 timestamps：什么都不写。
    r = _same(solve, _edge("new", "x"), [], [], now=_NOW)
    assert r["writes"] == []

    # 空串不写（官方 `if valid_at:`），另一个字段照常写。
    r = _same(solve, _edge("new", "x"), [], [], now=_NOW,
              timestamps={"valid_at": "",
                          "invalid_at": "2022-01-01T00:00:00Z"})
    assert r["resolved"]["valid_at"] is None
    assert r["writes"] == [{"uuid": "new", "field": "invalid_at",
                            "value": "2022-01-01T00:00:00Z"}]

    # naive 时间按 UTC 解释，输出一律带 Z。
    r = _same(solve, _edge("new", "x"), [], [], now=_NOW,
              timestamps={"valid_at": "2024-01-01T00:00:00"})
    assert r["resolved"]["valid_at"] == "2024-01-01T00:00:00Z"


def check_fast_path(solve):
    """fact 归一后逐字相同且两端点相同 -> 复用旧边，跳过矛盾处理。"""
    rel = [_edge("old", "Alice  lives\tin BERLIN",
                 valid="2024-01-01T00:00:00Z")]
    r = _same(solve,
              _edge("new", " alice lives in berlin\u3000"),
              rel, [],
              decisions={"duplicate_facts": [0], "contradicted_facts": [0]},
              now=_NOW)
    assert r["path"] == "fast"
    assert r["resolved"]["uuid"] == "old"
    assert r["resolved_is_new"] is False
    # 快速路径返回的 duplicate 列表是空的，且完全不失效。
    assert r["duplicate_ids"] == []
    assert r["duplicate_uuids"] == []
    assert r["invalidated"] == []
    assert r["writes"] == []

    # 端点不同就不走快速路径。
    rel = [_edge("old", "Alice lives in Berlin", source="alice",
                 target="paris")]
    r = _same(solve, _edge("new", "Alice lives in Berlin"), rel, [],
              decisions={"duplicate_facts": [], "contradicted_facts": []},
              now=_NOW)
    assert r["path"] == "resolve"

    # source 不同同样不行（两端点都要对上）。
    rel = [_edge("old", "Alice lives in Berlin", source="bob",
                 valid="2024-01-01T00:00:00Z")]
    r = _same(solve,
              _edge("new", "Alice lives in Berlin",
                    valid="2024-03-01T00:00:00Z"),
              rel, [],
              decisions={"duplicate_facts": [], "contradicted_facts": [0]},
              now=_NOW)
    assert r["path"] == "resolve"
    assert [x["uuid"] for x in r["invalidated"]] == ["old"]

    # 文本不同（多一个词）也不走。
    rel = [_edge("old", "Alice lives in Berlin now",
                 valid="2024-01-01T00:00:00Z")]
    r = _same(solve,
              _edge("new", "Alice lives in Berlin",
                    valid="2024-03-01T00:00:00Z"),
              rel, [],
              decisions={"duplicate_facts": [], "contradicted_facts": [0]},
              now=_NOW)
    assert r["path"] == "resolve"
    assert [x["uuid"] for x in r["invalidated"]] == ["old"]

    # 多条命中取 related 里的第一条。
    rel = [_edge("first", "Alice lives in Berlin"),
           _edge("second", "alice LIVES in berlin")]
    r = _same(solve, _edge("new", "Alice lives in Berlin"), rel, [],
              now=_NOW)
    assert r["resolved"]["uuid"] == "first"

    # 只扫 related：existing 里文本相同也不算快速路径。
    ex = [_edge("ex", "Alice lives in Berlin")]
    r = _same(solve,
              _edge("new", "Alice lives in Berlin",
                    valid="2024-03-01T00:00:00Z"),
              [], ex,
              decisions={"duplicate_facts": [], "contradicted_facts": []},
              now=_NOW)
    assert r["path"] == "resolve"
    assert r["resolved_is_new"] is True


def check_duplicate_merge(solve):
    """duplicate 命中 -> resolved 变成旧边，且不再重抽时间戳。"""
    rel = [_edge("other", "Alice works at Acme",
                 valid="2024-01-01T00:00:00Z"),
           _edge("dup", "Alice lives in Paris",
                 valid="2023-05-01T00:00:00Z")]
    r = _same(solve,
              _edge("new", "Alice lives in Paris",
                    valid="2024-03-01T00:00:00Z"),
              rel, [],
              decisions={"duplicate_facts": [1], "contradicted_facts": []},
              now=_NOW,
              timestamps={"valid_at": "2024-04-04T00:00:00Z"})
    assert r["resolved"]["uuid"] == "dup"
    assert r["resolved_is_new"] is False
    # 并进旧边时保留旧边的时间戳，抽取结果被丢弃。
    assert r["resolved"]["valid_at"] == "2023-05-01T00:00:00Z"
    assert r["writes"] == []

    # 旧边自己也没时间戳：抽取结果依然丢弃（并进旧边不重抽）。
    rel = [_edge("dup", "Alice lives in Paris")]
    r = _same(solve,
              _edge("new", "Alice lives in Paris now"),
              rel, [],
              decisions={"duplicate_facts": [0], "contradicted_facts": []},
              now=_NOW,
              timestamps={"valid_at": "2024-04-04T00:00:00Z"})
    assert r["path"] == "resolve"
    assert r["resolved"]["uuid"] == "dup"
    assert r["resolved"]["valid_at"] is None
    assert r["resolved"]["invalid_at"] is None
    assert r["writes"] == []

    # 新建边才抽时间戳（新边原本没有 valid_at）。
    r = _same(solve,
              _edge("new", "Alice lives in Berlin"),
              [_edge("old", "Alice lives in Paris",
                     valid="2024-01-01T00:00:00Z")],
              [],
              decisions={"duplicate_facts": [], "contradicted_facts": [0]},
              now=_NOW,
              timestamps={"valid_at": "2024-03-05T00:00:00Z"})
    assert r["resolved"]["uuid"] == "new"
    assert r["resolved"]["valid_at"] == "2024-03-05T00:00:00Z"
    assert r["writes"][0] == {"uuid": "new", "field": "valid_at",
                              "value": "2024-03-05T00:00:00Z"}
    assert [x["uuid"] for x in r["invalidated"]] == ["old"]
    assert r["edges"]["old"]["invalid_at"] == "2024-03-05T00:00:00Z"

    # 新边已自带一个时间戳：整段抽取跳过（官方 _extract_edge_timestamps 的门）。
    r = _same(solve, **_simple(
        timestamps={"valid_at": "2024-03-05T00:00:00Z",
                    "invalid_at": "2024-09-09T00:00:00Z"}))
    assert r["resolved"]["valid_at"] == "2024-03-01T00:00:00Z"
    assert r["resolved"]["invalid_at"] is None
    assert r["writes"] == [{"uuid": "old", "field": "invalid_at",
                            "value": "2024-03-01T00:00:00Z"},
                           {"uuid": "old", "field": "expired_at",
                            "value": _NOW}]


def check_duplicate_bounds(solve):
    """duplicate 下标越界只记账，不报错；有效下标保序可重复。"""
    rel = [_edge("a", "Alice lives in Rome", valid="2024-01-05T00:00:00Z"),
           _edge("b", "Alice works at Acme", valid="2024-01-01T00:00:00Z")]
    r = _same(solve,
              _edge("new", "Alice lives in Rome now",
                    valid="2024-03-01T00:00:00Z"),
              rel, [],
              decisions={"duplicate_facts": [-1, 5, 0, 0],
                         "contradicted_facts": [1, 1]},
              now=_NOW)
    assert r["invalid_duplicate_ids"] == [-1, 5]
    assert r["duplicate_ids"] == [0, 0]
    assert r["duplicate_uuids"] == ["a", "a"]
    assert r["resolved"]["uuid"] == "a"
    # 重复下标第二次进来时已被判过期（invalid == 新边 valid，含等号），
    # 不重复计入 invalidated。
    assert [x["uuid"] for x in r["invalidated"]] == ["b"]
    assert r["edges"]["b"]["invalid_at"] == "2024-01-05T00:00:00Z"

    # duplicate 的越界边界是 len(related)，不是两段总长。
    rel2 = [_edge("p", "Alice lives in Paris", valid="2024-01-01T00:00:00Z"),
            _edge("q", "Alice works at Acme", valid="2024-01-02T00:00:00Z")]
    ex2 = [_edge("y", "Alice is 30", valid="2024-01-03T00:00:00Z")]
    r = _same(solve,
              _edge("new", "Alice lives in Rome",
                    valid="2024-03-01T00:00:00Z"),
              rel2, ex2,
              decisions={"duplicate_facts": [1, 2],
                         "contradicted_facts": []},
              now=_NOW)
    assert r["invalid_duplicate_ids"] == [2]
    assert r["duplicate_ids"] == [1]
    assert r["resolved"]["uuid"] == "q"

    # 多个有效 duplicate：取第一个，不是最后一个。
    r = _same(solve,
              _edge("new", "Alice lives in Rome",
                    valid="2024-03-01T00:00:00Z"),
              rel2, ex2,
              decisions={"duplicate_facts": [1, 0],
                         "contradicted_facts": []},
              now=_NOW)
    assert r["resolved"]["uuid"] == "q"


def check_contradiction_offset(solve):
    """contradicted 下标跨两段连续编号；越界按 total-1 判。"""
    rel = [_edge("r0", "Alice lives in Paris", valid="2024-01-01T00:00:00Z"),
           _edge("r1", "Alice works at Acme", valid="2024-01-02T00:00:00Z")]
    ex = [_edge("x0", "Alice is 30", valid="2024-01-03T00:00:00Z"),
          _edge("x1", "Alice is 31", valid="2024-01-04T00:00:00Z")]
    r = _same(solve,
              _edge("new", "Alice lives in Rome",
                    valid="2024-03-01T00:00:00Z"),
              rel, ex,
              decisions={"duplicate_facts": [],
                         "contradicted_facts": [0, 2, 3, 4, -2, 1]},
              now=_NOW)
    assert r["invalid_contradiction_ids"] == [4, -2]
    assert r["candidate_uuids"] == ["r0", "x0", "x1", "r1"]
    # 排序后遍历，invalidated 顺序跟着变。
    assert r["sorted_candidate_uuids"] == ["r0", "r1", "x0", "x1"]
    assert [x["uuid"] for x in r["invalidated"]] == ["r0", "r1", "x0", "x1"]


def check_supersede(solve):
    """更晚的候选顶掉新边，且只顶一次（排序后取第一个严格更晚的）。"""
    rel = [_edge("early", "Alice lives in Paris",
                 valid="2024-01-01T00:00:00Z")]
    ex = [_edge("late", "Alice lives in Rome", valid="2024-08-01T00:00:00Z"),
          _edge("later", "Alice lives in Oslo", valid="2024-09-01T00:00:00Z")]
    r = _same(solve,
              _edge("new", "Alice lives in Berlin",
                    valid="2024-03-01T00:00:00Z"),
              rel, ex,
              decisions={"duplicate_facts": [],
                         "contradicted_facts": [0, 1, 2]},
              now=_NOW)
    assert r["resolved"]["invalid_at"] == "2024-08-01T00:00:00Z"
    assert r["resolved"]["expired_at"] == _NOW
    # 顶掉新边之后，反而把更早的候选也失效了。
    assert [x["uuid"] for x in r["invalidated"]] == ["early"]

    # 没有更晚的候选：新边不会被顶掉。
    r = _same(solve,
              _edge("new", "Alice lives in Berlin",
                    valid="2024-12-01T00:00:00Z"),
              rel, ex,
              decisions={"duplicate_facts": [],
                         "contradicted_facts": [0, 1]},
              now=_NOW)
    assert r["resolved"]["invalid_at"] is None
    assert r["resolved"]["expired_at"] is None


def check_no_sort_when_expired(solve):
    """新边已带 expired_at：不排序、不被顶掉，遍历序 = 决策序。"""
    rel = [_edge("r0", "Alice lives in Paris", valid="2024-01-01T00:00:00Z")]
    ex = [_edge("x0", "Alice is 30", valid="2024-04-10T00:00:00Z")]
    common = dict(
        extracted=_edge("new", "Alice lives in Berlin",
                        valid="2024-03-01T00:00:00Z",
                        expired="2020-01-01T00:00:00Z"),
        related_edges=rel,
        existing_edges=ex,
        decisions={"duplicate_facts": [], "contradicted_facts": [1, 0]},
        now=_NOW,
    )
    r = _same(solve, **common)
    assert r["sorted_candidate_uuids"] == ["x0", "r0"]
    # 已有 expired：不重写时间戳，也不被更晚的 x0 顶掉。
    assert r["resolved"]["expired_at"] == "2020-01-01T00:00:00Z"
    assert r["resolved"]["invalid_at"] is None
    assert [x["uuid"] for x in r["invalidated"]] == ["r0"]

    # 对照：同样输入但 expired 为空 -> 排序 + 被 x0 顶掉。
    common["extracted"] = _edge("new", "Alice lives in Berlin",
                                valid="2024-03-01T00:00:00Z")
    r = _same(solve, **common)
    assert r["sorted_candidate_uuids"] == ["r0", "x0"]
    assert r["resolved"]["invalid_at"] == "2024-04-10T00:00:00Z"
    assert r["resolved"]["expired_at"] == _NOW


def check_keep_existing_expired(solve):
    """候选已有 expired_at：失效时保留原值，不改写。"""
    rel = [_edge("r0", "Alice lives in Paris", valid="2024-01-01T00:00:00Z",
                 expired="2023-12-01T00:00:00Z")]
    r = _same(solve, **_simple(related_edges=rel))
    assert r["invalidated"][0]["expired_at"] == "2023-12-01T00:00:00Z"
    assert r["edges"]["r0"]["expired_at"] == "2023-12-01T00:00:00Z"


def check_dead_candidate_skipped(solve):
    """候选早已失效（invalid <= 新边 valid，含等号）-> 跳过。"""
    rel = [_edge("gone", "Alice lives in Paris",
                 valid="2024-01-01T00:00:00Z",
                 invalid="2024-03-01T00:00:00Z")]
    r = _same(solve, **_simple(related_edges=rel))
    assert r["invalidated"] == []
    assert r["edges"]["gone"]["invalid_at"] == "2024-03-01T00:00:00Z"

    # 早一点（严格小于）也跳过。
    rel = [_edge("gone", "Alice lives in Paris",
                 valid="2024-01-01T00:00:00Z",
                 invalid="2024-02-01T00:00:00Z")]
    r = _same(solve, **_simple(related_edges=rel))
    assert r["invalidated"] == []


def check_tie_not_invalidated(solve):
    """同一 valid_at 平局：谁都不失效谁（严格小于）。"""
    rel = [_edge("tie", "Alice lives in Paris",
                 valid="2024-03-01T00:00:00Z")]
    r = _same(solve, **_simple(related_edges=rel))
    assert r["invalidated"] == []
    assert r["edges"]["tie"]["invalid_at"] is None
    # 新边自己也不被这条平局候选顶掉（严格大于才顶）。
    assert r["resolved"]["invalid_at"] is None
    assert r["resolved"]["expired_at"] is None


def check_new_edge_dead_first(solve):
    """新边自述的失效早于候选生效 -> 候选不该被它失效（含等号）。"""
    rel = [_edge("mid", "Alice lives in Paris",
                 valid="2024-05-15T00:00:00Z"),
           _edge("early", "Alice lives in Rome",
                 valid="2024-04-01T00:00:00Z")]
    r = _same(solve,
              _edge("new", "Alice lives in Berlin",
                    valid="2024-06-01T00:00:00Z",
                    invalid="2024-05-01T00:00:00Z"),
              rel, [],
              decisions={"duplicate_facts": [],
                         "contradicted_facts": [0, 1]},
              now=_NOW)
    # 新边已是历史：expired 立刻落下，第 2 步不排序。
    assert r["resolved"]["expired_at"] == _NOW
    assert r["sorted_candidate_uuids"] == ["mid", "early"]
    # mid 生效晚于新边的失效时刻 -> 跳过；early 早于新边生效 -> 失效。
    assert [x["uuid"] for x in r["invalidated"]] == ["early"]

    # 等号：候选 valid 恰等于新边 invalid -> 也跳过。
    rel = [_edge("edge", "Alice lives in Paris",
                 valid="2024-05-01T00:00:00Z")]
    r = _same(solve,
              _edge("new", "Alice lives in Berlin",
                    valid="2024-06-01T00:00:00Z",
                    invalid="2024-05-01T00:00:00Z"),
              rel, [],
              decisions={"duplicate_facts": [],
                         "contradicted_facts": [0]},
              now=_NOW)
    assert r["invalidated"] == []


def check_unknown_valid(solve):
    """新边 valid_at 缺失：谁也顶不掉、谁也失效不了。"""
    rel = [_edge("r0", "Alice lives in Paris", valid="2024-01-01T00:00:00Z")]
    r = _same(solve,
              _edge("new", "Alice lives in Berlin"),
              rel, [],
              decisions={"duplicate_facts": [],
                         "contradicted_facts": [0]},
              now=_NOW)
    assert r["resolved"]["valid_at"] is None
    assert r["resolved"]["invalid_at"] is None
    assert r["invalidated"] == []

    # 候选 valid_at 缺失：排最后，也不被失效。
    rel = [_edge("non", "Alice lives in Paris"),
           _edge("dated", "Alice lives in Rome", valid="2024-01-01T00:00:00Z")]
    r = _same(solve,
              _edge("new", "Alice lives in Berlin",
                    valid="2024-03-01T00:00:00Z"),
              rel, [],
              decisions={"duplicate_facts": [],
                         "contradicted_facts": [0, 1]},
              now=_NOW)
    assert r["sorted_candidate_uuids"] == ["dated", "non"]
    assert [x["uuid"] for x in r["invalidated"]] == ["dated"]


def check_self_reference(solve):
    """候选里出现 resolved 自己：C 需要严格更早，不会自杀。"""
    rel = [_edge("dup", "Alice lives in Paris",
                 valid="2024-01-01T00:00:00Z")]
    r = _same(solve,
              _edge("new", "Alice lives in Berlin",
                    valid="2024-03-01T00:00:00Z"),
              rel, [],
              decisions={"duplicate_facts": [0], "contradicted_facts": [0]},
              now=_NOW)
    assert r["resolved"]["uuid"] == "dup"
    assert r["invalidated"] == []
    assert r["resolved"]["invalid_at"] is None


def check_errors(solve):
    def bad(fn):
        try:
            fn()
        except ValueError:
            return True
        except Exception as exc:                   # noqa: BLE001
            raise AssertionError("wrong error: %r" % (exc,))
        raise AssertionError("expected ValueError")

    assert bad(lambda: solve(_edge("new", "x"), [], [], now=None))
    assert bad(lambda: solve(_edge("new", "x"),
                             [_edge("old", "y", valid="2024-01-01T00:00:00Z")],
                             [], decisions=None, now=_NOW))


def check_random_cross(solve):
    """随机边集对拍 + 结构不变量。"""
    facts = ["Alice lives in %s" % city
             for city in ("Paris", "Berlin", "Rome", "Oslo")]
    stamps = [None, "2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z",
              "2024-03-01T00:00:00Z", "2024-04-01T00:00:00Z",
              "2024-05-01T00:00:00Z"]

    for seed in range(60):
        state = seed
        def pick(seq):
            nonlocal state
            state = (state * 1103515245 + 12345) % (1 << 31)
            return seq[state % len(seq)]

        n_rel = seed % 4
        n_ex = (seed // 4) % 3
        rel = [_edge("r%d" % i, pick(facts), valid=pick(stamps),
                     invalid=pick(stamps) if seed % 3 else None,
                     expired=pick([None, None, "2023-01-01T00:00:00Z"]))
               for i in range(n_rel)]
        ex = [_edge("x%d" % i, pick(facts), valid=pick(stamps),
                    invalid=pick(stamps) if seed % 2 else None)
              for i in range(n_ex)]
        total = n_rel + n_ex
        contras = [pick(range(-2, total + 2)) for _ in range(seed % 4)]
        dups = [pick(range(-1, max(n_rel, 1) + 1)) for _ in range(seed % 3)]
        kwargs = dict(
            extracted=_edge("new", pick(facts), valid=pick(stamps),
                            invalid=pick(stamps) if seed % 5 == 0 else None),
            related_edges=rel,
            existing_edges=ex,
            decisions={"duplicate_facts": dups,
                       "contradicted_facts": contras},
            now=_NOW,
            timestamps={"valid_at": pick(stamps)}
            if seed % 2 else None,
        )
        r = _same(solve, **kwargs)

        if r["path"] == "resolve":
            head = r["resolved"]
            for row in r["invalidated"]:
                assert row["uuid"] in r["candidate_uuids"]
                if head["valid_at"] is not None:
                    assert row["invalid_at"] == head["valid_at"]
            assert len(r["invalidated"]) <= len(r["candidate_uuids"])
            for row in r["edges"].values():
                if row["invalid_at"] is not None:
                    assert row["uuid"] != head["uuid"] or \
                        row is not None


def check_example(solve):
    """题面示例。"""
    r = _same(solve,
              _edge("new", "Alice lives in Berlin",
                    valid="2024-03-01T00:00:00Z"),
              [_edge("old", "Alice lives in Paris",
                     valid="2024-01-01T00:00:00Z")],
              [],
              decisions={"duplicate_facts": [], "contradicted_facts": [0]},
              now=_NOW)
    assert r["path"] == "resolve"
    assert r["resolved"]["uuid"] == "new"
    assert r["resolved"]["valid_at"] == "2024-03-01T00:00:00Z"
    assert r["resolved"]["invalid_at"] is None
    # 旧事实在 2024-03-01 被新事实顶掉，事务时刻是 now。
    assert r["invalidated"] == [{
        "uuid": "old",
        "fact": "Alice lives in Paris",
        "valid_at": "2024-01-01T00:00:00Z",
        "invalid_at": "2024-03-01T00:00:00Z",
        "expired_at": _NOW,
    }]
    assert r["edges"]["old"]["invalid_at"] == "2024-03-01T00:00:00Z"


check_example.example = True
