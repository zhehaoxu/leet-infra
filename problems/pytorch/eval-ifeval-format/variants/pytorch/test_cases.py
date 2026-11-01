import json
import re


def _py(instruction_ids, kwargs_list, response):
    """oracle：独立实现路径。json 用正则剥 tick；postscript 用逐行 search；
    two_responses 用列表推导与空段位置判定。"""
    results = []
    for iid, kw in zip(instruction_ids, kwargs_list):
        if iid == "detectable_format:json_format":
            s = response.strip()
            s = re.sub(r"^```(?:json|Json|JSON)?", "", s, count=1)
            s = re.sub(r"```$", "", s, count=1)
            s = s.strip()
            try:
                json.loads(s)
                ok = True
            except ValueError:
                ok = False
            results.append({"instruction_id": iid, "observed": ok, "pass": ok})
        elif iid == "detectable_content:postscript":
            marker = kw["postscript_marker"].strip()
            patterns = {
                "P.P.S": r"\s*p\.\s?p\.\s?s.*$",
                "P.S.": r"\s*p\.\s?s\..*$",
            }
            pat = patterns.get(marker, r"\s*" + marker.lower() + r".*$")
            n = 0
            for line in response.lower().splitlines():
                if re.search(pat, line) is not None:
                    n += 1
            results.append({"instruction_id": iid, "observed": n, "pass": n > 0})
        else:  # combination:two_responses
            parts = response.split("******")
            valid = [p for p in parts if p.strip()]
            inner_empty = any(
                not p.strip() for p in parts[1:-1]
            ) if len(parts) > 2 else False
            ok = (not inner_empty) and len(valid) == 2 \
                and valid[0].strip() != valid[1].strip()
            results.append({"instruction_id": iid, "observed": len(valid),
                            "pass": ok})
    return {"results": results, "all_pass": all(r["pass"] for r in results)}


def _compare(solve, instruction_ids, kwargs_list, response):
    led = solve(list(instruction_ids), [dict(k) for k in kwargs_list], response)
    ref = _py(instruction_ids, kwargs_list, response)
    assert led == ref, f"\nled={led}\nref={ref}"
    return led


def check_json_format(solve):
    iid = "detectable_format:json_format"
    cases = [
        # (response, expected pass)
        ('{"a": 1}', True),
        ('[1, 2, 3]', True),
        ('42', True),          # JSON 标量合法
        ('null', True),
        ('true', True),        # JSON 小写布尔合法
        ('True', False),       # Python 大写布尔非法
        ('NaN', True),         # Python json 扩展接受 NaN/Infinity
        ('Infinity', True),
        ("{'a': 1}", False),   # 单引号非法
        ('{"a": 1,}', False),  # 尾逗号非法
        ('', False),
        ('```', False),        # 剥完剩空串
        ('  {}  ', True),      # 外围空白容忍
        ('```json\n{"a": 1}\n```', True),   # markdown tick 包裹
        ('```JSON {}', True),  # 大写变体逐字匹配
        ('```Json {}', True),  # 混合大小写变体
        ('``````{}``````', False),  # 6 反引号只剥一次，剩 ```{} 非法
        ('```json{"a":1}```json', False),  # 尾部不是 ``` 不剥，整体非法
        ('```\n```json{}\n```', False),  # 内层 ```json 前有换行不剥
        (' {} ', True),  # strip 剥 unicode 空白，json.loads 本身不容忍
        ('```json {}\xa0', True),  # tick 内的 unicode 空白只能靠最终 strip 剥
    ]
    for resp, want in cases:
        led = _compare(solve, [iid], [{}], resp)
        assert led["results"][0]["pass"] is want, (resp, led)
        assert isinstance(led["results"][0]["observed"], bool)


def check_postscript(solve):
    iid = "detectable_content:postscript"
    cases = [
        # (response, marker, expected hits, expected pass)
        ("blah\nP.S. take care", "P.S.", 1, True),
        ("p.s. note", "P.S.", 1, True),        # 大小写不敏感
        ("p. s. note", "P.S.", 1, True),       # \\s? 恰一个空白
        ("p.  s. note", "P.S.", 0, False),     # 两空白不匹配（\\s? 只认一个）
        ("PS. note", "P.S.", 0, False),        # 无第一个点
        ("p.s.x", "P.S.", 1, True),            # s. 后无需空白
        ("P.P.S again", "P.P.S", 1, True),
        ("p. p. s again", "P.P.S", 1, True),   # 两处 \\s? 各一空白
        ("p..s again", "P.P.S", 0, False),     # p.. 不匹配
        ("p.  p. s again", "P.P.S", 0, False),  # 两空格 \\s? 只认一个
        ("P.S. x", "P.P.S", 0, False),         # marker 分支互斥
        ("ps: done", "PS:", 1, True),          # 自定义 marker 字面
        ("PSX", "PS:", 0, False),              # 无词边界但 marker 不中
        ("p.s. x", "p.s.", 1, True),           # 小写 marker 走 else 分支
        ("p. s. x", "p.s.", 0, False),         # else 分支 p\\.s\\. 无 \\s?
        ("p.s something", "P.S", 1, True),     # else 无尾点，无词边界
        ("p.something", "P.S", 1, True),       # quirk：p.something 也命中
        ("P.P.S. x", "P.P.S.", 1, True),       # 带尾点 marker 走 else 字面
        ("p.p.s. x", "P.P.S.", 1, True),
        ("P.P.S  x", "P.P.S.", 1, True),       # quirk：marker 不 re.escape，
                                               # p.p.s.. 的两个 . 通配吃两空格
        ("p. p.s. x", "P.P.S.", 0, False),     # p_p_ 骨架对不上
        ("pxs hello", "P.S", 1, True),         # quirk：p.s 的 . 通配 pxs
        ("line1 P.S. first\nbody", "P.S.", 1, True),  # MULTILINE 任意行
        ("a\np.p.s b\np.p.s c", "P.P.S", 2, True),    # 两行各计一次
        ("", "P.S.", 0, False),
        ("no marker here", "P.S.", 0, False),
    ]
    for resp, marker, want_n, want_pass in cases:
        led = _compare(solve, [iid], [{"postscript_marker": marker}], resp)
        assert led["results"][0]["observed"] == want_n, (resp, marker, led)
        assert led["results"][0]["pass"] is want_pass, (resp, marker, led)
    # marker 先 strip 再分支比较
    led = _compare(solve, [iid], [{"postscript_marker": " P.S. "}],
                   "p. s. x")
    assert led["results"][0]["pass"] is True


def check_two_responses(solve):
    iid = "combination:two_responses"
    cases = [
        # (response, expected valid count, expected pass)
        ("a******b", 2, True),
        (" aaa ****** bbb ", 2, True),      # strip 后比较
        ("same******same", 2, False),        # strip 后相等
        (" aaa ******aaa ", 2, False),       # strip 后相等（strip 前不等）
        ("Same******same", 2, True),         # 大小写不等
        ("only one", 1, False),
        ("a******b******c", 3, False),       # 三段
        ("******a******", 1, False),         # 首尾空段跳过，只剩 1 段
        ("a****** ******", 1, False),
        ("a****** ******b", 2, False),       # 中间空段直接 fail
        ("x ****** y ******", 2, True),      # 尾空段跳过仍两段
        ("a*****b", 1, False),               # 5 星不分隔
        ("*******", 1, False),               # 7 星 split 剩 ["", "*"]
        ("************", 0, False),          # 12 星 split 剩两个空段
        ("", 0, False),
        (" ****** ****** ", 0, False),       # 全空段
    ]
    for resp, want_n, want_pass in cases:
        led = _compare(solve, [iid], [{}], resp)
        assert led["results"][0]["observed"] == want_n, (resp, led)
        assert led["results"][0]["pass"] is want_pass, (resp, led)


def check_combined(solve):
    # json_format 要求整体可解析：postscript 只能藏进 JSON 字符串值里
    r = '{"verdict": "A", "note": "P.S. see appendix."}'
    led = _compare(solve, [
        "detectable_format:json_format",
        "detectable_content:postscript",
    ], [{}, {"postscript_marker": "P.S."}], r)
    assert led["all_pass"] is True
    assert led["results"][0]["observed"] is True
    assert led["results"][1]["observed"] == 1
    # postscript 在 JSON 外 -> json_format fail（整体不再是合法 JSON）
    led2 = _compare(solve, [
        "detectable_format:json_format",
        "detectable_content:postscript",
    ], [{}, {"postscript_marker": "P.S."}],
        '{"verdict": "A"}\nP.S. see appendix.')
    assert led2["all_pass"] is False
    assert led2["results"][0]["pass"] is False
    assert led2["results"][1]["pass"] is True
    # two_responses 两段 json：two pass，json fail（两段整体不可解析）
    led3 = _compare(solve, [
        "detectable_format:json_format",
        "combination:two_responses",
    ], [{}, {}], '{"a": 1}******{"a": 2}')
    assert led3["all_pass"] is False
    assert led3["results"][0]["pass"] is False
    assert led3["results"][1]["pass"] is True
    # 纯单段 json：json pass，two fail
    led4 = _compare(solve, [
        "detectable_format:json_format",
        "combination:two_responses",
    ], [{}, {}], '{"a": 1}')
    assert led4["all_pass"] is False
    assert led4["results"][0]["pass"] is True
    assert led4["results"][1]["pass"] is False


def check_oracle_random(solve):
    import random
    rng = random.Random(709)
    ids = ["detectable_format:json_format",
           "detectable_content:postscript",
           "combination:two_responses"]
    markers = ["P.S.", "P.P.S", "P.S", "PS:", "p.s.", "P.P.S.", "p.p.s"]
    json_ok = ['{"a": 1}', '[1, 2]', '"str"', '42', 'null', 'true',
               '{"k": [1, {"x": null}]}']
    json_bad = ["{'a': 1}", '{"a": 1,}', '', 'True', 'undefined']
    ticks = ['', '```json', '```Json', '```JSON', '```']
    bodies = ["body text here", "line one\nline two",
              "psx word", "p.something", "answer: 42", "p.p.s later"]

    def random_response():
        style = rng.randint(0, 3)
        if style == 0:  # 纯 json 片段，可选 tick 包裹
            frag = rng.choice(json_ok if rng.random() < 0.7 else json_bad)
            pre = rng.choice(ticks)
            suf = "```" if pre and rng.random() < 0.8 else ""
            pad = " " if rng.random() < 0.3 else ""
            return f"{pad}{pre}{frag}{suf}{pad}"
        if style == 1:  # 正文 + postscript 行
            line = rng.choice([
                "P.S. note", "p.s. note", "p. s. note", "p.  s. note",
                "P.P.S again", "p. p. s again", "ps: done", "PSX",
                "P.P.S. tail", "P.S", "p.something",
            ])
            return rng.choice(bodies) + "\n" + line
        if style == 2:  # 两段式
            seg = ["resp A", "resp B", "same", "  ", "", "resp C"]
            n_seg = rng.randint(1, 3)
            parts = [rng.choice(seg) for _ in range(n_seg)]
            sep = rng.choice(["******", "*****", "*******"])
            return sep.join(parts)
        # 混合：json + postscript
        frag = rng.choice(json_ok)
        return (f"```json{frag}```\n"
                + rng.choice(["P.S. tail", "p.p.s more", "plain"]))

    for _ in range(120):
        n = rng.randint(1, 4)
        ids_r = [rng.choice(ids) for _ in range(n)]
        kws = [{"postscript_marker": rng.choice(markers)}
               if iid == "detectable_content:postscript" else {}
               for iid in ids_r]
        resp = random_response()
        led = solve(ids_r, kws, resp)
        ref = _py(ids_r, kws, resp)
        assert led == ref, (ids_r, kws, resp, led, ref)


def check_invalid_inputs(solve):
    bad = [
        lambda: solve([], [], "text"),
        lambda: solve(["unknown:id"], [{}], "text"),
        lambda: solve(["detectable_format:json_format", "combination:two_responses"],
                      [{}], "text"),
        lambda: solve(["detectable_format:json_format"], "not a dict", "text"),
        lambda: solve(["detectable_format:json_format"], [{}], 42),
        lambda: solve(["detectable_content:postscript"], [{}], "text"),
        lambda: solve(["detectable_content:postscript"],
                      [{"postscript_marker": ""}], "text"),
        lambda: solve(["detectable_content:postscript"],
                      [{"postscript_marker": "   "}], "text"),
        lambda: solve(["detectable_content:postscript"],
                      [{"postscript_marker": None}], "text"),
        lambda: solve(["detectable_content:postscript"],
                      [{"postscript_marker": 3}], "text"),
    ]
    for fn in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
