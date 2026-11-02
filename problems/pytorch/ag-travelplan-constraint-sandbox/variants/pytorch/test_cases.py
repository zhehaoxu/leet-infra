"""TravelPlanner 约束沙盒的判据。所有期望值按上游 `evaluation/commonsense_constraint.py` 手工推算。"""

CITY_STATE = {"A": "S1", "B": "S1", "C": "S2", "D": "S2"}

ORDER = [
    "is_reasonable_visiting_city",
    "is_valid_restaurants",
    "is_valid_attractions",
    "is_valid_accommodation",
    "is_valid_transportation",
    "is_valid_information_in_current_city",
    "is_valid_information_in_sandbox",
    "is_not_absent",
]


def _query(**kw):
    query = {
        "org": "A",
        "dest": "S1",
        "days": 3,
        "visiting_city_number": 1,
        "people_number": 2,
        "budget": 1500,
    }
    query.update(kw)
    return query


def _trip():
    return [
        {
            "current_city": "from A to B",
            "transportation": "Self-driving, from A to B, time: 1.5 hours",
            "breakfast": "R1, A",
            "lunch": "R2, A",
            "dinner": "R3, A",
            "attraction": "X, A;",
            "accommodation": "H2, B",
        },
        {
            "current_city": "B",
            "transportation": "-",
            "breakfast": "R4, B",
            "lunch": "R5, B",
            "dinner": "R6, B",
            "attraction": "Y, B;",
            "accommodation": "H2, B",
        },
        {
            "current_city": "from B to A",
            "transportation": "Self-driving, from B to A, time: 1.5 hours",
            "breakfast": "R7, B",
            "lunch": "R8, B",
            "dinner": "R9, B",
            "attraction": "V, B;",
            "accommodation": "-",
        },
    ]


def _sandbox(**over):
    box = {
        "restaurants": [{"name": "R%d" % i, "city": city} for city in "AB" for i in range(1, 10)],
        "accommodations": [{"name": "H2", "city": "B", "minimum_nights": 2}],
        "attractions": [
            {"name": "X", "city": "A"},
            {"name": "Y", "city": "B"},
            {"name": "V", "city": "B"},
        ],
        "flights": {"F1": {"origin": "B", "dest": "A"}},
        "distances": {
            ("A", "B", "self-driving"): 120.0,
            ("B", "A", "self-driving"): 120.0,
        },
    }
    box.update(over)
    return box


def _run(solve, trip=None, query=None, sandbox=None):
    return solve(query or _query(), trip or _trip(), CITY_STATE, sandbox or _sandbox())


def _failed(solve, trip=None, query=None, sandbox=None):
    return _run(solve, trip, query, sandbox)["failed"]


def check_base_passes_everything(solve):
    """基座行程：8 条判据全过，passed 为真、failed 为空。"""
    out = _run(solve)
    assert out["passed"] is True
    assert out["failed"] == []
    assert [name for name in out["checks"]] == ORDER
    for name in ORDER:
        assert out["checks"][name] == {"ok": True, "msg": None}, (name, out["checks"][name])


def check_result_shape(solve):
    """返回体只有三键，每条判据都是 {ok, msg}，失败项按固定顺序收集。"""
    trip = _trip()
    trip[0]["dinner"] = "R1, A"
    out = _run(solve, trip)
    assert set(out) == {"checks", "failed", "passed"}
    assert out["passed"] is False
    assert out["failed"] == ["is_valid_restaurants"]
    assert out["checks"]["is_valid_restaurants"]["ok"] is False
    assert isinstance(out["checks"]["is_valid_restaurants"]["msg"], str)


def check_city_pair_expands_two_nodes(solve):
    """`from A to B` 展开成两个城市结点，所以 3 天的 'from' 行程得到 5 个城市的闭环。"""
    out = _run(solve)
    assert out["checks"]["is_reasonable_visiting_city"]["ok"] is True
    assert out["checks"]["is_valid_information_in_current_city"]["ok"] is True

    # 同一趟行程若把首日改成裸城市，只剩 3 个结点，中间城单例直接判非法
    trip = _trip()
    trip[0]["current_city"] = "A"
    trip[0]["transportation"] = "Self-driving, from A to B"
    trip[2]["current_city"] = "A"
    trip[2]["transportation"] = "-"
    bad = _run(solve, trip)
    assert bad["checks"]["is_reasonable_visiting_city"] == {
        "ok": False, "msg": "The city sequence is invalid."
    }


def check_first_day_must_start_from_org(solve):
    """第 0 天用 'from X to B' 且 X 不是出发地 -> 首日城市判据报错。"""
    trip = _trip()
    trip[0]["current_city"] = "from C to B"
    trip[2]["current_city"] = "from B to C"
    query = _query(dest="S2")
    out = _run(solve, trip, query)
    assert out["checks"]["is_reasonable_visiting_city"] == {
        "ok": False, "msg": "The first day's city should be A."
    }


def check_trip_must_be_closed_circle(solve):
    """首尾城市必须相同；不闭环时先于城市序列判据报错。"""
    trip = _trip()
    trip[2]["current_city"] = "B"
    out = _run(solve, trip)
    assert out["checks"]["is_reasonable_visiting_city"] == {
        "ok": False, "msg": "The trip should be a closed circle."
    }


def check_city_must_be_known(solve):
    """城市必须在 `city_state_map` 里；未知城市单独报错。"""
    trip = _trip()
    trip[2]["current_city"] = "from B to A"
    out = _run(solve, trip, sandbox=_sandbox(), query=_query())
    assert out["passed"] is True

    out = solve(_query(), _trip(), {"B": "S1"}, _sandbox())
    assert out["checks"]["is_reasonable_visiting_city"] == {
        "ok": False, "msg": "A is not a valid city."
    }


def check_long_trip_checks_middle_states(solve):
    """`days > 3` 时中间结点必须落在目的州；4 天段里 C 属于 S2 而不是 S1。"""
    trip = [
        {"current_city": "from A to B", "transportation": "Taxi, from A to B",
         "breakfast": "R1, A", "lunch": "R2, A", "dinner": "R3, A",
         "attraction": "X, A;", "accommodation": "H2, B"},
        {"current_city": "B", "transportation": "-",
         "breakfast": "R4, B", "lunch": "R5, B", "dinner": "R6, B",
         "attraction": "Y, B;", "accommodation": "H2, B"},
        {"current_city": "from B to C", "transportation": "Taxi, from B to C",
         "breakfast": "R7, B", "lunch": "R8, B", "dinner": "R9, B",
         "attraction": "V, B;", "accommodation": "H3, C"},
        {"current_city": "from C to A", "transportation": "Taxi, from C to A",
         "breakfast": "R1, C", "lunch": "R2, C", "dinner": "R3, C",
         "attraction": "W, C;", "accommodation": "-"},
    ]
    box = _sandbox(
        restaurants=[{"name": "R%d" % i, "city": city} for city in "ABC" for i in range(1, 10)],
        accommodations=[{"name": "H2", "city": "B", "minimum_nights": 2},
                        {"name": "H3", "city": "C", "minimum_nights": 1}],
        attractions=[{"name": "X", "city": "A"}, {"name": "Y", "city": "B"},
                     {"name": "V", "city": "B"}, {"name": "W", "city": "C"}],
        distances={("A", "B", "taxi"): 1.0, ("B", "C", "taxi"): 1.0, ("C", "A", "taxi"): 1.0},
    )
    query = _query(days=4, visiting_city_number=2)
    out = _run(solve, trip, query, box)
    assert out["failed"] == ["is_reasonable_visiting_city"]
    assert out["checks"]["is_reasonable_visiting_city"] == {
        "ok": False, "msg": "C is not in S1."
    }

    # 对照：4 天全在目的州内（A 出发、B 停留三段、回 A）就整趟通过
    rows = [
        ("from A to B", "Taxi, from A to B", "R1, A", "R2, A", "R3, A", "X, A;", "H2, B"),
        ("B", "-", "R4, B", "R5, B", "R6, B", "Y, B;", "H2, B"),
        ("B", "-", "R7, B", "R8, B", "R9, B", "V, B;", "H2, B"),
        ("from B to A", "Taxi, from B to A", "R1, B", "R2, B", "R3, B", "Z, B;", "-"),
    ]
    keys = ("current_city", "transportation", "breakfast", "lunch", "dinner",
            "attraction", "accommodation")
    trip = [dict(zip(keys, row)) for row in rows]
    box["attractions"].append({"name": "Z", "city": "B"})
    box["distances"][("B", "A", "taxi")] = 1.0
    ok = _run(solve, trip, _query(days=4, visiting_city_number=1), box)
    assert ok["passed"] is True, ok["failed"]


def check_days_must_cover_itinerary(solve):
    """行程条数少于 `query['days']` -> 缺日程，且 not_absent 直接短路成 Invalid Days。"""
    out = _run(solve, _trip()[:2], _query(days=3))
    assert out["checks"]["is_not_absent"] == {"ok": False, "msg": "Invalid Days"}
    assert out["checks"]["is_not_absent"]["ok"] is False
    assert out["checks"]["is_valid_restaurants"]["ok"] is True


def check_extra_days_are_ignored(solve):
    """行程比 `days` 长：多出来的那天完全不参与判据（重复餐厅也看不见）。"""
    trip = _trip() + [dict(_trip()[2], current_city="A", attraction="Y, B;")]
    out = _run(solve, trip, _query(days=3))
    assert out["passed"] is True
    out = _run(solve, trip, _query(days=4, visiting_city_number=1))
    assert out["checks"]["is_valid_restaurants"]["ok"] is False


def check_restaurant_repeat_across_days(solve):
    """三餐按「整串名字」跨天共用一张去重表，第 1 天与第 3 天撞同一家就报错。"""
    trip = _trip()
    trip[2]["breakfast"] = "R1, A"
    assert _failed(solve, trip) == ["is_valid_restaurants"]

    trip = _trip()
    trip[2]["breakfast"] = "R1, B"
    assert _failed(solve, trip) == []


def check_attraction_needs_trailing_semicolon(solve):
    """`split(';')[:-1]` 会丢掉最后一段：没有分号收尾的景点字符串等于没填。"""
    trip = _trip()
    trip[1]["attraction"] = "Y, B"
    assert _failed(solve, trip) == []

    trip = _trip()
    trip[1]["attraction"] = "Y, B;Y, B;"
    assert _failed(solve, trip) == ["is_valid_attractions"]

    # 缺尾分号时不仅不查重，也不参与「城市归属」和「沙箱存在性」两处校验
    trip = _trip()
    trip[1]["attraction"] = "NoSuchPlace, Z"
    assert _failed(solve, trip) == []


def check_accommodation_minimum_nights(solve):
    """连续入住天数按 `count_consecutive_values` 的 run 长度比 minimum nights。"""
    box = _sandbox(accommodations=[{"name": "H2", "city": "B", "minimum_nights": 3}])
    out = _run(solve, sandbox=box)
    assert out["checks"]["is_valid_accommodation"] == {
        "ok": False, "msg": "The accommodation H2, B do not obey the minumum nights rule."
    }


def check_accommodation_runs_split_on_change(solve):
    """住宿换一家就断开 run：两段各 1 晚，都不满足 minimum nights=2。"""
    trip = _trip()
    trip[1]["accommodation"] = "H3, B"
    box = _sandbox(accommodations=[
        {"name": "H2", "city": "B", "minimum_nights": 2},
        {"name": "H3", "city": "B", "minimum_nights": 2},
    ])
    out = _run(solve, trip, sandbox=box)
    assert out["checks"]["is_valid_accommodation"]["ok"] is False
    assert out["checks"]["is_valid_accommodation"]["msg"].startswith("The accommodation H2, B")


def check_accommodation_key_required(solve):
    """`accommodation` 键缺失（而不是 '-'）会让住宿判据整条报错。"""
    trip = _trip()
    del trip[1]["accommodation"]
    out = _run(solve, trip)
    assert out["checks"]["is_valid_accommodation"] == {"ok": False, "msg": "No Accommodation Info."}
    assert out["checks"]["is_not_absent"] == {"ok": False, "msg": "No Accommodation Info."}


def check_transport_day1_required(solve):
    """第一天交通必须是非空且非 '-'。"""
    trip = _trip()
    trip[0]["transportation"] = "-"
    out = _run(solve, trip)
    assert out["checks"]["is_valid_transportation"] == {
        "ok": False, "msg": "The transportation in day 1 should not be empty."
    }


def check_transport_conflicts(solve):
    """冲突只有两种组合：自驾+飞机、出租+自驾；出租+飞机是允许的。"""
    for first, last, expected in (
        ("Self-driving, from A to B", "Flight Number: F1, from B to A", False),
        ("Taxi, from A to B", "Self-driving, from B to A", False),
        ("Taxi, from A to B", "Flight Number: F1, from B to A", True),
    ):
        trip = _trip()
        trip[0]["transportation"] = first
        trip[2]["transportation"] = last
        box = _sandbox(distances={("A", "B", "taxi"): 5.0, ("B", "A", "taxi"): 5.0,
                                  ("B", "A", "self-driving"): 5.0, ("A", "B", "self-driving"): 5.0})
        out = _run(solve, trip, sandbox=box)
        assert out["checks"]["is_valid_transportation"]["ok"] is expected, first


def check_transport_match_priority(solve):
    """`transportation_match` 先匹配 taxi，再 self-driving，最后 flight。"""
    trip = _trip()
    trip[0]["transportation"] = "Taxi to airport, Flight Number: F1, from A to B"
    trip[2]["transportation"] = "Flight Number: F2, from B to A"
    box = _sandbox(flights={"F1": {"origin": "A", "dest": "B"},
                            "F2": {"origin": "B", "dest": "A"}},
                   distances={})
    out = _run(solve, trip, sandbox=box)
    # 这串同时含 taxi 与 flight，被记成 Taxi；Taxi + Flight 不算冲突
    assert out["checks"]["is_valid_transportation"]["ok"] is True
    assert out["checks"]["is_valid_information_in_sandbox"]["ok"] is True


def check_city_consistency_rules(solve):
    """当天每个字段都得提到当天城市；住宿只认「当天末城」，单城日不认别的城市。"""
    trip = _trip()
    trip[1]["breakfast"] = "R10, A"
    box = _sandbox(restaurants=_sandbox()["restaurants"] + [{"name": "R10", "city": "A"}])
    assert _failed(solve, trip, sandbox=box) == ["is_valid_information_in_current_city"]

    trip = _trip()
    trip[0]["accommodation"] = "H2, A"
    out = _run(solve, trip)
    assert out["checks"]["is_valid_information_in_current_city"] == {
        "ok": False, "msg": "The accommodation in day 1 is invalid city choice."
    }

    # 跨城日两个城市都算「当天城市」，写哪个都能过
    trip = _trip()
    trip[0]["breakfast"] = "R1, B"
    assert _failed(solve, trip) == []


def check_sandbox_matches_name_by_substring(solve):
    """餐厅/住宿/景点按「名字是库中名字的子串 + 城市相等」匹配，'R1' 命中 'R10'。"""
    box = _sandbox(restaurants=[{"name": "R10", "city": "A"}, {"name": "R2", "city": "A"},
                                {"name": "R3", "city": "A"}] +
                   [{"name": "R%d" % i, "city": "B"} for i in range(4, 10)])
    out = _run(solve, sandbox=box)
    assert out["checks"]["is_valid_information_in_sandbox"]["ok"] is True

    # 城市必须精确相等：这张库的 B 城没有 R1，行程写到 B 就找不到
    trip = _trip()
    trip[0]["breakfast"] = "R1, B"
    out = _run(solve, trip, sandbox=box)
    assert out["checks"]["is_valid_information_in_sandbox"]["ok"] is False


def check_sandbox_name_city_splits_at_last_comma(solve):
    """`(.*?),\\s*([^,]+)$` 把名字切在最后一个逗号上，多一段就整条匹配不上。"""
    trip = _trip()
    trip[0]["breakfast"] = "R1, A, extra"
    out = _run(solve, trip)
    assert out["checks"]["is_valid_information_in_sandbox"]["ok"] is False
    assert "breakfast" in out["checks"]["is_valid_information_in_sandbox"]["msg"]


def check_sandbox_city_strips_parenthesis(solve):
    """城市末尾的 `(State)` 会被 `extract_before_parenthesis` 剁掉。"""
    trip = _trip()
    trip[0]["breakfast"] = "R1, A (S1)"
    assert _failed(solve, trip) == []


def check_sandbox_flight_lookup(solve):
    """航班号从 'Flight Number: ' 后切到第一个逗号，并按起降城市双匹配。"""
    trip = _trip()
    trip[2]["transportation"] = "Flight Number: F1, from B to A"
    trip[0]["transportation"] = "Taxi, from A to B"
    box = _sandbox(distances={("A", "B", "taxi"): 1.0})
    out = _run(solve, trip, sandbox=box)
    assert out["checks"]["is_valid_information_in_sandbox"]["ok"] is True

    trip[2]["transportation"] = "Flight Number: F9, from B to A"
    out = _run(solve, trip, sandbox=box)
    assert out["checks"]["is_valid_information_in_sandbox"]["ok"] is False

    # 城市对不上（表里是 B->A，行程写 from A to B 仍是同一航班号）
    trip[2]["transportation"] = "Flight Number: F1, from A to B"
    out = _run(solve, trip, sandbox=box)
    assert out["checks"]["is_valid_information_in_sandbox"]["ok"] is False


def check_sandbox_distance_missing(solve):
    """自驾/出租的起讫城市必须在距离表里，缺项即判无效。"""
    trip = _trip()
    trip[0]["transportation"] = "Taxi, from A to B"
    trip[2]["transportation"] = "Taxi, from B to A"
    box = _sandbox(distances={("A", "B", "taxi"): 1.0})
    out = _run(solve, trip, sandbox=box)
    assert out["checks"]["is_valid_information_in_sandbox"] == {
        "ok": False, "msg": "The taxi in day 3 is invalid in the sandbox."
    }


def check_not_absent_needs_all_six_fields(solve):
    """缺字段与空值是两回事：缺键直接 No xxx Info.，空值才进后面的逐条规则。"""
    trip = _trip()
    del trip[0]["attraction"]
    out = _run(solve, trip)
    assert out["checks"]["is_not_absent"]["msg"] == "No Attraction Info."


def check_not_absent_day_rules(solve):
    """跨城日必须填交通、单城日必须有景点、非末日必须有住宿、非跨城日必须有三餐。"""
    trip = _trip()
    trip[2]["transportation"] = "-"
    out = _run(solve, trip)
    assert out["checks"]["is_not_absent"]["msg"] == "No transportation in day 3 is not allowed."

    trip = _trip()
    trip[1]["attraction"] = "-"
    out = _run(solve, trip)
    assert out["checks"]["is_not_absent"]["msg"] == "No attaction in day 2 is not allowed."

    trip = _trip()
    trip[1]["accommodation"] = "-"
    out = _run(solve, trip)
    assert out["checks"]["is_not_absent"]["msg"] == "No accommodation in day 2 is not allowed."

    trip = _trip()
    trip[1]["dinner"] = "-"
    out = _run(solve, trip)
    assert out["checks"]["is_not_absent"]["msg"] == "No meal in day 2 is not allowed."


def check_not_absent_fill_ratio(solve):
    """填充率的分母是 `6 * days`，分子却按 unit 的**全部键**数（连 current_city 也算）。"""
    rows = [
        ("from A to B", "Taxi, from A to B", "X, A;", "H2, B"),
        ("from B to B", "Taxi, from B to B", "Y, B;", "H2, B"),
        ("from B to A", "Taxi, from B to A", "V, B;", "-"),
    ]
    box = _sandbox(distances={("A", "B", "taxi"): 1.0, ("B", "B", "taxi"): 0.0,
                              ("B", "A", "taxi"): 1.0})

    def build(attractions):
        return [{"current_city": city, "transportation": transit,
                 "breakfast": "-", "lunch": "-", "dinner": "-",
                 "attraction": attractions[index], "accommodation": hotel}
                for index, (city, transit, _, hotel) in enumerate(rows)]

    # 只空三餐：剩 11/18，7 个键里的 current_city 把分子抬过了线
    mild = _run(solve, build(["X, A;", "Y, B;", "V, B;"]), sandbox=box)
    assert mild["checks"]["is_not_absent"]["ok"] is True
    assert mild["checks"]["is_valid_restaurants"]["ok"] is True

    # 景点也清空：只剩 8/18，跌破 50%
    out = _run(solve, build(["-", "-", "-"]), sandbox=box)
    assert out["checks"]["is_not_absent"] == {
        "ok": False, "msg": "The absent information is more than 50%."
    }
    assert out["checks"]["is_reasonable_visiting_city"]["ok"] is True


def check_city_number_discards_org(solve):
    """`visiting_city_number` 只数「除出发地外」的城市集合。"""
    out = _run(solve, _trip(), _query(visiting_city_number=2))
    assert out["checks"]["is_not_absent"] == {"ok": False, "msg": "Invalid City Number"}


def check_input_validation(solve):
    """结构化入参校验：布尔不是整数、days 至少 1、itinerary 不能空。"""
    for bad in (
        lambda: solve(_query(days=True), _trip(), CITY_STATE, _sandbox()),
        lambda: solve(_query(days=0), _trip(), CITY_STATE, _sandbox()),
        lambda: solve(_query(), [], CITY_STATE, _sandbox()),
        lambda: solve(_query(org=1), _trip(), CITY_STATE, _sandbox()),
        lambda: solve(_query(), [{"transportation": "-"}], CITY_STATE, _sandbox()),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("应当抛 ValueError")


def check_unparsable_current_city_raises(solve):
    """含 'from' 却解析不出城市对时直接抛错（上游是 `_before_paren(None)` 的 TypeError）。"""
    trip = _trip()
    trip[1]["current_city"] = "fromA to B"
    try:
        _run(solve, trip)
    except ValueError as exc:
        assert "current_city" in str(exc)
    else:
        raise AssertionError("应当抛 ValueError")


def check_unbalanced_parenthesis_kept(solve):
    """`extract_before_parenthesis` 只剁配对的括号；没有右括号就原样保留，城市因此对不上。"""
    trip = _trip()
    trip[0]["accommodation"] = "H2, B("
    out = _run(solve, trip)
    assert out["checks"]["is_valid_information_in_sandbox"]["ok"] is False
    assert out["checks"]["is_valid_information_in_current_city"]["ok"] is True


def check_empty_name_matches_anything(solve):
    """名字切出来是空串时 `'' in 库名` 恒真，城市对上就算命中（反直觉但上游如此）。"""
    trip = _trip()
    trip[0]["breakfast"] = ", A"
    assert _failed(solve, trip) == []

    trip = _trip()
    trip[0]["breakfast"] = ", Z"
    assert _failed(solve, trip) == ["is_valid_information_in_current_city",
                                    "is_valid_information_in_sandbox"]


def check_name_whitespace_stripped(solve):
    """名字与城市两侧空白会被 strip 掉，带空格写法照样命中。"""
    trip = _trip()
    trip[0]["breakfast"] = " R1 , A "
    assert _failed(solve, trip) == []


def check_transport_taxi_wins_over_self_driving(solve):
    """同一串里既有 taxi 又有 self-driving 时记成 Taxi，于是不再与飞机冲突。"""
    trip = _trip()
    trip[0]["transportation"] = "Self-driving then taxi, from A to B"
    trip[2]["transportation"] = "Flight Number: F2, from B to A"
    box = _sandbox(flights={"F2": {"origin": "B", "dest": "A"}},
                   distances={("A", "B", "self-driving"): 1.0})
    out = _run(solve, trip, sandbox=box)
    assert out["checks"]["is_valid_transportation"]["ok"] is True
    assert out["checks"]["is_valid_information_in_sandbox"]["ok"] is True


def check_transport_unknown_kind_is_neutral(solve):
    """认不出的交通方式归一成 None：既不参与冲突表，也不进沙箱查证。"""
    trip = _trip()
    trip[0]["transportation"] = "Bus, from A to B"
    out = _run(solve, trip)
    assert out["checks"]["is_valid_transportation"]["ok"] is True
    assert out["checks"]["is_valid_information_in_sandbox"]["ok"] is True


def check_one_day_trip_fails_sequence(solve):
    """城市结点少于 3 个一律判序列非法：单日游（1 个结点或 from A to A 两个结点）都过不了。"""
    trip = [{"current_city": "A", "transportation": "-", "breakfast": "R1, A",
             "lunch": "R2, A", "dinner": "R3, A", "attraction": "X, A;",
             "accommodation": "-"}]
    out = _run(solve, trip, _query(days=1, visiting_city_number=0))
    assert out["checks"]["is_reasonable_visiting_city"] == {
        "ok": False, "msg": "The city sequence is invalid."
    }

    trip[0]["current_city"] = "from A to A"
    trip[0]["transportation"] = "Taxi, from A to A"
    box = _sandbox(distances={("A", "A", "taxi"): 0.0})
    out = _run(solve, trip, _query(days=1, visiting_city_number=0), box)
    assert out["checks"]["is_reasonable_visiting_city"] == {
        "ok": False, "msg": "The city sequence is invalid."
    }


def check_state_gate_only_above_three_days(solve):
    """`days > 3` 才查中间结点属于目的州；3 天行程跑到另一个州也放行。"""
    rows = [
        ("from A to B", "Taxi, from A to B", "R1, A", "R2, A", "R3, A", "X, A;", "H2, B"),
        ("from B to C", "Taxi, from B to C", "R4, B", "R5, B", "R6, B", "Y, B;", "H3, C"),
        ("from C to A", "Taxi, from C to A", "R7, C", "R8, C", "R9, C", "V, C;", "-"),
    ]
    keys = ("current_city", "transportation", "breakfast", "lunch", "dinner",
            "attraction", "accommodation")
    trip = [dict(zip(keys, row)) for row in rows]
    box = _sandbox(
        restaurants=[{"name": "R%d" % i, "city": city} for city in "ABC" for i in range(1, 10)],
        accommodations=[{"name": "H2", "city": "B", "minimum_nights": 1},
                        {"name": "H3", "city": "C", "minimum_nights": 1}],
        attractions=[{"name": "X", "city": "A"}, {"name": "Y", "city": "B"},
                     {"name": "V", "city": "C"}],
        distances={("A", "B", "taxi"): 1.0, ("B", "C", "taxi"): 1.0, ("C", "A", "taxi"): 1.0},
    )
    out = _run(solve, trip, _query(visiting_city_number=2), box)
    assert out["checks"]["is_reasonable_visiting_city"]["ok"] is True


def check_place_city_must_equal(solve):
    """沙箱里城市用相等比较：库里写 'AB' 不能当成 'A'。"""
    box = _sandbox(restaurants=[{"name": "R1", "city": "AB"}, {"name": "R2", "city": "A"},
                                {"name": "R3", "city": "A"}] +
                   [{"name": "R%d" % i, "city": "B"} for i in range(4, 10)])
    out = _run(solve, sandbox=box)
    assert out["checks"]["is_valid_information_in_sandbox"]["ok"] is False


def check_restaurant_repeat_same_day(solve):
    """同一天午晚餐撞同一家也算重复。"""
    trip = _trip()
    trip[0]["dinner"] = "R2, A"
    assert _failed(solve, trip) == ["is_valid_restaurants"]


def check_min_nights_skipped_when_ambiguous(solve):
    """库里同名同城有多行时直接跳过 minimum nights 判据（不拿第一行凑数）。"""
    box = _sandbox(accommodations=[{"name": "H2", "city": "B", "minimum_nights": 9},
                                   {"name": "H2", "city": "B", "minimum_nights": 2}])
    out = _run(solve, sandbox=box)
    assert out["checks"]["is_valid_accommodation"]["ok"] is True


def check_flight_checks_both_cities(solve):
    """航班号命中还不够，起降城市也要对上。"""
    trip = _trip()
    trip[0]["transportation"] = "Taxi, from A to B"
    trip[2]["transportation"] = "Flight Number: F1, from B to A"
    box = _sandbox(flights={"F1": {"origin": "C", "dest": "A"}},
                   distances={("A", "B", "taxi"): 1.0})
    out = _run(solve, trip, sandbox=box)
    assert out["checks"]["is_valid_information_in_sandbox"]["ok"] is False

    box["flights"]["F1"]["origin"] = "B"
    out = _run(solve, trip, sandbox=box)
    assert out["checks"]["is_valid_information_in_sandbox"]["ok"] is True


def check_fill_ratio_boundary_passes(solve):
    """恰 9/18 = 0.5 不算「超过 50% 缺失」，是闭区间放行。"""
    rows = [
        ("from A to B", "Taxi, from A to B", "-", "H2, B"),
        ("from B to B", "Taxi, from B to B", "Y, B;", "H2, B"),
        ("from B to A", "Taxi, from B to A", "-", "-"),
    ]
    trip = [{"current_city": city, "transportation": transit, "breakfast": "-",
             "lunch": "-", "dinner": "-", "attraction": attraction, "accommodation": hotel}
            for city, transit, attraction, hotel in rows]
    box = _sandbox(distances={("A", "B", "taxi"): 1.0, ("B", "B", "taxi"): 0.0,
                              ("B", "A", "taxi"): 1.0})
    out = _run(solve, trip, sandbox=box)
    assert out["checks"]["is_not_absent"]["ok"] is True


def check_meal_rule_when_from_has_newline(solve):
    """三餐豁免只认字面 'from '；写成 'from\\nB to B' 时豁免失效，仍然要求三餐。"""
    trip = _trip()
    trip[1]["current_city"] = "from\nB to B"
    trip[1]["transportation"] = "Taxi, B"
    trip[1]["breakfast"] = "-"
    trip[1]["lunch"] = "-"
    trip[1]["dinner"] = "-"
    box = _sandbox(distances={("A", "B", "self-driving"): 1.0, ("B", "A", "self-driving"): 1.0,
                              ("B", "B", "taxi"): 0.0})
    out = _run(solve, trip, sandbox=box)
    assert out["checks"]["is_not_absent"] == {
        "ok": False, "msg": "No meal in day 2 is not allowed."
    }


def check_placeholder_day_counts_as_missing(solve):
    """占位串不算有效日程，`days` 数不满就先报 Invalid Days。"""
    trip = _trip()
    trip[1]["current_city"] = "You don't need to fill in the information for this or later days."
    out = _run(solve, trip)
    assert out["checks"]["is_not_absent"] == {"ok": False, "msg": "Invalid Days"}


def check_empty_string_is_not_missing_key(solve):
    """空串是「有键无值」：不走 No xxx Info.，而是掉进后面的逐日规则。"""
    trip = _trip()
    trip[1]["breakfast"] = ""
    out = _run(solve, trip)
    assert out["checks"]["is_not_absent"]["msg"] == "No meal in day 2 is not allowed."


def run_examples(solve):
    print(_run(solve)["passed"])
    trip = _trip()
    trip[0]["dinner"] = "R1, A"
    print(_run(solve, trip)["failed"])
    trip = _trip()
    trip[2]["transportation"] = "Flight Number: F9, from B to A"
    print(_run(solve, trip)["checks"]["is_valid_information_in_sandbox"])


check_base_passes_everything.example = True
check_restaurant_repeat_across_days.example = True
