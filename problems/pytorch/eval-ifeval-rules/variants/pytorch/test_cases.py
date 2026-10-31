import re


def _py(instruction_ids, kwargs_list, response):
    results = []
    for iid, kw in zip(instruction_ids, kwargs_list):
        if iid == "keywords:existence":
            hits = 0
            low = response.lower()
            for k in kw["keywords"]:
                if k.lower() in low:
                    hits += 1
            results.append({"instruction_id": iid, "observed": hits,
                            "pass": hits == len(kw["keywords"])})
        elif iid == "keywords:forbidden":
            hits = 0
            low = response.lower()
            for k in kw["forbidden_words"]:
                if k.lower() in low:
                    hits += 1
            results.append({"instruction_id": iid, "observed": hits, "pass": hits == 0})
        elif iid == "keywords:frequency":
            cnt = len(re.findall(r"\b" + re.escape(kw["keyword"]) + r"\b",
                                 response.lower()))
            ok = (cnt >= kw["frequency"]) if kw["relation"] == "at least" \
                else (cnt < kw["frequency"])
            results.append({"instruction_id": iid, "observed": cnt, "pass": ok})
        elif iid == "length_constraints:number_words":
            cnt = len(response.split())
            ok = (cnt >= kw["num_words"]) if kw["relation"] == "at least" \
                else (cnt < kw["num_words"])
            results.append({"instruction_id": iid, "observed": cnt, "pass": ok})
        elif iid == "format:end_phrase":
            ep = kw["end_phrase"]
            tail = response.strip()
            while tail.endswith('"'):
                tail = tail[:-1]
            ok = tail.lower().endswith(ep.lower())
            results.append({"instruction_id": iid, "observed": ok, "pass": ok})
        else:
            cnt = 0
            for line in response.splitlines():
                if line[:2] == "* ":
                    cnt += 1
            results.append({"instruction_id": iid, "observed": cnt,
                            "pass": cnt >= kw["num_bullets"]})
    return {"results": results, "all_pass": all(r["pass"] for r in results)}


def _compare(solve, instruction_ids, kwargs_list, response):
    led = solve(list(instruction_ids), [dict(k) for k in kwargs_list], response)
    ref = _py(instruction_ids, kwargs_list, response)
    assert led == ref, f"\nled={led}\nref={ref}"
    return led


def check_keywords_existence(solve):
    r = "The cat sat on the mat. The dog barked."
    led = _compare(solve, ["keywords:existence"], [{"keywords": ["cat", "dog"]}], r)
    assert led["results"][0]["observed"] == 2 and led["results"][0]["pass"] is True
    led2 = _compare(solve, ["keywords:existence"], [{"keywords": ["cat", "bird"]}], r)
    assert led2["results"][0]["pass"] is False
    # 大小写不敏感子串
    led3 = _compare(solve, ["keywords:existence"], [{"keywords": ["CAT"]}], r)
    assert led3["results"][0]["pass"] is True


def check_forbidden(solve):
    r = "Clean text with nothing bad inside."
    led = _compare(solve, ["keywords:forbidden"], [{"forbidden_words": ["porn", "xxx"]}], r)
    assert led["results"][0]["pass"] is True and led["results"][0]["observed"] == 0
    led2 = _compare(solve, ["keywords:forbidden"], [{"forbidden_words": ["bad"]}], r)
    assert led2["results"][0]["pass"] is False and led2["results"][0]["observed"] == 1


def check_frequency_word_boundary(solve):
    r = "Spam and eggs. More spam, spam!"
    # \b 词边界：spam 出现 3 次；"spa" 不因子串命中
    led = _compare(solve, ["keywords:frequency"],
                   [{"keyword": "spam", "frequency": 3, "relation": "at least"}], r)
    assert led["results"][0]["observed"] == 3 and led["results"][0]["pass"] is True
    led2 = _compare(solve, ["keywords:frequency"],
                    [{"keyword": "spa", "frequency": 1, "relation": "at least"}], r)
    assert led2["results"][0]["observed"] == 0
    led3 = _compare(solve, ["keywords:frequency"],
                    [{"keyword": "spam", "frequency": 3, "relation": "less than"}], r)
    assert led3["results"][0]["pass"] is False


def check_number_words(solve):
    r = "one two three four five"
    led = _compare(solve, ["length_constraints:number_words"],
                   [{"num_words": 5, "relation": "at least"}], r)
    assert led["results"][0]["observed"] == 5 and led["results"][0]["pass"] is True
    led2 = _compare(solve, ["length_constraints:number_words"],
                    [{"num_words": 5, "relation": "less than"}], r)
    assert led2["results"][0]["pass"] is False
    # 标点附着不切词：空白切分口径
    r2 = "Hello, world! Runs."
    led3 = _compare(solve, ["length_constraints:number_words"],
                    [{"num_words": 3, "relation": "at least"}], r2)
    assert led3["results"][0]["observed"] == 3


def check_end_phrase(solve):
    led = _compare(solve, ["format:end_phrase"],
                   [{"end_phrase": "Is there anything else?"}],
                   'Any response body here. Is there anything else?')
    assert led["results"][0]["pass"] is True
    # 尾部引号被剥
    led2 = _compare(solve, ["format:end_phrase"],
                    [{"end_phrase": "the end"}],
                    'Some text. "The End"')
    assert led2["results"][0]["pass"] is True
    # 中途出现不算
    led3 = _compare(solve, ["format:end_phrase"],
                    [{"end_phrase": "the end"}],
                    "The end is near. Not quite.")
    assert led3["results"][0]["pass"] is False


def check_bullets(solve):
    r = "* first\n* second\nplain line\n- not a bullet\n* third"
    led = _compare(solve, ["detectable_format:number_bullet_lists"],
                   [{"num_bullets": 3}], r)
    assert led["results"][0]["observed"] == 3 and led["results"][0]["pass"] is True
    led2 = _compare(solve, ["detectable_format:number_bullet_lists"],
                    [{"num_bullets": 4}], r)
    assert led2["results"][0]["pass"] is False


def check_combined_all_pass(solve):
    r = ("Roses are red.\n* item one\n* item two\n"
         "Violets are blue. No secrets here.")
    led = _compare(solve, [
        "length_constraints:number_words",
        "detectable_format:number_bullet_lists",
        "keywords:forbidden",
    ], [
        {"num_words": 10, "relation": "at least"},
        {"num_bullets": 2},
        {"forbidden_words": ["secret"]},
    ], r)
    assert led["all_pass"] is False  # forbidden 命中 'secrets'（子串口径）
    led2 = _compare(solve, [
        "length_constraints:number_words",
        "detectable_format:number_bullet_lists",
        "keywords:forbidden",
    ], [
        {"num_words": 10, "relation": "at least"},
        {"num_bullets": 2},
        {"forbidden_words": ["hidden"]},
    ], r)
    assert led2["all_pass"] is True


def check_oracle_random(solve):
    import random
    rng = random.Random(55)
    ids = ["keywords:existence", "keywords:forbidden", "keywords:frequency",
           "length_constraints:number_words", "format:end_phrase",
           "detectable_format:number_bullet_lists"]
    words = ["alpha", "beta", "gamma", "delta", "end"]
    for _ in range(120):
        n = rng.randint(1, 4)
        ids_r = [rng.choice(ids) for _ in range(n)]
        kws = []
        for iid in ids_r:
            if iid == "keywords:existence":
                kws.append({"keywords": [rng.choice(words) for _ in range(rng.randint(1, 3))]})
            elif iid == "keywords:forbidden":
                kws.append({"forbidden_words": [rng.choice(words)]})
            elif iid == "keywords:frequency":
                kws.append({"keyword": rng.choice(words), "frequency": rng.randint(0, 3),
                            "relation": rng.choice(["at least", "less than"])})
            elif iid == "length_constraints:number_words":
                kws.append({"num_words": rng.randint(1, 12),
                            "relation": rng.choice(["at least", "less than"])})
            elif iid == "format:end_phrase":
                kws.append({"end_phrase": rng.choice(["done.", "The end"])})
            else:
                kws.append({"num_bullets": rng.randint(1, 3)})
        body = " ".join(rng.choice(words + ["Some", "text", "here."])
                        for _ in range(rng.randint(3, 15)))
        led = solve(ids_r, kws, body)
        ref = _py(ids_r, kws, body)
        assert led == ref


def check_invalid_inputs(solve):
    bad = [
        lambda: solve([], [], "text"),
        lambda: solve(["unknown:id"], [{}], "text"),
        lambda: solve(["keywords:existence"], [{}], "text"),
        lambda: solve(["keywords:existence", "keywords:forbidden"],
                      [{"keywords": ["a"]}], "text"),
        lambda: solve(["keywords:frequency"],
                      [{"keyword": "x", "frequency": 1, "relation": "equal"}], "text"),
        lambda: solve(["keywords:frequency"],
                      [{"keyword": "x", "frequency": -1, "relation": "at least"}], "text"),
        lambda: solve(["length_constraints:number_words"],
                      [{"num_words": 0, "relation": "at least"}], "text"),
        lambda: solve(["format:end_phrase"], [{"end_phrase": ""}], "text"),
        lambda: solve(["detectable_format:number_bullet_lists"],
                      [{"num_bullets": 0}], "text"),
        lambda: solve(["format:end_phrase"], [{"end_phrase": "x"}], 42),
    ]
    for fn in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
