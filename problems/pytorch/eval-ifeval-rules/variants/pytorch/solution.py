import re


def solve(instruction_ids, kwargs_list, response):
    if not isinstance(response, str):
        raise ValueError("response must be a string")
    if not isinstance(instruction_ids, (list, tuple)) or len(instruction_ids) == 0:
        raise ValueError("instruction_ids must be a non-empty list")
    if len(instruction_ids) != len(kwargs_list):
        raise ValueError("instruction_ids/kwargs_list length mismatch")
    valid_ids = {
        "keywords:existence",
        "keywords:forbidden",
        "keywords:frequency",
        "length_constraints:number_words",
        "format:end_phrase",
        "detectable_format:number_bullet_lists",
    }
    for iid in instruction_ids:
        if iid not in valid_ids:
            raise ValueError(f"unknown instruction id: {iid}")
    if not all(isinstance(k, dict) for k in kwargs_list):
        raise ValueError("kwargs_list entries must be dicts")

    results = []
    for iid, kw in zip(instruction_ids, kwargs_list):
        if iid == "keywords:existence":
            kws = kw.get("keywords", [])
            if not isinstance(kws, (list, tuple)) or len(kws) == 0:
                raise ValueError("keywords:existence requires non-empty keywords")
            hits = sum(1 for k in kws if k.lower() in response.lower())
            results.append({"instruction_id": iid, "observed": hits,
                            "pass": hits == len(kws)})
        elif iid == "keywords:forbidden":
            kws = kw.get("forbidden_words", [])
            if not isinstance(kws, (list, tuple)) or len(kws) == 0:
                raise ValueError("keywords:forbidden requires non-empty forbidden_words")
            hits = sum(1 for k in kws if k.lower() in response.lower())
            results.append({"instruction_id": iid, "observed": hits, "pass": hits == 0})
        elif iid == "keywords:frequency":
            k = kw.get("keyword")
            if not isinstance(k, str) or not k:
                raise ValueError("keywords:frequency requires a keyword")
            n = kw.get("frequency")
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ValueError("frequency must be a non-negative int")
            rel = kw.get("relation")
            if rel not in ("at least", "less than"):
                raise ValueError("relation must be 'at least' or 'less than'")
            cnt = len(re.findall(r"\b" + re.escape(k) + r"\b", response.lower()))
            ok = cnt >= n if rel == "at least" else cnt < n
            results.append({"instruction_id": iid, "observed": cnt, "pass": ok})
        elif iid == "length_constraints:number_words":
            n = kw.get("num_words")
            if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
                raise ValueError("num_words must be a positive int")
            rel = kw.get("relation")
            if rel not in ("at least", "less than"):
                raise ValueError("relation must be 'at least' or 'less than'")
            cnt = len(response.split())
            ok = cnt >= n if rel == "at least" else cnt < n
            results.append({"instruction_id": iid, "observed": cnt, "pass": ok})
        elif iid == "format:end_phrase":
            ep = kw.get("end_phrase")
            if not isinstance(ep, str) or not ep:
                raise ValueError("end_phrase must be a non-empty string")
            ok = response.strip().rstrip('"').lower().endswith(ep.lower())
            results.append({"instruction_id": iid, "observed": ok, "pass": ok})
        else:  # detectable_format:number_bullet_lists
            nb = kw.get("num_bullets")
            if isinstance(nb, bool) or not isinstance(nb, int) or nb <= 0:
                raise ValueError("num_bullets must be a positive int")
            cnt = sum(1 for line in response.splitlines() if line.startswith("* "))
            results.append({"instruction_id": iid, "observed": cnt, "pass": cnt >= nb})

    return {
        "results": results,
        "all_pass": all(r["pass"] for r in results),
    }
