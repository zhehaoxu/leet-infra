"""Step-DPO（arXiv:2406.18629）步级偏好对构造账 -- 测试用例（期望值由官方
data_pipeline 三脚本 exec 沙箱实测钉死：stub openai/jsonlines/tqdm，
GPT-4o 调用剥除、筛选/解析/配对全入账）."""

QWEN_TMPL = ("<|im_start|>user\n{q}\nPlease reason step by step, and put your "
             "final answer within \\boxed{{}}.<|im_end|>\n"
             "<|im_start|>assistant\n{c}")
ALPACA_TMPL = ("### Instruction:\n{q}\n\n### Response:\n"
               "Let's think step by step.\n{c}")

Q1 = "What is 2+2?"
Q3 = "Compute the area of a triangle with diagram and [asy] draw((0,0)); [/asy]"
Q2 = "Solve x^2=4 with diagram shown"


def qwen_prompt(q, c):
    return QWEN_TMPL.format(q=q, c=c)


def alpaca_prompt(q, c):
    return ALPACA_TMPL.format(q=q, c=c)


def make_items():
    """官方 locate 筛选沙箱的 8 条预测项（result=True 的 i1 会被过滤）。"""
    return [
        {"result": False, "prompt": qwen_prompt(Q1, "We add 2 and 2.\n\nThe answer is 4."),
         "completion": "We add 2 and 2.\n\nThe answer is 4.",
         "gt_output": "2+2=4.\n\nThe answer is 4.", "prompt_answer": "4",
         "type": "gsm8k"},
        {"result": True, "prompt": qwen_prompt("ignored", "x"), "completion": "x",
         "gt_output": "x", "prompt_answer": "x", "type": "gsm8k"},
        {"result": False, "prompt": qwen_prompt(Q2, "Using the diagram.\nWrong."),
         "completion": "Using the diagram.\nWrong.",
         "gt_output": "x=2", "prompt_answer": "2", "type": "math"},
        {"result": False, "prompt": qwen_prompt(Q3, "Half of base times height.\n3."),
         "completion": "Half of base times height.\n3.",
         "gt_output": "6", "prompt_answer": "6", "type": "math"},
        {"result": False, "prompt": qwen_prompt(Q1, "2+2=5.\nThe answer is 5."),
         "completion": "2+2=5.\nThe answer is 5.",
         "gt_output": "4", "prompt_answer": "4", "type": "gsm8k"},
        {"result": False, "prompt": qwen_prompt(Q1, "Wrong again."),
         "completion": "Wrong again.", "gt_output": "4", "prompt_answer": "4",
         "type": "gsm8k"},
        {"result": False, "prompt": qwen_prompt("Odd question", "Nope."),
         "completion": "Nope.", "gt_output": "7", "prompt_answer": "7",
         "type": "math"},
        {"result": False, "prompt": qwen_prompt("Visited question", "Nope2."),
         "completion": "Nope2.", "gt_output": "8", "prompt_answer": "8",
         "type": "math"},
    ]


# ---------- mode="collect"：locate_error_by_gpt4.main 筛选账 ----------

def check_collect_basic(solve):
    res = solve(mode="collect", items=make_items(), prompt_style="qwen2-boxed",
                n_groups=2, remainder=0, visited=[6])
    # idx 是过滤后 pred_data 下标：0(Q1 首次) 放行；1(Q2) diagram 门 skip；
    # 2(Q3 asy 题) 放行；3(Q1 二次) 分片 3%2=1 skip（但 question2cnt 已增）；
    # 4(Q1 三次) 分片通过但门控 2 > 1 skip；5 分片 skip；6 visited skip
    sel = res["selected"]
    assert [s["idx"] for s in sel] == [0, 2], res
    # completion 补 "Step 1: " 头 + "\n\n" -> "\n" 折叠
    assert sel[0]["incorrect"] == "Step 1: We add 2 and 2.\nThe answer is 4.", sel
    assert sel[0]["folded_solution"] == "2+2=4.\nThe answer is 4.", sel
    # asy 题通过（有 [asy] 代码不跳）
    assert sel[1]["question"] == Q3, res
    assert res["cnt"] == 2


def check_collect_gates(solve):
    items = make_items()
    # diagram 门：Q2（含 diagram 无 asy）被跳过；per-question > 门严格：
    # Q1 第 2 次（idx3）仍放行（1 > 1 False），第 3 次（idx4）skip（2 > 1）
    res = solve(mode="collect", items=items, n_groups=1, visited=[6])
    assert [s["idx"] for s in res["selected"]] == [0, 2, 3, 5], res
    # question2cnt：diagram 项在增量前 continue 不计数；visited 项计数
    assert res["question2cnt"] == {Q1: 2, Q3: 1, "Odd question": 1,
                                   "Visited question": 1}, res
    # visited 项也耗 question2cnt 名额（增量在分片/visited 检查之前）：
    # visited=[3] 时 Q1 的 idx4 变成 2 > 1 被 skip
    res2 = solve(mode="collect", items=items, n_groups=1, visited=[3, 6])
    assert [s["idx"] for s in res2["selected"]] == [0, 2, 5], res2
    # 分片：n_groups=2 remainder=1 -> 奇数 idx 通过（idx1 是 diagram 门 skip）
    res3 = solve(mode="collect", items=items, n_groups=2, remainder=1,
                 visited=[6])
    assert [s["idx"] for s in res3["selected"]] == [3, 5], res3
    # max_count_total 截断：先保存再检查，cnt 达到即 break
    res4 = solve(mode="collect", items=items, n_groups=1, visited=[],
                 max_count_total=2)
    assert [s["idx"] for s in res4["selected"]] == [0, 2], res4
    assert res4["cnt"] == 2


def check_collect_alpaca(solve):
    items = [{"result": False,
              "prompt": alpaca_prompt("Alpaca q?", "Bad.\nThe answer is 9."),
              "completion": "Bad.\nThe answer is 9.", "gt_output": "ok.\n\nfine",
              "prompt_answer": "1", "type": "math"}]
    res = solve(mode="collect", items=items, prompt_style="alpaca")
    assert len(res["selected"]) == 1
    assert res["selected"][0]["question"] == "Alpaca q?", res
    assert res["selected"][0]["incorrect"] == "Step 1: Bad.\nThe answer is 9."
    assert res["selected"][0]["folded_solution"] == "ok.\nfine"


# ---------- mode="locate"：prepare_for_correction.main 解析账 ----------

GOOD_DECISION = "Step 1 is fine.\nFinal Decision:\nStep 1: correct; Step 2: incorrect\n"
NEUTRAL_DECISION = "Let me see.\nFinal Decision:\nStep 1: correct; Step 2: neutral; Step 3: incorrect\n"
SUMMARY_DECISION = ("Final decision: step 9: incorrect\n"
                    "Summary decision: step 2: incorrect\n")

PREFIX_A = "Let's think step by step.\nStep 1: We add 2 and 2.\n"
CONT_A = "Step 2: We get 5.\nStep 3: The answer is 5."


def check_locate_basic(solve):
    common = dict(full_prompt=qwen_prompt(Q1, PREFIX_A), completion=CONT_A,
                  prompt_style="qwen2-boxed", gt_output="4",
                  prompt_answer="4", item_type="gsm8k")
    # 三种 decision 头同解析：final decision / summary decision / neutral 前置
    for decision in (GOOD_DECISION, NEUTRAL_DECISION, SUMMARY_DECISION):
        res = solve(mode="locate", gpt4_output=decision, path="f0", **common)
        assert res["valid"] is True, (decision, res)
        assert res["step_num"] == 2, res
        assert res["instruction"] == Q1, res
        # prefix 截到并含 "Step 2:" 标头，Let's think 被剥
        assert res["prefix"] == "Step 1: We add 2 and 2.\nStep 2:", res
        assert res["output"] == " We get 5.\nStep 3: The answer is 5.", res
        assert res["ori_filepath"] == "f0", res
    # neutral-only 判定也通过（官方 any([neutral, incorrect])），
    # step_num 取 split("neutral")[0] 截出的步
    res = solve(mode="locate", gpt4_output="Final decision: step 2: neutral\n",
                **common)
    assert res["valid"] is True and res["step_num"] == 2, res
    # path 缺省 "n/a"
    res = solve(mode="locate", gpt4_output=GOOD_DECISION, **common)
    assert res["ori_filepath"] == "n/a", res
    # 透传字段逐项钉（gt_output/answer/type 原样透传）
    res = solve(mode="locate", gpt4_output=GOOD_DECISION, path="f0", **common)
    assert res["gt_output"] == "4" and res["answer"] == "4", res
    assert res["type"] == "gsm8k", res


def check_locate_invalid(solve):
    common = dict(full_prompt=qwen_prompt(Q1, PREFIX_A), completion=CONT_A,
                  prompt_style="qwen2-boxed", gt_output="4",
                  prompt_answer="4", item_type="gsm8k")
    # no_keyword：判定无 neutral/incorrect
    res = solve(mode="locate", gpt4_output="All correct. Final Decision:\nStep 1: correct\n", **common)
    assert res == {"valid": False, "reason": "no_keyword"}, res
    # ambiguous_step：关键词前两个 "step" token
    res = solve(mode="locate", gpt4_output="Final decision: step 2 and step 4: incorrect\n", **common)
    assert res == {"valid": False, "reason": "ambiguous_step"}, res
    # bad_int：step 非数字
    res = solve(mode="locate", gpt4_output="Final decision: step x: incorrect\n", **common)
    assert res == {"valid": False, "reason": "bad_int"}, res
    # no_step_marker：completion_full 无 "Step 5:"（官方静默丢的那类）
    res = solve(mode="locate", gpt4_output="Final decision: step 5: incorrect\n", **common)
    assert res == {"valid": False, "reason": "no_step_marker"}, res
    # 小数 step：int("2.5") 抛 ValueError -> bad_int
    res = solve(mode="locate", gpt4_output="Final decision: step 2.5: incorrect\n", **common)
    assert res == {"valid": False, "reason": "bad_int"}, res


def check_locate_replace_quirk(solve):
    # 官方 output = completion_full.replace(prefix_out, "") 是全局替换：
    # 后文重复的 "Step 2:" 段被一起删
    res = solve(mode="locate", gpt4_output="Final decision: step 2: incorrect\n",
                full_prompt=qwen_prompt(Q1, ""),
                completion="Step 2: A\nand again Step 2: B",
                prompt_style="qwen2-boxed", gt_output="4",
                prompt_answer="4", item_type="math")
    assert res["valid"] is True, res
    assert res["prefix"] == "Step 2:", res
    assert res["output"] == " A\nand again  B", res


def check_locate_alpaca(solve):
    prefix_al = "Let's think step by step.\nStep 1: Half of base times height.\n"
    res = solve(mode="locate", gpt4_output="Final decision: step 2: incorrect\n",
                full_prompt=alpaca_prompt("Area?", prefix_al),
                completion="Step 2: We get 3.\nThe answer is 3.",
                prompt_style="alpaca", gt_output="6", prompt_answer="6",
                item_type="math")
    assert res["valid"] is True, res
    assert res["instruction"] == "Area?", res
    assert res["prefix"] == "Step 1: Half of base times height.\nStep 2:", res
    assert res["output"] == " We get 3.\nThe answer is 3.", res


# ---------- mode="pair"：generate_dataset.main 对构造账 ----------

CORR_PREFIX = "Step 1: We add 2 and 2."


def make_pair_inputs():
    # 官方 correction prompt 的 assistant 位带 Let's think 头（identifier 剥之）
    corrections = [
        # 中间有非 "Step " 开头的行：首步切分是 split("\nStep ")[0]（含 sub note）
        {"result": True,
         "prompt": qwen_prompt(Q1, "Let's think step by step.\n" + CORR_PREFIX),
         "completion": "Step 2: We get 5 wrongly.\nsub note\nStep 3: The answer is 5."},
        # result=False 的 correction 不进组
        {"result": False,
         "prompt": qwen_prompt(Q1, "Let's think step by step.\n" + CORR_PREFIX),
         "completion": "Step 2: Fake correction."},
        {"result": True,
         "prompt": qwen_prompt(Q1, "Let's think step by step.\n" + CORR_PREFIX),
         "completion": "Step 2: We get 5 wrongly.\nStep 4: alt tail."},
        {"result": True, "prompt": qwen_prompt("Other q", "Step 9:"),
         "completion": "Step 10: unmatched"},
    ]
    sources = [
        {"instruction": Q1, "prefix": CORR_PREFIX,
         "output": "Step 1: We add 2 and 2.\nnote here\nStep 2: We get 5.\nThe answer is 5.",
         "type": "gsm8k", "answer": "4"},
        {"instruction": Q1, "prefix": CORR_PREFIX,
         "output": "Step 1: Different first.\nStep 3: Six.",
         "type": "gsm8k", "answer": "4"},
        {"instruction": "No such question", "prefix": "Step 1: nothing",
         "output": "Step 1: nothing\nStep 2: y", "type": "math", "answer": "0"},
    ]
    return corrections, sources


def check_pair_basic(solve):
    corrections, sources = make_pair_inputs()
    res = solve(mode="pair", corrections=corrections, sources=sources)
    # 同 identifier 两行 x 组内 2 条 correction（首步不同、都不去重）= 4 条
    assert len(res["items"]) == 4, res
    assert res["invalid_cnt"] == 1, res
    # cnt = 每行对组内 2 条 correction 的尝试（行2 重复计）= 4
    assert res["cnt"] == 4, res
    first = res["items"][0]
    assert first["dataset"] == "gsm8k" and first["prompt"] == Q1
    # prefix 加回 Let's think 头
    assert first["prefix"] == "Let's think step by step.\n" + CORR_PREFIX, res
    # chosen = correction 续写的首步（"\nStep " 之前，含中间非 Step 行）
    assert first["chosen"] == "Step 2: We get 5 wrongly.\nsub note", res
    assert first["original_chosen"] == \
        "Step 2: We get 5 wrongly.\nsub note\nStep 3: The answer is 5.", res
    # corr1 的首步（无 sub note）与 corr0 不同 -> 不被去重，也产 pair
    assert res["items"][1]["chosen"] == "Step 2: We get 5 wrongly.", res
    # rejected = source 行 output 的首步（"\nStep " 切分，含中间 note 行）
    assert [it["rejected"] for it in res["items"]] == [
        "Step 1: We add 2 and 2.\nnote here", "Step 1: We add 2 and 2.\nnote here",
        "Step 1: Different first.", "Step 1: Different first."], res
    assert first["answer"] == "4", res


def check_pair_dedup_only_chosen(solve):
    corrections, sources = make_pair_inputs()
    # rejected 不去重：同 rejected 出现在两条 pair 里也各保留
    res = solve(mode="pair", corrections=corrections, sources=sources)
    assert res["items"][0]["rejected"] == res["items"][0]["rejected"]
    # 去重键是 chosen_first_step：换成不同首步的 correction 不被去重
    corrections2 = [
        {"result": True, "prompt": qwen_prompt(Q1, CORR_PREFIX),
         "completion": "Step 2: Alpha.\nStep 3: x"},
        {"result": True, "prompt": qwen_prompt(Q1, CORR_PREFIX),
         "completion": "Step 2: Beta.\nStep 3: y"},
    ]
    res2 = solve(mode="pair", corrections=corrections2, sources=sources[:1])
    assert len(res2["items"]) == 2, res2
    assert [it["chosen"] for it in res2["items"]] == ["Step 2: Alpha.",
                                                      "Step 2: Beta."]
    # cnt 计数在去重之前：两条都尝试，第二条被去重但 cnt 仍 +1
    corrections3 = [
        {"result": True, "prompt": qwen_prompt(Q1, CORR_PREFIX),
         "completion": "Step 2: Alpha.\nStep 3: x"},
        {"result": True, "prompt": qwen_prompt(Q1, CORR_PREFIX),
         "completion": "Step 2: Alpha.\nStep 3: duplicate first step"},
    ]
    res3 = solve(mode="pair", corrections=corrections3, sources=sources[:1])
    assert len(res3["items"]) == 1 and res3["cnt"] == 2, res3


# ---------- 参数校验 ----------

def check_args_validation(solve):
    bad = [("mode", lambda: solve(mode="train")),
           ("items", lambda: solve(mode="collect", items=None)),
           ("style", lambda: solve(mode="collect", items=[],
                                   prompt_style="chatml")),
           ("n_groups", lambda: solve(mode="collect", items=[], n_groups=0)),
           ("remainder", lambda: solve(mode="collect", items=[], remainder=-1)),
           ("max_perq", lambda: solve(mode="collect", items=[],
                                      max_count_per_question=-1)),
           ("max_total", lambda: solve(mode="collect", items=[],
                                       max_count_total=1.5)),
           ("visited", lambda: solve(mode="collect", items=[],
                                     visited=["3"])),
           ("gpt4_output", lambda: solve(mode="locate", gpt4_output=None,
                                         full_prompt="", completion="")),
           ("corrections", lambda: solve(mode="pair", corrections=None,
                                         sources=[])),
           ("pair_style", lambda: solve(mode="pair", corrections=[],
                                        sources=[], prompt_style="alp")),
           ("locate_style", lambda: solve(mode="locate", gpt4_output="x",
                                          full_prompt="", completion="",
                                          prompt_style="chatml"))]
    for name, fn in bad:
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError(f"{name} must raise ValueError")
