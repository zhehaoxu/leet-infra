"""Step-DPO（arXiv:2406.18629, dvlab-research/Step-DPO）步级偏好对构造账。

Step-DPO 的核心主张：长 CoT 的错误往往只在某一步，对整条序列做 DPO 会把
正确的后续步也一起惩罚；正确做法是**从第一步错误处截断**，只对「错误步
的续写」构造 chosen/rejected 偏好对。官方 data_pipeline 三段确定性账
（GPT-4o 调用剥除，筛选/解析/配对全入参），本题逐段复刻：

solve(mode=...) 三个入口（官方逐条钉死）：

    mode="collect"（locate_error_by_gpt4.main 的筛选账）：
        items(result/prompt/completion/gt_output) + prompt_style +
        n_groups/remainder/max_count_per_question/max_count_total +
        visited -> {"selected", "question2cnt", "cnt"}。
        只收 **result 为 False** 的错误项（idx 是过滤后 pred_data 的下标）；
        question 从 prompt 按 style 拆出；question2cnt 门控是 **>（严格）**
        ——同题最多放行 max_count_per_question + 1 次，先查后增；
        question 含 "diagram" 且不含 "asy"（无 Asymptote 代码的图题）
        整条跳过；idx % n_groups != remainder 的分片跳过；idx 已在
        visited 的跳过；放行项 completion 补 **"Step 1: "** 头、
        gt_output 与补头后的 answer 各做 **"\\n\\n" -> "\\n"** 折叠进
        GPT-4 prompt 成分；**先保存再检查 cnt >= max_count_total 截断**。

    mode="locate"（prepare_for_correction.main 的单条解析账）：
        gpt4_output + full_prompt + completion + prompt_style (+gt_output/
        prompt_answer/type/path) -> {"valid": bool, ...}。
        correctness = gpt4_output.lower() 的 **split("final decision")[-1]
        .split("summary decision:")[-1]**.strip()（两种 decision 头都认，
        取最后一段）；不含 neutral/incorrect -> invalid(no_keyword)；
        step_num 段 = split("neutral")[0].split("incorrect")[0] 再取
        split("\\n")[-1].split(";")[-1]——**先 neutral 后 incorrect 的切分
        序**使两个关键词谁在前都能截到第一个错误步；段内 count("step") > 1
        -> invalid(ambiguous_step)；split("step")[-1].split(":")[0] 的
        int() 失败 -> invalid(bad_int)；completion_full = prompt 拆出的
        prefix + completion（**prefix 是 prompt 里已有的已生成部分、
        completion 是续写**）；不含 "Step {n}:" -> invalid(no_step_marker)
        （官方静默丢）；prefix_out = split("Step {n}:")[0] + "Step {n}:"，
        output = completion_full.replace(prefix_out, "")——**replace 全局
        替换**：后文重复出现的 "Step {n}:" 段被一起删掉；返回 prefix 剥
        "Let's think step by step.\\n"。

    mode="pair"（generate_dataset.main 的对构造账）：
        corrections(result/prompt/completion) + sources(instruction/prefix/
        output/type/answer) + prompt_style -> {"items", "invalid_cnt", "cnt"}。
        corrections 按 identifier（prompt + "||" + 剥 Let's think 的 prefix）
        分组；source 行 identifier 失配 -> invalid_cnt += 1 跳过；命中行的
        每条 correction：chosen = correction 续写、rejected = source 行的
        output（原错误续写），**首步切分是 split("\\nStep ")[0]**（取
        "\\nStep " 之前的全部），chosen_first_step 已见 -> **只跳过该条
        correction**（去重键是 chosen 首步；rejected 不去重）；cnt 计数
        在去重检查**之前**（含被去重的尝试）；产出 prefix 重新加回
        "Let's think step by step.\\n" 头、original_chosen 保留全文。
"""
import re

MODES = ("collect", "locate", "pair")
STYLES = ("qwen2-boxed", "alpaca")
LET_THINK = "Let's think step by step.\n"


def _split_prompt(full_prompt, style):
    """官方 prompt 拆分（两 style），返回 (question, prefix)。"""
    if style == "alpaca":
        question = full_prompt.split("### Instruction:")[1] \
            .split("### Response:")[0].strip()
        prefix = full_prompt.split("### Response:")[-1].lstrip()
    else:
        question = full_prompt.split("<|im_start|>user\n")[1] \
            .split("\nPlease reason step by step, and put your final answer "
                   "within \\boxed{}.<|im_end|>")[0].strip()
        prefix = full_prompt.split("<|im_start|>assistant\n")[-1].lstrip()
    return question, prefix


def solve(mode="collect", items=None, corrections=None, sources=None,
          gpt4_output=None, full_prompt=None, completion=None,
          gt_output=None, prompt_answer=None, item_type=None, path=None,
          prompt_style="qwen2-boxed", n_groups=1, remainder=0,
          max_count_per_question=1, max_count_total=10000, visited=None):
    """统一入口（判题协议）：按 mode 分发到 collect / locate / pair。"""
    if mode == "collect":
        if not isinstance(items, list):
            raise ValueError("items must be a list")
        if prompt_style not in STYLES:
            raise ValueError(f"prompt_style must be one of {STYLES}")
        if not isinstance(n_groups, int) or isinstance(n_groups, bool) \
                or n_groups < 1:
            raise ValueError("n_groups must be a positive int")
        if not isinstance(remainder, int) or isinstance(remainder, bool) \
                or remainder < 0:
            raise ValueError("remainder must be a non-negative int")
        if not isinstance(max_count_per_question, int) \
                or isinstance(max_count_per_question, bool) \
                or max_count_per_question < 0:
            raise ValueError("max_count_per_question must be a non-negative int")
        if not isinstance(max_count_total, int) \
                or isinstance(max_count_total, bool) or max_count_total < 0:
            raise ValueError("max_count_total must be a non-negative int")
        if visited is None:
            visited = []
        if not isinstance(visited, list) or \
                any(not isinstance(v, int) or isinstance(v, bool)
                    for v in visited):
            raise ValueError("visited must be a list of ints")
        pred_data = [it for it in items if not it["result"]]
        selected = []
        question2cnt = {}
        cnt = 0
        for idx, pred in enumerate(pred_data):
            question, _ = _split_prompt(pred["prompt"], prompt_style)
            # skip the invalid questions without diagram
            if "diagram" in question and "asy" not in question:
                continue
            if question in question2cnt and \
                    question2cnt[question] > max_count_per_question:
                continue
            if question not in question2cnt:
                question2cnt[question] = 0
            question2cnt[question] += 1
            # skip other threads
            if idx % n_groups != remainder:
                continue
            # skip the visited questions
            if idx in visited:
                continue
            incorrect = "Step 1: " + pred["completion"]
            selected.append({
                "idx": idx,
                "question": question,
                "incorrect": incorrect.replace("\n\n", "\n"),
                "folded_solution": pred["gt_output"].replace("\n\n", "\n"),
            })
            cnt += 1
            if cnt >= max_count_total:
                break
        return {"selected": selected, "question2cnt": question2cnt,
                "cnt": cnt}
    if mode == "locate":
        if not isinstance(gpt4_output, str):
            raise ValueError("gpt4_output must be a str")
        if not isinstance(full_prompt, str):
            raise ValueError("full_prompt must be a str")
        if not isinstance(completion, str):
            raise ValueError("completion must be a str")
        if prompt_style not in STYLES:
            raise ValueError(f"prompt_style must be one of {STYLES}")
        correctness = gpt4_output.lower()
        correctness = correctness.split("final decision")[-1] \
            .split("summary decision:")[-1].strip()
        if not any(x in correctness for x in ["neutral", "incorrect"]):
            return {"valid": False, "reason": "no_keyword"}
        step_num = correctness.split("neutral")[0].split("incorrect")[0]
        step_num = step_num.split("\n")[-1].split(";")[-1]
        if step_num.count("step") > 1:
            return {"valid": False, "reason": "ambiguous_step"}
        step_num = step_num.split("step")[-1].split(":")[0]
        try:
            step_num = int(step_num.strip())
        except ValueError:
            return {"valid": False, "reason": "bad_int"}
        question, prefix = _split_prompt(full_prompt, prompt_style)
        completion_full = prefix + completion
        marker = "Step {}:".format(step_num)
        if completion_full.count(marker) == 0:
            return {"valid": False, "reason": "no_step_marker"}
        prefix_out = completion_full.split(marker)[0] + marker
        output = completion_full.replace(prefix_out, "")
        return {
            "valid": True,
            "instruction": question,
            "prefix": prefix_out.replace(LET_THINK, ""),
            "output": output,
            "step_num": step_num,
            "gt_output": gt_output,
            "answer": prompt_answer,
            "type": item_type,
            "ori_filepath": path if path is not None else "n/a",
        }
    if mode == "pair":
        if not isinstance(corrections, list) or not isinstance(sources, list):
            raise ValueError("corrections and sources must be lists")
        if prompt_style not in STYLES:
            raise ValueError(f"prompt_style must be one of {STYLES}")
        identifier2items = {}
        for item in corrections:
            if item["result"]:
                question, prefix = _split_prompt(item["prompt"], prompt_style)
                prefix = prefix.replace(LET_THINK, "")
                identifier = question + "||" + prefix
                identifier2items.setdefault(identifier, []).append(item)
        new_items = []
        invalid_cnt = 0
        cnt = 0
        for line in sources:
            identifier = line["instruction"] + "||" + line["prefix"]
            if identifier not in identifier2items:
                invalid_cnt += 1
                continue
            items = identifier2items[identifier]
            visited_chosen = set()
            for item in items:
                cnt += 1
                chosen = item["completion"]
                rejected = line["output"]
                chosen_first_step = chosen.split("\nStep ")[0]
                rejected_first_step = rejected.split("\nStep ")[0]
                if chosen_first_step in visited_chosen:
                    continue
                visited_chosen.add(chosen_first_step)
                new_items.append({
                    "dataset": line["type"],
                    "prompt": line["instruction"],
                    "prefix": LET_THINK + line["prefix"],
                    "chosen": chosen_first_step,
                    "rejected": rejected_first_step,
                    "original_chosen": chosen,
                    "answer": line["answer"],
                })
        return {"items": new_items, "invalid_cnt": invalid_cnt, "cnt": cnt}
    raise ValueError(f"Unsupported mode {mode!r}")
